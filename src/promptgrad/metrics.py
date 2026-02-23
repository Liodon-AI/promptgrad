"""
Core metrics for prompt sensitivity analysis.

All functions operate on numpy arrays and are dependency-light.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RobustnessReport:
    """
    Full sensitivity analysis report for a single prompt.

    Attributes
    ----------
    prompt : str
        The original prompt.
    robustness_score : float
        0.0 (completely unstable) → 1.0 (perfectly robust).
    mean_cosine_similarity : float
        Average cosine similarity between original and perturbed embeddings.
    embedding_shift_std : float
        Standard deviation of L2 embedding shifts across perturbations.
    output_variance : Optional[float]
        Variance of LLM output embeddings (if outputs were provided).
    entropy : float
        Shannon entropy of the cosine-similarity distribution.
    token_importance : Dict[str, float]
        Token → importance score (higher = more sensitive).
    per_strategy : Dict[str, float]
        Per-perturbation-strategy mean cosine similarity.
    n_perturbations : int
        Total number of perturbations analysed.
    warnings : List[str]
        Human-readable instability warnings.
    """

    prompt: str
    robustness_score: float
    mean_cosine_similarity: float
    embedding_shift_std: float
    output_variance: Optional[float]
    entropy: float
    token_importance: Dict[str, float]
    per_strategy: Dict[str, float]
    n_perturbations: int
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_stable(self) -> bool:
        return self.robustness_score >= 0.75

    @property
    def stability_label(self) -> str:
        if self.robustness_score >= 0.90:
            return "VERY STABLE"
        if self.robustness_score >= 0.75:
            return "STABLE"
        if self.robustness_score >= 0.50:
            return "FRAGILE"
        return "UNSTABLE"

    def top_sensitive_tokens(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return the `n` tokens with highest importance scores."""
        return sorted(self.token_importance.items(), key=lambda x: x[1], reverse=True)[:n]

    def __repr__(self) -> str:
        return (
            f"RobustnessReport(score={self.robustness_score:.3f}, "
            f"label={self.stability_label!r}, "
            f"n_perturbations={self.n_perturbations})"
        )


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def cosine_similarity_distribution(
    original_embedding: np.ndarray,
    perturbed_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between the original and each perturbation.

    Parameters
    ----------
    original_embedding : shape (D,)
    perturbed_embeddings : shape (N, D)

    Returns
    -------
    similarities : shape (N,)
    """
    orig_norm = np.linalg.norm(original_embedding)
    if orig_norm == 0:
        return np.zeros(len(perturbed_embeddings))

    o = original_embedding / orig_norm
    norms = np.linalg.norm(perturbed_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    p = perturbed_embeddings / norms
    return (p @ o).astype(np.float32)


def embedding_entropy(similarities: np.ndarray, bins: int = 20) -> float:
    """
    Shannon entropy of the cosine similarity distribution.

    A high entropy → the distribution is spread out → prompt is sensitive.
    A low entropy → responses cluster tightly → prompt is robust.
    """
    # clip to [-1, 1] and shift to [0, 2] for histogram
    vals = np.clip(similarities, -1.0, 1.0) + 1.0
    counts, _ = np.histogram(vals, bins=bins, range=(0.0, 2.0), density=False)
    probs = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)
    # avoid log(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def robustness_score(
    similarities: np.ndarray,
    shift_std: float,
    entropy: float,
) -> float:
    """
    Composite robustness score in [0, 1].

    Combines:
      - mean cosine similarity (higher = more robust)
      - normalised embedding shift std (lower = more robust)
      - normalised entropy (lower = more robust)
    """
    mean_sim = float(np.mean(similarities))
    # convert to [0, 1]
    sim_component = (mean_sim + 1) / 2.0  # cosine ∈ [-1,1] → [0,1]

    # normalise shift std with a soft cap at 1.0
    shift_component = 1.0 - min(shift_std, 1.0)

    # entropy normalisation: max entropy for `bins=20` distribution ≈ ln(20)
    max_entropy = math.log(20)
    entropy_component = 1.0 - min(entropy / max_entropy, 1.0)

    # weighted average
    score = 0.5 * sim_component + 0.3 * shift_component + 0.2 * entropy_component
    return float(np.clip(score, 0.0, 1.0))


def token_importance_scores(
    prompt: str,
    original_embedding: np.ndarray,
    perturbation_results,      # List[PerturbationResult]
    perturbed_embeddings: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate per-token importance via ablation attribution.

    For each token that was modified/deleted in perturbations, accumulate
    the cosine distance to the original.  Tokens whose removal/replacement
    causes larger shifts get higher importance scores.
    """
    tokens = prompt.split()
    importance: Dict[str, float] = {}

    sims = cosine_similarity_distribution(original_embedding, perturbed_embeddings)

    for result, sim in zip(perturbation_results, sims):
        shift = 1.0 - float(sim)  # distance ∈ [0, 2] normalised to [0, 1] approx
        token_idx = result.token_index

        # Strategy-aware attribution
        if result.strategy == "word_deletion":
            word = result.metadata.get("deleted_word", "")
        elif result.strategy == "synonym_substitution":
            word = result.metadata.get("original_word", "")
        elif token_idx is not None and token_idx < len(tokens):
            word = tokens[token_idx]
        else:
            # Spread attribution across all tokens if no specific index
            for t in set(tokens):
                importance[t] = importance.get(t, 0.0) + shift / max(len(tokens), 1)
            continue

        if word:
            importance[word] = max(importance.get(word, 0.0), shift)

    # Normalise to [0, 1]
    if importance:
        max_val = max(importance.values())
        if max_val > 0:
            importance = {k: v / max_val for k, v in importance.items()}

    return importance


def per_strategy_scores(
    perturbation_results,      # List[PerturbationResult]
    similarities: np.ndarray,
) -> Dict[str, float]:
    """Mean cosine similarity grouped by perturbation strategy."""
    buckets: Dict[str, list] = {}
    for result, sim in zip(perturbation_results, similarities):
        buckets.setdefault(result.strategy, []).append(float(sim))
    return {strategy: float(np.mean(vals)) for strategy, vals in buckets.items()}


def detect_warnings(
    report_kwargs: dict,
) -> List[str]:
    """Generate human-readable warnings from metric values."""
    warnings = []
    score = report_kwargs.get("robustness_score", 1.0)
    entropy = report_kwargs.get("entropy", 0.0)
    shift_std = report_kwargs.get("embedding_shift_std", 0.0)
    per_strategy = report_kwargs.get("per_strategy", {})

    if score < 0.5:
        warnings.append("⚠️  Prompt is UNSTABLE — small changes cause large output shifts.")
    elif score < 0.75:
        warnings.append("⚠️  Prompt is FRAGILE — consider rephrasing for more consistent results.")

    if entropy > 2.0:
        warnings.append("⚠️  High embedding entropy — the prompt's intent is ambiguous to the model.")

    if shift_std > 0.3:
        warnings.append("⚠️  High variance in embedding shifts — certain phrasings are outliers.")

    for strategy, sim in per_strategy.items():
        if sim < 0.7:
            warnings.append(
                f"⚠️  Strategy '{strategy}' causes high divergence (mean cosine = {sim:.2f})."
            )

    if not warnings:
        warnings.append("✅  Prompt appears robust across tested perturbations.")

    return warnings
