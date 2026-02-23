"""
PromptAnalyzer — the main public interface for promptgrad.

Usage
-----
    from promptgrad import PromptAnalyzer

    analyzer = PromptAnalyzer()
    report = analyzer.analyze("Summarize the document in three sentences.")

    print(report.robustness_score)
    print(report.top_sensitive_tokens())
    analyzer.plot(report)
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from .embeddings import EmbeddingEngine, get_engine
from .metrics import (
    RobustnessReport,
    cosine_similarity_distribution,
    detect_warnings,
    embedding_entropy,
    per_strategy_scores,
    robustness_score,
    token_importance_scores,
)
from .perturbations import (
    ALL_PERTURBATIONS,
    PERTURBATION_REGISTRY,
    Perturbation,
    PerturbationResult,
)


class PromptAnalyzer:
    """
    Analyse the sensitivity of a prompt to linguistic perturbations.

    Parameters
    ----------
    embedding_backend : str
        "auto" (default) | "tfidf" | "sentence_transformers" | "openai"
    perturbations : list[str | Perturbation] | None
        Which perturbation strategies to use.  Pass ``None`` to use all
        built-in strategies, or a list of strategy names / instances.
    n_per_strategy : int
        Number of variants to generate per strategy (default 5).
    seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> from promptgrad import PromptAnalyzer
    >>> analyzer = PromptAnalyzer(embedding_backend="tfidf")
    >>> report = analyzer.analyze("List the top 5 programming languages.")
    >>> print(report.robustness_score)
    """

    def __init__(
        self,
        embedding_backend: str = "auto",
        perturbations: Optional[List[Union[str, Perturbation]]] = None,
        n_per_strategy: int = 5,
        seed: Optional[int] = 42,
        **engine_kwargs,
    ):
        self._engine: EmbeddingEngine = get_engine(embedding_backend, **engine_kwargs)
        self._n = n_per_strategy
        self._seed = seed

        if perturbations is None:
            self._perturbations = ALL_PERTURBATIONS
        else:
            resolved = []
            for p in perturbations:
                if isinstance(p, str):
                    if p not in PERTURBATION_REGISTRY:
                        raise ValueError(
                            f"Unknown perturbation {p!r}. "
                            f"Available: {list(PERTURBATION_REGISTRY)}"
                        )
                    resolved.append(PERTURBATION_REGISTRY[p])
                elif isinstance(p, Perturbation):
                    resolved.append(p)
                else:
                    raise TypeError(f"Expected str or Perturbation, got {type(p)}")
            self._perturbations = resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        prompt: str,
        output_texts: Optional[List[str]] = None,
    ) -> RobustnessReport:
        """
        Run a full sensitivity analysis on ``prompt``.

        Parameters
        ----------
        prompt : str
            The prompt to analyse.
        output_texts : list[str] | None
            Optional LLM outputs for the perturbed prompts.  When provided,
            ``output_variance`` in the report is computed from their embeddings.

        Returns
        -------
        RobustnessReport
        """
        # 1. Generate perturbations
        all_results: List[PerturbationResult] = []
        for strategy in self._perturbations:
            results = strategy.apply(prompt, n=self._n, seed=self._seed)
            all_results.extend(results)

        if not all_results:
            raise ValueError("No perturbations were generated — prompt may be too short.")

        perturbed_texts = [r.perturbed for r in all_results]

        # 2. Embed original + all perturbations together (batched)
        all_texts = [prompt] + perturbed_texts
        embeddings = self._engine.embed(all_texts)
        original_emb = embeddings[0]
        perturbed_embs = embeddings[1:]

        # 3. Core metrics
        sims = cosine_similarity_distribution(original_emb, perturbed_embs)
        shifts = np.linalg.norm(perturbed_embs - original_emb[np.newaxis, :], axis=1)
        shift_std = float(np.std(shifts))
        ent = embedding_entropy(sims)
        score = robustness_score(sims, shift_std, ent)
        per_strat = per_strategy_scores(all_results, sims)
        tok_imp = token_importance_scores(prompt, original_emb, all_results, perturbed_embs)

        # 4. Optional output variance
        output_variance: Optional[float] = None
        if output_texts:
            out_embs = self._engine.embed(output_texts)
            centroid = out_embs.mean(axis=0)
            output_variance = float(np.mean(np.linalg.norm(out_embs - centroid, axis=1) ** 2))

        # 5. Warnings
        kwargs = {
            "robustness_score": score,
            "entropy": ent,
            "embedding_shift_std": shift_std,
            "per_strategy": per_strat,
        }
        warnings = detect_warnings(kwargs)

        return RobustnessReport(
            prompt=prompt,
            robustness_score=score,
            mean_cosine_similarity=float(np.mean(sims)),
            embedding_shift_std=shift_std,
            output_variance=output_variance,
            entropy=ent,
            token_importance=tok_imp,
            per_strategy=per_strat,
            n_perturbations=len(all_results),
            warnings=warnings,
        )

    def analyze_batch(
        self,
        prompts: List[str],
        output_texts_list: Optional[List[Optional[List[str]]]] = None,
    ) -> List[RobustnessReport]:
        """Analyse a list of prompts and return a list of reports."""
        if output_texts_list is None:
            output_texts_list = [None] * len(prompts)
        return [
            self.analyze(p, ot)
            for p, ot in zip(prompts, output_texts_list)
        ]

    def compare(self, prompts: List[str]) -> dict:
        """
        Compare multiple prompts and return a ranked summary.

        Returns
        -------
        dict with keys "ranked" (list of (prompt, score) tuples, best first)
        and "reports" (dict mapping prompt → RobustnessReport).
        """
        reports = {p: self.analyze(p) for p in prompts}
        ranked = sorted(reports.items(), key=lambda x: x[1].robustness_score, reverse=True)
        return {
            "ranked": [(p, r.robustness_score) for p, r in ranked],
            "reports": reports,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        report: RobustnessReport,
        kind: str = "all",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Visualise the analysis results.

        Parameters
        ----------
        kind : "all" | "heatmap" | "importance" | "radar"
        save_path : str | None
            If given, save the figure to this path.
        show : bool
            Call ``plt.show()`` (default True; set False in notebooks).
        """
        from .viz import plot_report
        plot_report(report, kind=kind, save_path=save_path, show=show)
