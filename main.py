"""
promptgrad — Prompt Sensitivity Analyzer
=========================================
Perturbs prompts, measures embedding shifts, computes output variance,
detects unstable instructions, and produces robustness scores + visualizations.

Requirements:
    pip install torch transformers numpy matplotlib seaborn scipy tqdm

Usage:
    python promptgrad.py --prompt "You are a helpful assistant. Answer concisely."
    python promptgrad.py --prompt "Your prompt here" --model gpt2 --n_perturbations 50
"""

import argparse
import re
import math
import random
import warnings
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import entropy as scipy_entropy
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

PERTURBATION_STRATEGIES = [
    "synonym_swap",
    "word_deletion",
    "word_insertion",
    "char_typo",
    "case_flip",
    "punctuation_change",
    "word_shuffle",
    "negation_insert",
]

SYNONYM_MAP = {
    "helpful": ["useful", "supportive", "assistive", "beneficial"],
    "concise": ["brief", "short", "terse", "succinct"],
    "answer": ["respond", "reply", "address", "explain"],
    "provide": ["give", "offer", "supply", "deliver"],
    "always": ["consistently", "invariably", "constantly", "perpetually"],
    "never": ["not ever", "at no point", "under no circumstances"],
    "important": ["critical", "essential", "vital", "crucial"],
    "good": ["great", "excellent", "solid", "fine"],
    "bad": ["poor", "terrible", "awful", "subpar"],
    "make": ["create", "produce", "generate", "build"],
    "use": ["utilize", "employ", "apply", "leverage"],
    "ensure": ["guarantee", "make sure", "confirm", "verify"],
    "assistant": ["helper", "agent", "system", "bot"],
    "user": ["person", "human", "individual", "requester"],
    "task": ["job", "assignment", "objective", "goal"],
    "output": ["result", "response", "answer", "generation"],
    "format": ["structure", "layout", "style", "form"],
    "following": ["below", "subsequent", "next", "listed"],
    "instructions": ["directions", "guidelines", "commands", "rules"],
    "context": ["background", "situation", "setting", "environment"],
}


# ─────────────────────────────────────────────
#  PERTURBATION ENGINE
# ─────────────────────────────────────────────

class PerturbationEngine:
    """Applies diverse perturbations to prompts to probe sensitivity."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    def perturb(self, prompt: str, strategy: str, intensity: float = 0.3) -> str:
        fn = getattr(self, f"_strategy_{strategy}", self._strategy_synonym_swap)
        return fn(prompt, intensity)

    def _strategy_synonym_swap(self, prompt: str, intensity: float) -> str:
        words = prompt.split()
        for i, word in enumerate(words):
            clean = word.strip(".,!?;:").lower()
            if clean in SYNONYM_MAP and random.random() < intensity:
                synonym = random.choice(SYNONYM_MAP[clean])
                # preserve trailing punctuation
                punct = re.search(r'[.,!?;:]+$', word)
                words[i] = synonym + (punct.group() if punct else "")
        return " ".join(words)

    def _strategy_word_deletion(self, prompt: str, intensity: float) -> str:
        words = prompt.split()
        if len(words) <= 3:
            return prompt
        keep = [w for w in words if random.random() > intensity * 0.5]
        return " ".join(keep) if keep else prompt

    def _strategy_word_insertion(self, prompt: str, intensity: float) -> str:
        fillers = ["please", "kindly", "always", "carefully", "properly",
                   "strictly", "exactly", "clearly", "thoroughly", "well"]
        words = prompt.split()
        result = []
        for w in words:
            result.append(w)
            if random.random() < intensity * 0.3:
                result.insert(-1, random.choice(fillers))
        return " ".join(result)

    def _strategy_char_typo(self, prompt: str, intensity: float) -> str:
        chars = list(prompt)
        keyboard_neighbors = {
            'a': 'sq', 'e': 'wr', 'i': 'uo', 'o': 'ip', 'u': 'yi',
            'n': 'bm', 's': 'ad', 't': 'ry', 'r': 'te', 'l': 'ko',
        }
        for idx in range(len(chars)):
            if chars[idx].lower() in keyboard_neighbors and random.random() < intensity * 0.15:
                neighbor = random.choice(keyboard_neighbors[chars[idx].lower()])
                chars[idx] = neighbor.upper() if chars[idx].isupper() else neighbor
        return "".join(chars)

    def _strategy_case_flip(self, prompt: str, intensity: float) -> str:
        words = prompt.split()
        for i, w in enumerate(words):
            r = random.random()
            if r < intensity * 0.3:
                words[i] = w.upper()
            elif r < intensity * 0.5:
                words[i] = w.lower()
        return " ".join(words)

    def _strategy_punctuation_change(self, prompt: str, intensity: float) -> str:
        result = list(prompt)
        for i, c in enumerate(result):
            if c in ".,!?;:" and random.random() < intensity:
                result[i] = random.choice(".,!?;:")
        return "".join(result)

    def _strategy_word_shuffle(self, prompt: str, intensity: float) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', prompt)
        shuffled = []
        for sent in sentences:
            words = sent.split()
            if len(words) > 3 and random.random() < intensity:
                # shuffle non-first, non-last words
                mid = words[1:-1]
                random.shuffle(mid)
                words = [words[0]] + mid + [words[-1]]
            shuffled.append(" ".join(words))
        return " ".join(shuffled)

    def _strategy_negation_insert(self, prompt: str, intensity: float) -> str:
        words = prompt.split()
        negations = ["not", "never", "don't", "avoid", "refrain from"]
        for i, w in enumerate(words):
            if w.lower() in ["always", "must", "should", "will", "do"] and random.random() < intensity * 0.4:
                words.insert(i + 1, random.choice(negations[:2]))
                break
        return " ".join(words)

    def batch_perturb(
        self,
        prompt: str,
        n: int = 50,
        strategies: Optional[list] = None,
        intensity: float = 0.3,
    ) -> list[str]:
        strategies = strategies or PERTURBATION_STRATEGIES
        perturbed = []
        for i in range(n):
            strategy = strategies[i % len(strategies)]
            perturbed.append(self.perturb(prompt, strategy, intensity))
        return perturbed


# ─────────────────────────────────────────────
#  EMBEDDING ANALYZER
# ─────────────────────────────────────────────

class EmbeddingAnalyzer:
    """Encodes prompts, computes cosine distances and embedding variance."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[EmbeddingAnalyzer] Loading model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=512, return_tensors="pt").to(self.device)
            out = self.model(**enc)
            # Mean pooling
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb = F.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)

    def cosine_distances(self, anchor: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
        """Returns 1 - cosine_similarity for each row in others vs anchor."""
        sims = F.cosine_similarity(anchor.unsqueeze(0), others, dim=-1)
        return 1.0 - sims

    def embedding_variance(self, embeddings: torch.Tensor) -> float:
        """Mean variance across embedding dimensions."""
        return embeddings.var(dim=0).mean().item()


# ─────────────────────────────────────────────
#  OUTPUT VARIANCE ENGINE
# ─────────────────────────────────────────────

class OutputVarianceEngine:
    """Generates token probability distributions for prompt variants."""

    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[OutputVarianceEngine] Loading LM: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_next_token_probs(self, prompt: str, top_k: int = 50) -> torch.Tensor:
        """Returns top-k token probability distribution for the next token."""
        enc = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=1024).to(self.device)
        out = self.model(**enc)
        logits = out.logits[0, -1, :]  # last token position
        probs = F.softmax(logits, dim=-1)
        top_probs, _ = probs.topk(top_k)
        return top_probs.cpu()

    def compute_output_entropy(self, probs: torch.Tensor) -> float:
        p = probs.numpy().astype(np.float64)
        p = p / p.sum()
        return float(scipy_entropy(p))

    def batch_entropy(self, prompts: list[str], top_k: int = 50) -> list[float]:
        entropies = []
        for prompt in tqdm(prompts, desc="Computing output entropy", leave=False):
            probs = self.get_next_token_probs(prompt, top_k)
            entropies.append(self.compute_output_entropy(probs))
        return entropies

    def tv_distance(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """Total variation distance between two distributions."""
        # Pad to same length
        max_len = max(len(p), len(q))
        p_pad = F.pad(p, (0, max_len - len(p)))
        q_pad = F.pad(q, (0, max_len - len(q)))
        p_pad = p_pad / p_pad.sum()
        q_pad = q_pad / q_pad.sum()
        return 0.5 * (p_pad - q_pad).abs().sum().item()


# ─────────────────────────────────────────────
#  TOKEN IMPORTANCE
# ─────────────────────────────────────────────

class TokenImportanceAnalyzer:
    """
    Measures each token's importance via leave-one-out embedding ablation.
    Importance = cosine distance between full-prompt embedding and
                 embedding with that token removed.
    """

    def __init__(self, embedding_analyzer: EmbeddingAnalyzer):
        self.emb = embedding_analyzer

    def compute_importance(self, prompt: str) -> tuple[list[str], list[float]]:
        tokens = prompt.split()
        if not tokens:
            return [], []

        # Full prompt embedding
        full_emb = self.emb.encode([prompt])  # (1, D)

        importances = []
        for i in range(len(tokens)):
            ablated = " ".join(tokens[:i] + tokens[i + 1:])
            if not ablated.strip():
                importances.append(0.0)
                continue
            ablated_emb = self.emb.encode([ablated])  # (1, D)
            dist = self.emb.cosine_distances(full_emb[0], ablated_emb)[0].item()
            importances.append(dist)

        # Normalize to [0, 1]
        max_imp = max(importances) if max(importances) > 0 else 1.0
        importances_norm = [x / max_imp for x in importances]
        return tokens, importances_norm


# ─────────────────────────────────────────────
#  ROBUSTNESS SCORER
# ─────────────────────────────────────────────

class RobustnessScorer:
    """
    Aggregates all signals into a robustness score [0, 1].
    Higher = more robust (less sensitive to perturbations).

    Score = 1 - weighted_combination(
        mean_embedding_distance,
        normalized_embedding_variance,
        normalized_entropy_variance,
        mean_tv_distance
    )
    """

    WEIGHTS = {
        "mean_embedding_distance": 0.35,
        "embedding_variance":      0.25,
        "entropy_variance":        0.20,
        "mean_tv_distance":        0.20,
    }

    def score(
        self,
        mean_emb_dist: float,
        emb_variance: float,
        entropy_variance: float,
        mean_tv: float,
    ) -> dict:
        # Normalize each signal to [0, 1] using rough empirical ranges
        norm_emb_dist = min(mean_emb_dist / 0.5, 1.0)      # 0.5 = high distance
        norm_emb_var  = min(emb_variance / 0.1, 1.0)        # 0.1 = high variance
        norm_ent_var  = min(entropy_variance / 2.0, 1.0)    # 2.0 = high entropy var
        norm_tv       = min(mean_tv / 0.5, 1.0)             # 0.5 = high TV dist

        instability = (
            self.WEIGHTS["mean_embedding_distance"] * norm_emb_dist +
            self.WEIGHTS["embedding_variance"]      * norm_emb_var  +
            self.WEIGHTS["entropy_variance"]        * norm_ent_var  +
            self.WEIGHTS["mean_tv_distance"]        * norm_tv
        )

        robustness = 1.0 - instability

        label = (
            "VERY ROBUST"  if robustness >= 0.80 else
            "ROBUST"       if robustness >= 0.65 else
            "MODERATE"     if robustness >= 0.45 else
            "FRAGILE"      if robustness >= 0.25 else
            "VERY FRAGILE"
        )

        return {
            "robustness_score": round(robustness, 4),
            "instability_score": round(instability, 4),
            "label": label,
            "components": {
                "mean_embedding_distance": round(mean_emb_dist, 4),
                "embedding_variance":      round(emb_variance, 4),
                "entropy_variance":        round(entropy_variance, 4),
                "mean_tv_distance":        round(mean_tv, 4),
            }
        }


# ─────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────

class Visualizer:
    """Produces all output visualizations."""

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir

    def plot_all(
        self,
        prompt: str,
        tokens: list[str],
        token_importances: list[float],
        perturbed_prompts: list[str],
        cosine_distances: list[float],
        entropies: list[float],
        robustness_result: dict,
        strategies_used: list[str],
    ):
        fig = plt.figure(figsize=(20, 16), facecolor="#0d1117")
        fig.suptitle(
            "promptgrad — Prompt Sensitivity Analysis",
            fontsize=20, color="white", fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

        ax_score  = fig.add_subplot(gs[0, 0])
        ax_radar  = fig.add_subplot(gs[0, 1], polar=True)
        ax_dist   = fig.add_subplot(gs[0, 2])
        ax_tokens = fig.add_subplot(gs[1, :2])
        ax_heatmap = fig.add_subplot(gs[1, 2])
        ax_entropy = fig.add_subplot(gs[2, :2])
        ax_strat   = fig.add_subplot(gs[2, 2])

        self._plot_score_gauge(ax_score, robustness_result)
        self._plot_radar(ax_radar, robustness_result)
        self._plot_distance_hist(ax_dist, cosine_distances)
        self._plot_token_importance(ax_tokens, tokens, token_importances, prompt)
        self._plot_entropy_heatmap(ax_heatmap, entropies, strategies_used, len(PERTURBATION_STRATEGIES))
        self._plot_entropy_line(ax_entropy, entropies, perturbed_prompts)
        self._plot_strategy_breakdown(ax_strat, cosine_distances, strategies_used)

        plt.savefig(f"{self.output_dir}/promptgrad_analysis.png",
                    dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"\n[Visualizer] Saved → {self.output_dir}/promptgrad_analysis.png")

    # ── Individual plot methods ──

    def _style_ax(self, ax, title: str):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    def _plot_score_gauge(self, ax, result: dict):
        score = result["robustness_score"]
        label = result["label"]
        ax.set_facecolor("#161b22")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        # Color band
        color = (
            "#3fb950" if score >= 0.65 else
            "#d29922" if score >= 0.45 else
            "#f85149"
        )

        # Big number
        ax.text(0.5, 0.62, f"{score:.2f}", ha="center", va="center",
                fontsize=52, color=color, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.35, label, ha="center", va="center",
                fontsize=14, color=color, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.18, "Robustness Score", ha="center", va="center",
                fontsize=10, color="#8b949e", transform=ax.transAxes)

        # Progress bar
        bar_bg = plt.Rectangle((0.1, 0.08), 0.8, 0.06,
                                color="#30363d", transform=ax.transAxes, clip_on=False)
        bar_fg = plt.Rectangle((0.1, 0.08), 0.8 * score, 0.06,
                                color=color, transform=ax.transAxes, clip_on=False)
        ax.add_patch(bar_bg)
        ax.add_patch(bar_fg)
        ax.set_title("Prompt Robustness", color="white", fontsize=11, fontweight="bold", pad=8)

    def _plot_radar(self, ax, result: dict):
        ax.set_facecolor("#161b22")
        components = result["components"]
        labels = ["Embed Stability", "Embed Consistency", "Entropy Stability", "Output Similarity"]

        # Convert distances/variances to robustness axes (invert)
        values = [
            1 - min(components["mean_embedding_distance"] / 0.5, 1),
            1 - min(components["embedding_variance"] / 0.1, 1),
            1 - min(components["entropy_variance"] / 2.0, 1),
            1 - min(components["mean_tv_distance"] / 0.5, 1),
        ]
        values += values[:1]

        N = len(labels)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.plot(angles, values, color="#58a6ff", linewidth=2)
        ax.fill(angles, values, color="#58a6ff", alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color="#8b949e", fontsize=8)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="#8b949e", fontsize=6)
        ax.set_ylim(0, 1)
        ax.spines["polar"].set_color("#30363d")
        ax.set_title("Component Scores", color="white", fontsize=11, fontweight="bold", pad=15)
        ax.tick_params(colors="#30363d")
        ax.set_facecolor("#161b22")

    def _plot_distance_hist(self, ax, distances: list[float]):
        self._style_ax(ax, "Embedding Distance Distribution")
        ax.hist(distances, bins=20, color="#58a6ff", edgecolor="#30363d", alpha=0.85)
        ax.axvline(np.mean(distances), color="#f85149", linestyle="--",
                   linewidth=1.5, label=f"Mean: {np.mean(distances):.3f}")
        ax.axvline(np.median(distances), color="#3fb950", linestyle="--",
                   linewidth=1.5, label=f"Median: {np.median(distances):.3f}")
        ax.set_xlabel("Cosine Distance", color="#8b949e", fontsize=9)
        ax.set_ylabel("Count", color="#8b949e", fontsize=9)
        ax.legend(fontsize=8, facecolor="#161b22", labelcolor="white", edgecolor="#30363d")

    def _plot_token_importance(self, ax, tokens: list[str], importances: list[float], prompt: str):
        self._style_ax(ax, "Token Importance (Leave-One-Out Ablation)")
        if not tokens:
            return

        x = np.arange(len(tokens))
        colors = plt.cm.RdYlGn_r(np.array(importances))

        bars = ax.bar(x, importances, color=colors, edgecolor="#30363d", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha="right", color="#c9d1d9", fontsize=8)
        ax.set_ylabel("Importance Score", color="#8b949e", fontsize=9)
        ax.set_ylim(0, 1.15)

        for bar, val in zip(bars, importances):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom",
                        color="#8b949e", fontsize=7)

        # Highlight top 3
        top3 = np.argsort(importances)[-3:]
        for idx in top3:
            bars[idx].set_edgecolor("#f0e68c")
            bars[idx].set_linewidth(2)

    def _plot_entropy_heatmap(self, ax, entropies: list[float], strategies: list[str], n_strategies: int):
        self._style_ax(ax, "Entropy Heatmap by Strategy")
        n = len(entropies)
        n_strategies = len(PERTURBATION_STRATEGIES)
        n_per = n // n_strategies

        # Build matrix: strategies x perturbations_per_strategy
        rows = []
        strat_labels = []
        for i, strat in enumerate(PERTURBATION_STRATEGIES):
            chunk = entropies[i * n_per: (i + 1) * n_per]
            if chunk:
                rows.append(chunk)
                strat_labels.append(strat.replace("_", "\n"))

        if not rows:
            return

        max_len = max(len(r) for r in rows)
        matrix = np.array([r + [np.nan] * (max_len - len(r)) for r in rows])

        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, aspect="auto", cmap="YlOrRd",
                       vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        ax.set_yticks(range(len(strat_labels)))
        ax.set_yticklabels(strat_labels, color="#8b949e", fontsize=7)
        ax.set_xlabel("Perturbation #", color="#8b949e", fontsize=9)
        plt.colorbar(im, ax=ax, label="Entropy", shrink=0.8).ax.yaxis.set_tick_params(color="#8b949e")

    def _plot_entropy_line(self, ax, entropies: list[float], prompts: list[str]):
        self._style_ax(ax, "Output Entropy Across Perturbations")
        x = np.arange(len(entropies))
        ax.plot(x, entropies, color="#58a6ff", linewidth=1.2, alpha=0.7)

        # Rolling mean
        window = max(5, len(entropies) // 10)
        if len(entropies) >= window:
            rolling = np.convolve(entropies, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window - 1, len(entropies)), rolling,
                    color="#f0883e", linewidth=2, label=f"Rolling mean (w={window})")

        mean_e = np.mean(entropies)
        ax.axhline(mean_e, color="#f85149", linestyle="--",
                   linewidth=1.2, label=f"Mean entropy: {mean_e:.3f}")

        # Shade high-entropy regions
        threshold = mean_e + np.std(entropies)
        for i, e in enumerate(entropies):
            if e > threshold:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color="#f85149")

        ax.set_xlabel("Perturbation Index", color="#8b949e", fontsize=9)
        ax.set_ylabel("Entropy", color="#8b949e", fontsize=9)
        ax.legend(fontsize=8, facecolor="#161b22", labelcolor="white", edgecolor="#30363d")

    def _plot_strategy_breakdown(self, ax, distances: list[float], strategies: list[str]):
        self._style_ax(ax, "Mean Distance by Perturbation Strategy")
        n = len(distances)
        n_strategies = len(PERTURBATION_STRATEGIES)
        n_per = n // n_strategies

        strat_means = []
        strat_labels = []
        for i, strat in enumerate(PERTURBATION_STRATEGIES):
            chunk = distances[i * n_per: (i + 1) * n_per]
            if chunk:
                strat_means.append(np.mean(chunk))
                strat_labels.append(strat.replace("_", "\n"))

        if not strat_means:
            return

        colors_vals = plt.cm.plasma(np.linspace(0.2, 0.9, len(strat_means)))
        y = np.arange(len(strat_means))
        ax.barh(y, strat_means, color=colors_vals, edgecolor="#30363d")
        ax.set_yticks(y)
        ax.set_yticklabels(strat_labels, color="#8b949e", fontsize=7)
        ax.set_xlabel("Mean Cosine Distance", color="#8b949e", fontsize=9)

        # Most dangerous strategy
        max_idx = int(np.argmax(strat_means))
        ax.get_children()[max_idx].set_edgecolor("#f85149")
        ax.get_children()[max_idx].set_linewidth(2)


# ─────────────────────────────────────────────
#  MAIN ANALYZER
# ─────────────────────────────────────────────

class PromptGrad:
    """
    Main entry point for prompt sensitivity analysis.

    Example:
        analyzer = PromptGrad(lm_model="gpt2")
        results = analyzer.analyze("You are a helpful assistant. Always be concise.")
    """

    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        lm_model: str = "gpt2",
        n_perturbations: int = 40,
        intensity: float = 0.3,
        seed: int = 42,
        output_dir: str = ".",
    ):
        self.n_perturbations = n_perturbations
        self.intensity = intensity
        self.output_dir = output_dir

        self.perturber = PerturbationEngine(seed=seed)
        self.embed_analyzer = EmbeddingAnalyzer(model_name=embed_model)
        self.output_engine = OutputVarianceEngine(model_name=lm_model)
        self.token_analyzer = TokenImportanceAnalyzer(self.embed_analyzer)
        self.scorer = RobustnessScorer()
        self.viz = Visualizer(output_dir=output_dir)

    def analyze(self, prompt: str) -> dict:
        print("\n" + "═" * 60)
        print("  promptgrad — Prompt Sensitivity Analysis")
        print("═" * 60)
        print(f"\n  Prompt: \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")
        print(f"  Perturbations: {self.n_perturbations} | Intensity: {self.intensity}")
        print()

        # 1. Generate perturbations
        print("[1/5] Generating perturbations...")
        perturbed = self.perturber.batch_perturb(
            prompt, self.n_perturbations, intensity=self.intensity
        )

        # 2. Compute embeddings and distances
        print("[2/5] Computing embedding shifts...")
        all_texts = [prompt] + perturbed
        all_embeddings = self.embed_analyzer.encode(all_texts)
        anchor_emb = all_embeddings[0]
        perturbed_embs = all_embeddings[1:]

        cosine_dists = self.embed_analyzer.cosine_distances(anchor_emb, perturbed_embs)
        cosine_dists_list = cosine_dists.tolist()
        emb_variance = self.embed_analyzer.embedding_variance(perturbed_embs)
        mean_emb_dist = float(cosine_dists.mean())

        # 3. Output entropy
        print("[3/5] Computing output entropy...")
        anchor_probs = self.output_engine.get_next_token_probs(prompt)
        perturbed_subset = perturbed[:min(self.n_perturbations, 40)]
        entropies = self.output_engine.batch_entropy(perturbed_subset)
        entropy_variance = float(np.var(entropies))

        # TV distances for a subset
        tv_distances = []
        for p_text in tqdm(perturbed[:20], desc="Computing TV distances", leave=False):
            p_probs = self.output_engine.get_next_token_probs(p_text)
            tv_distances.append(self.output_engine.tv_distance(anchor_probs, p_probs))
        mean_tv = float(np.mean(tv_distances)) if tv_distances else 0.0

        # 4. Token importance
        print("[4/5] Analyzing token importance...")
        tokens, importances = self.token_analyzer.compute_importance(prompt)

        # 5. Score + visualize
        print("[5/5] Scoring and visualizing...")
        result = self.scorer.score(mean_emb_dist, emb_variance, entropy_variance, mean_tv)

        strategies_used = PERTURBATION_STRATEGIES * (self.n_perturbations // len(PERTURBATION_STRATEGIES) + 1)
        strategies_used = strategies_used[:self.n_perturbations]

        self.viz.plot_all(
            prompt=prompt,
            tokens=tokens,
            token_importances=importances,
            perturbed_prompts=perturbed,
            cosine_distances=cosine_dists_list,
            entropies=entropies,
            robustness_result=result,
            strategies_used=strategies_used,
        )

        # Print summary
        self._print_summary(result, tokens, importances, mean_emb_dist,
                            entropy_variance, entropies)

        return {
            "prompt": prompt,
            "robustness": result,
            "tokens": tokens,
            "token_importances": importances,
            "cosine_distances": cosine_dists_list,
            "entropies": entropies,
            "mean_tv_distance": mean_tv,
        }

    def _print_summary(self, result, tokens, importances, mean_emb_dist,
                       entropy_variance, entropies):
        score = result["robustness_score"]
        label = result["label"]
        comp = result["components"]

        bar_width = 40
        filled = int(score * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        print("\n" + "═" * 60)
        print("  RESULTS")
        print("═" * 60)
        print(f"\n  Robustness Score:  {score:.4f}  [{bar}]")
        print(f"  Label:             {label}")
        print(f"\n  ── Component Breakdown ──")
        print(f"  Mean Embed Distance : {comp['mean_embedding_distance']:.4f}")
        print(f"  Embedding Variance  : {comp['embedding_variance']:.4f}")
        print(f"  Entropy Variance    : {comp['entropy_variance']:.4f}")
        print(f"  Mean TV Distance    : {comp['mean_tv_distance']:.4f}")

        if tokens and importances:
            top3_idx = np.argsort(importances)[-3:][::-1]
            print(f"\n  ── Most Important Tokens ──")
            for i, idx in enumerate(top3_idx):
                print(f"  #{i+1}: \"{tokens[idx]}\"  (importance: {importances[idx]:.4f})")

        print(f"\n  Mean Output Entropy : {np.mean(entropies):.4f}")
        print(f"  Max Output Entropy  : {np.max(entropies):.4f}")
        print(f"  Entropy Variance    : {entropy_variance:.4f}")

        if score < 0.45:
            print("\n  ⚠  WARNING: Prompt is fragile. Small changes cause large output shifts.")
            print("     Consider: more explicit constraints, fewer ambiguous terms,")
            print("               or structural anchoring (e.g., XML tags, numbered rules).")
        elif score >= 0.75:
            print("\n  ✓  Prompt appears robust to typical perturbations.")

        print("\n" + "═" * 60 + "\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="promptgrad — Prompt Sensitivity Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python promptgrad.py --prompt "You are a helpful assistant. Always answer concisely."
  python promptgrad.py --prompt "Summarize the following document:" --n 80 --intensity 0.5
  python promptgrad.py --prompt "Translate to French:" --model gpt2 --embed distilbert-base-uncased
        """
    )
    parser.add_argument("--prompt", type=str,
                        default="You are a helpful assistant. Always provide concise and accurate answers.",
                        help="The prompt to analyze")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace causal LM for output variance (default: gpt2)")
    parser.add_argument("--embed", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace embedding model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--n", type=int, default=40,
                        help="Number of perturbations (default: 40)")
    parser.add_argument("--intensity", type=float, default=0.3,
                        help="Perturbation intensity 0–1 (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out", type=str, default=".",
                        help="Output directory for plot (default: .)")

    args = parser.parse_args()

    analyzer = PromptGrad(
        embed_model=args.embed,
        lm_model=args.model,
        n_perturbations=args.n,
        intensity=args.intensity,
        seed=args.seed,
        output_dir=args.out,
    )
    analyzer.analyze(args.prompt)


if __name__ == "__main__":
    main()
