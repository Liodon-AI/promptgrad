"""
Visualization helpers for promptgrad reports.

Requires matplotlib (included in the `viz` extra).
"""

from __future__ import annotations

import math
from typing import Optional

from .metrics import RobustnessReport


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting.\n"
            "Install it with:  pip install promptgrad[viz]"
        ) from e


def _plot_entropy_heatmap(ax, report: RobustnessReport):
    """
    Horizontal token heatmap — colour = importance score.
    """
    import numpy as np

    tokens = report.prompt.split()
    scores = [report.token_importance.get(t, 0.0) for t in tokens]

    data = [scores]
    cmap = ax.imshow(
        data,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_title("Token Sensitivity Heatmap", fontsize=11, fontweight="bold")
    return cmap


def _plot_token_importance_bar(ax, report: RobustnessReport):
    """Bar chart of top-10 token importance scores."""
    import numpy as np

    top = report.top_sensitive_tokens(10)
    if not top:
        ax.text(0.5, 0.5, "No token importance data", ha="center", va="center")
        return

    tokens, scores = zip(*top)
    colours = ["#d73027" if s > 0.7 else "#fc8d59" if s > 0.4 else "#91cf60" for s in scores]
    bars = ax.barh(range(len(tokens)), scores, color=colours)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    ax.set_title("Token Importance (Top 10)", fontsize=11, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)


def _plot_strategy_radar(ax, report: RobustnessReport):
    """Radar / spider chart of per-strategy cosine similarity."""
    import numpy as np

    strategies = list(report.per_strategy.keys())
    values = [report.per_strategy[s] for s in strategies]
    if not strategies:
        ax.text(0.5, 0.5, "No strategy data", ha="center", va="center")
        return

    N = len(strategies)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(strategies, fontsize=8)
    ax.set_ylim(0, 1)
    ax.plot(angles, values_plot, "o-", linewidth=2, color="#2171b5")
    ax.fill(angles, values_plot, alpha=0.25, color="#2171b5")
    ax.set_title("Per-Strategy Robustness", fontsize=11, fontweight="bold", pad=20)
    # Add reference circle at 0.75
    ref = [0.75] * (N + 1)
    ax.plot(angles, ref, "--", color="gray", linewidth=0.8, alpha=0.6)


def _plot_score_gauge(ax, report: RobustnessReport):
    """Semi-circular gauge for the overall robustness score."""
    import numpy as np

    score = report.robustness_score
    label = report.stability_label

    # Draw background arc
    theta = np.linspace(0, math.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#dddddd", linewidth=15, solid_capstyle="round")

    # Colour bands
    colours = [
        (0.00, 0.50, "#d73027"),  # red
        (0.50, 0.75, "#fc8d59"),  # orange
        (0.75, 0.90, "#91cf60"),  # light green
        (0.90, 1.00, "#1a9641"),  # green
    ]
    for lo, hi, colour in colours:
        t = np.linspace(lo * math.pi, hi * math.pi, 100)
        ax.plot(np.cos(t), np.sin(t), color=colour, linewidth=15, solid_capstyle="butt")

    # Needle
    needle_angle = (1.0 - score) * math.pi
    ax.annotate(
        "",
        xy=(0.75 * math.cos(needle_angle), 0.75 * math.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=2),
    )
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, -0.15, f"{score:.2f}", ha="center", fontsize=22, fontweight="bold")
    ax.text(0, -0.35, label, ha="center", fontsize=11, color="gray")
    ax.set_title("Robustness Score", fontsize=11, fontweight="bold")


def plot_report(
    report: RobustnessReport,
    kind: str = "all",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Render one or all visualisations for a RobustnessReport.

    Parameters
    ----------
    kind : "all" | "heatmap" | "importance" | "radar" | "gauge"
    """
    plt = _require_matplotlib()
    import matplotlib.pyplot as _plt

    if kind == "all":
        fig = _plt.figure(figsize=(16, 10))
        fig.suptitle(
            f"promptgrad — Sensitivity Analysis\n\"{report.prompt[:80]}{'…' if len(report.prompt) > 80 else ''}\"",
            fontsize=12,
            fontweight="bold",
            y=1.01,
        )

        # Layout: 2×2 grid
        ax_gauge = fig.add_subplot(2, 2, 1)
        ax_heatmap = fig.add_subplot(2, 2, 2)
        ax_bar = fig.add_subplot(2, 2, 3)
        ax_radar = fig.add_subplot(2, 2, 4, projection="polar")

        _plot_score_gauge(ax_gauge, report)
        im = _plot_entropy_heatmap(ax_heatmap, report)
        _plot_token_importance_bar(ax_bar, report)
        _plot_strategy_radar(ax_radar, report)

        # Warnings text box
        warning_text = "\n".join(report.warnings)
        fig.text(
            0.01, -0.02, warning_text,
            fontsize=8, color="#555555",
            verticalalignment="top",
            wrap=True,
        )

        _plt.colorbar(im, ax=ax_heatmap, orientation="horizontal", fraction=0.05, pad=0.2, label="Sensitivity")
        _plt.tight_layout()

    elif kind == "heatmap":
        fig, ax = _plt.subplots(figsize=(12, 2))
        im = _plot_entropy_heatmap(ax, report)
        _plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.4, label="Sensitivity")
        _plt.tight_layout()

    elif kind == "importance":
        fig, ax = _plt.subplots(figsize=(8, 6))
        _plot_token_importance_bar(ax, report)
        _plt.tight_layout()

    elif kind == "radar":
        fig, ax = _plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        _plot_strategy_radar(ax, report)
        _plt.tight_layout()

    elif kind == "gauge":
        fig, ax = _plt.subplots(figsize=(5, 4))
        _plot_score_gauge(ax, report)
        _plt.tight_layout()

    else:
        raise ValueError(f"Unknown plot kind {kind!r}. Choose from: all, heatmap, importance, radar, gauge")

    if save_path:
        _plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        _plt.show()
    