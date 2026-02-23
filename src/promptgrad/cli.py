"""
promptgrad CLI

Usage examples
--------------
    # Analyse a prompt
    promptgrad analyze "Summarize the article in 3 sentences."

    # Analyse with sentence-transformers backend, save plot
    promptgrad analyze "List the top 5 frameworks." --backend sentence_transformers --plot report.png

    # Compare multiple prompts from a file (one per line)
    promptgrad compare prompts.txt

    # Print version
    promptgrad --version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_report(report) -> None:
    from .metrics import RobustnessReport

    print("\n" + "═" * 60)
    print(f"  promptgrad — Sensitivity Analysis")
    print("═" * 60)
    print(f"  Prompt       : {report.prompt[:70]}{'…' if len(report.prompt) > 70 else ''}")
    print(f"  Score        : {report.robustness_score:.3f}  [{report.stability_label}]")
    print(f"  Mean cosine  : {report.mean_cosine_similarity:.3f}")
    print(f"  Entropy      : {report.entropy:.3f}")
    print(f"  Shift std    : {report.embedding_shift_std:.3f}")
    print(f"  Perturbations: {report.n_perturbations}")

    print("\n  Per-strategy robustness:")
    for strategy, sim in report.per_strategy.items():
        bar = "█" * int(sim * 20)
        print(f"    {strategy:<28} {bar:<20} {sim:.3f}")

    print("\n  Top sensitive tokens:")
    for token, score in report.top_sensitive_tokens(5):
        marker = "🔴" if score > 0.7 else "🟡" if score > 0.4 else "🟢"
        print(f"    {marker}  {token:<20} {score:.3f}")

    print("\n  Warnings:")
    for w in report.warnings:
        print(f"    {w}")
    print("═" * 60 + "\n")


def cmd_analyze(args) -> None:
    from .analyzer import PromptAnalyzer

    analyzer = PromptAnalyzer(
        embedding_backend=args.backend,
        n_per_strategy=args.n,
        seed=args.seed,
    )
    report = analyzer.analyze(args.prompt)
    _print_report(report)

    if args.json:
        data = {
            "prompt": report.prompt,
            "robustness_score": report.robustness_score,
            "stability_label": report.stability_label,
            "mean_cosine_similarity": report.mean_cosine_similarity,
            "entropy": report.entropy,
            "embedding_shift_std": report.embedding_shift_std,
            "n_perturbations": report.n_perturbations,
            "per_strategy": report.per_strategy,
            "token_importance": report.token_importance,
            "warnings": report.warnings,
        }
        out_path = args.json if args.json != "-" else None
        if out_path:
            Path(out_path).write_text(json.dumps(data, indent=2))
            print(f"  JSON saved to {out_path}")
        else:
            print(json.dumps(data, indent=2))

    if args.plot:
        report_obj = report
        analyzer.plot(report_obj, save_path=args.plot, show=False)
        print(f"  Plot saved to {args.plot}")


def cmd_compare(args) -> None:
    from .analyzer import PromptAnalyzer

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not prompts:
        print("Error: no prompts found in file.", file=sys.stderr)
        sys.exit(1)

    analyzer = PromptAnalyzer(embedding_backend=args.backend, seed=args.seed)
    result = analyzer.compare(prompts)

    print("\n" + "═" * 60)
    print("  promptgrad — Prompt Comparison")
    print("═" * 60)
    for rank, (prompt, score) in enumerate(result["ranked"], 1):
        label = result["reports"][prompt].stability_label
        bar = "█" * int(score * 30)
        print(f"  #{rank}  {bar:<30} {score:.3f}  {label}")
        print(f"       {prompt[:70]}")
    print("═" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="promptgrad",
        description="Prompt Sensitivity Analyzer",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Analyse a single prompt.")
    p_analyze.add_argument("prompt", help="The prompt to analyse.")
    p_analyze.add_argument(
        "--backend", default="auto",
        choices=["auto", "tfidf", "sentence_transformers", "openai"],
        help="Embedding backend (default: auto).",
    )
    p_analyze.add_argument("--n", type=int, default=5, help="Variants per strategy (default 5).")
    p_analyze.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_analyze.add_argument("--plot", metavar="PATH", help="Save plot to PATH.")
    p_analyze.add_argument(
        "--json", metavar="PATH", nargs="?", const="-",
        help="Output JSON report to PATH (use '-' for stdout).",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # --- compare ---
    p_compare = subparsers.add_parser("compare", help="Compare prompts from a file.")
    p_compare.add_argument("file", help="Text file with one prompt per line.")
    p_compare.add_argument("--backend", default="auto")
    p_compare.add_argument("--seed", type=int, default=42)
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
