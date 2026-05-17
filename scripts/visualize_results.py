#!/usr/bin/env python3
"""Generate public figures from the reported experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001


EXPERIMENT_ID = "restaurant_strategy"
FIGURE_NAME = "strategy-comparison.png"


def load_results(path: Path) -> dict:
    """Load and validate the public result document needed by this figure."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("reported metrics must use schema version 1")
    experiments = data.get("experiments")
    if not isinstance(experiments, list) or not any(
        experiment.get("id") == EXPERIMENT_ID
        for experiment in experiments
        if isinstance(experiment, dict)
    ):
        raise ValueError(f"reported metrics must include {EXPERIMENT_ID!r}")
    return data


def plot_strategy_comparison(data: dict, output: Path, dpi: int) -> None:
    """Plot restaurant baseline and residual-strategy macro-F1 scores."""
    experiment = next(
        experiment for experiment in data["experiments"] if experiment["id"] == EXPERIMENT_ID
    )
    rows = experiment["rows"]
    labels = [
        "Baseline",
        "+Fact",
        "+Comparison",
        "+Intensity",
        "+Fact\n+Comparison",
        "All\nResiduals",
    ]
    scores = [row["macro_f1"] for row in rows]
    baseline = scores[0]

    figure, axis = plt.subplots(figsize=(10, 6))
    try:
        bars = axis.bar(
            range(len(rows)),
            scores,
            width=0.62,
            color=["#3498db", *("#e74c3c" for _ in rows[1:])],
            edgecolor="white",
        )
        for bar, score in zip(bars, scores):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.00035,
                f"{score:.4f}\n(Δ{score - baseline:+.4f})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        axis.set_ylabel("Macro-F1", color="#e74c3c", fontsize=13)
        axis.set_title(
            "Pragmatic Residual Strategies — Macro-F1 Comparison\n"
            f"({experiment['dataset']} {experiment['split'].title()} Set)",
            fontsize=14,
            fontweight="bold",
        )
        axis.set_xticks(range(len(rows)))
        axis.set_xticklabels(labels)
        spread = max(scores) - min(scores)
        margin = max(spread, 0.005)
        axis.set_ylim(min(scores) - margin, max(scores) + 2 * margin)
        axis.axhline(
            y=baseline,
            color="#3498db",
            linestyle="--",
            alpha=0.4,
            linewidth=1,
            label="Baseline",
        )
        axis.legend(loc="upper left", fontsize=10)
        figure.tight_layout()
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)


def generate_all(results_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    """Generate the complete public figure set and return the written paths."""
    data = load_results(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / FIGURE_NAME
    plot_strategy_comparison(data, output, dpi)
    return [output]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    for output in generate_all(args.results, args.output_dir, args.dpi):
        print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
