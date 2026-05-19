#!/usr/bin/env python3
"""Generate public figures from the reported experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001


STRATEGY_EXPERIMENT_ID = "restaurant_strategy"
CROSS_DOMAIN_EXPERIMENT_IDS = (
    "restaurant_strategy",
    "laptop_strategy",
    "twitter_strategy",
)
MULTI_BACKBONE_EXPERIMENT_ID = "multi_backbone"
PUBLIC_FIGURE_NAMES = (
    "strategy-comparison.png",
    "cross-domain.png",
    "multi-backbone.png",
)


def load_results(path: Path) -> dict:
    """Load and validate the public result document needed by this figure."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("reported metrics must use schema version 1")
    experiments = data.get("experiments")
    required_ids = {*CROSS_DOMAIN_EXPERIMENT_IDS, MULTI_BACKBONE_EXPERIMENT_ID}
    experiment_ids = (
        {experiment.get("id") for experiment in experiments if isinstance(experiment, dict)}
        if isinstance(experiments, list)
        else set()
    )
    missing = sorted(required_ids - experiment_ids)
    if missing:
        raise ValueError(f"reported metrics must include experiments: {', '.join(missing)}")
    return data


def plot_strategy_comparison(data: dict, output: Path, dpi: int) -> None:
    """Plot restaurant baseline and residual-strategy macro-F1 scores."""
    experiment = next(
        experiment
        for experiment in data["experiments"]
        if experiment["id"] == STRATEGY_EXPERIMENT_ID
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


def _strategy_label(name: str) -> str:
    return name.replace("+", " + ").replace("_", " ").title()


def _domain_label(dataset: str) -> str:
    return dataset.removeprefix("SemEval-2014 ").removesuffix(" ABSA")


def plot_cross_domain(data: dict, output: Path, dpi: int) -> None:
    """Plot macro-F1 for every residual strategy across the three domains."""
    experiments_by_id = {experiment["id"]: experiment for experiment in data["experiments"]}
    experiments = [experiments_by_id[item] for item in CROSS_DOMAIN_EXPERIMENT_IDS]
    figure, axes = plt.subplots(1, len(experiments), figsize=(18, 6))
    try:
        for axis, experiment in zip(axes, experiments):
            rows = experiment["rows"]
            scores = [row["macro_f1"] for row in rows]
            baseline = scores[0]
            deltas = [score - baseline for score in scores]
            colors = [
                "#95a5a6" if index == 0 else "#27ae60" if delta > 0 else "#e74c3c"
                for index, delta in enumerate(deltas)
            ]
            bars = axis.bar(
                range(len(rows)),
                scores,
                width=0.62,
                color=colors,
                edgecolor="white",
                alpha=0.9,
            )
            for index, (bar, score, delta) in enumerate(zip(bars, scores, deltas)):
                label = f"{score:.4f}" if index == 0 else f"{score:.4f}\n({delta:+.4f})"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    score + 0.0005,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=colors[index],
                    fontweight="bold" if index == 0 else "normal",
                )

            spread = max(scores) - min(scores)
            margin = max(spread, 0.005)
            axis.set_ylim(min(scores) - margin, max(scores) + 2.5 * margin)
            axis.axhline(baseline, color="#7f8c8d", linestyle="--", alpha=0.5)
            axis.set_xticks(range(len(rows)))
            axis.set_xticklabels(
                [_strategy_label(row["name"]) for row in rows],
                fontsize=8,
                rotation=30,
                ha="right",
            )
            axis.set_ylabel("Macro-F1")
            axis.set_title(
                f"{_domain_label(experiment['dataset'])}\nResidual strategy comparison",
                fontweight="bold",
                fontsize=12,
            )

        figure.suptitle("Cross-Domain Macro-F1", fontsize=15, fontweight="bold")
        figure.tight_layout()
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_multi_backbone(data: dict, output: Path, dpi: int) -> None:
    """Plot base and best residual macro-F1 for every reported backbone."""
    experiment = next(
        experiment
        for experiment in data["experiments"]
        if experiment["id"] == MULTI_BACKBONE_EXPERIMENT_ID
    )
    rows = experiment["rows"]
    base_scores = [row["macro_f1"] for row in rows]
    residual_scores = [row["residual_macro_f1"] for row in rows]
    positions = list(range(len(rows)))
    width = 0.36

    figure, axis = plt.subplots(figsize=(14, 7))
    try:
        base_bars = axis.bar(
            [position - width / 2 for position in positions],
            base_scores,
            width,
            color="#95a5a6",
            edgecolor="white",
            label="Base",
            alpha=0.9,
        )
        residual_bars = axis.bar(
            [position + width / 2 for position in positions],
            residual_scores,
            width,
            color="#27ae60",
            edgecolor="white",
            label="Best residual",
            alpha=0.9,
        )
        for row, base_bar, residual_bar in zip(rows, base_bars, residual_bars):
            delta = row["residual_macro_f1"] - row["macro_f1"]
            axis.text(
                residual_bar.get_x() + residual_bar.get_width() / 2,
                residual_bar.get_height() + 0.003,
                f"+{delta:.4f}\n{row['best_strategy']}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1e8449",
            )
            axis.text(
                base_bar.get_x() + base_bar.get_width() / 2,
                base_bar.get_height() + 0.002,
                f"{row['macro_f1']:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#566573",
            )

        minimum = min(base_scores + residual_scores)
        maximum = max(base_scores + residual_scores)
        margin = max(maximum - minimum, 0.03)
        axis.set_ylim(minimum - margin * 0.45, maximum + margin * 0.7)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [row["name"] for row in rows],
            rotation=25,
            ha="right",
            fontsize=9,
        )
        axis.set_ylabel("Macro-F1")
        axis.set_title(
            "Residual Compatibility Across Backbones\n"
            f"{experiment['dataset']} {experiment['split'].title()} Set",
            fontsize=14,
            fontweight="bold",
        )
        axis.legend(loc="upper right", fontsize=10)
        figure.tight_layout()
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)


def generate_all(results_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    """Generate the complete public figure set and return the written paths."""
    data = load_results(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / name for name in PUBLIC_FIGURE_NAMES]
    plot_strategy_comparison(data, outputs[0], dpi)
    plot_cross_domain(data, outputs[1], dpi)
    plot_multi_backbone(data, outputs[2], dpi)
    return outputs


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
