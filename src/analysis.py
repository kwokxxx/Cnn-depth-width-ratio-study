from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .plot_results import aggregate_summary
except ImportError:
    from plot_results import aggregate_summary


def write_markdown_tables(summary_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    grouped = aggregate_summary(summary)

    ranking = grouped.sort_values("best_test_accuracy_mean", ascending=False)
    columns = [
        "dataset",
        "model_family",
        "width_schedule",
        "depth",
        "width",
        "effective_width",
        "effective_depth_width_ratio",
        "base_depth_width_ratio",
        "downsample_count",
        "final_spatial_size",
        "parameters",
        "flops_mean",
        "best_test_accuracy_mean",
        "best_test_accuracy_std",
        "final_generalization_gap_mean",
    ]
    present_columns = [column for column in columns if column in ranking.columns]
    (output_dir / "model_ranking.md").write_text(
        ranking[present_columns].to_markdown(index=False, floatfmt=".4f"),
        encoding="utf-8",
    )

    per_family = (
        grouped.groupby(["dataset", "model_family", "width_schedule"], as_index=False)
        .agg(
            best_accuracy=("best_test_accuracy_mean", "max"),
            mean_accuracy=("best_test_accuracy_mean", "mean"),
            mean_gap=("final_generalization_gap_mean", "mean"),
            mean_flops=("flops_mean", "mean"),
        )
        .sort_values("best_accuracy", ascending=False)
    )
    (output_dir / "family_summary.md").write_text(
        per_family.to_markdown(index=False, floatfmt=".4f"),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write paper-ready Markdown tables from sweep outputs.")
    parser.add_argument("--summary", type=str, default="results/sweep_summary.csv")
    parser.add_argument("--output-dir", type=str, default="results/tables")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_markdown_tables(Path(args.summary), Path(args.output_dir))
