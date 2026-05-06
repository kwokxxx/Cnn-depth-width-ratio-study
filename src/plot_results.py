from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GROUP_COLUMNS = ["dataset", "model_family", "width_schedule", "depth", "width", "parameters"]


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["depth_width_ratio"] = frame["depth"] / frame["width"]
    if "target_parameters" in frame.columns:
        numeric_target = pd.to_numeric(frame["target_parameters"], errors="coerce")
        frame["parameter_budget_error"] = (frame["parameters"] - numeric_target).abs() / numeric_target
    else:
        frame["parameter_budget_error"] = 0.0
    return frame


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    summary = add_derived_columns(summary)
    present_group_columns = [column for column in GROUP_COLUMNS if column in summary.columns]
    grouped = (
        summary.groupby(present_group_columns + ["depth_width_ratio"], as_index=False)
        .agg(
            best_test_accuracy_mean=("best_test_accuracy", "mean"),
            best_test_accuracy_std=("best_test_accuracy", "std"),
            final_test_accuracy_mean=("final_test_accuracy", "mean"),
            final_train_accuracy_mean=("final_train_accuracy", "mean"),
            final_generalization_gap_mean=("final_generalization_gap", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            flops_mean=("flops", "mean"),
            parameter_budget_error_mean=("parameter_budget_error", "mean"),
        )
        .sort_values(["dataset", "model_family", "width_schedule", "depth_width_ratio"])
    )
    return grouped


def plot_metric_by_ratio(
    grouped: pd.DataFrame,
    output_dir: Path,
    metric: str,
    ylabel: str,
    filename: str,
    include_errorbar: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    for key, subset in grouped.groupby(["dataset", "model_family", "width_schedule"]):
        dataset, family, schedule = key
        label = f"{dataset} / {family} / {schedule}"
        subset = subset.sort_values("depth_width_ratio")
        if include_errorbar and "best_test_accuracy_std" in subset.columns:
            plt.errorbar(
                subset["depth_width_ratio"],
                subset[metric],
                yerr=subset["best_test_accuracy_std"].fillna(0.0),
                marker="o",
                linewidth=2,
                capsize=4,
                label=label,
            )
        else:
            plt.plot(subset["depth_width_ratio"], subset[metric], marker="o", linewidth=2, label=label)

    plt.xlabel("Depth-to-width ratio")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=220)
    plt.close()


def plot_accuracy_vs_cost(grouped: pd.DataFrame, output_dir: Path) -> None:
    cost_frame = grouped[grouped["flops_mean"] > 0].copy()
    if cost_frame.empty:
        return
    plt.figure(figsize=(7, 5))
    for key, subset in cost_frame.groupby(["dataset", "model_family", "width_schedule"]):
        dataset, family, schedule = key
        plt.scatter(
            subset["flops_mean"],
            subset["best_test_accuracy_mean"],
            s=70,
            label=f"{dataset} / {family} / {schedule}",
        )
        for _, row in subset.iterrows():
            plt.annotate(f"d{int(row['depth'])}", (row["flops_mean"], row["best_test_accuracy_mean"]), fontsize=7)
    plt.xlabel("Estimated FLOPs per image")
    plt.ylabel("Best test accuracy")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_flops.png", dpi=220)
    plt.close()


def plot_summary(summary_path: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    grouped = aggregate_summary(summary)

    plot_metric_by_ratio(
        grouped,
        output_dir,
        metric="best_test_accuracy_mean",
        ylabel="Best test accuracy",
        filename="accuracy_vs_ratio.png",
        include_errorbar=True,
    )
    plot_metric_by_ratio(
        grouped,
        output_dir,
        metric="final_generalization_gap_mean",
        ylabel="Final train-test accuracy gap",
        filename="generalization_gap_vs_ratio.png",
    )
    plot_metric_by_ratio(
        grouped,
        output_dir,
        metric="elapsed_seconds_mean",
        ylabel="Training seconds",
        filename="training_time_vs_ratio.png",
    )
    if grouped["parameter_budget_error_mean"].notna().any():
        plot_metric_by_ratio(
            grouped,
            output_dir,
            metric="parameter_budget_error_mean",
            ylabel="Relative parameter budget error",
            filename="budget_error_vs_ratio.png",
        )
    plot_accuracy_vs_cost(grouped, output_dir)

    grouped.to_csv(output_dir / "aggregated_summary.csv", index=False)
    return grouped


def plot_histories(results_dir: Path, output_dir: Path) -> None:
    history_files = sorted(results_dir.glob("*_history.csv"))
    if not history_files:
        return

    plt.figure(figsize=(8.5, 5.5))
    for history_file in history_files:
        history = pd.read_csv(history_file)
        label = (
            f"{history['model_family'].iloc[0]}-{history['width_schedule'].iloc[0]} "
            f"d={history['depth'].iloc[0]}, w={history['width'].iloc[0]}, seed={history['seed'].iloc[0]}"
        )
        plt.plot(history["epoch"], history["test_accuracy"], linewidth=1.4, alpha=0.75, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "test_accuracy_curves.png", dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from sweep outputs.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--summary", type=str, default="results/sweep_summary.csv")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    plot_summary(Path(args.summary), output_dir)
    plot_histories(results_dir, output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
