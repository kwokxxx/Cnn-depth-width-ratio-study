from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

try:
    from .budget import find_width_for_budget
    from .data import dataset_num_classes
    from .train import train_once
except ImportError:
    from budget import find_width_for_budget
    from data import dataset_num_classes
    from train import train_once


DEFAULT_CONFIGS = [
    {"depth": 2, "width": 128},
    {"depth": 4, "width": 64},
    {"depth": 6, "width": 48},
    {"depth": 8, "width": 40},
    {"depth": 10, "width": 32},
]


def load_configs(config_path: str | None) -> list[dict[str, Any]]:
    if config_path is None:
        return DEFAULT_CONFIGS
    path = Path(config_path)
    configs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(configs, list):
        raise ValueError("config file must contain a JSON list")
    return configs


def parse_seed_list(raw: str) -> list[int]:
    return [int(seed.strip()) for seed in raw.split(",") if seed.strip()]


def parse_depth_list(raw: str) -> list[int]:
    return [int(depth.strip()) for depth in raw.split(",") if depth.strip()]


def build_budget_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if args.target_params is None:
        return configs

    num_classes = dataset_num_classes(args.dataset)
    for depth in parse_depth_list(args.depths):
        match = find_width_for_budget(
            depth=depth,
            target_parameters=args.target_params,
            num_classes=num_classes,
            dropout=args.dropout,
            use_batch_norm=not args.no_batch_norm,
            model_family=args.model_family,
            width_schedule=args.width_schedule,
            width_slope=args.width_slope,
            min_width=args.min_width,
            max_width=args.max_width,
        )
        configs.append(
            {
                "depth": match.depth,
                "width": match.width,
                "target_parameters": match.target_parameters,
                "budget_parameters": match.parameters,
                "budget_absolute_error": match.absolute_error,
                "budget_relative_error": match.relative_error,
            }
        )
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sweep over CNN depth-width configurations.")
    parser.add_argument("--config", type=str, default=None, help="JSON list of config dictionaries.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--model-family", choices=["plain", "residual"], default="plain")
    parser.add_argument("--width-schedule", choices=["constant", "stagewise", "linear"], default="stagewise")
    parser.add_argument("--width-slope", type=int, default=8)
    parser.add_argument("--target-params", type=int, default=None)
    parser.add_argument("--depths", type=str, default="2,4,6,8,10,12")
    parser.add_argument("--min-width", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--measure-flops", action="store_true")
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument("--track-grad-norm", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_budget_configs(args) if args.target_params is not None else load_configs(args.config)
    seeds = parse_seed_list(args.seeds)

    summaries = []
    for config in configs:
        for seed in seeds:
            run_args = Namespace(
                dataset=config.get("dataset", args.dataset),
                model_family=config.get("model_family", args.model_family),
                width_schedule=config.get("width_schedule", args.width_schedule),
                depth=int(config["depth"]),
                width=int(config["width"]),
                width_slope=int(config.get("width_slope", args.width_slope)),
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                optimizer=args.optimizer,
                scheduler=args.scheduler,
                lr=args.lr,
                momentum=args.momentum,
                nesterov=args.nesterov,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                label_smoothing=args.label_smoothing,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                num_workers=args.num_workers,
                device=args.device,
                download=args.download,
                no_augment=args.no_augment,
                no_batch_norm=args.no_batch_norm,
                no_progress=args.no_progress,
                measure_flops=args.measure_flops,
                measure_latency=args.measure_latency,
                track_grad_norm=args.track_grad_norm,
                train_limit=args.train_limit,
                test_limit=args.test_limit,
            )
            summary = train_once(run_args)
            summary.update({key: value for key, value in config.items() if key.startswith("budget_")})
            summary["target_parameters"] = config.get("target_parameters", "")
            summaries.append(summary)

    summary_path = output_dir / "sweep_summary.csv"
    with summary_path.open("w", newline="") as file:
        fieldnames = sorted({key for summary in summaries for key in summary.keys()})
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
