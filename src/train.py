from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from .data import build_dataloaders, dataset_num_classes
    from .engine import get_device, run_epoch, set_seed
    from .metrics import count_parameters, estimate_flops, measure_inference_latency
    from .models import make_model
except ImportError:
    from data import build_dataloaders, dataset_num_classes
    from engine import get_device, run_epoch, set_seed
    from metrics import count_parameters, estimate_flops, measure_inference_latency
    from models import make_model


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"unknown optimizer: {args.optimizer}")


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[max(1, int(args.epochs * 0.5)), max(1, int(args.epochs * 0.75))],
            gamma=0.1,
        )
    if args.scheduler == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    raise ValueError(f"unknown scheduler: {args.scheduler}")


def train_once(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = dataset_num_classes(args.dataset)
    train_loader, test_loader = build_dataloaders(
        dataset_name=args.dataset,
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        augment=not args.no_augment,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    model = make_model(
        depth=args.depth,
        width=args.width,
        num_classes=num_classes,
        dropout=args.dropout,
        use_batch_norm=not args.no_batch_norm,
        model_family=args.model_family,
        width_schedule=args.width_schedule,
        width_slope=args.width_slope,
        max_downsamples=args.max_downsamples,
    ).to(device)

    parameter_count = count_parameters(model)
    model_metadata = model.metadata
    flops_per_example = estimate_flops(model, (1, 3, 32, 32), device) if args.measure_flops else 0
    latency_seconds = (
        measure_inference_latency(model, (args.batch_size, 3, 32, 32), device) if args.measure_latency else 0.0
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            show_progress=not args.no_progress,
            track_gradient_norm=args.track_grad_norm,
        )
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            show_progress=not args.no_progress,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "dataset": args.dataset,
            "model_family": args.model_family,
            "width_schedule": args.width_schedule,
            "depth": args.depth,
            "width": args.width,
            "width_slope": args.width_slope,
            "channel_schedule": model_metadata["channel_schedule"],
            "base_depth_width_ratio": model_metadata["base_depth_width_ratio"],
            "effective_width": model_metadata["effective_width"],
            "effective_depth_width_ratio": model_metadata["effective_depth_width_ratio"],
            "downsample_count": model_metadata["downsample_count"],
            "downsample_indices": model_metadata["downsample_indices"],
            "final_spatial_size": model_metadata["final_spatial_size"],
            "seed": args.seed,
            "parameters": parameter_count,
            "flops": flops_per_example,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "train_examples_per_second": train_metrics["examples_per_second"],
            "test_examples_per_second": test_metrics["examples_per_second"],
            "train_gradient_norm": train_metrics["gradient_norm"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} dataset={args.dataset} family={args.model_family} "
            f"schedule={args.width_schedule} depth={args.depth} width={args.width} "
            f"train_acc={row['train_accuracy']:.4f} test_acc={row['test_accuracy']:.4f}"
        )

    elapsed_seconds = time.perf_counter() - start_time
    run_name = (
        f"{args.dataset}_{args.model_family}_{args.width_schedule}_"
        f"d{args.depth}_w{args.width}_seed{args.seed}"
    )
    history_path = output_dir / f"{run_name}_history.csv"
    with history_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    best_row = max(history, key=lambda row: row["test_accuracy"])
    summary = {
        "run_name": run_name,
        "dataset": args.dataset,
        "model_family": args.model_family,
        "width_schedule": args.width_schedule,
        "depth": args.depth,
        "width": args.width,
        "width_slope": args.width_slope,
        "channel_schedule": model_metadata["channel_schedule"],
        "base_depth_width_ratio": model_metadata["base_depth_width_ratio"],
        "effective_width": model_metadata["effective_width"],
        "effective_depth_width_ratio": model_metadata["effective_depth_width_ratio"],
        "downsample_count": model_metadata["downsample_count"],
        "downsample_indices": model_metadata["downsample_indices"],
        "final_spatial_size": model_metadata["final_spatial_size"],
        "seed": args.seed,
        "parameters": parameter_count,
        "flops": flops_per_example,
        "latency_seconds_per_batch": latency_seconds,
        "epochs": args.epochs,
        "best_epoch": best_row["epoch"],
        "best_test_accuracy": best_row["test_accuracy"],
        "best_test_loss": best_row["test_loss"],
        "final_test_accuracy": history[-1]["test_accuracy"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_train_loss": history[-1]["train_loss"],
        "final_test_loss": history[-1]["test_loss"],
        "final_generalization_gap": history[-1]["train_accuracy"] - history[-1]["test_accuracy"],
        "elapsed_seconds": elapsed_seconds,
        "history_path": str(history_path),
        "device": str(device),
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
    }
    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one CNN depth-width configuration.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--model-family", choices=["plain", "residual"], default="plain")
    parser.add_argument("--width-schedule", choices=["constant", "stagewise", "linear"], default="stagewise")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--width-slope", type=int, default=8)
    parser.add_argument("--max-downsamples", type=int, default=3)
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
    parser.add_argument("--seed", type=int, default=0)
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


if __name__ == "__main__":
    train_once(parse_args())
