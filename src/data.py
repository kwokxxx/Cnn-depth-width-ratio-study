from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "classes": 100,
    },
}


def dataset_num_classes(dataset_name: str) -> int:
    return int(DATASET_STATS[dataset_name]["classes"])


def build_transforms(dataset_name: str, augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    stats = DATASET_STATS[dataset_name]
    train_steps = []
    if augment:
        train_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )
    test_steps = [
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ]
    return transforms.Compose(train_steps), transforms.Compose(test_steps)


def build_dataset(
    dataset_name: str,
    data_dir: Path,
    train: bool,
    download: bool,
    transform: transforms.Compose,
) -> Dataset:
    if dataset_name == "cifar10":
        return datasets.CIFAR10(root=data_dir, train=train, download=download, transform=transform)
    if dataset_name == "cifar100":
        return datasets.CIFAR100(root=data_dir, train=train, download=download, transform=transform)
    raise ValueError(f"unknown dataset: {dataset_name}")


def maybe_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def build_dataloaders(
    dataset_name: str,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    download: bool = False,
    augment: bool = True,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_transform, test_transform = build_transforms(dataset_name, augment=augment)
    train_set = build_dataset(dataset_name, data_dir, train=True, download=download, transform=train_transform)
    test_set = build_dataset(dataset_name, data_dir, train=False, download=download, transform=test_transform)
    train_set = maybe_subset(train_set, train_limit)
    test_set = maybe_subset(test_set, test_limit)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
