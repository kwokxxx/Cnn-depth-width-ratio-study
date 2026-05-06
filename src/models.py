from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CNNConfig:
    """Architecture configuration for CIFAR-scale CNN experiments."""

    depth: int
    width: int
    num_classes: int = 10
    input_channels: int = 3
    dropout: float = 0.0
    use_batch_norm: bool = True
    model_family: str = "plain"
    width_schedule: str = "stagewise"
    width_slope: int = 8
    max_width_multiplier: int = 8


def make_channel_schedule(config: CNNConfig) -> list[int]:
    if config.depth < 1:
        raise ValueError("depth must be at least 1")
    if config.width < 1:
        raise ValueError("width must be at least 1")

    channels: list[int] = []
    for block_index in range(config.depth):
        if config.width_schedule == "constant":
            multiplier = 1
            out_channels = config.width
        elif config.width_schedule == "stagewise":
            stage = min(block_index // 2, 3)
            multiplier = min(2**stage, config.max_width_multiplier)
            out_channels = config.width * multiplier
        elif config.width_schedule == "linear":
            out_channels = config.width + config.width_slope * block_index
        else:
            raise ValueError(f"unknown width schedule: {config.width_schedule}")
        channels.append(int(out_channels))
    return channels


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool,
        dropout: float,
        stride: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=not use_batch_norm,
            ),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool,
        dropout: float,
        stride: int = 1,
    ) -> None:
        super().__init__()
        bias = not use_batch_norm
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.residual = nn.Sequential(*layers)

        if stride != 1 or in_channels != out_channels:
            projection: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
            ]
            if use_batch_norm:
                projection.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*projection)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.residual(x) + self.shortcut(x))


class ConfigurableCNN(nn.Module):
    """CNN whose depth, width, width schedule, and skip connections are configurable."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config
        self.channels = make_channel_schedule(config)

        layers: list[nn.Module] = []
        in_channels = config.input_channels
        for block_index, out_channels in enumerate(self.channels):
            is_transition = block_index > 0 and block_index % 2 == 0
            stride = 2 if is_transition else 1

            if config.model_family == "plain":
                layers.append(
                    ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        use_batch_norm=config.use_batch_norm,
                        dropout=config.dropout,
                        stride=stride,
                    )
                )
            elif config.model_family == "residual":
                layers.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        use_batch_norm=config.use_batch_norm,
                        dropout=config.dropout,
                        stride=stride,
                    )
                )
            else:
                raise ValueError(f"unknown model family: {config.model_family}")

            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, config.num_classes),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)


def make_model(
    depth: int,
    width: int,
    num_classes: int = 10,
    dropout: float = 0.0,
    use_batch_norm: bool = True,
    model_family: str = "plain",
    width_schedule: str = "stagewise",
    width_slope: int = 8,
) -> ConfigurableCNN:
    config = CNNConfig(
        depth=depth,
        width=width,
        num_classes=num_classes,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        model_family=model_family,
        width_schedule=width_schedule,
        width_slope=width_slope,
    )
    return ConfigurableCNN(config)
