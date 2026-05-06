from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def compute_gradient_norm(model: nn.Module) -> float:
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        parameter_norm = parameter.grad.detach().data.norm(2).item()
        squared_norm += parameter_norm**2
    return squared_norm**0.5


@contextmanager
def temporary_eval(model: nn.Module) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def estimate_flops(model: nn.Module, input_shape: tuple[int, int, int, int], device: torch.device) -> int:
    flops = 0
    handles = []

    def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal flops
        batch_size, out_channels, out_height, out_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        in_channels = module.in_channels // module.groups
        flops += batch_size * out_channels * out_height * out_width * in_channels * kernel_height * kernel_width * 2

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal flops
        batch_size = output.shape[0]
        flops += batch_size * module.in_features * module.out_features * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    with temporary_eval(model), torch.no_grad():
        dummy = torch.zeros(input_shape, device=device)
        model(dummy)

    for handle in handles:
        handle.remove()
    return int(flops // input_shape[0])


def measure_inference_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    warmup_steps: int = 5,
    timed_steps: int = 20,
) -> float:
    with temporary_eval(model), torch.no_grad():
        dummy = torch.randn(input_shape, device=device)
        for _ in range(warmup_steps):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(timed_steps):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / timed_steps
