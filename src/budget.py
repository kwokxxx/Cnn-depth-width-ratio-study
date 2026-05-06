from __future__ import annotations

from dataclasses import dataclass

try:
    from .metrics import count_parameters
    from .models import make_model
except ImportError:
    from metrics import count_parameters
    from models import make_model


@dataclass(frozen=True)
class BudgetMatch:
    depth: int
    width: int
    parameters: int
    target_parameters: int
    absolute_error: int
    relative_error: float


def parameter_count_for_width(
    depth: int,
    width: int,
    num_classes: int,
    dropout: float,
    use_batch_norm: bool,
    model_family: str,
    width_schedule: str,
    width_slope: int,
) -> int:
    model = make_model(
        depth=depth,
        width=width,
        num_classes=num_classes,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        model_family=model_family,
        width_schedule=width_schedule,
        width_slope=width_slope,
    )
    return count_parameters(model)


def find_width_for_budget(
    depth: int,
    target_parameters: int,
    num_classes: int,
    dropout: float,
    use_batch_norm: bool,
    model_family: str,
    width_schedule: str,
    width_slope: int,
    min_width: int = 4,
    max_width: int = 512,
) -> BudgetMatch:
    best_width = min_width
    best_parameters = parameter_count_for_width(
        depth,
        min_width,
        num_classes,
        dropout,
        use_batch_norm,
        model_family,
        width_schedule,
        width_slope,
    )
    best_error = abs(best_parameters - target_parameters)

    for width in range(min_width, max_width + 1):
        parameters = parameter_count_for_width(
            depth,
            width,
            num_classes,
            dropout,
            use_batch_norm,
            model_family,
            width_schedule,
            width_slope,
        )
        error = abs(parameters - target_parameters)
        if error < best_error:
            best_width = width
            best_parameters = parameters
            best_error = error
        if parameters > target_parameters and width > best_width:
            break

    return BudgetMatch(
        depth=depth,
        width=best_width,
        parameters=best_parameters,
        target_parameters=target_parameters,
        absolute_error=best_error,
        relative_error=best_error / target_parameters,
    )
