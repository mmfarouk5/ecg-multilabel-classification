"""
Learning rate scheduler factory for ECG classification training.
"""

from typing import Any, Dict

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)


def build_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
) -> _LRScheduler:
    """
    Build a learning rate scheduler from config.

    Args:
        optimizer: Optimizer to schedule.
        config: Full config dict with ``training.scheduler`` and
            ``training.scheduler_params``.

    Returns:
        LR scheduler instance.
    """
    train_cfg = config["training"]
    name = train_cfg.get("scheduler", "cosine").lower()
    params = train_cfg.get("scheduler_params", {})

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=params.get("T_max", train_cfg.get("epochs", 50)),
            eta_min=params.get("eta_min", 1e-6),
        )
    elif name == "step":
        return StepLR(
            optimizer,
            step_size=params.get("step_size", 10),
            gamma=params.get("gamma", 0.1),
        )
    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=params.get("patience", 5),
            factor=params.get("factor", 0.5),
            min_lr=params.get("min_lr", 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")
