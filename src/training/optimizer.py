"""
Optimizer factory for ECG classification training.
"""

from typing import Any, Dict

import torch.nn as nn
import torch.optim as optim


def build_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> optim.Optimizer:
    """
    Build an optimizer from config.

    Args:
        model: Model whose parameters to optimize.
        config: Full config dict with ``training`` key containing
            ``optimizer``, ``learning_rate``, ``weight_decay``.

    Returns:
        Optimizer instance.
    """
    train_cfg = config["training"]
    name = train_cfg.get("optimizer", "adam").lower()
    lr = train_cfg.get("learning_rate", 1e-3)
    wd = train_cfg.get("weight_decay", 1e-4)

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        momentum = train_cfg.get("momentum", 0.9)
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
