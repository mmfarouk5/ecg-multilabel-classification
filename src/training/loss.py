"""
Loss functions for ECG multi-label classification.

Provides Weighted BCE and Focal Loss implementations, plus a factory
function ``build_loss`` for config-driven construction.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy with logits loss with per-class weights.

    Args:
        class_weights: Tensor of shape ``(num_classes,)`` with per-class
            weights. If None, uniform weights are used.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: Raw predictions ``(batch, num_classes)``.
            targets: Binary targets ``(batch, num_classes)``.

        Returns:
            Scalar loss.
        """
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    Reduces the loss contribution from easy-to-classify examples,
    focusing training on hard negatives.

    ``FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)``

    Args:
        alpha: Weighting factor. Can be a scalar or per-class tensor.
        gamma: Focusing parameter (gamma >= 0).
        class_weights: Optional per-class weights.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw predictions ``(batch, num_classes)``.
            targets: Binary targets ``(batch, num_classes)``.

        Returns:
            Scalar loss.
        """
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        return loss.mean()


def build_loss(
    config: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Build a loss function from config.

    Args:
        config: Full config dict with ``training.loss`` key.
        class_weights: Optional per-class weights tensor.

    Returns:
        Loss module.
    """
    loss_name = config["training"]["loss"]

    if loss_name == "weighted_bce":
        return WeightedBCEWithLogitsLoss(class_weights=class_weights)
    elif loss_name == "focal":
        focal_params = config["training"].get("focal_loss_params", {})
        return FocalLoss(
            alpha=focal_params.get("alpha", 1.0),
            gamma=focal_params.get("gamma", 2.0),
            class_weights=class_weights,
        )
    elif loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
