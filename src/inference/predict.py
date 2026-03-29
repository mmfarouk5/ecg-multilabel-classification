"""
Inference pipeline for ECG classification.

Loads a trained model checkpoint and runs predictions on new ECG signals.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.preprocessing import preprocess_pipeline
from src.models import build_model

logger = logging.getLogger(__name__)


def load_trained_model(
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        config: Configuration dict. If None, loaded from checkpoint.
        device: Device to load model onto.

    Returns:
        Loaded model in eval mode.
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if config is None:
        config = checkpoint.get("config", {})

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded model '%s' from %s (epoch %d, val_loss=%.4f)",
        config.get("model", {}).get("name", "unknown"),
        checkpoint_path,
        checkpoint.get("epoch", -1),
        checkpoint.get("val_loss", -1),
    )
    return model


@torch.no_grad()
def predict_signal(
    model: nn.Module,
    signal: np.ndarray,
    config: Dict[str, Any],
    threshold: float = 0.5,
    label_classes: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run inference on a single ECG signal.

    Args:
        model: Trained model in eval mode.
        signal: Raw ECG signal ``(seq_len, n_leads)`` or
                ``(1, seq_len, n_leads)``.
        config: Configuration for preprocessing.
        threshold: Prediction threshold.
        label_classes: Optional list of class names.
        device: Inference device.

    Returns:
        Dictionary with ``probabilities``, ``predictions``,
        ``predicted_classes``, and ``logits``.
    """
    device = device or next(model.parameters()).device

    # Ensure batch dimension
    if signal.ndim == 2:
        signal = signal[np.newaxis, ...]  # (1, seq_len, n_leads)

    # Preprocess
    signal = preprocess_pipeline(signal, config)

    # Convert to tensor: (1, seq_len, n_leads) → (1, n_leads, seq_len)
    tensor = torch.tensor(signal, dtype=torch.float32).permute(0, 2, 1).to(device)

    # Forward pass
    logits = model(tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    preds = (probs >= threshold).astype(int)

    result = {
        "logits": logits.cpu().numpy()[0],
        "probabilities": probs,
        "predictions": preds,
    }

    if label_classes:
        predicted = [cls for cls, p in zip(label_classes, preds) if p == 1]
        result["predicted_classes"] = predicted
        result["class_probabilities"] = dict(zip(label_classes, probs.round(4)))

    return result


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    signals: np.ndarray,
    config: Dict[str, Any],
    threshold: float = 0.5,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """
    Run batch inference on multiple ECG signals.

    Args:
        model: Trained model.
        signals: Raw signals ``(N, seq_len, n_leads)``.
        config: Configuration for preprocessing.
        threshold: Prediction threshold.
        batch_size: Inference batch size.
        device: Inference device.

    Returns:
        Dictionary with ``probabilities`` and ``predictions`` arrays.
    """
    device = device or next(model.parameters()).device

    # Preprocess all signals
    signals = preprocess_pipeline(signals, config)

    all_probs = []
    n_samples = len(signals)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = signals[start:end]

        tensor = torch.tensor(batch, dtype=torch.float32).permute(0, 2, 1).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    preds = (probs >= threshold).astype(int)

    return {
        "probabilities": probs,
        "predictions": preds,
    }
