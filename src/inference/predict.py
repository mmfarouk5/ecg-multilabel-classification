"""
Inference pipeline for ECG classification.

Loads a trained model checkpoint and runs predictions on new ECG signals.
Supports single-model and ensemble inference with per-class optimal thresholds.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.preprocessing import preprocess_pipeline
from src.models import build_model

logger = logging.getLogger(__name__)

# ── Checkpoint helpers ──────────────────────────────────────────

def _remap_state_dict(state_dict: dict) -> dict:
    """
    Normalise checkpoint keys produced by older training runs.

    Handles two common key mismatches:
    - ``module.`` prefix from ``DataParallel`` wrapping.
    - ``backbone.`` → ``lead_backbone.`` rename in LeadwiseCNN.
    """
    remapped = {}
    for key, value in state_dict.items():
        clean = key
        if clean.startswith("module."):
            clean = clean[len("module."):]
        if clean.startswith("backbone."):
            clean = clean.replace("backbone.", "lead_backbone.", 1)
        remapped[clean] = value
    return remapped


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
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    if config is None:
        config = checkpoint.get("config", {})

    model = build_model(config)

    state_dict = checkpoint["model_state_dict"]
    state_dict = _remap_state_dict(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            "Checkpoint key mismatch — missing=%d  unexpected=%d",
            len(missing), len(unexpected),
        )

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


# ── Threshold loading ───────────────────────────────────────────

def load_optimal_thresholds(
    results_dir: str,
    num_classes: int = 5,
    default: float = 0.5,
) -> np.ndarray:
    """
    Load per-class optimal thresholds from saved evaluation results.

    Falls back to a uniform ``default`` threshold if the file is missing.

    Args:
        results_dir: Directory containing ``optimal_thresholds.npy``.
        num_classes: Expected number of classes (sanity check).
        default: Fallback threshold value.

    Returns:
        Array of shape ``(num_classes,)`` with per-class thresholds.
    """
    path = Path(results_dir) / "optimal_thresholds.npy"
    if path.exists():
        thresholds = np.load(path)
        if thresholds.shape == (num_classes,):
            logger.info("Loaded per-class thresholds from %s: %s", path, thresholds.round(3))
            return thresholds
        logger.warning(
            "Threshold shape %s != expected (%d,). Using default=%.2f.",
            thresholds.shape, num_classes, default,
        )
    else:
        logger.info("No optimal thresholds found at %s. Using default=%.2f.", path, default)
    return np.full(num_classes, default, dtype=np.float32)


# ── Single-model inference ──────────────────────────────────────

@torch.inference_mode()
def predict_signal(
    model: nn.Module,
    signal: np.ndarray,
    config: Dict[str, Any],
    threshold: Union[float, np.ndarray] = 0.5,
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
        threshold: Prediction threshold — scalar or per-class array
                   ``(num_classes,)``.
        label_classes: Optional list of class names.
        device: Inference device.

    Returns:
        Dictionary with ``probabilities``, ``predictions``,
        ``predicted_classes``, ``class_probabilities``, and ``logits``.
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

    # Apply threshold (scalar or per-class)
    if isinstance(threshold, np.ndarray):
        preds = (probs >= threshold).astype(int)
    else:
        preds = (probs >= threshold).astype(int)

    result: Dict[str, Any] = {
        "logits": logits.cpu().numpy()[0],
        "probabilities": probs,
        "predictions": preds,
    }

    if label_classes:
        predicted = [cls for cls, p in zip(label_classes, preds) if p == 1]
        result["predicted_classes"] = predicted
        result["class_probabilities"] = dict(zip(label_classes, probs.round(4)))

    return result


@torch.inference_mode()
def predict_batch(
    model: nn.Module,
    signals: np.ndarray,
    config: Dict[str, Any],
    threshold: Union[float, np.ndarray] = 0.5,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """
    Run batch inference on multiple ECG signals.

    Args:
        model: Trained model.
        signals: Raw signals ``(N, seq_len, n_leads)``.
        config: Configuration for preprocessing.
        threshold: Prediction threshold — scalar or per-class array.
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


# ── Ensemble inference ──────────────────────────────────────────

def load_ensemble(
    model_names: Sequence[str],
    configs_dir: str = "configs",
    models_dir: str = "outputs/models",
    device: Optional[torch.device] = None,
) -> List[nn.Module]:
    """
    Load multiple trained models for ensemble inference.

    Args:
        model_names: Ordered list of model names (e.g. ``["leadwise_cnn", "cnn_1d", "lstm"]``).
        configs_dir: Path to the configs directory.
        models_dir: Path to the models checkpoint directory.
        device: Target device.

    Returns:
        List of loaded models in eval mode.
    """
    device = device or torch.device("cpu")
    models = []
    for name in model_names:
        cfg_path = Path(configs_dir) / f"{name}.yaml"
        ckpt_path = Path(models_dir) / f"best_{name}.pt"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        model = load_trained_model(str(ckpt_path), config=config, device=device)
        models.append(model)
    logger.info("Loaded ensemble of %d models: %s", len(models), model_names)
    return models


@torch.inference_mode()
def predict_ensemble(
    models: List[nn.Module],
    signal: np.ndarray,
    config: Dict[str, Any],
    weights: Optional[Sequence[float]] = None,
    threshold: Union[float, np.ndarray] = 0.5,
    label_classes: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run ensemble inference on a single ECG signal.

    Averages sigmoid probabilities (optionally weighted) across models
    and applies per-class or scalar thresholds.

    Args:
        models: List of loaded models in eval mode.
        signal: Raw ECG signal ``(seq_len, n_leads)`` or ``(1, seq_len, n_leads)``.
        config: Configuration for preprocessing.
        weights: Per-model weights (normalised internally). Equal if None.
        threshold: Scalar or per-class threshold array.
        label_classes: Optional list of class names.
        device: Inference device.

    Returns:
        Dictionary with ensemble ``probabilities``, ``predictions``,
        and optional ``predicted_classes`` / ``class_probabilities``.
    """
    if not models:
        raise ValueError("models list must not be empty")

    device = device or next(models[0].parameters()).device

    # Normalise weights
    n = len(models)
    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()

    # Ensure batch dimension
    if signal.ndim == 2:
        signal = signal[np.newaxis, ...]

    # Preprocess once
    processed = preprocess_pipeline(signal, config)
    tensor = torch.tensor(processed, dtype=torch.float32).permute(0, 2, 1).to(device)

    # Collect weighted probabilities
    ensemble_probs = np.zeros(models[0](tensor).shape[-1], dtype=np.float64)
    for i, model in enumerate(models):
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        ensemble_probs += w[i] * probs

    ensemble_probs = ensemble_probs.astype(np.float32)

    if isinstance(threshold, np.ndarray):
        preds = (ensemble_probs >= threshold).astype(int)
    else:
        preds = (ensemble_probs >= threshold).astype(int)

    result: Dict[str, Any] = {
        "probabilities": ensemble_probs,
        "predictions": preds,
    }

    if label_classes:
        predicted = [cls for cls, p in zip(label_classes, preds) if p == 1]
        result["predicted_classes"] = predicted
        result["class_probabilities"] = dict(zip(label_classes, ensemble_probs.round(4)))

    return result
