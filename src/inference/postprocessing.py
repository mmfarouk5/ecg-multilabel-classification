"""
Post-processing utilities for ECG classification predictions.

Provides threshold optimization, confidence filtering, and
prediction formatting.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def apply_thresholds(
    probabilities: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    default_threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply per-class thresholds to probabilities.

    Args:
        probabilities: Prediction probabilities ``(N, num_classes)`` or ``(num_classes,)``.
        thresholds: Per-class thresholds ``(num_classes,)``. If None, uses default.
        default_threshold: Default threshold if per-class not provided.

    Returns:
        Binary predictions with the same shape as input.
    """
    if thresholds is None:
        thresholds = np.full(probabilities.shape[-1], default_threshold)

    return (probabilities >= thresholds).astype(np.float32)


def filter_by_confidence(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    min_confidence: float = 0.3,
) -> np.ndarray:
    """
    Filter predictions to keep only high-confidence positives.

    Sets predictions to 0 where the probability is below
    ``min_confidence``, even if it was above the threshold.

    Args:
        probabilities: ``(N, num_classes)`` or ``(num_classes,)``.
        predictions: Binary predictions.
        min_confidence: Minimum probability to keep a positive prediction.

    Returns:
        Filtered binary predictions.
    """
    filtered = predictions.copy()
    filtered[probabilities < min_confidence] = 0
    return filtered


def format_predictions(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    label_classes: List[str],
    top_k: Optional[int] = None,
) -> List[Dict[str, object]]:
    """
    Format predictions into human-readable dictionaries.

    Args:
        probabilities: ``(N, num_classes)``.
        predictions: ``(N, num_classes)``.
        label_classes: Class names.
        top_k: If set, only include top-k most probable classes.

    Returns:
        List of dicts, one per sample, with keys:
        ``predicted_classes``, ``probabilities``, ``all_scores``.
    """
    results = []
    batch_size = probabilities.shape[0] if probabilities.ndim > 1 else 1

    if probabilities.ndim == 1:
        probabilities = probabilities[np.newaxis, :]
        predictions = predictions[np.newaxis, :]

    for i in range(batch_size):
        probs = probabilities[i]
        preds = predictions[i]

        # All class scores
        scores = {cls: round(float(p), 4) for cls, p in zip(label_classes, probs)}

        # Predicted classes
        predicted = [cls for cls, pred in zip(label_classes, preds) if pred == 1]

        # Top-k
        if top_k is not None:
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_classes = [(label_classes[j], round(float(probs[j]), 4)) for j in top_indices]
        else:
            top_classes = None

        result = {
            "predicted_classes": predicted,
            "probabilities": scores,
        }
        if top_classes:
            result["top_k"] = top_classes

        results.append(result)

    return results
