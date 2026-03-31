"""
Evaluation metrics for multi-label ECG classification.

Provides per-class and aggregate metrics including accuracy, F1,
precision, recall, and ROC-AUC.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    multilabel_confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_classes: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive multi-label classification metrics.

    Args:
        y_true: Ground truth binary labels ``(N, num_classes)``.
        y_pred: Predicted binary labels ``(N, num_classes)``.
        y_prob: Predicted probabilities ``(N, num_classes)``.
        label_classes: Optional list of class names.
        threshold: Threshold for converting probabilities to binary predictions.

    Returns:
        Dictionary of metric names to values.
    """
    num_classes = y_true.shape[1]
    if label_classes is None:
        label_classes = [f"class_{i}" for i in range(num_classes)]

    metrics = {}

    # Subset accuracy (exact match)
    metrics["subset_accuracy"] = accuracy_score(y_true, y_pred)

    # Sample-averaged metrics
    metrics["f1_macro"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_micro"] = f1_score(
        y_true, y_pred, average="micro", zero_division=0)
    metrics["f1_weighted"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0)

    # ROC-AUC (per-class, then macro)
    try:
        metrics["roc_auc_macro"] = roc_auc_score(
            y_true, y_prob, average="macro")
        metrics["roc_auc_weighted"] = roc_auc_score(
            y_true, y_prob, average="weighted")

        # Per-class ROC-AUC
        for i, cls_name in enumerate(label_classes):
            if y_true[:, i].sum() > 0:  # Only compute if positive samples exist
                metrics[f"roc_auc_{cls_name}"] = roc_auc_score(
                    y_true[:, i], y_prob[:, i])
            else:
                metrics[f"roc_auc_{cls_name}"] = float("nan")
    except ValueError as e:
        logger.warning("Could not compute ROC-AUC: %s", e)
        metrics["roc_auc_macro"] = float("nan")

    # Average Precision (per-class)
    try:
        metrics["avg_precision_macro"] = average_precision_score(
            y_true, y_prob, average="macro"
        )
    except ValueError:
        metrics["avg_precision_macro"] = float("nan")

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, cls_name in enumerate(label_classes):
        metrics[f"f1_{cls_name}"] = per_class_f1[i]

    # Per-class Precision & Recall
    per_class_prec = precision_score(
        y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, cls_name in enumerate(label_classes):
        metrics[f"precision_{cls_name}"] = per_class_prec[i]
        metrics[f"recall_{cls_name}"] = per_class_rec[i]

    return metrics


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find optimal per-class thresholds maximizing F1 score.

    Args:
        y_true: Ground truth binary labels ``(N, num_classes)``.
        y_prob: Predicted probabilities ``(N, num_classes)``.
        thresholds: Candidate thresholds to search over.

    Returns:
        Tuple of:
        - Optimal thresholds per class ``(num_classes,)``.
        - Best F1 scores per class ``(num_classes,)``.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.951, 0.01)

    num_classes = y_true.shape[1]
    best_thresholds = np.zeros(num_classes)
    best_f1s = np.zeros(num_classes)

    for c in range(num_classes):
        best_f1 = 0.0
        best_t = 0.5
        for t in thresholds:
            preds = (y_prob[:, c] >= t).astype(int)
            f1 = f1_score(y_true[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[c] = best_t
        best_f1s[c] = best_f1

    return best_thresholds, best_f1s


def format_metrics_table(
    metrics: Dict[str, float],
    label_classes: List[str],
) -> str:
    """
    Format metrics as a readable table string.

    Args:
        metrics: Dictionary of metric names to values.
        label_classes: List of class names.

    Returns:
        Formatted table string.
    """
    lines = [
        "=" * 60,
        "EVALUATION METRICS",
        "=" * 60,
        f"  Subset Accuracy:   {metrics.get('subset_accuracy', 0):.4f}",
        f"  Macro F1:          {metrics.get('f1_macro', 0):.4f}",
        f"  Micro F1:          {metrics.get('f1_micro', 0):.4f}",
        f"  Weighted F1:       {metrics.get('f1_weighted', 0):.4f}",
        f"  Macro Precision:   {metrics.get('precision_macro', 0):.4f}",
        f"  Macro Recall:      {metrics.get('recall_macro', 0):.4f}",
        f"  ROC-AUC Macro:     {metrics.get('roc_auc_macro', 0):.4f}",
        "",
        "  Per-Class Metrics:",
        f"  {'Class':<12} {'F1':>8} {'Precision':>10} {'Recall':>8} {'ROC-AUC':>9}",
        "  " + "-" * 49,
    ]

    for cls in label_classes:
        f1 = metrics.get(f"f1_{cls}", 0)
        prec = metrics.get(f"precision_{cls}", 0)
        rec = metrics.get(f"recall_{cls}", 0)
        auc = metrics.get(f"roc_auc_{cls}", 0)
        lines.append(
            f"  {cls:<12} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {auc:>9.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)
