"""
Confusion matrix visualization for multi-label ECG classification.

Generates per-class binary confusion matrices and a combined heatmap.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_classes: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrices (Per Class)",
) -> plt.Figure:
    """
    Plot per-class binary confusion matrices in a grid.

    For multi-label classification, each class gets its own
    binary confusion matrix (positive vs negative).

    Args:
        y_true: Ground truth binary labels ``(N, num_classes)``.
        y_pred: Predicted binary labels ``(N, num_classes)``.
        label_classes: List of class names.
        save_path: If provided, save figure to this path.
        title: Overall title.

    Returns:
        Matplotlib Figure object.
    """
    n_classes = len(label_classes)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_classes == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (cls, ax) in enumerate(zip(label_classes, axes)):
        cm = sk_confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
        )
        ax.set_title(cls, fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrices saved to %s", save_path)

    return fig


def plot_multilabel_confusion_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_classes: List[str],
    save_path: Optional[str] = None,
    title: str = "Multi-Label Classification Summary",
) -> plt.Figure:
    """
    Plot a summary heatmap of TP, FP, FN, TN per class.

    Args:
        y_true: Ground truth binary labels ``(N, num_classes)``.
        y_pred: Predicted binary labels ``(N, num_classes)``.
        label_classes: List of class names.
        save_path: If provided, save figure.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    summary = np.zeros((len(label_classes), 4))

    for i in range(len(label_classes)):
        tn, fp, fn, tp = sk_confusion_matrix(
            y_true[:, i], y_pred[:, i], labels=[0, 1]
        ).ravel()
        summary[i] = [tp, fp, fn, tn]

    fig, ax = plt.subplots(figsize=(8, max(4, len(label_classes) * 0.8)))
    sns.heatmap(
        summary, annot=True, fmt=".0f", cmap="YlOrRd",
        xticklabels=["TP", "FP", "FN", "TN"],
        yticklabels=label_classes,
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion summary saved to %s", save_path)

    return fig
