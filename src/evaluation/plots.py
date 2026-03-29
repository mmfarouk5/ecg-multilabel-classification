"""
Visualization utilities for ECG classification evaluation.

Generates ROC curves, precision-recall curves, training history plots,
and metric comparison charts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_classes: List[str],
    save_path: Optional[str] = None,
    title: str = "ROC Curves (Per Class)",
) -> plt.Figure:
    """
    Plot per-class ROC curves with AUC values.

    Args:
        y_true: Ground truth labels ``(N, num_classes)``.
        y_prob: Predicted probabilities ``(N, num_classes)``.
        label_classes: List of class names.
        save_path: If provided, save figure to this path.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, len(label_classes)))

    for i, (cls, color) in enumerate(zip(label_classes, colors)):
        if y_true[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curves saved to %s", save_path)

    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_classes: List[str],
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curves",
) -> plt.Figure:
    """
    Plot per-class Precision-Recall curves.

    Args:
        y_true: Ground truth labels ``(N, num_classes)``.
        y_prob: Predicted probabilities ``(N, num_classes)``.
        label_classes: List of class names.
        save_path: If provided, save figure to this path.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(label_classes)))

    for i, (cls, color) in enumerate(zip(label_classes, colors)):
        if y_true[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{cls} (AP={pr_auc:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("PR curves saved to %s", save_path)

    return fig


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
    title: str = "Training History",
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with ``train_loss``, ``val_loss``, ``learning_rate``.
        save_path: If provided, save figure to this path.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train", lw=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning Rate
    if "learning_rate" in history:
        ax2.plot(epochs, history["learning_rate"], "g-", lw=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training history saved to %s", save_path)

    return fig


def plot_metrics_comparison(
    model_metrics: Dict[str, Dict[str, float]],
    metric_keys: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
) -> plt.Figure:
    """
    Plot a bar chart comparing metrics across models.

    Args:
        model_metrics: Dict mapping model name → metrics dict.
        metric_keys: Which metrics to compare. Defaults to common ones.
        save_path: If provided, save figure.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    if metric_keys is None:
        metric_keys = ["f1_macro", "roc_auc_macro", "precision_macro", "recall_macro"]

    model_names = list(model_metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        values = [model_metrics[model_name].get(k, 0) for k in metric_keys]
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([k.replace("_", " ").title() for k in metric_keys], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Metrics comparison saved to %s", save_path)

    return fig
