"""
src.evaluation — Evaluation pipeline for ECG classification.
"""

from src.evaluation.metrics import compute_metrics, find_optimal_thresholds, format_metrics_table
from src.evaluation.evaluator import Evaluator
from src.evaluation.plots import (
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_training_history,
    plot_metrics_comparison,
)
from src.evaluation.confusion_matrix import (
    plot_confusion_matrices,
    plot_multilabel_confusion_summary,
)

__all__ = [
    "compute_metrics",
    "find_optimal_thresholds",
    "format_metrics_table",
    "Evaluator",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_training_history",
    "plot_metrics_comparison",
    "plot_confusion_matrices",
    "plot_multilabel_confusion_summary",
]
