"""
Evaluator module for ECG classification.

Provides an ``Evaluator`` class that loads a trained model, runs
inference on a test set, and computes comprehensive metrics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics, find_optimal_thresholds, format_metrics_table

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for trained ECG classification models.

    Args:
        model: Trained model.
        device: Device for inference.
        label_classes: List of class names.
        threshold: Default prediction threshold.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        label_classes: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.label_classes = label_classes or []
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Run inference on a DataLoader.

        Args:
            dataloader: DataLoader to run predictions on.

        Returns:
            Dictionary with keys:
            - ``logits``: Raw logits ``(N, num_classes)``
            - ``probabilities``: Sigmoid probabilities ``(N, num_classes)``
            - ``predictions``: Binary predictions ``(N, num_classes)``
            - ``labels``: Ground truth labels ``(N, num_classes)``
        """
        all_logits = []
        all_labels = []

        for signals, labels in dataloader:
            signals = signals.to(self.device)
            logits = self.model(signals)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        preds = (probs >= self.threshold).astype(np.float32)

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
            "labels": labels,
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        optimize_thresholds: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full evaluation: predictions + metrics.

        Args:
            dataloader: Test/validation DataLoader.
            optimize_thresholds: If True, find per-class optimal thresholds.

        Returns:
            Dictionary with ``metrics``, ``predictions``, and optionally
            ``optimal_thresholds``.
        """
        results = self.predict(dataloader)

        if optimize_thresholds:
            thresholds, _ = find_optimal_thresholds(
                results["labels"], results["probabilities"]
            )
            results["predictions"] = (results["probabilities"] >= thresholds).astype(np.float32)
            results["optimal_thresholds"] = thresholds
            logger.info("Optimal thresholds: %s",
                        dict(zip(self.label_classes, thresholds.round(3))))

        metrics = compute_metrics(
            y_true=results["labels"],
            y_pred=results["predictions"],
            y_prob=results["probabilities"],
            label_classes=self.label_classes,
        )

        results["metrics"] = metrics

        # Print formatted table
        table = format_metrics_table(metrics, self.label_classes)
        logger.info("\n%s", table)

        return results

    def save_results(self, results: Dict[str, Any], save_dir: str) -> None:
        """
        Save evaluation results to disk.

        Args:
            results: Results dictionary from :meth:`evaluate`.
            save_dir: Directory to save results.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        metrics = {k: float(v) if isinstance(v, (float, np.floating)) else v
                   for k, v in results["metrics"].items()}
        with open(save_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save predictions
        np.save(save_path / "probabilities.npy", results["probabilities"])
        np.save(save_path / "predictions.npy", results["predictions"])
        np.save(save_path / "labels.npy", results["labels"])

        if "optimal_thresholds" in results:
            np.save(save_path / "optimal_thresholds.npy", results["optimal_thresholds"])

        logger.info("Results saved to %s", save_path)
