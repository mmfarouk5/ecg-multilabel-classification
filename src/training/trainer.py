"""
Training engine for ECG multi-label classification.

Handles the full training loop including:
- Forward/backward passes with AMP
- Gradient clipping
- Validation evaluation
- Early stopping (with min_delta, monitor, mode support)
- Checkpoint saving
- TensorBoard logging
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.utils import get_device, unwrap_model

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping handler with configurable monitoring.

    Tracks a monitored metric and stops training when it has not
    improved by at least ``min_delta`` for ``patience`` epochs.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``"min"`` (lower is better, e.g. loss) or
              ``"max"`` (higher is better, e.g. F1).
        restore_best_weights: Whether to restore model weights from the
            best epoch when stopping.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_state_dict: Optional[dict] = None

        if mode == "min":
            self.is_improvement = lambda current, best: current < best - min_delta
        elif mode == "max":
            self.is_improvement = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def __call__(
        self,
        score: float,
        model: nn.Module,
        epoch: int,
    ) -> bool:
        """
        Check whether training should stop.

        Args:
            score: Current value of the monitored metric.
            model: The model (for saving best weights).
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        if self.best_score is None or self.is_improvement(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered at epoch %d "
                    "(no improvement for %d epochs, best=%.4f at epoch %d)",
                    epoch + 1, self.patience, self.best_score, self.best_epoch + 1,
                )
                return True
            return False

    def restore(self, model: nn.Module) -> None:
        """Restore model to the best weights if available."""
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logger.info(
                "Restored best model weights from epoch %d (score=%.4f)",
                self.best_epoch + 1, self.best_score,
            )

    @property
    def status(self) -> str:
        """Human-readable status string."""
        if self.best_score is None:
            return "not started"
        return (
            f"best={self.best_score:.4f} at epoch {self.best_epoch + 1}, "
            f"patience {self.counter}/{self.patience}"
        )


class Trainer:
    """
    Core training engine for ECG classification.

    Args:
        model: The model to train.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: LR scheduler (optional).
        config: Full configuration dictionary.
        device: Device to train on.
        writer: TensorBoard SummaryWriter (optional).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        self.config = config or {}
        train_cfg = self.config.get("training", {})

        self.device = device or get_device()
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer

        # AMP (only supported on CUDA, not MPS)
        amp_requested = train_cfg.get("amp", False)
        if amp_requested and self.device.type == "mps":
            logger.warning(
                "AMP is not supported on MPS device — disabling AMP automatically.")
        self.use_amp = amp_requested and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.grad_clip = train_cfg.get("gradient_clip", 0.0)
        self.non_blocking = bool(train_cfg.get(
            "pin_memory", False) and self.device.type == "cuda")

        # Early stopping — supports both new structured config and legacy flat key
        es_cfg = train_cfg.get("early_stopping", {})
        if isinstance(es_cfg, dict) and es_cfg.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 10),
                min_delta=es_cfg.get("min_delta", 0.0),
                mode=es_cfg.get("mode", "min"),
                restore_best_weights=es_cfg.get("restore_best_weights", True),
            )
            self.es_monitor = es_cfg.get("monitor", "val_loss")
        elif "early_stopping_patience" in train_cfg:
            # Legacy fallback
            self.early_stopping = EarlyStopping(
                patience=train_cfg["early_stopping_patience"],
                min_delta=0.0,
                mode="min",
                restore_best_weights=True,
            )
            self.es_monitor = "val_loss"
        else:
            self.early_stopping = None
            self.es_monitor = None

        # Output paths
        out_cfg = self.config.get("output", {})
        self.models_dir = Path(out_cfg.get("models_dir", "outputs/models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Track best for checkpointing
        self.best_val_loss = float("inf")

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1_macro": [],
            "learning_rate": [],
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Run a single training epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number (0-indexed).

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for signals, labels in train_loader:
            signals = signals.to(self.device, non_blocking=self.non_blocking)
            labels = labels.to(self.device, non_blocking=self.non_blocking)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast():
                    logits = self.model(signals)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(signals)
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        collect_outputs: bool = False,
    ) -> Dict[str, Any]:
        """
        Run validation.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary with ``val_loss`` and optional predictions.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_logits = [] if collect_outputs else None
        all_labels = [] if collect_outputs else None

        for signals, labels in val_loader:
            signals = signals.to(self.device, non_blocking=self.non_blocking)
            labels = labels.to(self.device, non_blocking=self.non_blocking)

            logits = self.model(signals)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1
            if collect_outputs:
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        avg_loss = total_loss / max(n_batches, 1)
        result = {"val_loss": avg_loss}
        if collect_outputs and all_logits and all_labels:
            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            result["logits"] = logits_cat
            result["labels"] = labels_cat

            # Compute F1 macro for early stopping
            from sklearn.metrics import f1_score
            from src.evaluation.metrics import find_optimal_thresholds

            probs = torch.sigmoid(logits_cat).numpy()
            labels_np = labels_cat.numpy()
            thresholds, _ = find_optimal_thresholds(labels_np, probs)
            preds = (probs >= thresholds).astype(float)
            result["val_f1_macro"] = float(
                f1_score(labels_np, preds, average="macro", zero_division=0)
            )
        return result

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> Dict[str, list]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs (overrides config).

        Returns:
            Training history dictionary.
        """
        epochs = epochs or self.config.get("training", {}).get("epochs", 50)
        model_name = self.config.get("model", {}).get("name", "model")

        logger.info("Starting training: %d epochs on %s", epochs, self.device)
        logger.info("Model: %s | AMP: %s | Grad clip: %s",
                    model_name, self.use_amp, self.grad_clip)
        if self.early_stopping:
            logger.info(
                "Early stopping: monitor=%s, patience=%d, min_delta=%.4f, mode=%s",
                self.es_monitor, self.early_stopping.patience,
                self.early_stopping.min_delta,
                self.early_stopping.mode if hasattr(
                    self.early_stopping, 'mode') else 'min',
            )

        epoch_pbar = tqdm(
            range(epochs),
            desc=f"{model_name}",
            unit="ep",
            ncols=100,
        )

        for epoch in epoch_pbar:

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            collect_outputs = bool(
                self.es_monitor and self.es_monitor != "val_loss")
            val_results = self.validate(
                val_loader, collect_outputs=collect_outputs)
            val_loss = val_results["val_loss"]

            # LR scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            # Compute and log val_f1_macro if available
            val_f1 = val_results.get("val_f1_macro")
            if val_f1 is not None:
                self.history["val_f1_macro"].append(val_f1)

            # Determine monitored score for early stopping
            monitor_score = val_loss
            if self.es_monitor and self.es_monitor != "val_loss":
                monitor_score = val_results.get(self.es_monitor, val_loss)

            # Update progress bar
            epoch_pbar.set_postfix(
                loss=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{self.best_val_loss:.4f}",
            )

            # TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            # Checkpointing (always save best by val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    self.models_dir / f"best_{model_name}.pt",
                    epoch, val_loss,
                )

            # Early stopping check
            if self.early_stopping:
                should_stop = self.early_stopping(
                    monitor_score, self.model, epoch)
                if should_stop:
                    break

        # Restore best weights if early stopping was used
        if self.early_stopping and self.early_stopping.restore_best_weights:
            self.early_stopping.restore(self.model)

        # Save final checkpoint
        self.save_checkpoint(
            self.models_dir / f"final_{model_name}.pt",
            epoch, val_loss,
        )

        logger.info("Training complete. Best val loss: %.4f",
                    self.best_val_loss)
        return self.history

    def save_checkpoint(
        self, path: Path, epoch: int, val_loss: float
    ) -> None:
        """
        Save a model checkpoint.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch.
            val_loss: Current validation loss.
        """
        # Unwrap DataParallel if needed
        model_to_save = unwrap_model(self.model)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a model checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info("Loaded checkpoint from %s (epoch %d)",
                    path, checkpoint["epoch"])
        return checkpoint
