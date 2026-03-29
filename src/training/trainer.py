"""
Training engine for ECG multi-label classification.

Handles the full training loop including:
- Forward/backward passes with AMP
- Gradient clipping
- Validation evaluation
- Early stopping
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
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


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

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer

        # AMP
        self.use_amp = train_cfg.get("amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.grad_clip = train_cfg.get("gradient_clip", 0.0)

        # Early stopping
        self.patience = train_cfg.get("early_stopping_patience", 10)
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        # Output paths
        out_cfg = self.config.get("output", {})
        self.models_dir = Path(out_cfg.get("models_dir", "outputs/models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
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

        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits = self.model(signals)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(signals)
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary with ``val_loss`` and predictions.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_logits = []
        all_labels = []

        for signals, labels in val_loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(signals)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / max(n_batches, 1)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return {
            "val_loss": avg_loss,
            "logits": all_logits,
            "labels": all_labels,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> Dict[str, list]:
        """
        Full training loop.

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

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_results = self.validate(val_loader)
            val_loss = val_results["val_loss"]

            # LR scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            elapsed = time.time() - start_time

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            logger.info(
                "Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f | LR: %.2e | Time: %.1fs",
                epoch + 1, epochs, train_loss, val_loss, current_lr, elapsed,
            )

            # TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            # Early stopping + checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(
                    self.models_dir / f"best_{model_name}.pt",
                    epoch, val_loss,
                )
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

        # Save final checkpoint
        self.save_checkpoint(
            self.models_dir / f"final_{model_name}.pt",
            epoch, val_loss,
        )

        logger.info("Training complete. Best val loss: %.4f", self.best_val_loss)
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
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
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

        logger.info("Loaded checkpoint from %s (epoch %d)", path, checkpoint["epoch"])
        return checkpoint
