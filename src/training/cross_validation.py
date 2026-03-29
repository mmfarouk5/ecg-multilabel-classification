"""
K-fold cross-validation for ECG classification.

Trains the model on each fold and aggregates results.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ECGDataset
from src.data.label_processing import compute_class_weights, encode_labels
from src.data.loader import load_metadata, load_raw_signals, load_scp_statements, aggregate_diagnostics
from src.data.preprocessing import preprocess_pipeline
from src.data.split import get_kfold_splits
from src.models import build_model
from src.training.loss import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def run_cross_validation(
    config: Dict[str, Any],
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run K-fold cross-validation.

    Args:
        config: Full configuration dictionary.
        max_samples: If set, limit the number of samples loaded.

    Returns:
        List of per-fold results dictionaries containing:
        ``fold``, ``best_val_loss``, ``history``.
    """
    data_cfg = config["data"]
    split_cfg = config["split"]
    train_cfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    metadata = load_metadata(data_cfg["raw_dir"])
    signals = load_raw_signals(
        metadata, data_cfg["raw_dir"],
        sampling_rate=data_cfg["sampling_rate"],
        max_samples=max_samples,
    )
    if max_samples:
        metadata = metadata.iloc[:max_samples]

    scp_df = load_scp_statements(data_cfg["raw_dir"])
    diag_labels = aggregate_diagnostics(metadata, scp_df, data_cfg["label_type"])
    label_matrix, label_classes = encode_labels(diag_labels, label_type=data_cfg["label_type"])

    # Preprocess
    signals = preprocess_pipeline(signals, config)

    # K-fold splits
    folds = get_kfold_splits(
        metadata,
        n_folds=split_cfg.get("n_kfolds", 5),
        test_fold=split_cfg["test_fold"],
        label_matrix=label_matrix,
        seed=split_cfg.get("seed", 42),
    )

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info("=" * 60)
        logger.info("FOLD %d / %d", fold_idx + 1, len(folds))
        logger.info("  Train: %d samples | Val: %d samples", len(train_idx), len(val_idx))

        # DataLoaders
        train_ds = ECGDataset(signals[train_idx], label_matrix[train_idx])
        val_ds = ECGDataset(signals[val_idx], label_matrix[val_idx])

        train_loader = DataLoader(
            train_ds, batch_size=train_cfg["batch_size"],
            shuffle=True, drop_last=True,
            num_workers=train_cfg.get("num_workers", 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 0),
        )

        # Fresh model for each fold
        model = build_model(config)
        class_weights = torch.tensor(
            compute_class_weights(label_matrix[train_idx])
        ).to(device)

        criterion = build_loss(config, class_weights=class_weights)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        # Override model save name for this fold
        fold_config = {**config}
        fold_config["model"] = {**config["model"], "name": f"{config['model']['name']}_fold{fold_idx}"}

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=fold_config,
            device=device,
        )

        history = trainer.fit(train_loader, val_loader)

        fold_results.append({
            "fold": fold_idx,
            "best_val_loss": trainer.best_val_loss,
            "history": history,
        })

        logger.info("Fold %d complete. Best val loss: %.4f", fold_idx + 1, trainer.best_val_loss)

    # Summary
    losses = [r["best_val_loss"] for r in fold_results]
    logger.info("=" * 60)
    logger.info("Cross-validation complete.")
    logger.info("  Mean val loss: %.4f ± %.4f", np.mean(losses), np.std(losses))

    return fold_results
