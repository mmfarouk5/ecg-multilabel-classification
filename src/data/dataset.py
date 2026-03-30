"""
PyTorch Dataset and DataLoader utilities for ECG signals.

Provides ``ECGDataset`` and a convenience ``get_dataloaders`` function.
Supports two modes:
  1. **Cached** (fast): loads preprocessed ``.npy`` files from ``data/processed/``
  2. **On-the-fly** (fallback): loads raw signals and preprocesses from scratch
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG signals and multi-label targets.

    Args:
        signals: Numpy array of shape ``(N, seq_len, n_leads)``.
        labels: Numpy array of shape ``(N, num_classes)``.
        transform: Optional callable applied to each signal tensor.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Any] = None,
    ):
        self.signals = signals
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of:
            - signal tensor of shape ``(n_leads, seq_len)`` (channels first).
            - label tensor of shape ``(num_classes,)``.
        """
        # Shape: (seq_len, n_leads) → (n_leads, seq_len) for Conv1d
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).permute(1, 0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform is not None:
            signal = self.transform(signal)

        return signal, label


def _load_from_cache(
    processed_dir: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, Dict[str, np.ndarray]]:
    """
    Load preprocessed data from cached .npy files.

    Args:
        processed_dir: Path to the ``data/processed/`` directory.

    Returns:
        Tuple of (signals, label_matrix, label_classes, class_weights, splits).
    """
    p = Path(processed_dir)

    signals = np.load(p / "signals.npy")
    labels = np.load(p / "labels.npy")
    class_weights = np.load(p / "class_weights.npy")

    with open(p / "label_classes.json") as f:
        label_classes = json.load(f)

    splits = {
        "train": np.load(p / "train_indices.npy"),
        "val": np.load(p / "val_indices.npy"),
        "test": np.load(p / "test_indices.npy"),
    }

    logger.info(
        "Loaded cached data from %s — signals=%s, labels=%s",
        processed_dir, signals.shape, labels.shape,
    )
    return signals, labels, label_classes, class_weights, splits


def _is_cache_valid(processed_dir: str) -> bool:
    """Check if all required cached files exist."""
    p = Path(processed_dir)
    required = [
        "signals.npy", "labels.npy", "class_weights.npy",
        "label_classes.json", "metadata.json",
        "train_indices.npy", "val_indices.npy", "test_indices.npy",
    ]
    return all((p / f).exists() for f in required)


def get_dataloaders(
    config: Dict[str, Any],
    max_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from config.

    If preprocessed data exists in ``data/processed/``, loads from cache
    (fast path). Otherwise falls back to loading raw data and
    preprocessing on-the-fly.

    Args:
        config: Full configuration dictionary.
        max_samples: If set, only use this many samples (for debugging).

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``, each
        mapping to a ``DataLoader``.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    processed_dir = data_cfg.get("processed_dir", "data/processed")

    # ── Try cached data first ────────────────────────────────
    if max_samples is None and _is_cache_valid(processed_dir):
        logger.info("Using cached preprocessed data from %s", processed_dir)
        signals, label_matrix, label_classes, class_weights, splits = _load_from_cache(processed_dir)
    else:
        # ── Fallback: process from scratch ───────────────────
        if max_samples is not None:
            logger.info("max_samples=%d specified, processing from scratch", max_samples)
        else:
            logger.info(
                "No cached data found at %s. Run 'python scripts/preprocess_data.py' "
                "to preprocess and cache data for faster subsequent runs.",
                processed_dir,
            )

        from src.data.loader import load_metadata, load_raw_signals, load_scp_statements, aggregate_diagnostics
        from src.data.preprocessing import preprocess_pipeline
        from src.data.label_processing import encode_labels
        from src.data.split import train_val_test_split

        split_cfg = config["split"]

        metadata = load_metadata(data_cfg["raw_dir"])
        signals = load_raw_signals(
            metadata, data_cfg["raw_dir"],
            sampling_rate=data_cfg["sampling_rate"],
            max_samples=max_samples,
        )
        if max_samples is not None:
            metadata = metadata.iloc[:max_samples]

        scp_df = load_scp_statements(data_cfg["raw_dir"])
        diag_labels = aggregate_diagnostics(metadata, scp_df, data_cfg["label_type"])
        label_matrix, label_classes = encode_labels(diag_labels, label_type=data_cfg["label_type"])

        signals = preprocess_pipeline(signals, config)

        splits = train_val_test_split(
            metadata,
            val_fold=split_cfg["val_fold"],
            test_fold=split_cfg["test_fold"],
        )

    # ── Build DataLoaders ────────────────────────────────────
    dataloaders = {}
    for split_name in ["train", "val", "test"]:
        idx = splits[split_name]
        dataset = ECGDataset(signals[idx], label_matrix[idx])
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=(split_name == "train"),
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=train_cfg.get("pin_memory", False),
            drop_last=(split_name == "train"),
        )
        logger.info(
            "Created %s DataLoader: %d samples, %d batches",
            split_name, len(dataset), len(dataloaders[split_name]),
        )

    return dataloaders
