"""
PyTorch Dataset and DataLoader utilities for ECG signals.

Provides ``ECGDataset`` and a convenience ``get_dataloaders`` function
that orchestrates loading, preprocessing, encoding, splitting, and
DataLoader creation from a config dict.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def get_dataloaders(
    config: Dict[str, Any],
    max_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from config.

    End-to-end pipeline:
    1. Load metadata and signals
    2. Aggregate diagnostic labels
    3. Encode labels as binary matrix
    4. Preprocess signals
    5. Split into train/val/test
    6. Create DataLoaders

    Args:
        config: Full configuration dictionary.
        max_samples: If set, only load this many samples (for debugging).

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``, each
        mapping to a ``DataLoader``.
    """
    from src.data.loader import load_metadata, load_raw_signals, load_scp_statements, aggregate_diagnostics
    from src.data.preprocessing import preprocess_pipeline
    from src.data.label_processing import encode_labels
    from src.data.split import train_val_test_split

    data_cfg = config["data"]
    split_cfg = config["split"]
    train_cfg = config["training"]

    # 1. Load metadata
    metadata = load_metadata(data_cfg["raw_dir"])

    # 2. Load signals
    signals = load_raw_signals(
        metadata, data_cfg["raw_dir"],
        sampling_rate=data_cfg["sampling_rate"],
        max_samples=max_samples,
    )

    if max_samples is not None:
        metadata = metadata.iloc[:max_samples]

    # 3. Aggregate and encode labels
    scp_df = load_scp_statements(data_cfg["raw_dir"])
    diag_labels = aggregate_diagnostics(metadata, scp_df, data_cfg["label_type"])
    label_matrix, label_classes = encode_labels(diag_labels, label_type=data_cfg["label_type"])

    # 4. Preprocess signals
    signals = preprocess_pipeline(signals, config)

    # 5. Split
    splits = train_val_test_split(
        metadata,
        val_fold=split_cfg["val_fold"],
        test_fold=split_cfg["test_fold"],
    )

    # 6. Create DataLoaders
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
