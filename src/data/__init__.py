"""
src.data — Data pipeline for PTB-XL ECG classification.

Submodules:
    download: Dataset download utilities.
    loader: Raw signal and metadata loading.
    preprocessing: Signal preprocessing pipeline.
    label_processing: Multi-label encoding.
    split: Train/val/test splitting.
    dataset: PyTorch Dataset and DataLoader creation.
"""

from src.data.loader import (
    load_metadata,
    load_raw_signals,
    load_scp_statements,
    aggregate_diagnostics,
)
from src.data.preprocessing import preprocess_pipeline
from src.data.label_processing import encode_labels, get_label_classes, compute_class_weights
from src.data.split import train_val_test_split, get_kfold_splits
from src.data.dataset import ECGDataset, get_dataloaders

__all__ = [
    "load_metadata",
    "load_raw_signals",
    "load_scp_statements",
    "aggregate_diagnostics",
    "preprocess_pipeline",
    "encode_labels",
    "get_label_classes",
    "compute_class_weights",
    "train_val_test_split",
    "get_kfold_splits",
    "ECGDataset",
    "get_dataloaders",
]
