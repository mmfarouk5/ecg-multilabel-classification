"""
Preprocess and cache ECG data to disk.

Run this ONCE before training experiments. It loads raw signals,
preprocesses them, encodes labels, computes split indices, and saves
everything to ``data/processed/`` as ``.npy`` files.

Usage:
    python scripts/preprocess_data.py --config configs/default.yaml

Subsequent experiment runs will load directly from the cached files,
skipping the expensive wfdb loading and signal filtering steps.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_processing import encode_labels, get_label_classes, compute_class_weights
from src.data.loader import load_metadata, load_raw_signals, load_scp_statements, aggregate_diagnostics
from src.data.preprocessing import preprocess_pipeline
from src.data.split import train_val_test_split
from src.utils import resolve_runtime_paths


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_and_save(config_path: str) -> None:
    """
    Load raw data, preprocess, and save to disk.

    Saves the following files to ``data/processed/``:
    - ``signals.npy`` — preprocessed signals ``(N, seq_len, 12)``
    - ``labels.npy`` — multi-label binary matrix ``(N, num_classes)``
    - ``label_classes.json`` — ordered class names
    - ``class_weights.npy`` — inverse-frequency class weights
    - ``train_indices.npy`` — train split indices
    - ``val_indices.npy`` — validation split indices
    - ``test_indices.npy`` — test split indices
    - ``metadata.json`` — preprocessing metadata (config snapshot)

    Args:
        config_path: Path to YAML config file.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = resolve_runtime_paths(
        config, project_root=PROJECT_ROOT, logger=logger)

    data_cfg = config["data"]
    split_cfg = config["split"]

    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load metadata ─────────────────────────────────────
    logger.info("Step 1/5: Loading metadata...")
    metadata = load_metadata(data_cfg["raw_dir"])
    logger.info("  → %d records loaded", len(metadata))

    # ── 2. Load raw signals ──────────────────────────────────
    logger.info("Step 2/5: Loading raw signals (this takes a few minutes)...")
    signals = load_raw_signals(
        metadata, data_cfg["raw_dir"],
        sampling_rate=data_cfg["sampling_rate"],
    )
    logger.info("  → Shape: %s", signals.shape)

    # ── 3. Preprocess ────────────────────────────────────────
    logger.info("Step 3/5: Preprocessing signals...")
    signals = preprocess_pipeline(signals, config)
    logger.info("  → Preprocessed shape: %s", signals.shape)

    # ── 4. Labels ────────────────────────────────────────────
    logger.info("Step 4/5: Encoding labels...")
    scp_df = load_scp_statements(data_cfg["raw_dir"])
    diag_labels = aggregate_diagnostics(
        metadata, scp_df, data_cfg["label_type"])
    label_matrix, label_classes = encode_labels(
        diag_labels, label_type=data_cfg["label_type"])
    class_weights = compute_class_weights(label_matrix)

    # Label distribution
    for i, cls in enumerate(label_classes):
        count = int(label_matrix[:, i].sum())
        logger.info("  → %s: %d samples (%.1f%%)", cls,
                    count, count / len(label_matrix) * 100)

    # ── 5. Splits ────────────────────────────────────────────
    logger.info("Step 5/5: Computing splits...")
    splits = train_val_test_split(
        metadata,
        val_fold=split_cfg["val_fold"],
        test_fold=split_cfg["test_fold"],
    )

    # ── Save everything ──────────────────────────────────────
    logger.info("Saving to %s ...", processed_dir)

    np.save(processed_dir / "signals.npy", signals)
    np.save(processed_dir / "labels.npy", label_matrix)
    np.save(processed_dir / "class_weights.npy", class_weights)
    np.save(processed_dir / "train_indices.npy", splits["train"])
    np.save(processed_dir / "val_indices.npy", splits["val"])
    np.save(processed_dir / "test_indices.npy", splits["test"])

    with open(processed_dir / "label_classes.json", "w") as f:
        json.dump(label_classes, f, indent=2)

    # Save metadata snapshot so we know how this was processed
    meta = {
        "sampling_rate": data_cfg["sampling_rate"],
        "label_type": data_cfg["label_type"],
        "preprocessing": config.get("preprocessing", {}),
        "split": split_cfg,
        "n_samples": int(signals.shape[0]),
        "signal_shape": list(signals.shape),
        "n_classes": len(label_classes),
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }
    with open(processed_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_mb = signals.nbytes / 1024 / 1024
    logger.info("=" * 50)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("  Signals:  %s (%.0f MB)", signals.shape, total_mb)
    logger.info("  Labels:   %s", label_matrix.shape)
    logger.info("  Train:    %d | Val: %d | Test: %d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))
    logger.info("  Saved to: %s", processed_dir)
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess PTB-XL data and save to disk")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config file (only data/preprocessing/split sections are used)")
    args = parser.parse_args()

    preprocess_and_save(args.config)
