"""
Validation script for Phase 1 (Data Pipeline) & Phase 2 (Preprocessing).

Validates:
1. Metadata loading
2. Signal loading (subset of 200 samples)
3. Preprocessing pipeline
4. Label encoding + distribution
5. Train/val/test split
6. DataLoader creation + batch iteration
7. ECG signal visualization (raw vs preprocessed)
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("validate")

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    # Load config
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info("✓ Config loaded from %s", config_path)

    data_dir = str(PROJECT_ROOT / config["data"]["raw_dir"])
    sr = config["data"]["sampling_rate"]

    # ---- 1. Load metadata ----
    from src.data.loader import load_metadata, load_scp_statements, aggregate_diagnostics, load_raw_signals
    metadata = load_metadata(data_dir)
    logger.info("✓ Metadata shape: %s", metadata.shape)
    logger.info("  First 3 columns: %s", list(metadata.columns[:5]))

    # ---- 2. Load signals (200-sample subset) ----
    MAX_SAMPLES = 200
    signals_raw = load_raw_signals(metadata, data_dir, sampling_rate=sr, max_samples=MAX_SAMPLES)
    expected_seq_len = 1000 if sr == 100 else 5000
    assert signals_raw.shape == (MAX_SAMPLES, expected_seq_len, 12), \
        f"Unexpected shape: {signals_raw.shape}"
    logger.info("✓ Signals loaded: shape=%s, dtype=%s", signals_raw.shape, signals_raw.dtype)

    # ---- 3. Aggregate & encode labels ----
    from src.data.label_processing import encode_labels, get_label_distribution, compute_class_weights
    scp_df = load_scp_statements(data_dir)
    metadata_subset = metadata.iloc[:MAX_SAMPLES]
    diag_labels = aggregate_diagnostics(metadata_subset, scp_df, config["data"]["label_type"])
    label_matrix, label_classes = encode_labels(diag_labels, label_type=config["data"]["label_type"])
    assert label_matrix.shape == (MAX_SAMPLES, len(label_classes)), \
        f"Unexpected label shape: {label_matrix.shape}"
    logger.info("✓ Labels encoded: shape=%s, classes=%s", label_matrix.shape, label_classes)

    # Label distribution
    dist = get_label_distribution(label_matrix, label_classes)
    logger.info("  Label distribution:\n%s", dist.to_string(index=False))

    # Check no all-zero rows
    empty_rows = (label_matrix.sum(axis=1) == 0).sum()
    logger.info("  Records with no labels: %d", empty_rows)

    # Class weights
    weights = compute_class_weights(label_matrix)
    logger.info("✓ Class weights: %s", dict(zip(label_classes, weights.round(3))))

    # ---- 4. Preprocess signals ----
    from src.data.preprocessing import preprocess_pipeline
    signals_clean = preprocess_pipeline(signals_raw.copy(), config)
    assert signals_clean.shape == signals_raw.shape, \
        f"Shape mismatch after preprocessing: {signals_clean.shape} vs {signals_raw.shape}"
    logger.info("✓ Preprocessing passed. Output shape: %s", signals_clean.shape)

    # ---- 5. Split ----
    from src.data.split import train_val_test_split
    splits = train_val_test_split(
        metadata_subset,
        val_fold=config["split"]["val_fold"],
        test_fold=config["split"]["test_fold"],
    )
    total_split = sum(len(v) for v in splits.values())
    assert total_split == MAX_SAMPLES, f"Split total {total_split} != {MAX_SAMPLES}"
    logger.info("✓ Split — train: %d, val: %d, test: %d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # ---- 6. DataLoader ----
    from src.data.dataset import ECGDataset
    from torch.utils.data import DataLoader
    train_ds = ECGDataset(signals_clean[splits["train"]], label_matrix[splits["train"]])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    batch_x, batch_y = next(iter(train_loader))
    logger.info("✓ DataLoader batch — X: %s, Y: %s", batch_x.shape, batch_y.shape)
    assert batch_x.shape[1] == 12, "Expected 12 leads (channels first)"
    assert batch_x.shape[2] == expected_seq_len, f"Expected seq_len={expected_seq_len}"

    # ---- 7. Visualize ----
    fig_dir = PROJECT_ROOT / config["output"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    sample_idx = 0
    leads_to_show = [0, 1, 6, 7]  # I, II, V1, V2

    for ax_idx, lead in enumerate(leads_to_show):
        axes[ax_idx].plot(signals_raw[sample_idx, :, lead], alpha=0.5, label="Raw", linewidth=0.8)
        axes[ax_idx].plot(signals_clean[sample_idx, :, lead], label="Preprocessed", linewidth=0.8)
        axes[ax_idx].set_ylabel(LEAD_NAMES[lead])
        axes[ax_idx].legend(loc="upper right", fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)

    axes[0].set_title("ECG Signal: Raw vs Preprocessed")
    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    fig_path = fig_dir / "sample_ecg.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("✓ ECG visualization saved to %s", fig_path)

    logger.info("\n" + "=" * 60)
    logger.info("  ALL PHASE 1 & 2 VALIDATIONS PASSED ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
