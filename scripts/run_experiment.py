"""
Experiment runner for ECG classification.

End-to-end script that trains a model and runs evaluation,
producing all outputs (models, logs, figures, metrics).
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders, _is_cache_valid, _load_from_cache
from src.data.label_processing import compute_class_weights, encode_labels, get_label_classes
from src.data.loader import load_metadata, load_raw_signals, load_scp_statements, aggregate_diagnostics
from src.evaluation.confusion_matrix import plot_confusion_matrices, plot_multilabel_confusion_summary
from src.evaluation.evaluator import Evaluator
from src.evaluation.plots import plot_roc_curves, plot_training_history, plot_precision_recall_curves
from src.models import build_model
from src.training.loss import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils import get_device, resolve_runtime_paths, zip_output_directories


logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def run_experiment(
    config_path: str,
    max_samples: int = None,
    create_archives: bool = True,
) -> dict:
    """
    Run a full experiment: train + evaluate + generate outputs.

    Args:
        config_path: Path to YAML config file.
        max_samples: Limit samples for debugging.
        create_archives: If True, create zip archives for output folders.

    Returns:
        Dictionary with training history and evaluation metrics.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = resolve_runtime_paths(
        config, project_root=PROJECT_ROOT, logger=logger)

    seed = config.get("experiment", {}).get("seed", 42)
    deterministic = config.get("experiment", {}).get("deterministic", True)
    set_seed(seed, deterministic=deterministic)

    exp_name = config.get("experiment", {}).get("name", "default")
    model_name = config["model"]["name"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    device = get_device()
    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s | MODEL: %s | DEVICE: %s",
                exp_name, model_name, device)
    logger.info("Reproducibility mode (deterministic): %s", deterministic)
    logger.info("=" * 60)

    # ── Data Pipeline ────────────────────────────────────────
    dataloaders = get_dataloaders(config, max_samples=max_samples)

    # Load class weights + label classes — from cache or compute
    processed_dir = config["data"].get("processed_dir", "data/processed")
    if max_samples is None and _is_cache_valid(processed_dir):
        import json
        class_weights = torch.tensor(
            np.load(Path(processed_dir) / "class_weights.npy")).to(device)
        with open(Path(processed_dir) / "label_classes.json") as f:
            label_classes = json.load(f)
        logger.info("Loaded class weights and label classes from cache")
    else:
        data_dir = config["data"]["raw_dir"]
        metadata = load_metadata(data_dir)
        if max_samples:
            metadata = metadata.iloc[:max_samples]
        scp_df = load_scp_statements(data_dir)
        diag_labels = aggregate_diagnostics(
            metadata, scp_df, config["data"]["label_type"])
        label_matrix, label_classes = encode_labels(
            diag_labels, label_type=config["data"]["label_type"])
        class_weights = torch.tensor(
            compute_class_weights(label_matrix)).to(device)

    # ── Model ────────────────────────────────────────────────
    model = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")

    # ── Training ─────────────────────────────────────────────
    criterion = build_loss(config, class_weights=class_weights)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    writer = None
    if config.get("experiment", {}).get("use_tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(config["output"]["logs_dir"]) / model_name
        writer = SummaryWriter(log_dir=str(log_dir))

    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer,
        scheduler=scheduler, config=config, device=device, writer=writer,
    )

    history = trainer.fit(dataloaders["train"], dataloaders["val"])

    if writer:
        writer.close()

    # ── Evaluation ───────────────────────────────────────────
    # Load best model
    best_ckpt = Path(config["output"]["models_dir"]) / f"best_{model_name}.pt"
    if best_ckpt.exists():
        trainer.load_checkpoint(str(best_ckpt))

    evaluator = Evaluator(
        model=model, device=device,
        label_classes=label_classes,
    )

    test_results = evaluator.evaluate(
        dataloaders["test"], optimize_thresholds=True)

    # ── Outputs ──────────────────────────────────────────────
    figures_dir = Path(config["output"]["figures_dir"]) / model_name
    results_dir = Path(config["output"]["results_dir"]) / model_name

    # Save results
    evaluator.save_results(test_results, str(results_dir))

    # ROC curves
    plot_roc_curves(
        test_results["labels"], test_results["probabilities"],
        label_classes,
        save_path=str(figures_dir / "roc_curves.png"),
        title=f"ROC Curves — {model_name}",
    )

    # PR curves
    plot_precision_recall_curves(
        test_results["labels"], test_results["probabilities"],
        label_classes,
        save_path=str(figures_dir / "pr_curves.png"),
        title=f"Precision-Recall — {model_name}",
    )

    # Confusion matrices
    plot_confusion_matrices(
        test_results["labels"], test_results["predictions"],
        label_classes,
        save_path=str(figures_dir / "confusion_matrices.png"),
    )

    plot_multilabel_confusion_summary(
        test_results["labels"], test_results["predictions"],
        label_classes,
        save_path=str(figures_dir / "confusion_summary.png"),
    )

    # Training history
    plot_training_history(
        history,
        save_path=str(figures_dir / "training_history.png"),
        title=f"Training History — {model_name}",
    )

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE: %s / %s", exp_name, model_name)
    logger.info("  Results: %s", results_dir)
    logger.info("  Figures: %s", figures_dir)

    archives = {}
    if create_archives:
        archives = zip_output_directories(config["output"])
        if archives:
            logger.info("  Archives:")
            for key, archive_path in archives.items():
                logger.info("    %s -> %s", key, archive_path)

    logger.info("=" * 60)

    return {
        "history": history,
        "metrics": test_results["metrics"],
        "archives": {k: str(v) for k, v in archives.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ECG classification experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_experiment(args.config, max_samples=args.max_samples)
