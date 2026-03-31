"""
Sequential multi-model training for ECG classification.

Trains all 7 models sequentially and produces a comparison summary,
matching the workflow from experiment_2.ipynb.

Usage:
    python scripts/run_all_models.py
    python scripts/run_all_models.py --models cnn_1d leadwise_cnn
    python scripts/run_all_models.py --max-samples 1000  # For debugging
"""

from src.data.label_processing import compute_class_weights, encode_labels
from src.data.loader import load_metadata, load_scp_statements, aggregate_diagnostics
from src.data.dataset import get_dataloaders, _is_cache_valid
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer
from src.training.scheduler import build_scheduler
from src.training.optimizer import build_optimizer
from src.training.loss import build_loss
from src.models import build_model
from src.utils import get_device
import argparse
import gc
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import yaml

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger(__name__)

# Models to train in order (matching notebook)
ALL_MODELS = [
    "cnn_1d",
    "leadwise_cnn",
    "pretrained_resnet",
    "lstm",
    "transformer",
    "cnn_lstm",
    "cnn_transformer",
]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def load_config(model_name: str) -> dict:
    """Load config for a specific model."""
    config_path = PROJECT_ROOT / "configs" / f"{model_name}.yaml"
    if not config_path.exists():
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
        logger.warning(
            "Config for %s not found, using default.yaml", model_name)

    with open(config_path) as f:
        return yaml.safe_load(f)


def train_single_model(
    model_name: str,
    dataloaders: dict,
    class_weights: torch.Tensor,
    label_classes: List[str],
    device: torch.device,
    max_samples: int = None,
) -> Dict[str, Any]:
    """
    Train a single model and return results.

    Args:
        model_name: Name of the model to train.
        dataloaders: Dictionary with train/val/test DataLoaders.
        class_weights: Tensor of class weights.
        label_classes: List of class names.
        device: Device to train on.
        max_samples: Optional limit for debugging.

    Returns:
        Dictionary with model results including metrics and history.
    """
    config = load_config(model_name)

    # Ensure model name is set correctly
    config["model"]["name"] = model_name

    seed = config.get("experiment", {}).get("seed", 42)
    deterministic = config.get("experiment", {}).get("deterministic", True)
    set_seed(seed, deterministic=deterministic)

    logger.info("=" * 80)
    logger.info("  Training: %s", model_name)
    logger.info("=" * 80)

    # Build model
    model = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")

    # Build training components
    criterion = build_loss(config, class_weights=class_weights)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Train
    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer,
        scheduler=scheduler, config=config, device=device,
    )

    history = trainer.fit(dataloaders["train"], dataloaders["val"])

    # Load best checkpoint for evaluation
    best_ckpt = Path(config["output"]["models_dir"]) / f"best_{model_name}.pt"
    if best_ckpt.exists():
        trainer.load_checkpoint(str(best_ckpt))

    # Evaluate
    evaluator = Evaluator(
        model=model, device=device,
        label_classes=label_classes,
    )

    test_results = evaluator.evaluate(
        dataloaders["test"], optimize_thresholds=True)

    # Save results
    results_dir = Path(config["output"]["results_dir"]) / model_name
    evaluator.save_results(test_results, str(results_dir))

    # Save history arrays (matching notebook format)
    np.save(results_dir / "history_train_loss.npy", history["train_loss"])
    np.save(results_dir / "history_val_loss.npy", history["val_loss"])
    if history.get("val_f1_macro"):
        np.save(results_dir / "history_val_f1_macro.npy",
                history["val_f1_macro"])

    metrics = test_results["metrics"]
    logger.info("  Results for %s:", model_name)
    logger.info("    F1 Macro:    %.4f", metrics.get("f1_macro", 0))
    logger.info("    ROC-AUC:     %.4f", metrics.get("roc_auc_macro", 0))
    logger.info("    Accuracy:    %.4f", metrics.get("subset_accuracy", 0))

    # Cleanup GPU memory
    del model, trainer, optimizer, scheduler, criterion
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_name": model_name,
        "metrics": metrics,
        "history": history,
        "config": config,
    }


def print_comparison_table(all_results: Dict[str, Dict]) -> None:
    """Print model comparison table to console."""
    print("\n" + "=" * 100)
    print(f"{'Model':<25} {'F1 Macro':>12} {'ROC-AUC':>12} {'Subset Acc':>12} {'Precision':>12} {'Recall':>12}")
    print("-" * 100)

    # Sort by F1 macro
    ranked = sorted(
        all_results.items(),
        key=lambda kv: kv[1]["metrics"].get("f1_macro", 0),
        reverse=True,
    )

    best_model = None
    best_f1 = 0.0

    for model_name, result in ranked:
        m = result["metrics"]
        f1 = m.get("f1_macro", 0)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model_name

        print(
            f"{model_name:<25} "
            f"{f1:>12.4f} "
            f"{m.get('roc_auc_macro', 0):>12.4f} "
            f"{m.get('subset_accuracy', 0):>12.4f} "
            f"{m.get('precision_macro', 0):>12.4f} "
            f"{m.get('recall_macro', 0):>12.4f}"
        )

    print("-" * 100)
    print(f"Best model: {best_model} (F1 Macro: {best_f1:.4f})")
    print("=" * 100 + "\n")


def run_all_models(
    models: List[str] = None,
    max_samples: int = None,
) -> Dict[str, Dict]:
    """
    Train all specified models sequentially.

    Args:
        models: List of model names to train. Defaults to ALL_MODELS.
        max_samples: Optional limit for debugging.

    Returns:
        Dictionary mapping model names to their results.
    """
    if models is None:
        models = ALL_MODELS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    device = get_device()
    logger.info("=" * 80)
    logger.info("  ECG Multi-Label Classification — Sequential Training")
    logger.info("  Device: %s", device)
    logger.info("  Models: %s", ", ".join(models))
    logger.info("=" * 80)

    # Load first model config for data paths
    base_config = load_config(models[0])

    # Load data once (reuse for all models)
    logger.info("Loading data...")
    dataloaders = get_dataloaders(base_config, max_samples=max_samples)

    # Load class weights and label classes
    processed_dir = base_config["data"].get("processed_dir", "data/processed")
    if max_samples is None and _is_cache_valid(processed_dir):
        class_weights = torch.tensor(
            np.load(Path(processed_dir) / "class_weights.npy")).to(device)
        with open(Path(processed_dir) / "label_classes.json") as f:
            label_classes = json.load(f)
        logger.info("Loaded class weights and label classes from cache")
    else:
        data_dir = base_config["data"]["raw_dir"]
        metadata = load_metadata(data_dir)
        if max_samples:
            metadata = metadata.iloc[:max_samples]
        scp_df = load_scp_statements(data_dir)
        diag_labels = aggregate_diagnostics(
            metadata, scp_df, base_config["data"]["label_type"])
        label_matrix, label_classes = encode_labels(
            diag_labels, label_type=base_config["data"]["label_type"])
        class_weights = torch.tensor(
            compute_class_weights(label_matrix)).to(device)

    logger.info("Classes: %s", label_classes)
    logger.info("Class weights: %s", class_weights.cpu().numpy().round(3))

    # Train each model
    all_results = {}
    for i, model_name in enumerate(models, 1):
        logger.info("\n[%d/%d] Starting %s", i, len(models), model_name)

        try:
            result = train_single_model(
                model_name=model_name,
                dataloaders=dataloaders,
                class_weights=class_weights,
                label_classes=label_classes,
                device=device,
                max_samples=max_samples,
            )
            all_results[model_name] = result
        except Exception as e:
            logger.error("Failed to train %s: %s", model_name, e)
            all_results[model_name] = {"error": str(e)}

    # Save comparison summary
    output_dir = Path(base_config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        name: result.get("metrics", {})
        for name, result in all_results.items()
        if "metrics" in result
    }
    with open(output_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info("Saved comparison to %s", output_dir / "model_comparison.json")

    # Print comparison table
    successful_results = {
        k: v for k, v in all_results.items() if "metrics" in v
    }
    if successful_results:
        print_comparison_table(successful_results)

    logger.info("=" * 80)
    logger.info("  FINISHED %d/%d MODELS",
                len(successful_results), len(models))
    logger.info("=" * 80)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all ECG classification models sequentially"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=ALL_MODELS,
        help="Specific models to train (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples for debugging",
    )
    args = parser.parse_args()

    run_all_models(models=args.models, max_samples=args.max_samples)
