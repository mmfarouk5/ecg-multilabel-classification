"""
Training entry point for ECG classification.

Sets up the full training pipeline from config: loads data, builds
model/loss/optimizer/scheduler, and runs training.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders, _is_cache_valid, _load_from_cache
from src.data.label_processing import compute_class_weights, encode_labels
from src.data.loader import load_metadata, load_scp_statements, aggregate_diagnostics
from src.models import build_model
from src.training.loss import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils import get_device

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(config_path: str, max_samples: int = None) -> dict:
    """
    Run the full training pipeline.

    Args:
        config_path: Path to the YAML config file.
        max_samples: If set, only use this many samples (for quick testing).

    Returns:
        Training history dictionary.
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    device = get_device()
    logger.info("Device: %s | Seed: %d", device, seed)

    # Data
    dataloaders = get_dataloaders(config, max_samples=max_samples)

    # Load class weights — from cache if available, else compute
    processed_dir = config["data"].get("processed_dir", "data/processed")
    if max_samples is None and _is_cache_valid(processed_dir):
        class_weights = torch.tensor(np.load(Path(processed_dir) / "class_weights.npy")).to(device)
        logger.info("Loaded class weights from cache")
    else:
        data_dir = config["data"]["raw_dir"]
        metadata = load_metadata(data_dir)
        if max_samples:
            metadata = metadata.iloc[:max_samples]
        scp_df = load_scp_statements(data_dir)
        diag_labels = aggregate_diagnostics(metadata, scp_df, config["data"]["label_type"])
        label_matrix, _ = encode_labels(diag_labels, label_type=config["data"]["label_type"])
        class_weights = torch.tensor(compute_class_weights(label_matrix)).to(device)

    # Model
    model = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s | Parameters: %s", config["model"]["name"], f"{total_params:,}")

    # Loss, Optimizer, Scheduler
    criterion = build_loss(config, class_weights=class_weights)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # TensorBoard writer
    writer = None
    if config.get("experiment", {}).get("use_tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(config["output"]["logs_dir"]) / config["experiment"].get("name", "default")
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info("TensorBoard logging to %s", log_dir)

    # Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        writer=writer,
    )

    # Train
    history = trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
    )

    if writer:
        writer.close()

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG classification model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to load (for debugging)",
    )
    args = parser.parse_args()

    train(args.config, max_samples=args.max_samples)
