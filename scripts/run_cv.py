"""
Cross-validation experiment runner.

Runs K-fold cross-validation and aggregates results across folds.
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

from src.training.cross_validation import run_cross_validation
from src.utils import resolve_runtime_paths


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config_path: str, max_samples: int = None):
    """
    Run cross-validation experiment.

    Args:
        config_path: Path to YAML config file.
        max_samples: Limit samples for debugging.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = resolve_runtime_paths(
        config, project_root=PROJECT_ROOT, logger=logger)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("Starting cross-validation experiment")
    fold_results = run_cross_validation(config, max_samples=max_samples)

    # Save summary
    results_dir = Path(config["output"]["results_dir"]) / "cross_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": config["model"]["name"],
        "n_folds": len(fold_results),
        "val_losses": [r["best_val_loss"] for r in fold_results],
        "mean_val_loss": float(np.mean([r["best_val_loss"] for r in fold_results])),
        "std_val_loss": float(np.std([r["best_val_loss"] for r in fold_results])),
    }

    with open(results_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("CV Summary: %s", json.dumps(summary, indent=2))
    logger.info("Results saved to %s", results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-validation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    main(args.config, max_samples=args.max_samples)
