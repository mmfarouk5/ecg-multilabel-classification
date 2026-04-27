"""
Ensemble evaluation runner for ECG multi-label classification.

Builds an equal-weight (or custom-weight) probability ensemble across
multiple trained models and evaluates on the test split.

Usage:
    python scripts/run_ensemble.py
    python scripts/run_ensemble.py --models leadwise_cnn cnn_1d lstm
    python scripts/run_ensemble.py --weights 0.5 0.3 0.2
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders, _is_cache_valid
from src.data.label_processing import encode_labels, get_label_classes
from src.data.loader import aggregate_diagnostics, load_metadata, load_scp_statements
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_metrics, find_optimal_thresholds
from src.models import build_model
from src.utils import get_device

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["leadwise_cnn", "cnn_1d", "lstm"]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_model_config(model_name: str, fallback_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg_path = PROJECT_ROOT / "configs" / f"{model_name}.yaml"
    if cfg_path.exists():
        return _load_yaml(cfg_path)

    cfg = dict(fallback_config)
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["model"]["name"] = model_name
    logger.warning("Config for %s not found. Falling back to base config.", model_name)
    return cfg


def _load_label_classes(config: Dict[str, Any], max_samples: int | None) -> List[str]:
    processed_dir = config["data"].get("processed_dir", "data/processed")
    if max_samples is None and _is_cache_valid(processed_dir):
        with open(Path(processed_dir) / "label_classes.json") as f:
            return json.load(f)

    label_type = config["data"].get("label_type", "diagnostic_superclass")
    try:
        return get_label_classes(label_type=label_type)
    except ValueError:
        metadata = load_metadata(config["data"]["raw_dir"])
        if max_samples is not None:
            metadata = metadata.iloc[:max_samples]
        scp_df = load_scp_statements(config["data"]["raw_dir"])
        diag_labels = aggregate_diagnostics(metadata, scp_df, label_type)
        _, label_classes = encode_labels(diag_labels, label_type=label_type)
        return label_classes


def _normalize_weights(weights: Sequence[float], n_models: int) -> np.ndarray:
    if len(weights) != n_models:
        raise ValueError(
            f"weights length ({len(weights)}) must match number of models ({n_models})"
        )
    arr = np.asarray(weights, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError("weights must be non-negative")
    if arr.sum() == 0:
        raise ValueError("weights must sum to a positive value")
    return arr / arr.sum()


def _load_checkpoint_state(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    remapped = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        if clean_key.startswith("backbone."):
            clean_key = clean_key.replace("backbone.", "lead_backbone.", 1)
        remapped[clean_key] = value

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        logger.warning(
            "Checkpoint load for %s had mismatched keys | missing=%d unexpected=%d",
            ckpt_path.name, len(missing), len(unexpected),
        )


def run_ensemble(
    config_path: str,
    models: Sequence[str],
    weights: Sequence[float] | None = None,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    base_config = _load_yaml(Path(config_path))
    label_classes = _load_label_classes(base_config, max_samples=max_samples)

    dataloaders = get_dataloaders(base_config, max_samples=max_samples)
    device = get_device()

    if weights is None:
        model_weights = np.ones(len(models), dtype=np.float64) / len(models)
    else:
        model_weights = _normalize_weights(weights, len(models))

    logger.info("Running ensemble on device=%s", device)
    logger.info("Models: %s", ", ".join(models))
    logger.info("Weights: %s", np.round(model_weights, 4).tolist())

    probs_list = []
    labels_ref = None

    for model_name in models:
        model_config = _load_model_config(model_name, base_config)
        model = build_model(model_config).to(device).eval()

        ckpt_dir = Path(model_config["output"]["models_dir"])
        ckpt_path = ckpt_dir / f"best_{model_name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        _load_checkpoint_state(model, ckpt_path, device)
        evaluator = Evaluator(model=model, device=device, label_classes=label_classes)
        results = evaluator.predict(dataloaders["test"])

        if labels_ref is None:
            labels_ref = results["labels"]
        elif not np.array_equal(labels_ref, results["labels"]):
            raise ValueError(
                f"Test labels mismatch for model '{model_name}'. "
                "Ensure all models are evaluated on the same split."
            )

        probs_list.append(results["probabilities"])

    stacked_probs = np.stack(probs_list, axis=0)
    ensemble_probs = np.average(stacked_probs, axis=0, weights=model_weights)
    thresholds, _ = find_optimal_thresholds(labels_ref, ensemble_probs)
    ensemble_preds = (ensemble_probs >= thresholds).astype(np.float32)
    metrics = compute_metrics(
        y_true=labels_ref,
        y_pred=ensemble_preds,
        y_prob=ensemble_probs,
        label_classes=label_classes,
    )

    model_tag = "_".join(models)
    out_root = Path(base_config["output"]["results_dir"]) / f"ensemble_{model_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "metrics.json", "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()},
            f,
            indent=2,
        )
    np.save(out_root / "probabilities.npy", ensemble_probs)
    np.save(out_root / "predictions.npy", ensemble_preds)
    np.save(out_root / "labels.npy", labels_ref)
    np.save(out_root / "optimal_thresholds.npy", thresholds)

    with open(out_root / "ensemble_metadata.json", "w") as f:
        json.dump(
            {
                "models": list(models),
                "weights": model_weights.tolist(),
                "label_classes": label_classes,
            },
            f,
            indent=2,
        )

    logger.info("Ensemble results saved to %s", out_root)
    logger.info(
        "Ensemble metrics | f1_macro=%.4f roc_auc_macro=%.4f subset_accuracy=%.4f",
        metrics.get("f1_macro", 0.0),
        metrics.get("roc_auc_macro", 0.0),
        metrics.get("subset_accuracy", 0.0),
    )

    return {
        "metrics": metrics,
        "output_dir": str(out_root),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run probability ensemble on test split")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    run_ensemble(
        config_path=args.config,
        models=args.models,
        weights=args.weights,
        max_samples=args.max_samples,
    )
