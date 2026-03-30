"""
Ablation sweep runner for ECG multi-label classification.

Runs a small, high-impact experiment matrix and writes a ranked summary
to ``outputs/ablation/ablation_summary.json`` and ``.csv``.

Usage:
    python scripts/run_ablation.py --config configs/default.yaml
    python scripts/run_ablation.py --models pretrained_resnet leadwise_cnn
    python scripts/run_ablation.py --max-samples 2000
"""

from scripts.run_experiment import run_experiment
import argparse
import copy
import csv
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger(__name__)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override values into base dict."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build_ablation_grid() -> List[Dict[str, Any]]:
    """Return default ablation variants."""
    return [
        {
            "name": "baseline",
            "overrides": {
                "training": {
                    "loss": "weighted_bce",
                    "class_aware_sampling": {"enabled": False},
                },
                "experiment": {"deterministic": True},
            },
        },
        {
            "name": "throughput",
            "overrides": {
                "training": {
                    "loss": "weighted_bce",
                    "class_aware_sampling": {"enabled": False},
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                },
                "experiment": {"deterministic": False},
            },
        },
        {
            "name": "asymmetric_focal",
            "overrides": {
                "training": {
                    "loss": "asymmetric_focal",
                    "class_aware_sampling": {"enabled": False},
                    "asymmetric_focal_loss_params": {
                        "gamma_pos": 1.0,
                        "gamma_neg": 4.0,
                        "clip": 0.05,
                    },
                },
                "experiment": {"deterministic": True},
            },
        },
        {
            "name": "asymmetric_focal_sampler",
            "overrides": {
                "training": {
                    "loss": "asymmetric_focal",
                    "class_aware_sampling": {"enabled": True},
                    "asymmetric_focal_loss_params": {
                        "gamma_pos": 1.0,
                        "gamma_neg": 4.0,
                        "clip": 0.05,
                    },
                },
                "experiment": {"deterministic": True},
            },
        },
    ]


def _write_temp_config(config: Dict[str, Any]) -> str:
    """Persist config to a temporary yaml path and return it."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    with tmp as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return tmp.name


def _save_summary(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(rows, key=lambda x: x.get("f1_macro", 0.0), reverse=True)

    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(ranked, f, indent=2)

    csv_fields = [
        "rank",
        "model",
        "ablation",
        "f1_macro",
        "f1_micro",
        "roc_auc_macro",
        "subset_accuracy",
        "precision_macro",
        "recall_macro",
        "config_path",
    ]
    with open(out_dir / "ablation_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for i, row in enumerate(ranked, start=1):
            writer.writerow({"rank": i, **row})


def main(config_path: str, models: List[str], max_samples: int | None = None) -> None:
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    grid = _build_ablation_grid()
    all_rows: List[Dict[str, Any]] = []

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    for model_name in models:
        for variant in grid:
            run_cfg = _deep_update(base_cfg, variant["overrides"])

            run_cfg.setdefault("model", {})
            run_cfg["model"]["name"] = model_name

            run_cfg.setdefault("experiment", {})
            run_cfg["experiment"]["name"] = f"ablation_{variant['name']}_{model_name}"

            # Isolate artifacts per run to avoid overwriting.
            run_cfg.setdefault("output", {})
            base_out = Path(base_cfg.get("output", {}).get("dir", "outputs"))
            run_root = base_out / "ablation_runs" / variant["name"]
            run_cfg["output"]["dir"] = str(run_root)
            run_cfg["output"]["figures_dir"] = str(run_root / "figures")
            run_cfg["output"]["logs_dir"] = str(run_root / "logs")
            run_cfg["output"]["models_dir"] = str(run_root / "models")
            run_cfg["output"]["results_dir"] = str(run_root / "results")

            tmp_cfg_path = _write_temp_config(run_cfg)
            logger.info("Running model=%s | ablation=%s",
                        model_name, variant["name"])
            result = run_experiment(tmp_cfg_path, max_samples=max_samples)
            metrics = result.get("metrics", {})

            all_rows.append(
                {
                    "model": model_name,
                    "ablation": variant["name"],
                    "f1_macro": float(metrics.get("f1_macro", 0.0)),
                    "f1_micro": float(metrics.get("f1_micro", 0.0)),
                    "roc_auc_macro": float(metrics.get("roc_auc_macro", 0.0)),
                    "subset_accuracy": float(metrics.get("subset_accuracy", 0.0)),
                    "precision_macro": float(metrics.get("precision_macro", 0.0)),
                    "recall_macro": float(metrics.get("recall_macro", 0.0)),
                    "config_path": tmp_cfg_path,
                }
            )

    summary_dir = Path(base_cfg.get("output", {}).get(
        "dir", "outputs")) / "ablation"
    _save_summary(all_rows, summary_dir)

    ranked = sorted(all_rows, key=lambda x: x.get(
        "f1_macro", 0.0), reverse=True)
    print("\nTop ablation runs by f1_macro:")
    for i, row in enumerate(ranked[:8], start=1):
        print(
            f"{i:>2}. {row['model']:<18} {row['ablation']:<24} "
            f"F1={row['f1_macro']:.4f} AUC={row['roc_auc_macro']:.4f}"
        )
    print(f"\nSaved summary to: {summary_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ablation sweeps for ECG models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pretrained_resnet", "leadwise_cnn"],
        help="Model names to sweep (registry names).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    main(args.config, args.models, max_samples=args.max_samples)
