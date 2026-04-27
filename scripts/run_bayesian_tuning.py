"""
Bayesian hyperparameter tuning for ECG multi-label classification.

Uses Optuna's TPE sampler (Bayesian optimization) to maximize validation
macro-F1 for a chosen model/config.

Usage:
    python scripts/run_bayesian_tuning.py --config configs/leadwise_cnn.yaml --n-trials 30
    python scripts/run_bayesian_tuning.py --config configs/leadwise_cnn.yaml --n-trials 50 --run-best
"""

import argparse
import copy
import gc
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd
import torch
import yaml

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import run_experiment
from src.data.dataset import get_dataloaders, _is_cache_valid
from src.data.label_processing import compute_class_weights, encode_labels
from src.data.loader import aggregate_diagnostics, load_metadata, load_scp_statements
from src.models import build_model
from src.training.loss import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils import get_device


logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_class_weights(config: Dict[str, Any], device: torch.device, max_samples: int | None) -> torch.Tensor:
    processed_dir = config["data"].get("processed_dir", "data/processed")
    if max_samples is None and _is_cache_valid(processed_dir):
        logger.info("Loading class weights from cache: %s", processed_dir)
        return torch.tensor(np.load(Path(processed_dir) / "class_weights.npy")).to(device)

    data_dir = config["data"]["raw_dir"]
    metadata = load_metadata(data_dir)
    if max_samples is not None:
        metadata = metadata.iloc[:max_samples]
    scp_df = load_scp_statements(data_dir)
    diag_labels = aggregate_diagnostics(
        metadata, scp_df, config["data"]["label_type"])
    label_matrix, _ = encode_labels(
        diag_labels, label_type=config["data"]["label_type"])
    return torch.tensor(compute_class_weights(label_matrix)).to(device)


def _apply_hparams(base_cfg: Dict[str, Any], hparams: Dict[str, Any], output_root: Path, trial_id: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    train_cfg = cfg["training"]
    model_params = cfg.setdefault("model", {}).setdefault("params", {})

    train_cfg["learning_rate"] = float(hparams["learning_rate"])
    train_cfg["weight_decay"] = float(hparams["weight_decay"])
    train_cfg["batch_size"] = int(hparams["batch_size"])
    train_cfg["optimizer"] = str(hparams["optimizer"])
    train_cfg["gradient_clip"] = float(hparams["gradient_clip"])
    train_cfg["scheduler"] = str(hparams["scheduler"])

    if train_cfg["scheduler"] == "cosine":
        train_cfg["scheduler_params"] = {
            "T_max": int(train_cfg.get("epochs", 50)),
            "eta_min": float(hparams["eta_min"]),
        }
    else:
        train_cfg["scheduler_params"] = {
            "patience": int(hparams["plateau_patience"]),
            "factor": float(hparams["plateau_factor"]),
            "min_lr": float(hparams["plateau_min_lr"]),
        }

    train_cfg.setdefault("class_aware_sampling", {})
    train_cfg["class_aware_sampling"]["enabled"] = bool(
        hparams["class_aware_sampling"])

    if train_cfg.get("loss") == "asymmetric_focal":
        af = train_cfg.setdefault("asymmetric_focal_loss_params", {})
        af["gamma_pos"] = float(hparams["gamma_pos"])
        af["gamma_neg"] = float(hparams["gamma_neg"])
        af["clip"] = float(hparams["clip"])

    if "dropout" in model_params and "dropout" in hparams:
        model_params["dropout"] = float(hparams["dropout"])

    exp_cfg = cfg.setdefault("experiment", {})
    exp_cfg["name"] = f"bayes_trial_{trial_id}"
    exp_cfg["deterministic"] = True
    exp_cfg["use_tensorboard"] = False
    exp_cfg["use_wandb"] = False

    trial_root = output_root / "trials" / f"trial_{trial_id:04d}"
    out_cfg = cfg.setdefault("output", {})
    out_cfg["dir"] = str(trial_root)
    out_cfg["figures_dir"] = str(trial_root / "figures")
    out_cfg["logs_dir"] = str(trial_root / "logs")
    out_cfg["models_dir"] = str(trial_root / "models")
    out_cfg["results_dir"] = str(trial_root / "results")

    return cfg


def _suggest_hparams(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    hparams: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "adam"]),
        "gradient_clip": trial.suggest_float("gradient_clip", 0.5, 2.0),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "plateau"]),
        "class_aware_sampling": trial.suggest_categorical("class_aware_sampling", [False, True]),
    }

    if hparams["scheduler"] == "cosine":
        hparams["eta_min"] = trial.suggest_float(
            "eta_min", 1e-6, 1e-4, log=True)
    else:
        hparams["plateau_patience"] = trial.suggest_int(
            "plateau_patience", 3, 8)
        hparams["plateau_factor"] = trial.suggest_float(
            "plateau_factor", 0.2, 0.8)
        hparams["plateau_min_lr"] = trial.suggest_float(
            "plateau_min_lr", 1e-6, 1e-4, log=True)

    if base_cfg["training"].get("loss") == "asymmetric_focal":
        hparams["gamma_pos"] = trial.suggest_float("gamma_pos", 0.5, 2.5)
        hparams["gamma_neg"] = trial.suggest_float("gamma_neg", 2.0, 6.0)
        hparams["clip"] = trial.suggest_float("clip", 0.0, 0.1)

    if "dropout" in base_cfg.get("model", {}).get("params", {}):
        hparams["dropout"] = trial.suggest_float("dropout", 0.1, 0.6)

    return hparams


def run_bayesian_tuning(
    config_path: str,
    n_trials: int = 30,
    timeout: Optional[int] = None,
    max_samples: Optional[int] = None,
    tune_epochs: Optional[int] = None,
    run_best: bool = False,
) -> Dict[str, Any]:
    base_config = _load_config(config_path)
    if tune_epochs is not None:
        base_config["training"]["epochs"] = int(tune_epochs)

    model_name = base_config["model"]["name"]
    seed = int(base_config.get("experiment", {}).get("seed", 42))
    study_name = f"bayes_{model_name}"

    output_root = Path(base_config["output"]["dir"]) / "hpo" / study_name
    output_root.mkdir(parents=True, exist_ok=True)

    device = get_device()
    class_weights = _load_class_weights(
        base_config, device=device, max_samples=max_samples)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.info("Starting Bayesian tuning | study=%s | model=%s | device=%s",
                study_name, model_name, device)

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name=study_name)

    def objective(trial: optuna.Trial) -> float:
        hparams = _suggest_hparams(trial, base_config)
        trial_config = _apply_hparams(
            base_config, hparams, output_root=output_root, trial_id=trial.number)
        set_seed(seed, deterministic=True)

        dataloaders = get_dataloaders(trial_config, max_samples=max_samples)
        model = build_model(trial_config)
        criterion = build_loss(trial_config, class_weights=class_weights)
        optimizer = build_optimizer(model, trial_config)
        scheduler = build_scheduler(optimizer, trial_config)

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=trial_config,
            device=device,
            writer=None,
        )
        history = trainer.fit(dataloaders["train"], dataloaders["val"])

        if history.get("val_f1_macro"):
            score = float(max(history["val_f1_macro"]))
            trial.set_user_attr("best_val_f1_macro", score)
        else:
            score = float(-min(history["val_loss"]))
            trial.set_user_attr("best_val_f1_macro", float("nan"))

        trial.set_user_attr("best_val_loss", float(min(history["val_loss"])))

        del model, criterion, optimizer, scheduler, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return score

    study.optimize(objective, n_trials=n_trials,
                   timeout=timeout, show_progress_bar=True)

    best_hparams = _suggest_hparams(
        trial=optuna.trial.FixedTrial(study.best_trial.params),
        base_cfg=base_config,
    )
    best_config = _apply_hparams(
        base_cfg=base_config,
        hparams=best_hparams,
        output_root=output_root,
        trial_id=int(study.best_trial.number),
    )
    best_config_path = output_root / "best_config.yaml"
    with open(best_config_path, "w") as f:
        yaml.safe_dump(best_config, f, sort_keys=False)

    summary = {
        "study_name": study_name,
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "best_params": study.best_trial.params,
        "best_config_path": str(best_config_path),
    }
    with open(output_root / "best_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    trial_rows = []
    for t in study.trials:
        row = {
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
        }
        row.update({f"param_{k}": v for k, v in t.params.items()})
        row.update({f"attr_{k}": v for k, v in t.user_attrs.items()})
        trial_rows.append(row)
    pd.DataFrame(trial_rows).to_csv(output_root / "trials.csv", index=False)

    logger.info("Best trial #%d | score=%.4f",
                study.best_trial.number, study.best_value)
    logger.info("Saved tuning outputs to %s", output_root)

    if run_best:
        final_result = run_experiment(
            str(best_config_path), max_samples=max_samples)
        with open(output_root / "best_run_metrics.json", "w") as f:
            json.dump(final_result.get("metrics", {}), f, indent=2)
        logger.info("Best-config experiment completed.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter tuning with Optuna")
    parser.add_argument("--config", type=str,
                        default="configs/leadwise_cnn.yaml")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Total tuning timeout (seconds)")
    parser.add_argument("--max-samples", type=int,
                        default=None, help="Use subset for faster tuning")
    parser.add_argument("--tune-epochs", type=int, default=None,
                        help="Override training epochs during tuning")
    parser.add_argument("--run-best", action="store_true",
                        help="Run full experiment with best found config")
    args = parser.parse_args()

    run_bayesian_tuning(
        config_path=args.config,
        n_trials=args.n_trials,
        timeout=args.timeout,
        max_samples=args.max_samples,
        tune_epochs=args.tune_epochs,
        run_best=args.run_best,
    )
