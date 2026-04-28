"""
ECG Diagnosis AI — FastAPI Web Server

A web server for ECG multi-label classification using FastAPI.
Uses an ensemble of Leadwise CNN + CNN 1D + LSTM for inference.
Falls back to any available single model if ensemble checkpoints are missing.

Usage:
    uvicorn webapp.main:app --port 8000
    # or
    python -m webapp.main [--port 8000]
"""

from src.utils import resolve_runtime_paths
import argparse
import csv
import io
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBAPP_DIR / "static"

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _pick_path(env_var: str, candidates: List[Path]) -> Path:
    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value).expanduser()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# ── Ensemble configuration ─────────────────────────────────────
ENSEMBLE_MODELS = ["leadwise_cnn", "cnn_1d", "lstm"]
ENSEMBLE_WEIGHTS: Optional[List[float]] = None  # Equal weights; set e.g. [0.4, 0.35, 0.25] to customise

CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = _pick_path(
    "ECG_CHECKPOINT_PATH",  # still honour legacy env var
    [
        PROJECT_ROOT / "outputs" / "models",
        Path("/kaggle/working/outputs/models"),
    ],
)
# If the env var pointed at a file, use its parent directory
if MODELS_DIR.is_file():
    MODELS_DIR = MODELS_DIR.parent

PROCESSED_DIR = _pick_path(
    "ECG_PROCESSED_DIR",
    [
        PROJECT_ROOT / "data" / "processed",
        Path("/kaggle/working/data/processed"),
    ],
)
MODEL_COMPARISON_PATH = _pick_path(
    "ECG_MODEL_COMPARISON_PATH",
    [
        PROJECT_ROOT / "outputs" / "results" / "model_comparison.json",
        PROJECT_ROOT / "outputs" / "model_comparison.json",
        Path("/kaggle/working/outputs/results/model_comparison.json"),
    ],
)

# ── Label definitions ──────────────────────────────────────────
LABEL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LABEL_DESCRIPTIONS = {
    "NORM": {
        "full_name": "Normal ECG",
        "description": "No significant cardiac abnormalities detected. The heart rhythm, rate, and waveform morphology are within normal limits.",
        "severity": "normal",
        "color": "#00e676",
    },
    "MI": {
        "full_name": "Myocardial Infarction",
        "description": "Evidence of myocardial infarction (heart attack). Characterized by ST-segment elevation/depression, pathological Q waves, or T-wave inversions indicating myocardial damage.",
        "severity": "critical",
        "color": "#ff5252",
    },
    "STTC": {
        "full_name": "ST/T Changes",
        "description": "ST-segment and/or T-wave changes detected. These may indicate ischemia, electrolyte abnormalities, or other cardiac conditions affecting repolarization.",
        "severity": "warning",
        "color": "#ffa726",
    },
    "CD": {
        "full_name": "Conduction Disturbance",
        "description": "Abnormal conduction pathways detected. This includes bundle branch blocks, AV blocks, or other delays in the electrical conduction system of the heart.",
        "severity": "warning",
        "color": "#ffca28",
    },
    "HYP": {
        "full_name": "Hypertrophy",
        "description": "Signs of cardiac hypertrophy (enlarged heart chambers). High-voltage QRS complexes and specific waveform patterns suggest increased cardiac muscle mass.",
        "severity": "warning",
        "color": "#ab47bc",
    },
}

ECG_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

# ── Global model / data caches ─────────────────────────────────
_ensemble_models: Optional[List] = None
_ensemble_names: Optional[List[str]] = None
_ensemble_config = None
_signals = None
_labels = None
_test_indices = None
_optimal_thresholds = None


def get_config() -> Dict[str, Any]:
    """Load the first available ensemble model config for preprocessing."""
    global _ensemble_config
    if _ensemble_config is not None:
        return _ensemble_config

    # Use the first ensemble model's config as the canonical preprocessing config
    for name in ENSEMBLE_MODELS:
        cfg_path = CONFIGS_DIR / f"{name}.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                _ensemble_config = yaml.safe_load(f)
            _ensemble_config = resolve_runtime_paths(
                _ensemble_config, project_root=PROJECT_ROOT, logger=logger)
            return _ensemble_config

    raise FileNotFoundError(
        f"No config found for any ensemble model {ENSEMBLE_MODELS} in {CONFIGS_DIR}"
    )


def get_ensemble_models() -> List:
    """Load ensemble models, falling back to any available single model."""
    global _ensemble_models, _ensemble_names
    if _ensemble_models is not None:
        return _ensemble_models

    from src.inference.predict import load_trained_model

    device = torch.device("cpu")
    loaded_models = []
    loaded_names = []

    for name in ENSEMBLE_MODELS:
        cfg_path = CONFIGS_DIR / f"{name}.yaml"
        ckpt_path = MODELS_DIR / f"best_{name}.pt"
        if not cfg_path.exists() or not ckpt_path.exists():
            logger.warning(
                "Skipping %s — config=%s checkpoint=%s",
                name, cfg_path.exists(), ckpt_path.exists(),
            )
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = load_trained_model(str(ckpt_path), config=cfg, device=device)
        loaded_models.append(model)
        loaded_names.append(name)
        logger.info("✓ Loaded %s", name)

    if not loaded_models:
        raise RuntimeError(
            f"No checkpoints found for any ensemble model in {MODELS_DIR}. "
            f"Expected: {', '.join(f'best_{n}.pt' for n in ENSEMBLE_MODELS)}"
        )

    _ensemble_models = loaded_models
    _ensemble_names = loaded_names
    logger.info(
        "✓ Ensemble ready: %d model(s) — %s",
        len(loaded_models), ", ".join(loaded_names),
    )
    return _ensemble_models


def get_optimal_thresholds() -> np.ndarray:
    """Load per-class optimal thresholds from saved results."""
    global _optimal_thresholds
    if _optimal_thresholds is not None:
        return _optimal_thresholds

    from src.inference.predict import load_optimal_thresholds

    # Try ensemble thresholds first, then single-model
    results_root = PROJECT_ROOT / "outputs" / "results"
    ensemble_dir = results_root / "ensemble_leadwise_cnn_cnn_1d_lstm"
    single_dir = results_root / "leadwise_cnn"

    for candidate in [ensemble_dir, single_dir]:
        thresh_path = candidate / "optimal_thresholds.npy"
        if thresh_path.exists():
            _optimal_thresholds = load_optimal_thresholds(
                str(candidate), num_classes=len(LABEL_CLASSES)
            )
            logger.info("✓ Loaded optimal thresholds from %s", candidate.name)
            return _optimal_thresholds

    _optimal_thresholds = np.full(len(LABEL_CLASSES), 0.5, dtype=np.float32)
    logger.info("Using default threshold=0.5 (no saved thresholds found)")
    return _optimal_thresholds


def get_sample_data():
    global _signals, _labels, _test_indices
    if _signals is not None:
        return _signals, _labels, _test_indices

    logger.info("Loading processed data for sample endpoint ...")
    _signals = np.load(str(PROCESSED_DIR / "signals.npy"), mmap_mode="r")
    _labels = np.load(str(PROCESSED_DIR / "labels.npy"))
    _test_indices = np.load(str(PROCESSED_DIR / "test_indices.npy"))
    logger.info(
        "✓ Loaded %d signals, %d test indices", len(
            _signals), len(_test_indices)
    )
    return _signals, _labels, _test_indices


@torch.inference_mode()
def run_inference(signal: np.ndarray) -> Dict[str, Any]:
    """Run ensemble preprocessing + inference on a (1000, 12) ECG signal.

    Averages sigmoid probabilities across all loaded models and applies
    per-class optimal thresholds for improved accuracy.
    """
    from src.inference.predict import predict_ensemble

    models = get_ensemble_models()
    config = get_config()
    thresholds = get_optimal_thresholds()

    result = predict_ensemble(
        models, signal, config,
        weights=ENSEMBLE_WEIGHTS,
        threshold=thresholds,
        label_classes=LABEL_CLASSES,
        device=torch.device("cpu"),
    )

    probs = result["probabilities"]
    preds = result["predictions"]

    classes = []
    for i, cls_name in enumerate(LABEL_CLASSES):
        info = LABEL_DESCRIPTIONS[cls_name]
        prob = float(probs[i])
        predicted = bool(preds[i])
        t = float(thresholds[i]) if isinstance(thresholds, np.ndarray) else float(thresholds)
        confidence = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"

        classes.append({
            "name": cls_name,
            "full_name": info["full_name"],
            "description": info["description"],
            "severity": info["severity"],
            "color": info["color"],
            "probability": round(prob, 4),
            "predicted": predicted,
            "confidence": confidence,
            "threshold": round(t, 3),
        })

    predicted_classes = [c["full_name"] for c in classes if c["predicted"]]
    return {
        "classes": classes,
        "predicted_classes": predicted_classes,
        "num_predicted": len(predicted_classes),
        "ensemble_models": _ensemble_names or [],
    }


def parse_csv_signal(body: bytes) -> np.ndarray:
    """Parse CSV body into a (N, 12) numpy array."""
    text = body.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    rows = []
    for row in reader:
        if not row or all(v.strip() == "" for v in row):
            continue
        rows.append([float(v) for v in row])

    signal = np.array(rows, dtype=np.float32)
    if signal.shape[1] != 12:
        raise ValueError(
            f"Expected 12 columns (leads), got {signal.shape[1]}.")
    if signal.shape[0] < 100:
        raise ValueError(
            f"Signal too short. Expected ~1000 timesteps, got {signal.shape[0]}."
        )
    # Pad or truncate to 1000
    if signal.shape[0] > 1000:
        signal = signal[:1000]
    elif signal.shape[0] < 1000:
        pad = np.zeros((1000 - signal.shape[0], 12), dtype=np.float32)
        signal = np.concatenate([signal, pad], axis=0)
    return signal


def _build_ground_truth(label):
    """Build ground truth list from a label vector."""
    ground_truth = []
    for i, cls_name in enumerate(LABEL_CLASSES):
        info = LABEL_DESCRIPTIONS[cls_name]
        ground_truth.append({
            "name": cls_name,
            "full_name": info["full_name"],
            "present": bool(label[i]),
        })
    return ground_truth


# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(
    title="ECG Diagnosis AI",
    description="Multi-label ECG classification using deep learning",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup():
    """Pre-load ensemble models, thresholds, and sample data on startup."""
    logger.info("Pre-loading ensemble models...")
    get_ensemble_models()
    logger.info("Pre-loading optimal thresholds...")
    get_optimal_thresholds()
    logger.info("Pre-loading sample data...")
    get_sample_data()
    logger.info("=" * 50)
    logger.info("  🫀 ECG Diagnosis AI Server — Ready (Ensemble Mode)")
    logger.info("=" * 50)


# ── Routes ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend page."""
    return HTMLResponse(content=(STATIC_DIR / "index.html").read_text())


@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "ensemble_loaded": _ensemble_models is not None,
        "ensemble_size": len(_ensemble_models) if _ensemble_models else 0,
        "ensemble_models": _ensemble_names or [],
    }


@app.get("/api/model-info")
async def model_info():
    """Return model comparison metrics and architecture details."""
    result = {
        "current_model": "Ensemble (" + " + ".join(_ensemble_names or ENSEMBLE_MODELS) + ")",
        "ensemble_models": _ensemble_names or ENSEMBLE_MODELS,
        "label_classes": LABEL_CLASSES,
        "label_details": LABEL_DESCRIPTIONS,
        "lead_names": ECG_LEAD_NAMES,
        "input_shape": {"timesteps": 1000, "leads": 12, "sampling_rate": 100},
    }
    if MODEL_COMPARISON_PATH.exists():
        with open(MODEL_COMPARISON_PATH) as f:
            result["model_comparison"] = json.load(f)
    return result


@app.get("/api/sample")
async def get_sample():
    """Return a random test ECG sample with its ground truth labels."""
    try:
        signals, labels, test_indices = get_sample_data()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Could not load sample data: {e}")

    idx = int(random.choice(test_indices))
    signal = np.array(signals[idx]).astype(float)
    label = labels[idx].tolist()

    return {
        "signal": signal.tolist(),
        "ground_truth": _build_ground_truth(label),
        "sample_index": idx,
        "shape": list(signal.shape),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a CSV file with 12-lead ECG data and return predictions.

    The CSV should have 1000 rows and 12 columns (no header).
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Only CSV files are accepted.")

    try:
        content = await file.read()
        signal = parse_csv_signal(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

    try:
        result = run_inference(signal)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    result["signal"] = signal.tolist()
    return result


@app.post("/api/predict-sample")
async def predict_sample():
    """Get a random sample and predict on it (combined endpoint)."""
    try:
        signals, labels, test_indices = get_sample_data()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Could not load sample data: {e}")

    idx = int(random.choice(test_indices))
    signal = np.array(signals[idx]).astype(np.float32)
    label = labels[idx].tolist()

    try:
        result = run_inference(signal)
    except Exception as e:
        logger.exception("Sample prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Sample prediction error: {e}")

    result["signal"] = signal.tolist()
    result["ground_truth"] = _build_ground_truth(label)
    result["sample_index"] = idx
    return result


# ── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(
        description="ECG Diagnosis AI — FastAPI Server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to serve on")
    parser.add_argument("--host", type=str,
                        default="localhost", help="Host to bind to")
    args = parser.parse_args()

    uvicorn.run(
        "webapp.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
