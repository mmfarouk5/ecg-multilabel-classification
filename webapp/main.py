"""
ECG Diagnosis AI — Web Server

A lightweight web server for ECG multi-label classification using only
the Python standard library (no FastAPI/Flask required). Uses the trained
Leadwise CNN model to classify 12-lead ECG signals.

Usage:
    python -m webapp.main [--port 8000]
"""

import argparse
import csv
import io
import json
import logging
import mimetypes
import os
import random
import sys
from functools import lru_cache
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
import yaml

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBAPP_DIR / "static"
CONFIG_PATH = PROJECT_ROOT / "configs" / "leadwise_cnn.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "models" / "best_leadwise_cnn.pt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "model_comparison.json"

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
_model = None
_config = None
_signals = None
_labels = None
_test_indices = None


def get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        with open(CONFIG_PATH, "r") as f:
            _config = yaml.safe_load(f)
    return _config


def get_model():
    global _model
    if _model is not None:
        return _model

    logger.info("Loading model from %s ...", CHECKPOINT_PATH)
    from src.models import build_model

    config = get_config()
    device = torch.device("cpu")
    checkpoint = torch.load(
        str(CHECKPOINT_PATH), map_location=device, weights_only=False
    )

    model = build_model(config)

    # Remap checkpoint keys: the model was trained with 'backbone'
    # but the current code uses 'lead_backbone'
    state_dict = checkpoint["model_state_dict"]
    remapped = {}
    for k, v in state_dict.items():
        new_key = k.replace("backbone.", "lead_backbone.", 1) if k.startswith("backbone.") else k
        remapped[new_key] = v

    model.load_state_dict(remapped)
    model.to(device)
    model.eval()

    _model = model
    logger.info("✓ Model loaded successfully.")
    return _model


def get_sample_data():
    global _signals, _labels, _test_indices
    if _signals is not None:
        return _signals, _labels, _test_indices

    logger.info("Loading processed data for sample endpoint ...")
    _signals = np.load(str(PROCESSED_DIR / "signals.npy"), mmap_mode="r")
    _labels = np.load(str(PROCESSED_DIR / "labels.npy"))
    _test_indices = np.load(str(PROCESSED_DIR / "test_indices.npy"))
    logger.info(
        "✓ Loaded %d signals, %d test indices", len(_signals), len(_test_indices)
    )
    return _signals, _labels, _test_indices


@torch.no_grad()
def run_inference(signal: np.ndarray) -> Dict[str, Any]:
    """Run preprocessing + inference on a (1000, 12) ECG signal."""
    from src.data.preprocessing import preprocess_pipeline

    model = get_model()
    config = get_config()
    device = torch.device("cpu")

    batch = signal[np.newaxis, ...].astype(np.float32)
    batch = preprocess_pipeline(batch, config)

    tensor = torch.tensor(batch, dtype=torch.float32).permute(0, 2, 1).to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    preds = (probs >= 0.5).astype(int)

    classes = []
    for i, cls_name in enumerate(LABEL_CLASSES):
        info = LABEL_DESCRIPTIONS[cls_name]
        prob = float(probs[i])
        predicted = bool(preds[i])
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
        })

    predicted_classes = [c["full_name"] for c in classes if c["predicted"]]
    return {
        "classes": classes,
        "predicted_classes": predicted_classes,
        "num_predicted": len(predicted_classes),
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
        raise ValueError(f"Expected 12 columns (leads), got {signal.shape[1]}.")
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


def parse_multipart(content_type: str, body: bytes):
    """Minimal multipart/form-data parser to extract the file upload."""
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):]
            break
    if not boundary:
        raise ValueError("No boundary found in content-type")

    boundary_bytes = ("--" + boundary).encode()
    parts = body.split(boundary_bytes)

    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        # Split headers from body
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers_raw = part[:header_end].decode("utf-8", errors="replace")
        file_body = part[header_end + 4:]
        # Remove trailing \r\n-- if present
        if file_body.endswith(b"--\r\n"):
            file_body = file_body[:-4]
        elif file_body.endswith(b"\r\n"):
            file_body = file_body[:-2]

        if 'name="file"' in headers_raw or "filename=" in headers_raw:
            return file_body

    raise ValueError("No file part found in multipart upload")


# ── HTTP Handler ───────────────────────────────────────────────
class ECGHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the ECG diagnosis API."""

    def log_message(self, format, *args):
        logger.info(format, *args)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, detail: str):
        self._send_json({"detail": detail}, status)

    def _serve_static(self, path: str):
        """Serve a static file."""
        if path == "/" or path == "":
            file_path = STATIC_DIR / "index.html"
        else:
            # Remove leading slash
            clean = path.lstrip("/")
            # Remove /static/ prefix if present
            if clean.startswith("static/"):
                clean = clean[len("static/"):]
            file_path = STATIC_DIR / clean

        # Security: prevent directory traversal
        try:
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(STATIC_DIR.resolve())):
                self.send_error(403)
                return
        except Exception:
            self.send_error(403)
            return

        if not file_path.is_file():
            self.send_error(404)
            return

        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"

        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json({"status": "ok", "model_loaded": _model is not None})

        elif path == "/api/model-info":
            result = {
                "current_model": "Leadwise CNN",
                "label_classes": LABEL_CLASSES,
                "label_details": LABEL_DESCRIPTIONS,
                "lead_names": ECG_LEAD_NAMES,
                "input_shape": {"timesteps": 1000, "leads": 12, "sampling_rate": 100},
            }
            if MODEL_COMPARISON_PATH.exists():
                with open(MODEL_COMPARISON_PATH) as f:
                    result["model_comparison"] = json.load(f)
            self._send_json(result)

        elif path == "/api/sample":
            try:
                signals, labels, test_indices = get_sample_data()
            except Exception as e:
                self._send_error_json(500, f"Could not load sample data: {e}")
                return

            idx = int(random.choice(test_indices))
            signal = np.array(signals[idx]).astype(float)
            label = labels[idx].tolist()

            ground_truth = []
            for i, cls_name in enumerate(LABEL_CLASSES):
                info = LABEL_DESCRIPTIONS[cls_name]
                ground_truth.append({
                    "name": cls_name,
                    "full_name": info["full_name"],
                    "present": bool(label[i]),
                })

            self._send_json({
                "signal": signal.tolist(),
                "ground_truth": ground_truth,
                "sample_index": idx,
                "shape": list(signal.shape),
            })

        else:
            self._serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        content_type = self.headers.get("Content-Type", "")

        if path == "/api/predict":
            try:
                if "multipart/form-data" in content_type:
                    csv_bytes = parse_multipart(content_type, body)
                else:
                    csv_bytes = body

                signal = parse_csv_signal(csv_bytes)
                result = run_inference(signal)
                result["signal"] = signal.tolist()
                self._send_json(result)
            except ValueError as e:
                self._send_error_json(400, str(e))
            except Exception as e:
                logger.exception("Inference failed")
                self._send_error_json(500, f"Inference error: {e}")

        elif path == "/api/predict-sample":
            try:
                signals, labels, test_indices = get_sample_data()
                idx = int(random.choice(test_indices))
                signal = np.array(signals[idx]).astype(np.float32)
                label = labels[idx].tolist()

                result = run_inference(signal)

                ground_truth = []
                for i, cls_name in enumerate(LABEL_CLASSES):
                    info = LABEL_DESCRIPTIONS[cls_name]
                    ground_truth.append({
                        "name": cls_name,
                        "full_name": info["full_name"],
                        "present": bool(label[i]),
                    })

                result["signal"] = signal.tolist()
                result["ground_truth"] = ground_truth
                result["sample_index"] = idx
                self._send_json(result)
            except Exception as e:
                logger.exception("Sample prediction failed")
                self._send_error_json(500, f"Sample prediction error: {e}")

        else:
            self._send_error_json(404, "Not found")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ECG Diagnosis AI — Web Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    args = parser.parse_args()

    # Pre-load model on startup
    logger.info("Pre-loading model...")
    get_model()
    logger.info("Pre-loading sample data...")
    get_sample_data()

    server = HTTPServer((args.host, args.port), ECGHandler)
    logger.info("=" * 50)
    logger.info("  🫀 ECG Diagnosis AI Server")
    logger.info("  http://localhost:%d", args.port)
    logger.info("=" * 50)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
