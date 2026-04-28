#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python executable not found. Set PYTHON_BIN or install python." >&2
    exit 1
  fi
fi

DEFAULT_CONFIG="configs/default.yaml"
TOP_MODEL_CONFIG="configs/leadwise_cnn.yaml"
HPO_CONFIG="configs/leadwise_cnn.yaml"
HPO_TRIALS=30
MAX_SAMPLES=""
RUN_BEST_AFTER_HPO=0
FORCE_PREPROCESS=0

usage() {
  cat <<'EOF'
Run all ECG project experiments end-to-end.

Usage:
  bash scripts/run_full_pipeline.sh [options]

Options:
  --max-samples N         Use a subset for faster debug runs (supported steps only)
  --hpo-trials N          Number of Optuna Bayesian tuning trials (default: 30)
  --run-best-after-hpo    Run full experiment with best found HPO config
  --force-preprocess      Rebuild cached preprocessing even if cache exists
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --hpo-trials)
      HPO_TRIALS="$2"
      shift 2
      ;;
    --run-best-after-hpo)
      RUN_BEST_AFTER_HPO=1
      shift
      ;;
    --force-preprocess)
      FORCE_PREPROCESS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/pipeline_logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

MAX_SAMPLES_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLES_ARGS=(--max-samples "${MAX_SAMPLES}")
fi

run_step() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  echo
  echo "============================================================"
  echo "STEP: ${name}"
  echo "CMD : $*"
  echo "LOG : ${log_file}"
  echo "============================================================"
  "$@" 2>&1 | tee "${log_file}"
}

resolve_processed_dir() {
  CONFIG_PATH_ENV="${DEFAULT_CONFIG}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import yaml
from src.utils import resolve_runtime_paths

project_root = Path.cwd()
config_path = os.environ["CONFIG_PATH_ENV"]
with open(config_path) as f:
    config = yaml.safe_load(f)
config = resolve_runtime_paths(config, project_root=project_root, create_dirs=False)
print(config["data"]["processed_dir"])
PY
}

cache_is_valid() {
  local processed_dir="$1"
  local required
  for required in \
    signals.npy \
    labels.npy \
    class_weights.npy \
    label_classes.json \
    metadata.json \
    train_indices.npy \
    val_indices.npy \
    test_indices.npy; do
    if [[ ! -f "${processed_dir}/${required}" ]]; then
      return 1
    fi
  done
  return 0
}

echo "Project root: ${PROJECT_ROOT}"
echo "Python      : ${PYTHON_BIN}"
echo "Logs        : ${LOG_DIR}"
if [[ -n "${PTBXL_DATA_DIR:-}" ]]; then
  echo "PTBXL_DATA_DIR: ${PTBXL_DATA_DIR}"
fi

if [[ -d "/kaggle" ]]; then
  echo "Runtime     : Kaggle detected"
  echo "Hint        : Add PTB-XL dataset in notebook settings under Input."
else
  echo "Runtime     : Local/other"
fi

PROCESSED_DIR="$(resolve_processed_dir)"
echo "Processed dir: ${PROCESSED_DIR}"

if [[ ${FORCE_PREPROCESS} -eq 1 ]]; then
  run_step "01_preprocess" \
    "${PYTHON_BIN}" scripts/preprocess_data.py --config "${DEFAULT_CONFIG}"
elif cache_is_valid "${PROCESSED_DIR}"; then
  echo
  echo "============================================================"
  echo "STEP: 01_preprocess"
  echo "Using existing cached preprocessing artifacts in ${PROCESSED_DIR}"
  echo "============================================================"
else
  run_step "01_preprocess" \
    "${PYTHON_BIN}" scripts/preprocess_data.py --config "${DEFAULT_CONFIG}"
fi

run_step "02_all_models" \
  "${PYTHON_BIN}" scripts/run_all_models.py "${MAX_SAMPLES_ARGS[@]}"

run_step "03_ensemble" \
  "${PYTHON_BIN}" scripts/run_ensemble.py \
    --config "${DEFAULT_CONFIG}" \
    --models leadwise_cnn cnn_1d lstm \
    "${MAX_SAMPLES_ARGS[@]}"

run_step "04_cross_validation" \
  "${PYTHON_BIN}" scripts/run_cv.py \
    --config "${TOP_MODEL_CONFIG}" \
    "${MAX_SAMPLES_ARGS[@]}"

run_step "05_ablation" \
  "${PYTHON_BIN}" scripts/run_ablation.py \
    --config "${DEFAULT_CONFIG}" \
    "${MAX_SAMPLES_ARGS[@]}"

HPO_ARGS=(
  "${PYTHON_BIN}" scripts/run_bayesian_tuning.py
  --config "${HPO_CONFIG}"
  --n-trials "${HPO_TRIALS}"
)
if [[ ${RUN_BEST_AFTER_HPO} -eq 1 ]]; then
  HPO_ARGS+=(--run-best)
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  HPO_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

run_step "06_bayesian_tuning" "${HPO_ARGS[@]}"

echo
echo "Pipeline complete."
echo "Main outputs:"
echo "  outputs/models"
echo "  outputs/results"
echo "  outputs/figures"
echo "  outputs/archives"
echo "Logs:"
echo "  ${LOG_DIR}"
