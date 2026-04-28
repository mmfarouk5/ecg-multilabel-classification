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

CONFIG_PATH="configs/kaggle_notebook.yaml"
MAX_SAMPLES=""
FORCE_PREPROCESS=0
WORKFLOW="full"
HPO_TRIALS=30
RUN_BEST_AFTER_HPO=0
DEFAULT_KAGGLE_PTBXL_DIR="/kaggle/input/datasets/khyeh0719/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

usage() {
  cat <<'__USAGE__'
Run a Kaggle-friendly ECG notebook workflow.

Usage:
  bash scripts/run_kaggle_notebook.sh [options]

Options:
  --config PATH          Config file to use (default: configs/kaggle_notebook.yaml)
  --workflow MODE        'full' or 'single' (default: full)
  --single-model         Shortcut for --workflow single
  --full-pipeline        Shortcut for --workflow full
  --max-samples N        Limit training to a subset for quicker debugging
  --hpo-trials N         Number of Optuna trials when workflow=full (default: 30)
  --run-best-after-hpo   Run best HPO config after tuning when workflow=full
  --force-preprocess     Rebuild cached full-dataset preprocessing artifacts
  -h, --help             Show this help
__USAGE__
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --workflow)
      WORKFLOW="$2"
      shift 2
      ;;
    --single-model)
      WORKFLOW="single"
      shift
      ;;
    --full-pipeline)
      WORKFLOW="full"
      shift
      ;;
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

if [[ "${WORKFLOW}" != "single" && "${WORKFLOW}" != "full" ]]; then
  echo "Unsupported workflow: ${WORKFLOW}" >&2
  echo "Use --workflow single or --workflow full." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/kaggle_logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

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

discover_kaggle_ptbxl_dir() {
  if [[ ! -d "/kaggle/input" ]]; then
    return 1
  fi

  while IFS= read -r csv_path; do
    local candidate_dir
    candidate_dir="$(dirname "${csv_path}")"
    if [[ -f "${candidate_dir}/scp_statements.csv" ]]; then
      printf '%s\n' "${candidate_dir}"
      return 0
    fi
  done < <(find /kaggle/input -maxdepth 5 -type f -name 'ptbxl_database.csv' | sort)

  return 1
}

if [[ -z "${PTBXL_DATA_DIR:-}" && -d "${DEFAULT_KAGGLE_PTBXL_DIR}" ]]; then
  export PTBXL_DATA_DIR="${DEFAULT_KAGGLE_PTBXL_DIR}"
fi

if [[ -z "${PTBXL_DATA_DIR:-}" ]]; then
  DISCOVERED_DATA_DIR="$(discover_kaggle_ptbxl_dir || true)"
  if [[ -n "${DISCOVERED_DATA_DIR}" ]]; then
    export PTBXL_DATA_DIR="${DISCOVERED_DATA_DIR}"
  fi
fi

RUNTIME_INFO="$(
  CONFIG_PATH_ENV="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import yaml
from src.utils import resolve_runtime_paths

project_root = Path.cwd()
config_path = os.environ["CONFIG_PATH_ENV"]
with open(config_path) as f:
    config = yaml.safe_load(f)
config = resolve_runtime_paths(config, project_root=project_root, create_dirs=False)
print(config["data"]["raw_dir"])
print(config["data"]["processed_dir"])
print(config["output"]["models_dir"])
print(config["output"]["results_dir"])
print(config["output"]["figures_dir"])
PY
)"

RAW_DIR="$(printf '%s\n' "${RUNTIME_INFO}" | sed -n '1p')"
PROCESSED_DIR="$(printf '%s\n' "${RUNTIME_INFO}" | sed -n '2p')"
MODELS_DIR="$(printf '%s\n' "${RUNTIME_INFO}" | sed -n '3p')"
RESULTS_DIR="$(printf '%s\n' "${RUNTIME_INFO}" | sed -n '4p')"
FIGURES_DIR="$(printf '%s\n' "${RUNTIME_INFO}" | sed -n '5p')"

if [[ ! -f "${RAW_DIR}/ptbxl_database.csv" || ! -f "${RAW_DIR}/scp_statements.csv" ]]; then
  echo "PTB-XL dataset was not found." >&2
  echo "Resolved dataset path: ${RAW_DIR}" >&2
  echo "On Kaggle, add the PTB-XL dataset under Notebook Input and rerun." >&2
  exit 1
fi

HAS_CUDA="$("${PYTHON_BIN}" - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)"

if [[ "${HAS_CUDA}" != "1" && -z "${MAX_SAMPLES}" ]]; then
  if [[ "${WORKFLOW}" == "single" ]]; then
    MAX_SAMPLES="4000"
    echo "CUDA was not detected. Falling back to a quicker CPU-sized run with --max-samples ${MAX_SAMPLES}."
  else
    echo "CUDA was not detected. Full pipeline will run on CPU and may take a very long time." >&2
  fi
fi

echo "Project root : ${PROJECT_ROOT}"
echo "Python       : ${PYTHON_BIN}"
echo "Config       : ${CONFIG_PATH}"
echo "Workflow     : ${WORKFLOW}"
echo "Dataset      : ${RAW_DIR}"
if [[ -n "${PTBXL_DATA_DIR:-}" ]]; then
  echo "PTBXL_DATA_DIR: ${PTBXL_DATA_DIR}"
fi
echo "Processed dir: ${PROCESSED_DIR}"
echo "Logs         : ${LOG_DIR}"
if [[ "${HAS_CUDA}" == "1" ]]; then
  echo "Device       : CUDA"
else
  echo "Device       : CPU"
fi

MAX_SAMPLES_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLES_ARGS=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${WORKFLOW}" == "single" ]]; then
  if [[ -z "${MAX_SAMPLES}" ]]; then
    CACHE_VALID=1
    for required in \
      signals.npy \
      labels.npy \
      class_weights.npy \
      label_classes.json \
      metadata.json \
      train_indices.npy \
      val_indices.npy \
      test_indices.npy; do
      if [[ ! -f "${PROCESSED_DIR}/${required}" ]]; then
        CACHE_VALID=0
        break
      fi
    done

    if [[ ${FORCE_PREPROCESS} -eq 1 || ${CACHE_VALID} -eq 0 ]]; then
      run_step "01_preprocess" \
        "${PYTHON_BIN}" scripts/preprocess_data.py --config "${CONFIG_PATH}"
    else
      echo
      echo "============================================================"
      echo "STEP: 01_preprocess"
      echo "Using existing cached preprocessing artifacts in ${PROCESSED_DIR}"
      echo "============================================================"
    fi
  else
    echo
    echo "============================================================"
    echo "STEP: 01_preprocess"
    echo "Skipping full-dataset preprocessing because a subset run was requested."
    echo "============================================================"
  fi
else
  echo
  echo "============================================================"
  echo "STEP: 01_preprocess"
  echo "Full pipeline will manage preprocessing and reuse cache when available."
  echo "============================================================"
fi

if [[ "${WORKFLOW}" == "single" ]]; then
  run_step "02_train_single_model" \
    "${PYTHON_BIN}" scripts/run_experiment.py --config "${CONFIG_PATH}" "${MAX_SAMPLES_ARGS[@]}"
else
  FULL_PIPELINE_ARGS=(
    bash scripts/run_full_pipeline.sh
    --hpo-trials "${HPO_TRIALS}"
  )
  if [[ ${FORCE_PREPROCESS} -eq 1 ]]; then
    FULL_PIPELINE_ARGS+=(--force-preprocess)
  fi
  if [[ ${RUN_BEST_AFTER_HPO} -eq 1 ]]; then
    FULL_PIPELINE_ARGS+=(--run-best-after-hpo)
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    FULL_PIPELINE_ARGS+=(--max-samples "${MAX_SAMPLES}")
  fi

  run_step "02_full_pipeline" "${FULL_PIPELINE_ARGS[@]}"
fi

echo
echo "Notebook run complete."
echo "Main outputs:"
echo "  ${MODELS_DIR}"
echo "  ${RESULTS_DIR}"
echo "  ${FIGURES_DIR}"
echo "  /kaggle/working/outputs/archives"
echo "Logs:"
echo "  ${PROJECT_ROOT}/${LOG_DIR}"
