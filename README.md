# ECG Multi-Label Classification (PTB-XL)

End-to-end pipeline for multi-label ECG diagnosis from 12-lead signals. This repo includes preprocessing, training and evaluation, and a FastAPI web app for inference. Current outputs in this workspace include single-model runs for `cnn_1d`, `cnn_lstm`, `cnn_transformer`, `leadwise_cnn`, `lstm`, and `pretrained_resnet`, plus an ensemble of `leadwise_cnn` + `cnn_1d` + `lstm`.

## Project Overview

This project predicts 5 PTB-XL diagnostic superclasses:

- `NORM` (Normal ECG)
- `MI` (Myocardial Infarction)
- `STTC` (ST/T Changes)
- `CD` (Conduction Disturbance)
- `HYP` (Hypertrophy)

Core goals:

1. Build a robust multi-label ECG pipeline from raw waveform files to deployable inference.
2. Compare architecture families under a consistent training and evaluation setup.
3. Prioritize reproducibility with config-driven experiments and artifact tracking.

## Current Pipeline

1. Ingest raw PTB-XL records and metadata.
2. Preprocess signals (bandpass, baseline-wander removal, normalization, outlier clipping).
3. Split by PTB-XL stratified folds.
4. Train a selected model from a YAML config.
5. Evaluate on the test fold and compute metrics.
6. Persist artifacts (metrics, predictions, probabilities, thresholds, histories, plots).

## Data Requirements

### Dataset

- Source: PTB-XL (PhysioNet, v1.0.1)
- Expected path (default config): [data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1](data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1)

### Required files in dataset root

- ptbxl_database.csv
- scp_statements.csv
- waveform records referenced by filename_lr / filename_hr

### Signal assumptions

- 12 leads
- 10-second windows
- default training sampling rate: 100 Hz
- training tensors are shaped as (batch, channels=12, seq_len)

### Split protocol

Default split follows PTB-XL strat_fold:

- Train: folds 1-8
- Validation: fold 9
- Test: fold 10

### Cached preprocessed artifacts

After preprocessing, the project stores:

- signals.npy, labels.npy
- train_indices.npy, val_indices.npy, test_indices.npy
- class_weights.npy, label_classes.json, metadata.json

under [data/processed](data/processed).

## Installation

### Prerequisites

- Python 3.10+
- Recommended: CUDA-capable GPU for training

### Setup

```bash
git clone https://github.com/<your-username>/ecg-multilabel-classification.git
cd ecg-multilabel-classification

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Download PTB-XL

Option A (programmatic download helper):

```bash
python -m src.data.download
```

Option B: download manually from PhysioNet and place it under [data/raw](data/raw).

### Kaggle path compatibility

Training and evaluation scripts resolve paths automatically for Kaggle:

- Raw PTB-XL data is auto-detected from /kaggle/input/...
- Writable artifacts are directed to /kaggle/working/...

Optional overrides:

- PTBXL_DATA_DIR for dataset root
- ECG_CHECKPOINT_PATH, ECG_PROCESSED_DIR, ECG_MODEL_COMPARISON_PATH for web app assets

## How to Run

### 1. Preprocess and cache dataset

```bash
python scripts/preprocess_data.py --config configs/default.yaml
```

### 2. Train and evaluate one model

```bash
python scripts/run_experiment.py --config configs/leadwise_cnn.yaml
```

### 3. Train all models and produce comparison

```bash
python scripts/run_all_models.py
```

### 4. Run ablation sweep

```bash
python scripts/run_ablation.py --config configs/default.yaml
```

### 5. Run top-model ensemble (leadwise_cnn + cnn_1d + lstm)

```bash
python scripts/run_ensemble.py --models leadwise_cnn cnn_1d lstm
```

Requires checkpoints in [outputs/models](outputs/models).

### 6. Run Bayesian hyperparameter tuning (Optuna TPE)

```bash
python scripts/run_bayesian_tuning.py --config configs/leadwise_cnn.yaml --n-trials 30
```

### 7. Run cross-validation

```bash
python scripts/run_cv.py --config configs/leadwise_cnn.yaml
```

### 8. Launch web app

```bash
python run_server.py
```

Then open http://localhost:8000

### 9. Programmatic inference

```python
import numpy as np
import yaml
import torch
from src.inference.predict import load_trained_model, predict_signal

with open("configs/leadwise_cnn.yaml") as f:
    config = yaml.safe_load(f)

model = load_trained_model(
    "outputs/models/best_leadwise_cnn.pt",
    config=config,
    device=torch.device("cpu"),
)

# ECG array shape: (1000, 12)
signal = np.load("path/to/ecg_signal.npy")
result = predict_signal(model, signal, config, threshold=0.5)
print(result["predictions"], result["probabilities"])
```

## Outputs and Artifacts

### Single-model comparison summary

- [outputs/model_comparison.json](outputs/model_comparison.json)
- Contains metrics for models with completed runs in [outputs/results](outputs/results).
- Current models in this file: `cnn_1d`, `cnn_lstm`, `cnn_transformer`, `leadwise_cnn`, `lstm`, `pretrained_resnet`.

### Per-model results bundle

Each model folder in [outputs/results](outputs/results) contains:

- metrics.json (summary metrics for the test set)
- labels.npy (ground truth)
- probabilities.npy (predicted probabilities)
- predictions.npy (thresholded predictions)
- optimal_thresholds.npy (per-class thresholds)
- history_train_loss.npy, history_val_loss.npy, history_val_f1_macro.npy

### Ensemble results

- [outputs/results/ensemble_leadwise_cnn_cnn_1d_lstm](outputs/results/ensemble_leadwise_cnn_cnn_1d_lstm)
- Contains metrics.json plus predictions and probabilities, and ensemble_metadata.json with model list and weights.

### Cross-validation summary

- [outputs/results/cross_validation/cv_summary.json](outputs/results/cross_validation/cv_summary.json)
- Reports per-fold validation loss statistics for `leadwise_cnn`.

### Ablation runs

- [outputs/ablation_runs](outputs/ablation_runs) contains baseline, asymmetric_focal, asymmetric_focal_sampler, and throughput variants.
- Each variant has results for `leadwise_cnn` and `pretrained_resnet` with the same metrics.json schema.

### Hyperparameter tuning

- [outputs/hpo/bayes_leadwise_cnn](outputs/hpo/bayes_leadwise_cnn) includes:
  - best_summary.json (best trial, value, params)
  - best_config.yaml (Kaggle paths from the tuning environment)
  - trials.csv and per-trial outputs

### Archives

- [outputs/archives](outputs/archives) contains zipped model and result bundles.

## Example Output

Example from [outputs/results/cnn_1d/metrics.json](outputs/results/cnn_1d/metrics.json):

```json
{
  "subset_accuracy": 0.5928279618701771,
  "f1_macro": 0.7307163453461925,
  "roc_auc_macro": 0.9188427440939044,
  "f1_NORM": 0.8581349206349206,
  "f1_MI": 0.7363717605004468
}
```

### Field Notes

- subset_accuracy: exact match across all labels
- f1_macro, f1_micro, f1_weighted: F1 aggregation variants
- precision_macro, recall_macro: macro-averaged precision and recall
- roc_auc_macro, roc_auc_weighted: macro and weighted ROC-AUC
- roc_auc_<LABEL>: per-class ROC-AUC
- avg_precision_macro: macro average precision
- f1_<LABEL>, precision_<LABEL>, recall_<LABEL>: per-class metrics

## Evaluation Metrics

Metrics stored in each metrics.json:

- subset_accuracy
- f1_macro, f1_micro, f1_weighted
- precision_macro, recall_macro
- roc_auc_macro, roc_auc_weighted, roc_auc_<LABEL>
- avg_precision_macro
- f1_<LABEL>, precision_<LABEL>, recall_<LABEL>

## Latest Results (from outputs/model_comparison.json)

| Rank (F1 Macro) | Model | F1 Macro | ROC-AUC Macro | Subset Accuracy |
|---|---|---:|---:|---:|
| 1 | cnn_1d | 0.7307 | 0.9188 | 0.5928 |
| 2 | leadwise_cnn | 0.7205 | 0.9075 | 0.5801 |
| 3 | cnn_transformer | 0.7196 | 0.9152 | 0.5552 |
| 4 | lstm | 0.7177 | 0.9097 | 0.5638 |
| 5 | cnn_lstm | 0.7151 | 0.9115 | 0.5311 |
| 6 | pretrained_resnet | 0.5258 | 0.7205 | 0.0576 |

Ensemble result from [outputs/results/ensemble_leadwise_cnn_cnn_1d_lstm/metrics.json](outputs/results/ensemble_leadwise_cnn_cnn_1d_lstm/metrics.json):

- f1_macro: 0.7488
- roc_auc_macro: 0.9248
- subset_accuracy: 0.6015

## Known Issues and Limitations

- No single-model outputs for `transformer` are present in [outputs/results](outputs/results) or [outputs/model_comparison.json](outputs/model_comparison.json).
- Cross-validation output currently includes only validation loss summary; per-fold metrics and test metrics are not stored.
- The HPO best_config.yaml uses Kaggle absolute paths and needs editing for local reuse.

## Future Improvements

- Add transformer baseline outputs and include them in model_comparison.
- Save fold-level metrics and confusion matrices for cross-validation.
- Add output manifests with dataset version hashes and run metadata.

## Project Structure

- [configs](configs) experiment configs
- [data/raw](data/raw) PTB-XL data
- [data/processed](data/processed) cached artifacts
- [outputs](outputs) metrics, predictions, models, archives
- [scripts](scripts) CLI runners
- [src](src) core library
- [webapp](webapp) FastAPI UI
- [run_server.py](run_server.py) app launcher
- [requirements.txt](requirements.txt)

## Acknowledgments

- PTB-XL dataset (PhysioNet)
- PyTorch ecosystem contributors
