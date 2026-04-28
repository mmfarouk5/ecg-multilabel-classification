# ECG Multi-Label Classification (PTB-XL)

End-to-end multi-label ECG diagnosis from 12-lead waveforms. This repository includes preprocessing, training and evaluation, and a FastAPI web app for inference. Current outputs in this workspace include single-model results for `cnn_1d`, `cnn_lstm`, `cnn_transformer`, `leadwise_cnn`, `lstm`, and `pretrained_resnet`, plus an ensemble of `leadwise_cnn` + `cnn_1d` + `lstm`.

## Project Overview

This project predicts five PTB-XL diagnostic superclasses:

- `NORM` (Normal ECG)
- `MI` (Myocardial Infarction)
- `STTC` (ST/T Changes)
- `CD` (Conduction Disturbance)
- `HYP` (Hypertrophy)

Core goals:

1. Build a robust multi-label ECG pipeline from raw waveform files to deployable inference.
2. Compare architecture families under a consistent training and evaluation setup.
3. Prioritize reproducibility with config-driven experiments and artifact tracking.

## Methodology

### Problem Definition

Given a 12-lead ECG recording (10 seconds at 100 Hz by default), predict a multi-label diagnosis across the five PTB-XL superclasses. Each record can map to multiple labels, so outputs are sigmoid probabilities and thresholded multi-hot vectors.

### Data Pipeline

1. Load PTB-XL metadata and SCP statements, map SCP codes to diagnostic labels, and encode labels into a binary matrix using `diagnostic_superclass` by default.
2. Load waveform signals with wfdb at 100 Hz or 500 Hz. Shapes are `(N, seq_len, 12)` where `seq_len` is 1000 at 100 Hz or 5000 at 500 Hz.
3. Preprocess signals in this order:
   - Baseline wander removal (high-pass Butterworth).
   - Bandpass filtering (Butterworth).
   - Outlier clipping at a percentile threshold.
   - Z-score normalization per lead (with zero-std protection).
4. Split by PTB-XL `strat_fold` by default (train folds 1-8, val fold 9, test fold 10). Cross-validation uses multi-label stratified K-fold on non-test folds.
5. Cache preprocessed data in [data/processed](data/processed); subsequent runs reuse cached tensors.

Edge handling:

- Unsupported sampling rates raise errors (only 100 or 500 Hz are supported).
- Normalization replaces zero standard deviations with 1 to avoid division by zero.
- The web app pads or truncates uploaded signals to 1000 timesteps and requires 12 columns.

### Model Architectures

- `cnn_1d`: stacked Conv1d blocks with BN, ReLU, MaxPool, then adaptive average pooling and FC.
- `leadwise_cnn`: shared per-lead Conv1d backbone, concatenated lead features, FC head.
- `resnet`: 1D residual blocks with downsampling stages and global pooling.
- `pretrained_resnet`: xresnet1d-style stem and residual stages, optional pretrained backbone loading and freezing.
- `lstm`: bidirectional LSTM with temporal attention and FC head.
- `cnn_lstm`: CNN downsampling followed by a bidirectional LSTM, last hidden state to FC.
- `cnn_transformer`: CNN downsampling, projection + positional encoding, CLS token pooling, Transformer encoder.

### Training Strategy

- Configuration-driven via [configs](configs). Defaults are in [configs/default.yaml](configs/default.yaml).
- Device selection: CUDA, then Apple MPS, then CPU. AMP is enabled only on CUDA.
- Loss options: weighted BCE, focal, asymmetric focal, or BCE, chosen per config.
- Optimizers: Adam, AdamW, or SGD. Schedulers: cosine, step, or plateau.
- Optional class-aware sampling with a weighted sampler for rare-label upweighting.
- Early stopping supports `val_loss` or `val_f1_macro` monitoring; `val_f1_macro` is computed using per-class thresholds optimized on the validation set each epoch.
- Best checkpoint is saved by validation loss in [outputs/models](outputs/models) with a best_ prefix; final checkpoint saved at end, non-best checkpoints are pruned.

### Inference Pipeline

- `predict_signal` and `predict_batch` apply the same preprocessing pipeline, run a forward pass, apply sigmoid, and threshold probabilities (default threshold 0.5).
- `postprocessing` provides per-class thresholds, confidence filtering, and prediction formatting.
- The web app loads the `leadwise_cnn` checkpoint, runs CPU inference, and formats predictions with confidence labels.

### Output Generation and Schema

- `Evaluator` computes metrics and saves results to disk. When `optimize_thresholds=True`, per-class thresholds are selected by grid search (0.05 to 0.95, step 0.01) to maximize per-class F1.
- Per-model results include metrics, probabilities, predictions, labels, and optional optimal thresholds. Model histories are saved by [scripts/run_all_models.py](scripts/run_all_models.py).
- The multi-model comparison summary is written by [scripts/run_all_models.py](scripts/run_all_models.py) to [outputs/model_comparison.json](outputs/model_comparison.json).

### Evaluation Approach

- Metrics: subset accuracy, macro/micro/weighted F1, macro precision and recall, macro and weighted ROC-AUC, per-class ROC-AUC, and macro average precision.
- Default split is PTB-XL fold-based. Cross-validation uses multi-label stratified folds excluding the test fold.

## Pipeline Overview

```text
PTB-XL records + SCP statements
  -> label encoding + class weights
  -> waveform loading (100 or 500 Hz)
  -> preprocessing (baseline removal, bandpass, clip, normalize)
  -> split by strat_fold
  -> train (config-driven)
  -> evaluate + threshold optimization
  -> outputs (metrics, predictions, models, histories)
```

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

Path resolution is Kaggle-aware. Raw data and writable outputs are auto-detected from `/kaggle/input` and `/kaggle/working` when running in Kaggle.

Optional overrides:

- `PTBXL_DATA_DIR` for dataset root
- `ECG_CONFIG_PATH`, `ECG_CHECKPOINT_PATH`, `ECG_PROCESSED_DIR`, `ECG_MODEL_COMPARISON_PATH` for the web app

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

### Cached processed data

- [data/processed/signals.npy](data/processed/signals.npy)
- [data/processed/labels.npy](data/processed/labels.npy)
- [data/processed/class_weights.npy](data/processed/class_weights.npy)
- [data/processed/label_classes.json](data/processed/label_classes.json)
- [data/processed/train_indices.npy](data/processed/train_indices.npy)
- [data/processed/val_indices.npy](data/processed/val_indices.npy)
- [data/processed/test_indices.npy](data/processed/test_indices.npy)
- [data/processed/metadata.json](data/processed/metadata.json)

### Per-model results bundle

Each model directory under [outputs/results](outputs/results) contains the same schema. Examples from [outputs/results/cnn_1d](outputs/results/cnn_1d):

- [outputs/results/cnn_1d/metrics.json](outputs/results/cnn_1d/metrics.json)
- [outputs/results/cnn_1d/labels.npy](outputs/results/cnn_1d/labels.npy)
- [outputs/results/cnn_1d/probabilities.npy](outputs/results/cnn_1d/probabilities.npy)
- [outputs/results/cnn_1d/predictions.npy](outputs/results/cnn_1d/predictions.npy)
- [outputs/results/cnn_1d/optimal_thresholds.npy](outputs/results/cnn_1d/optimal_thresholds.npy)
- [outputs/results/cnn_1d/history_train_loss.npy](outputs/results/cnn_1d/history_train_loss.npy)
- [outputs/results/cnn_1d/history_val_loss.npy](outputs/results/cnn_1d/history_val_loss.npy)
- [outputs/results/cnn_1d/history_val_f1_macro.npy](outputs/results/cnn_1d/history_val_f1_macro.npy)

### Single-model comparison summary

- [outputs/model_comparison.json](outputs/model_comparison.json)
- Contains metrics for models with completed runs.

### Ensemble results

- [outputs/results/ensemble_leadwise_cnn_cnn_1d_lstm](outputs/results/ensemble_leadwise_cnn_cnn_1d_lstm)
- Includes metrics, predictions, probabilities, thresholds, and ensemble metadata.

### Cross-validation summary

- [outputs/results/cross_validation/cv_summary.json](outputs/results/cross_validation/cv_summary.json)

### Ablation runs

- [outputs/ablation_runs](outputs/ablation_runs) contains baseline, asymmetric_focal, asymmetric_focal_sampler, and throughput variants.

### Hyperparameter tuning

- [outputs/hpo/bayes_leadwise_cnn](outputs/hpo/bayes_leadwise_cnn)
- [outputs/hpo/bayes_leadwise_cnn/best_summary.json](outputs/hpo/bayes_leadwise_cnn/best_summary.json)
- [outputs/hpo/bayes_leadwise_cnn/best_config.yaml](outputs/hpo/bayes_leadwise_cnn/best_config.yaml)
- [outputs/hpo/bayes_leadwise_cnn/trials.csv](outputs/hpo/bayes_leadwise_cnn/trials.csv)

### Archives

- [outputs/archives/models.zip](outputs/archives/models.zip)
- [outputs/archives/results.zip](outputs/archives/results.zip)

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

## Evaluation Metrics and Results

Metrics stored in each metrics output file (example: [outputs/results/cnn_1d/metrics.json](outputs/results/cnn_1d/metrics.json)):

- subset_accuracy
- f1_macro, f1_micro, f1_weighted
- precision_macro, recall_macro
- roc_auc_macro, roc_auc_weighted, roc_auc_<LABEL>
- avg_precision_macro
- f1_<LABEL>, precision_<LABEL>, recall_<LABEL>

### Latest single-model results (from outputs/model_comparison.json)

| Rank (F1 Macro) | Model | F1 Macro | ROC-AUC Macro | Subset Accuracy |
|---|---|---:|---:|---:|
| 1 | cnn_1d | 0.7307 | 0.9188 | 0.5928 |
| 2 | leadwise_cnn | 0.7205 | 0.9075 | 0.5801 |
| 3 | cnn_transformer | 0.7196 | 0.9152 | 0.5552 |
| 4 | lstm | 0.7177 | 0.9097 | 0.5638 |
| 5 | cnn_lstm | 0.7151 | 0.9115 | 0.5311 |
| 6 | pretrained_resnet | 0.5258 | 0.7205 | 0.0576 |

### Ensemble results

From [outputs/results/ensemble_leadwise_cnn_cnn_1d_lstm/metrics.json](outputs/results/ensemble_leadwise_cnn_cnn_1d_lstm/metrics.json):

- f1_macro: 0.7488
- roc_auc_macro: 0.9248
- subset_accuracy: 0.6015

### Cross-validation summary

From [outputs/results/cross_validation/cv_summary.json](outputs/results/cross_validation/cv_summary.json):

- n_folds: 5
- mean_val_loss: 0.0563
- std_val_loss: 0.0009

## Known Issues and Limitations

- Per-class thresholds are optimized on the test split during evaluation, which can inflate test metrics.
- Cross-validation outputs include validation loss only, without per-fold metrics.
- Ablation summary outputs are not present in this snapshot; only per-variant result folders are available.
- Figure and log artifacts are not present at the top-level outputs snapshot; they are produced by [scripts/run_experiment.py](scripts/run_experiment.py).

## Future Improvements

- Optimize thresholds on the validation split and apply them to the test split.
- Save per-fold metrics and confusion matrices for cross-validation.
- Add output schema validation for metrics and predictions.
- Emit ablation summary outputs and surface them in the README.

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
