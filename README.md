# ECG Multi-Label Classification (PTB-XL)

End-to-end multi-label ECG diagnosis from 12-lead waveforms. This repository includes signal preprocessing, config-driven training across seven deep learning architectures, threshold-optimised evaluation, weighted probability ensemble, Bayesian hyperparameter tuning, and a production-ready FastAPI web application for inference.

## Project Overview

### Problem Definition

**Task:** Given a 12-lead, 10-second ECG recording, predict one or more cardiac diagnostic superclasses simultaneously (multi-label classification).

**Why it matters:** Automated ECG interpretation can accelerate clinical triage, reduce diagnostic error in resource-limited settings, and serve as a decision-support tool for cardiologists. The PTB-XL dataset (21,799 records from 18,869 patients) provides a large, clinician-annotated benchmark for this task.

**Diagnostic superclasses (5 labels):**

| Label | Full Name             | Description                                          |
|-------|-----------------------|------------------------------------------------------|
| NORM  | Normal ECG            | No significant cardiac abnormalities                 |
| MI    | Myocardial Infarction | ST changes, pathological Q-waves, T-wave inversions  |
| STTC  | ST/T Changes          | Repolarisation abnormalities (ischaemia, electrolyte) |
| CD    | Conduction Disturbance | Bundle branch blocks, AV blocks                     |
| HYP   | Hypertrophy           | Enlarged heart chambers, high-voltage QRS            |

Each record can carry **zero or more** labels, making this a **multi-label** problem. Outputs are sigmoid probabilities thresholded by per-class optimal cutoffs.

---

## Methodology

### Data Pipeline

```text
PTB-XL PhysioNet download
  → metadata + SCP statement loading
  → diagnostic superclass label encoding (binary matrix)
  → waveform loading via wfdb (100 Hz default)
  → preprocessing pipeline
  → stratified split (train / val / test)
  → cached to data/processed/
```

**Steps in detail:**

1. **Metadata & Labels:** Load PTB-XL metadata CSV and SCP statement definitions. Map each record's SCP codes → diagnostic superclass labels → binary label matrix `(N, 5)`.
2. **Waveform Loading:** Load raw ECG signals at 100 Hz (shape `(N, 1000, 12)`) or 500 Hz (`(N, 5000, 12)`). Only these two rates are supported.
3. **Preprocessing Pipeline** (applied in order):
   - **Baseline wander removal** — 4th-order high-pass Butterworth at 0.5 Hz
   - **Bandpass filter** — 4th-order Butterworth [0.5, 40.0] Hz
   - **Outlier clipping** — symmetric clip at the 99th percentile of absolute amplitude
   - **Z-score normalisation** — per-sample, per-lead (zero-std protection: std=0 → std=1)
4. **Splitting:** Default uses PTB-XL `strat_fold` (train: folds 1–8, val: fold 9, test: fold 10). Cross-validation uses multi-label stratified K-fold on non-test folds.
5. **Caching:** Preprocessed signals, labels, class weights, split indices, and metadata are cached under `data/processed/`. Subsequent runs load directly from cache.

**Edge-case handling:**
- Unsupported sampling rates raise explicit errors.
- Per-lead normalisation replaces zero standard deviation with 1 to prevent NaN.
- The web app pads or truncates uploaded signals to exactly 1000 timesteps, rejecting signals with <100 steps or ≠12 leads.

### Model Architectures

Seven architectures are implemented in a shared registry (`src/models/`), all instantiated from YAML config via `build_model()`:

| Model | Key Design | Parameters |
|-------|-----------|-----------|
| `cnn_1d` | Stacked Conv1d → BN → ReLU → MaxPool, adaptive avg pool, FC head | ~3.5M |
| `leadwise_cnn` | **Shared** per-lead Conv1d backbone × 12 leads → concatenate → FC head | ~0.8M |
| `resnet` | 1D residual blocks with downsampling, global average pooling | ~1.1M |
| `pretrained_resnet` | xresnet1d-style with optional pretrained backbone & freezing | ~25M |
| `lstm` | Bidirectional LSTM with temporal attention and FC head | ~1.6M |
| `cnn_lstm` | CNN downsampling → bidirectional LSTM → last hidden → FC | ~3.2M |
| `cnn_transformer` | CNN downsampling → projection + positional encoding → Transformer encoder with CLS token | ~2.7M |

**Architecture selection rationale:**
- **CNNs** capture local morphological patterns (QRS width, ST elevation).
- **LSTMs** model temporal dependencies across the full 10-second window.
- **Hybrid CNN+LSTM / CNN+Transformer** combine local feature extraction with global sequence modelling.
- **Leadwise CNN** exploits the clinical practice of interpreting each lead independently before fusing findings — this proved to be the most parameter-efficient model.

### Training Strategy

All training is configuration-driven via YAML files under `configs/`.

| Aspect | Details |
|--------|---------|
| **Loss** | Asymmetric Focal Loss (default: γ⁺=1, γ⁻=4, clip=0.05), Focal Loss, Weighted BCE, or plain BCE |
| **Optimiser** | AdamW (default), Adam, or SGD |
| **Scheduler** | Cosine annealing (default), step decay, or ReduceLROnPlateau |
| **AMP** | Automatic mixed precision on CUDA (disabled on MPS/CPU) |
| **Gradient clipping** | Max-norm clipping (default: 1.0) |
| **Class-aware sampling** | Optional weighted sampler that upweights samples with rare labels |
| **Early stopping** | Monitors `val_f1_macro` (default) or `val_loss` with configurable patience, min_delta, and best-weight restoration |
| **Checkpointing** | Best checkpoint saved by validation loss; non-best checkpoints pruned automatically |
| **Reproducibility** | Seed-locked (default 42), `torch.backends.cudnn.deterministic=True` |
| **Multi-GPU** | `DataParallel` wrapping when >1 CUDA GPU detected (configurable exclusion list) |

### Inference Pipeline

The inference module (`src/inference/`) provides a unified API for single-model and ensemble prediction:

```text
Raw ECG signal (seq_len, 12)
  → add batch dimension
  → preprocessing pipeline (same as training)
  → channel-first transpose (n_leads, seq_len)
  → forward pass → sigmoid → probabilities
  → apply per-class optimal thresholds
  → binary predictions + formatted output
```

**Key features:**
- **Centralised checkpoint loading** with automatic state-dict key remapping (handles `DataParallel` `module.` prefix and legacy `backbone.` → `lead_backbone.` rename).
- **Per-class optimal thresholds** loaded from saved evaluation results (grid-search optimised on F1).
- **Ensemble inference** (`predict_ensemble`) averages sigmoid probabilities across multiple models with configurable weights.
- Uses `@torch.inference_mode()` for maximum inference performance.
- The **web app** automatically loads the best available thresholds (ensemble first, then single-model fallback).

### Output Generation and Schema

**Per-model results bundle** (saved under `outputs/results/<model_name>/`):

| File | Contents |
|------|----------|
| `metrics.json` | All evaluation metrics (see schema below) |
| `probabilities.npy` | Sigmoid probabilities `(N, 5)` |
| `predictions.npy` | Binary predictions `(N, 5)` |
| `labels.npy` | Ground truth `(N, 5)` |
| `optimal_thresholds.npy` | Per-class thresholds `(5,)` |
| `history_*.npy` | Training loss, val loss, val F1 per epoch |

**Metrics JSON schema:**
```json
{
  "subset_accuracy": 0.5928,
  "f1_macro": 0.7307,
  "f1_micro": 0.7649,
  "f1_weighted": 0.7659,
  "precision_macro": 0.7214,
  "recall_macro": 0.7458,
  "roc_auc_macro": 0.9188,
  "roc_auc_weighted": 0.9260,
  "roc_auc_NORM": 0.9440,
  "roc_auc_MI": 0.9190,
  "roc_auc_STTC": 0.9289,
  "roc_auc_CD": 0.9176,
  "roc_auc_HYP": 0.8847,
  "avg_precision_macro": 0.7959,
  "f1_NORM": 0.8581,
  "f1_MI": 0.7364,
  "f1_STTC": 0.7544,
  "f1_CD": 0.7366,
  "f1_HYP": 0.5681,
  "precision_NORM": 0.8222,
  "recall_NORM": 0.8973,
  "precision_MI": 0.7279,
  "recall_MI": 0.7450,
  "precision_STTC": 0.7632,
  "recall_STTC": 0.7457,
  "precision_CD": 0.7890,
  "recall_CD": 0.6908,
  "precision_HYP": 0.5044,
  "recall_HYP": 0.6502
}
```

All float values are Python `float`. Required fields: `subset_accuracy`, `f1_macro`, `roc_auc_macro`, per-class `f1_<LABEL>`, `precision_<LABEL>`, `recall_<LABEL>`, `roc_auc_<LABEL>`.

### Evaluation Approach

| Metric | Purpose |
|--------|---------|
| **F1 Macro** | Primary metric — unweighted average across all 5 classes (penalises poor minority-class performance) |
| **F1 Micro** | Global TP/FP/FN aggregation (favours majority classes) |
| **F1 Weighted** | Prevalence-weighted F1 |
| **ROC-AUC Macro/Weighted** | Discrimination ability across operating points |
| **Subset Accuracy** | Exact-match accuracy (all 5 labels must match simultaneously) |
| **Average Precision Macro** | Area under the precision-recall curve |
| **Per-class F1/Precision/Recall/AUC** | Fine-grained per-condition analysis |

**Threshold optimisation:** Per-class thresholds are grid-searched (0.05–0.95, step 0.01) to maximise per-class F1 on the test split. This inflates reported test metrics; a future improvement is to optimise on validation and apply to test.

**Validation strategy:** Default hold-out (PTB-XL fold-based). 5-fold multi-label stratified cross-validation available for model stability assessment.

---

## Results

### Single-Model Comparison

Trained with asymmetric focal loss, cosine annealing, AdamW, and per-class threshold optimisation.

| Rank | Model | F1 Macro | ROC-AUC Macro | Subset Accuracy | F1 HYP (hardest) |
|:---:|-------|:---:|:---:|:---:|:---:|
| 1 | `cnn_1d` | **0.7307** | 0.9188 | **0.5928** | 0.5681 |
| 2 | `leadwise_cnn` | 0.7205 | 0.9075 | 0.5801 | 0.6025 |
| 3 | `cnn_transformer` | 0.7196 | **0.9152** | 0.5552 | 0.5302 |
| 4 | `lstm` | 0.7177 | 0.9097 | 0.5638 | 0.5017 |
| 5 | `cnn_lstm` | 0.7151 | 0.9115 | 0.5311 | 0.5420 |
| 6 | `pretrained_resnet` | 0.5258 | 0.7205 | 0.0576 | 0.3318 |

> **Notes:**
> - Top 5 models are within 1.6% F1 of each other — architecture matters less than training strategy for this task.
> - `pretrained_resnet` underperforms significantly, likely due to domain mismatch (ImageNet pretraining ≠ ECG signals) and the large parameter count overfitting.
> - HYP is the hardest class across all models (lowest F1), consistent with the low prevalence and subtle ECG morphology.

### Ensemble Results

**Probability-averaged ensemble** of `leadwise_cnn + cnn_1d + lstm` with equal weights and per-class threshold optimisation:

| Metric | Ensemble | Best Single (cnn_1d) | Δ |
|--------|:---:|:---:|:---:|
| **F1 Macro** | **0.7488** | 0.7307 | **+0.0181** |
| **ROC-AUC Macro** | **0.9248** | 0.9188 | **+0.0060** |
| **Subset Accuracy** | **0.6015** | 0.5928 | **+0.0087** |
| **Avg Precision Macro** | **0.8127** | 0.7959 | **+0.0168** |

### Bayesian Hyperparameter Tuning

Optuna TPE sampler on `leadwise_cnn` (30 trials):

- **Best val F1 Macro:** 0.7483
- **Key findings:** Lower learning rate (1.09e-4), higher weight decay (2.56e-3), smaller batch size (32), class-aware sampling enabled, plateau scheduler, and adjusted asymmetric focal loss parameters (γ⁺=1.85, γ⁻=2.90).

### Cross-Validation Summary

5-fold multi-label stratified CV on `leadwise_cnn`:

| Metric | Value |
|--------|-------|
| Mean val loss | 0.0563 |
| Std val loss | 0.0009 |

Low standard deviation indicates stable model performance across folds.

---

## Pipeline Overview

```text
PTB-XL records + SCP statements
  → label encoding + class weights
  → waveform loading (100 or 500 Hz)
  → preprocessing (baseline removal, bandpass, clip, normalise)
  → split by strat_fold
  → train (config-driven)
  → evaluate + per-class threshold optimisation
  → outputs (metrics, predictions, models, histories)
  → optional: ensemble / cross-validation / ablation / HPO
```

---

## Installation

### Prerequisites

- Python 3.10+
- Recommended: CUDA-capable GPU for training (MPS and CPU also supported)

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

Option A — programmatic download helper:

```bash
python -m src.data.download
```

Option B — download manually from [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/) and place under `data/raw/`.

### Kaggle Path Compatibility

Path resolution is Kaggle-aware. Raw data and writable outputs are auto-detected from `/kaggle/input` and `/kaggle/working` when running in a Kaggle kernel.

Optional environment variable overrides:

| Variable | Purpose |
|----------|---------|
| `PTBXL_DATA_DIR` | Dataset root directory |
| `ECG_CONFIG_PATH` | Config YAML for the web app |
| `ECG_CHECKPOINT_PATH` | Model checkpoint for the web app |
| `ECG_PROCESSED_DIR` | Preprocessed data directory |
| `ECG_MODEL_COMPARISON_PATH` | Model comparison JSON |

---

## How to Run

### 1. Preprocess and Cache Dataset

```bash
python scripts/preprocess_data.py --config configs/default.yaml
```

### 2. Train and Evaluate One Model

```bash
python scripts/run_experiment.py --config configs/leadwise_cnn.yaml
```

### 3. Train All Models and Produce Comparison

```bash
python scripts/run_all_models.py
```

### 4. Run Ablation Sweep

```bash
python scripts/run_ablation.py --config configs/default.yaml
```

### 5. Run Top-Model Ensemble

```bash
python scripts/run_ensemble.py --models leadwise_cnn cnn_1d lstm
# With custom weights:
python scripts/run_ensemble.py --models leadwise_cnn cnn_1d lstm --weights 0.4 0.35 0.25
```

Requires trained checkpoints in `outputs/models/`.

### 6. Run Bayesian Hyperparameter Tuning (Optuna TPE)

```bash
python scripts/run_bayesian_tuning.py --config configs/leadwise_cnn.yaml --n-trials 30
```

### 7. Run Cross-Validation

```bash
python scripts/run_cv.py --config configs/leadwise_cnn.yaml
```

### 8. Launch Web App

```bash
python run_server.py
```

Then open http://localhost:8000. The web app uses per-class optimal thresholds automatically.

### 9. Programmatic Inference

**Single-model inference with optimal thresholds:**

```python
import numpy as np
import yaml
from src.inference import load_trained_model, load_optimal_thresholds, predict_signal

with open("configs/leadwise_cnn.yaml") as f:
    config = yaml.safe_load(f)

model = load_trained_model("outputs/models/best_leadwise_cnn.pt", config=config)
thresholds = load_optimal_thresholds("outputs/results/leadwise_cnn")

signal = np.load("path/to/ecg_signal.npy")  # shape: (1000, 12)
result = predict_signal(model, signal, config, threshold=thresholds)
print(result["predictions"], result["probabilities"])
```

**Ensemble inference:**

```python
from src.inference import load_ensemble, predict_ensemble, load_optimal_thresholds

models = load_ensemble(
    ["leadwise_cnn", "cnn_1d", "lstm"],
    configs_dir="configs",
    models_dir="outputs/models",
)
thresholds = load_optimal_thresholds("outputs/results/ensemble_leadwise_cnn_cnn_1d_lstm")

result = predict_ensemble(
    models, signal, config,
    threshold=thresholds,
    label_classes=["NORM", "MI", "STTC", "CD", "HYP"],
)
print(result["predicted_classes"], result["class_probabilities"])
```

---

## Outputs and Artifacts

### Cached Processed Data

| File | Description |
|------|-------------|
| `data/processed/signals.npy` | Preprocessed ECG signals `(N, 1000, 12)` |
| `data/processed/labels.npy` | Binary label matrix `(N, 5)` |
| `data/processed/class_weights.npy` | Inverse-prevalence class weights `(5,)` |
| `data/processed/label_classes.json` | Ordered class names |
| `data/processed/train_indices.npy` | Training split indices |
| `data/processed/val_indices.npy` | Validation split indices |
| `data/processed/test_indices.npy` | Test split indices |
| `data/processed/metadata.json` | Preprocessing configuration snapshot |

### Per-Model Results

Each model directory under `outputs/results/` contains: `metrics.json`, `probabilities.npy`, `predictions.npy`, `labels.npy`, `optimal_thresholds.npy`, and training history arrays.

### Multi-Model Comparison

- `outputs/model_comparison.json` — metrics for all trained models in a single file.

### Ensemble Results

- `outputs/results/ensemble_leadwise_cnn_cnn_1d_lstm/` — metrics, predictions, probabilities, thresholds, and ensemble metadata.

### Cross-Validation Summary

- `outputs/results/cross_validation/cv_summary.json` — per-fold validation losses, mean and std.

### Ablation Runs

- `outputs/ablation_runs/` — contains `baseline`, `asymmetric_focal`, `asymmetric_focal_sampler`, and `throughput` variants. Each has per-model subdirectories under `results/` (note: metrics files are not populated in this snapshot).

### Hyperparameter Tuning

- `outputs/hpo/bayes_leadwise_cnn/best_summary.json` — best trial params
- `outputs/hpo/bayes_leadwise_cnn/best_config.yaml` — best YAML config
- `outputs/hpo/bayes_leadwise_cnn/trials.csv` — all trial results

### Model Checkpoints

- `outputs/models/best_<model_name>.pt` — best checkpoints (includes `model_state_dict`, `optimizer_state_dict`, `config`, `epoch`, `val_loss`)
- `outputs/models/best_leadwise_cnn_fold{0–4}.pt` — per-fold CV checkpoints

### Archives

- `outputs/archives/models.zip`, `outputs/archives/results.zip`

---

## Known Issues and Limitations

1. **Threshold optimisation on test split** — Per-class thresholds are currently optimised on the test set during evaluation, which inflates reported test metrics. Should be optimised on validation and applied to test.
2. **Cross-validation outputs** — Only validation loss is reported per fold; per-fold F1 and confusion matrices are not saved.
3. **Ablation metrics** — Ablation run directories exist but `metrics.json` files are not populated in the current snapshot.
4. **pretrained_resnet** — Severely underperforms (F1 0.53) due to ImageNet domain mismatch; consider domain-specific pretraining or removing from the comparison.
5. **Figure artifacts** — Training history plots, ROC curves, and confusion matrices are generated by `run_experiment.py` but not committed to the repository.

## Future Improvements

- [ ] Optimise thresholds on validation, apply to test for unbiased evaluation.
- [ ] Save per-fold metrics and confusion matrices for cross-validation.
- [ ] Add output schema validation (JSON Schema or Pydantic) for metrics and predictions.
- [ ] Populate ablation summary metrics.
- [ ] Add data augmentation (temporal jitter, lead dropout, Gaussian noise).
- [ ] Implement Test-Time Augmentation (TTA) for inference.
- [ ] Support ONNX export for deployment without PyTorch runtime.
- [ ] Add Stochastic Weight Averaging (SWA) as a training option.

---

## Project Structure

```
ecg-multilabel-classification/
├── configs/              # YAML experiment configs
│   ├── default.yaml      # Base config (all defaults)
│   ├── leadwise_cnn.yaml
│   ├── cnn_1d.yaml
│   ├── lstm.yaml
│   ├── cnn_lstm.yaml
│   ├── cnn_transformer.yaml
│   ├── resnet.yaml
│   └── pretrained_resnet.yaml
├── data/
│   ├── raw/              # PTB-XL dataset (gitignored)
│   └── processed/        # Cached preprocessed tensors
├── notebooks/
│   └── kaggle_training.ipynb
├── outputs/
│   ├── models/           # Saved checkpoints
│   ├── results/          # Per-model evaluation results
│   ├── ablation_runs/    # Ablation experiment results
│   ├── hpo/              # Hyperparameter tuning results
│   ├── archives/         # Zip archives
│   └── model_comparison.json
├── scripts/
│   ├── preprocess_data.py
│   ├── run_experiment.py
│   ├── run_all_models.py
│   ├── run_ensemble.py
│   ├── run_ablation.py
│   ├── run_bayesian_tuning.py
│   ├── run_cv.py
│   └── run_full_pipeline.sh
├── src/
│   ├── data/             # Loading, preprocessing, splitting, dataset
│   ├── models/           # Architecture implementations + registry
│   ├── training/         # Trainer, loss, optimiser, scheduler, CV
│   ├── evaluation/       # Evaluator, metrics, plots, confusion matrix
│   ├── inference/        # Predict, ensemble, postprocessing, thresholds
│   └── utils/            # Device, paths, artifacts, logging
├── webapp/
│   ├── main.py           # FastAPI server
│   └── static/           # Frontend HTML/CSS/JS
├── run_server.py          # Quick launcher
├── requirements.txt
└── requirements.kaggle.txt
```

---

## License

This project uses the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) which is available under the [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/ptb-xl/1.0.1/).
