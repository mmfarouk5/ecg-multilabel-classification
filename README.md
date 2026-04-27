# ECG Multi-Label Classification (PTB-XL)

End-to-end deep learning project for **multi-label ECG diagnosis** from 12-lead signals.  
The repository trains and compares 7 neural architectures on PTB-XL, serves a FastAPI web app for interactive inference, and includes reusable preprocessing, training, and evaluation modules.

## Project Overview

This project predicts 5 PTB-XL diagnostic superclasses:

- `NORM` (Normal ECG)
- `MI` (Myocardial Infarction)
- `STTC` (ST/T Changes)
- `CD` (Conduction Disturbance)
- `HYP` (Hypertrophy)

Core goals:

1. Build a robust multi-label ECG pipeline from raw waveform files to deployable inference.
2. Compare architecture families under a consistent training/evaluation setup.
3. Prioritize reproducibility with config-driven experiments, deterministic options, and artifact tracking.

## Data Requirements

### Dataset

- **Source**: PTB-XL (PhysioNet, v1.0.1)
- **Expected path (default config)**:  
  `data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1`

### Required files in dataset root

- `ptbxl_database.csv`
- `scp_statements.csv`
- waveform records referenced by `filename_lr` / `filename_hr`

### Signal assumptions

- 12 leads
- 10-second windows
- default training sampling rate: **100 Hz** (`sampling_rate: 100`)
- training tensors are shaped as `(batch, channels=12, seq_len)`

### Split protocol

Default split follows PTB-XL `strat_fold`:

- Train: folds 1-8
- Validation: fold 9
- Test: fold 10

### Cached preprocessed artifacts

After preprocessing, the project stores:

- `signals.npy`, `labels.npy`
- `train_indices.npy`, `val_indices.npy`, `test_indices.npy`
- `class_weights.npy`, `label_classes.json`, `metadata.json`

under `data/processed/`.

## Installation

### Prerequisites

- Python **3.10+**
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

Option B: download manually from PhysioNet and place it under `data/raw/`.

### Kaggle path compatibility

Training/evaluation scripts now resolve paths automatically for Kaggle:

- Raw PTB-XL data is auto-detected from `/kaggle/input/...`
- Writable artifacts are directed to `/kaggle/working/...`

Optional overrides:

- `PTBXL_DATA_DIR` for dataset root
- `ECG_CHECKPOINT_PATH`, `ECG_PROCESSED_DIR`, `ECG_MODEL_COMPARISON_PATH` for web app assets

## Usage Examples

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

> Requires checkpoints at `outputs/models/best_<model>.pt` for each selected model.

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

Then open: `http://localhost:8000`

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

## Model Architecture Summary

| Model | Family | Key idea |
|---|---|---|
| `cnn_1d` | CNN | Deep 1D conv blocks + global pooling baseline |
| `leadwise_cnn` | CNN | Shared per-lead encoder, then late fusion across 12 leads |
| `pretrained_resnet` | ResNet | 1D residual backbone with optional transfer learning |
| `lstm` | RNN | BiLSTM with temporal attention |
| `transformer` | Transformer | Sequence projection + CLS token + encoder stack |
| `cnn_lstm` | Hybrid | CNN feature extractor followed by BiLSTM temporal modeling |
| `cnn_transformer` | Hybrid | CNN downsampling + transformer encoder for long-range context |

Training characteristics across configs:

- Loss: weighted BCE / focal / asymmetric focal (default experiments use asymmetric focal)
- Optimizer: AdamW
- Scheduler: cosine annealing
- Early stopping on `val_f1_macro`
- Optional AMP on CUDA

## Results Summary

The table below is taken from `outputs/results/model_comparison.json` in this repository snapshot.

| Rank | Model | F1 Macro | ROC-AUC Macro | Subset Accuracy |
|---|---|---:|---:|---:|
| 1 | **leadwise_cnn** | **0.7377** | 0.9166 | **0.6037** |
| 2 | cnn_1d | 0.7310 | **0.9193** | 0.5819 |
| 3 | lstm | 0.7309 | 0.9176 | 0.6015 |
| 4 | cnn_transformer | 0.7274 | 0.9176 | 0.5915 |
| 5 | pretrained_resnet | 0.7240 | 0.9140 | 0.5815 |
| 6 | cnn_lstm | 0.7207 | 0.9126 | 0.5874 |
| 7 | transformer | 0.6874 | 0.8855 | 0.5452 |

Key takeaway: **Leadwise CNN** provides the best overall macro-F1 and exact-match accuracy, while `cnn_1d` achieves the highest macro ROC-AUC.

## Code Quality and Reproducibility Review

Strengths observed in this codebase:

- Clear module boundaries (`src/data`, `src/models`, `src/training`, `src/evaluation`, `src/inference`)
- Config-driven experiments via YAML files per model
- Reproducibility controls (seed setting, deterministic mode toggle, fixed fold-based split)
- Practical performance features (cached preprocessing, AMP support, early stopping, checkpointing)
- Consistent artifact outputs (metrics, predictions, figures, saved models)

Best-practice improvements that would strengthen portfolio readiness further:

1. Add automated test coverage for data pipeline, model factory, and training/evaluation smoke tests.
2. Add CI (lint + tests) and a pinned environment file for exact dependency reproducibility.
3. Add a `LICENSE` file and model-card style documentation for clinical-use disclaimers.

## Project Structure

```text
.
├── configs/                # Experiment configs (per model + default)
├── data/
│   ├── raw/                # PTB-XL raw files
│   └── processed/          # Cached numpy artifacts
├── outputs/
│   ├── models/             # Checkpoints
│   ├── results/            # Metrics/predictions
│   └── figures/            # Curves/plots
├── scripts/                # CLI experiment runners
├── src/                    # Core library code
├── webapp/                 # FastAPI app + static frontend
├── run_server.py           # One-command local launcher
└── requirements.txt
```

## Acknowledgments

- PTB-XL dataset (PhysioNet)
- PyTorch ecosystem contributors
