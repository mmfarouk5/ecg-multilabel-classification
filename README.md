# ECG Multi-Label Classification

A deep learning framework for automated multi-label classification of 12-lead electrocardiogram (ECG) signals using the PTB-XL dataset. This project implements and compares 7 different neural network architectures to classify ECG recordings into 5 diagnostic superclasses.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Experimental Results](#experimental-results)
- [Key Findings](#key-findings)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

## Overview

ECG interpretation is a critical task in cardiology, requiring expertise to identify various cardiac abnormalities. This project develops an automated multi-label classification system that can simultaneously detect multiple cardiac conditions from a single ECG recording.

### Key Features

- **Multi-label Classification**: Handles cases where a patient may have multiple cardiac conditions simultaneously
- **7 Deep Learning Architectures**: CNN, Leadwise CNN, ResNet, BiLSTM, Transformer, CNN-BiLSTM, CNN-Transformer
- **Asymmetric Focal Loss**: Custom loss function designed for imbalanced multi-label classification
- **Comprehensive Preprocessing**: Bandpass filtering, baseline wander removal, and Z-score normalization
- **Per-class Threshold Optimization**: Fine-grained threshold search for optimal F1 scores

## Dataset

### PTB-XL Database

We use the **PTB-XL** dataset, one of the largest publicly available 12-lead ECG datasets:

| Attribute | Value |
|-----------|-------|
| Total Recordings | 21,837 |
| Patients | 18,885 |
| Sampling Rate | 100 Hz (downsampled from 500 Hz) |
| Duration | 10 seconds per recording |
| Leads | 12 standard ECG leads |

### Diagnostic Superclasses (5 classes)

| Class | Description | Distribution |
|-------|-------------|--------------|
| **NORM** | Normal ECG | ~43.8% |
| **MI** | Myocardial Infarction | ~25.2% |
| **STTC** | ST/T Changes | ~24.1% |
| **CD** | Conduction Disturbance | ~22.5% |
| **HYP** | Hypertrophy | ~12.2% |

### Data Split

Following the PTB-XL recommended stratified split:
- **Training**: Folds 1-8 (17,441 samples)
- **Validation**: Fold 9 (2,193 samples)
- **Test**: Fold 10 (2,203 samples)

## Models

### 1. CNN 1D
Standard 1D convolutional neural network with 4 blocks of Conv-BatchNorm-ReLU-MaxPool layers, followed by global average pooling and a fully connected classifier.

### 2. Leadwise CNN (Best Model)
Processes each of the 12 ECG leads independently through a shared CNN backbone, then concatenates the per-lead features before classification. This architecture captures lead-specific patterns while maintaining parameter efficiency.

### 3. Pretrained ResNet 1D (xresnet-D)
Deep residual network adapted for 1D signals with pre-activation residual blocks, a multi-layer stem, and 4 stages of residual blocks with increasing filter sizes.

### 4. BiLSTM with Attention
Bidirectional LSTM network with a learnable attention mechanism that weights different time steps based on their relevance to the classification task.

### 5. Transformer
Vision Transformer-inspired architecture with positional encoding, CLS token, and multi-head self-attention layers adapted for ECG time series.

### 6. CNN-BiLSTM
Hybrid architecture combining CNN feature extraction with BiLSTM temporal modeling. The CNN extracts local features while the BiLSTM captures long-range dependencies.

### 7. CNN-Transformer
Combines CNN for downsampling and local feature extraction with a Transformer encoder for capturing global temporal relationships.

## Experimental Results

### Experiment 1: Ablation Study

Compared different training configurations on Leadwise CNN and Pretrained ResNet:

| Configuration | Model | F1 Macro | ROC-AUC |
|--------------|-------|----------|---------|
| Asymmetric Focal | Leadwise CNN | **0.7305** | 0.9099 |
| Asymmetric Focal | Pretrained ResNet | 0.7257 | 0.9113 |
| Baseline (BCE) | Pretrained ResNet | 0.7243 | 0.9120 |
| Throughput | Pretrained ResNet | 0.7225 | 0.9127 |
| Throughput | Leadwise CNN | 0.7219 | 0.9122 |
| Asymmetric Focal + Sampler | Leadwise CNN | 0.7206 | 0.9070 |
| Baseline (BCE) | Leadwise CNN | 0.7194 | 0.9102 |

### Experiment 2: Full Model Comparison

All 7 models trained with Asymmetric Focal Loss:

| Rank | Model | F1 Macro | ROC-AUC | Subset Accuracy |
|------|-------|----------|---------|-----------------|
| 1 | **Leadwise CNN** | **0.7377** | **0.9166** | **0.6037** |
| 2 | CNN 1D | 0.7310 | 0.9193 | 0.5819 |
| 3 | BiLSTM | 0.7309 | 0.9176 | 0.6015 |
| 4 | CNN-Transformer | 0.7274 | 0.9176 | 0.5915 |
| 5 | Pretrained ResNet | 0.7240 | 0.9140 | 0.5815 |
| 6 | CNN-BiLSTM | 0.7207 | 0.9126 | 0.5874 |
| 7 | Transformer | 0.6874 | 0.8855 | 0.5452 |

### Best Model Performance (Leadwise CNN)

| Metric | Score |
|--------|-------|
| F1 Macro | 0.7377 |
| F1 Micro | 0.7700 |
| ROC-AUC Macro | 0.9166 |
| Precision Macro | 0.7243 |
| Recall Macro | 0.7523 |
| Subset Accuracy | 0.6037 |

## Key Findings

### 1. Leadwise CNN Architecture Excels
The Leadwise CNN outperformed all other architectures by processing each ECG lead independently. This approach:
- Captures lead-specific morphological patterns
- Reduces inter-lead interference during feature extraction
- Maintains parameter efficiency through weight sharing

### 2. Asymmetric Focal Loss Improves Performance
The Asymmetric Focal Loss (γ⁺=1.0, γ⁻=4.0, clip=0.05) consistently improved results by:
- Down-weighting easy negative samples more aggressively
- Better handling of class imbalance in multi-label setting
- Focusing learning on hard positive examples

### 3. Simpler Models Outperform Complex Ones
Interestingly, simpler architectures (CNN, Leadwise CNN, BiLSTM) outperformed more complex ones (Transformer, CNN-Transformer). This suggests:
- ECG patterns may be more local than global
- The dataset size may not support very deep models
- Inductive biases of CNNs and RNNs are well-suited for ECG signals

### 4. Per-Class Threshold Optimization is Critical
Using a fine-grained threshold search (0.01 steps from 0.05 to 0.95) for each class improved F1 scores compared to using a fixed 0.5 threshold.

## Web Application

The project includes an interactive **web application** for real-time ECG diagnosis. Upload a 12-lead ECG signal or try a random sample from the PTB-XL dataset, and the AI will predict cardiac conditions instantly.

### Features

- **Drag & Drop CSV Upload** — Upload a 12-lead ECG signal (1000 timesteps × 12 columns at 100 Hz)
- **Random Sample Mode** — Try real ECG recordings from the PTB-XL test set with ground truth comparison
- **12-Lead ECG Visualization** — Clinical-style grid display of all leads rendered on HTML Canvas
- **Diagnosis Results** — Per-class probability gauges with color-coded severity and clinical descriptions
- **Zero Extra Dependencies** — Built with Python's standard library HTTP server (no Flask/FastAPI required)

### Quick Start

```bash
# One-command launcher (creates venv, installs deps, starts server)
python3 run_server.py

# Then open in browser
open http://localhost:8000
```

Or manually:

```bash
source .venv/bin/activate
python -m webapp.main --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the web UI |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/model-info` | Model details and comparison metrics |
| `GET` | `/api/sample` | Random PTB-XL test sample with ground truth |
| `POST` | `/api/predict` | Upload CSV file → get predictions |
| `POST` | `/api/predict-sample` | Random sample + instant inference |

### CSV Format

The uploaded CSV file should contain:
- **1000 rows** (10 seconds at 100 Hz sampling rate)
- **12 columns** (one per ECG lead: I, II, III, aVR, aVL, aVF, V1–V6)
- **No header row** — numeric values only

### Diagnostic Classes

| Class | Condition | Severity |
|-------|-----------|----------|
| **NORM** | Normal ECG | ✅ Normal |
| **MI** | Myocardial Infarction | 🔴 Critical |
| **STTC** | ST/T Changes | 🟡 Warning |
| **CD** | Conduction Disturbance | 🟡 Warning |
| **HYP** | Hypertrophy | 🟡 Warning |

## Project Structure

```
ecg-multilabel-classification/
├── configs/                    # YAML configuration files
│   ├── default.yaml           # Base configuration
│   ├── cnn_1d.yaml
│   ├── leadwise_cnn.yaml
│   ├── pretrained_resnet.yaml
│   ├── lstm.yaml
│   ├── transformer.yaml
│   ├── cnn_lstm.yaml
│   └── cnn_transformer.yaml
├── data/
│   ├── raw/                   # PTB-XL dataset
│   └── processed/             # Preprocessed signals
├── notebooks/
│   ├── experiment_1.ipynb     # Ablation study
│   ├── experiment_2.ipynb     # Full model comparison
│   └── kaggle_training.ipynb  # Kaggle GPU training
├── outputs/
│   ├── models/                # Saved model checkpoints
│   ├── results/               # Predictions and metrics
│   └── figures/               # Training plots
├── src/
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Neural network architectures
│   ├── training/              # Training loop and losses
│   ├── evaluation/            # Metrics and evaluation
│   ├── inference/             # Model inference
│   └── utils/                 # Utilities and helpers
├── webapp/                     # Web application
│   ├── main.py                # HTTP server with API endpoints
│   └── static/                # Frontend assets
│       ├── index.html         # Single-page application
│       ├── style.css          # Premium dark medical theme
│       └── app.js             # Frontend logic & ECG rendering
├── scripts/                   # Training and evaluation scripts
├── run_server.py              # One-click web app launcher
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ecg-multilabel-classification.git
cd ecg-multilabel-classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download PTB-XL Dataset

```bash
# Download from PhysioNet
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.1/

# Or download from Kaggle
# https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset
```

## Usage

### Web Application (Recommended)

```bash
# One-command launcher
python3 run_server.py

# Open http://localhost:8000 in your browser
```

### Training a Model

```bash
# Train with default configuration
python scripts/train.py --config configs/leadwise_cnn.yaml

# Train with custom parameters
python scripts/train.py --config configs/leadwise_cnn.yaml \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py --model outputs/models/best_leadwise_cnn.pt
```

### Kaggle Training

For training on Kaggle with free GPU:
1. Upload `notebooks/experiment_2.ipynb` to Kaggle
2. Add the PTB-XL dataset
3. Run all cells
4. Download the `outputs_archive.zip` with trained models

## Training Configuration

Key hyperparameters (from `configs/default.yaml`):

```yaml
training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  loss: "asymmetric_focal"
  asymmetric_focal_loss_params:
    gamma_pos: 1.0      # Focus on hard positives
    gamma_neg: 4.0      # Strongly down-weight easy negatives
    clip: 0.05          # Probability clipping
  early_stopping:
    monitor: "val_f1_macro"
    patience: 10
    mode: "max"
```

## Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- WFDB >= 4.1.0
- SciPy >= 1.10.0
- scikit-learn >= 1.2.0
- Matplotlib >= 3.7.0
- PyYAML >= 6.0
- TensorBoard >= 2.13.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PTB-XL dataset from PhysioNet
- Kaggle for free GPU resources
- PyTorch team for the deep learning framework
