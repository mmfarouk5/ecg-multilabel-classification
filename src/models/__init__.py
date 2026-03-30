"""
Model registry for ECG classification models.

Provides a centralized ``MODEL_REGISTRY`` dictionary and a factory
function ``build_model`` to instantiate models by name from config.
"""

from typing import Any, Dict

import torch.nn as nn

from src.models.cnn_1d import CNN1D
from src.models.leadwise_cnn import LeadwiseCNN
from src.models.resnet import ResNet1D
from src.models.pretrained_resnet import PretrainedResNet1D
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.models.cnn_lstm import CNNLSTM
from src.models.cnn_transformer import CNNTransformer

# ── Model Registry ──────────────────────────────────────────
MODEL_REGISTRY: Dict[str, type] = {
    "cnn_1d": CNN1D,
    "leadwise_cnn": LeadwiseCNN,
    "resnet": ResNet1D,
    "pretrained_resnet": PretrainedResNet1D,
    "lstm": LSTMModel,
    "transformer": TransformerModel,
    "cnn_lstm": CNNLSTM,
    "cnn_transformer": CNNTransformer,
}


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Instantiate a model from the registry using config.

    Args:
        config: Full configuration dictionary. Must contain a ``model``
            key with at least ``name``, ``input_channels``, and
            ``num_classes``.

    Returns:
        Instantiated model module.

    Raises:
        ValueError: If the model name is not in the registry.
    """
    model_cfg = config["model"]
    name = model_cfg["name"]

    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        )

    ModelClass = MODEL_REGISTRY[name]

    # Merge common params with model-specific params
    kwargs = {
        "input_channels": model_cfg.get("input_channels", 12),
        "num_classes": model_cfg.get("num_classes", 5),
    }
    kwargs.update(model_cfg.get("params", {}))

    model = ModelClass(**kwargs)
    return model


__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "CNN1D",
    "LeadwiseCNN",
    "ResNet1D",
    "PretrainedResNet1D",
    "LSTMModel",
    "TransformerModel",
    "CNNLSTM",
    "CNNTransformer",
]
