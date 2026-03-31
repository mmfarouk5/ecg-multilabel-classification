"""
Shared utility functions for ECG classification pipeline.
"""

from typing import Optional, Set, Tuple

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """
    Select the best available device: CUDA → MPS (Apple Silicon) → CPU.

    Returns:
        torch.device for training/inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Default models that should not use DataParallel (often faster on single GPU)
DEFAULT_FORCE_SINGLE_GPU_MODELS = {"cnn_1d", "leadwise_cnn", "lstm"}


def wrap_model_for_parallelism(
    model: nn.Module,
    model_name: str,
    multi_gpu_mode: str = "dataparallel",
    force_single_gpu_models: Optional[Set[str]] = None,
) -> Tuple[nn.Module, str]:
    """
    Wrap model with DataParallel if multiple GPUs are available.

    Args:
        model: The PyTorch model to potentially wrap.
        model_name: Name of the model (used to check exclusion list).
        multi_gpu_mode: "dataparallel" to use nn.DataParallel, "single" to force single GPU.
        force_single_gpu_models: Set of model names that should not use DataParallel.

    Returns:
        Tuple of (wrapped_model, parallel_mode_string).
    """
    if force_single_gpu_models is None:
        force_single_gpu_models = DEFAULT_FORCE_SINGLE_GPU_MODELS

    n_gpu = torch.cuda.device_count()

    if n_gpu <= 1:
        return model, "single"

    if model_name in force_single_gpu_models:
        return model, "single_forced"

    if multi_gpu_mode == "dataparallel":
        return nn.DataParallel(model), "dataparallel"

    return model, "single"


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap a DataParallel model to get the underlying module.

    Args:
        model: Model that may be wrapped in DataParallel.

    Returns:
        The underlying model (model.module if DataParallel, else model).
    """
    return model.module if hasattr(model, "module") else model
