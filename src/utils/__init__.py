"""
Shared utility functions for ECG classification pipeline.
"""

import torch


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
