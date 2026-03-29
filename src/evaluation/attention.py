"""
Attention visualization for Transformer-based ECG models.

Extracts and plots attention weights from the Transformer model
to provide interpretability of model predictions.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@torch.no_grad()
def extract_attention(
    model: nn.Module,
    signal: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    """
    Extract attention weights from a Transformer model.

    Args:
        model: Transformer model with ``return_attention`` support.
        signal: Input signal ``(1, n_leads, seq_len)`` or ``(n_leads, seq_len)``.
        device: Device for inference.

    Returns:
        Attention weights array ``(n_heads, seq_len+1, seq_len+1)``
        or None if the model doesn't support attention extraction.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    if signal.ndim == 2:
        signal = signal.unsqueeze(0)
    signal = signal.to(device)

    try:
        logits, attn_weights = model(signal, return_attention=True)
        return attn_weights.cpu().numpy()[0]  # Remove batch dim
    except TypeError:
        logger.warning("Model does not support return_attention=True")
        return None


def plot_attention_map(
    attention_weights: np.ndarray,
    sample_idx: int = 0,
    head_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    title: str = "Attention Weights",
    sampling_rate: int = 100,
) -> plt.Figure:
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention array ``(n_heads, seq_len+1, seq_len+1)``.
        sample_idx: Sample index (for labeling).
        head_idx: If specified, plot only this attention head. Otherwise,
            plot average across all heads.
        save_path: If provided, save figure.
        title: Plot title.
        sampling_rate: Sampling rate for x-axis labeling.

    Returns:
        Matplotlib Figure.
    """
    if head_idx is not None:
        attn = attention_weights[head_idx]
        subtitle = f"Head {head_idx}"
    else:
        attn = attention_weights.mean(axis=0)  # Average over heads
        subtitle = "Mean across heads"

    # Focus on CLS token attention to sequence positions
    # CLS is at position 0, sequence starts at 1
    cls_attn = attn[0, 1:]  # Attention from CLS to all sequence positions

    fig, ax = plt.subplots(figsize=(14, 3))
    time_axis = np.arange(len(cls_attn)) / sampling_rate

    ax.plot(time_axis, cls_attn, "b-", lw=1.0, alpha=0.8)
    ax.fill_between(time_axis, cls_attn, alpha=0.3)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title(f"{title} — {subtitle} (Sample {sample_idx})", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Attention map saved to %s", save_path)

    return fig


def plot_attention_multi_head(
    attention_weights: np.ndarray,
    n_heads_to_show: int = 4,
    save_path: Optional[str] = None,
    title: str = "Multi-Head Attention",
    sampling_rate: int = 100,
) -> plt.Figure:
    """
    Plot CLS attention for multiple attention heads.

    Args:
        attention_weights: ``(n_heads, seq_len+1, seq_len+1)``.
        n_heads_to_show: Number of heads to display.
        save_path: If provided, save figure.
        title: Plot title.
        sampling_rate: Sampling rate.

    Returns:
        Matplotlib Figure.
    """
    n_heads = min(n_heads_to_show, attention_weights.shape[0])
    fig, axes = plt.subplots(n_heads, 1, figsize=(14, 3 * n_heads), sharex=True)

    if n_heads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        cls_attn = attention_weights[i, 0, 1:]
        time_axis = np.arange(len(cls_attn)) / sampling_rate

        ax.plot(time_axis, cls_attn, lw=1.0, alpha=0.8)
        ax.fill_between(time_axis, cls_attn, alpha=0.3)
        ax.set_ylabel(f"Head {i}", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Multi-head attention saved to %s", save_path)

    return fig
