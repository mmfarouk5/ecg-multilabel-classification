"""
ECG signal preprocessing pipeline.

Provides bandpass filtering, baseline wander removal, outlier clipping,
and Z-score normalization for 12-lead ECG signals.
"""

import logging
from typing import Dict, Any

import numpy as np
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """
    Design a Butterworth bandpass filter.

    Args:
        lowcut: Low cutoff frequency in Hz.
        highcut: High cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Tuple of filter coefficients (b, a).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 100.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to an ECG signal.

    Args:
        signal: ECG signal array of shape ``(seq_len, n_leads)`` or
                ``(N, seq_len, n_leads)``.
        lowcut: Low cutoff frequency in Hz.
        highcut: High cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal with the same shape as input.
    """
    b, a = _butter_bandpass(lowcut, highcut, fs, order)

    if signal.ndim == 2:
        # Single record: (seq_len, n_leads)
        return filtfilt(b, a, signal, axis=0).astype(np.float32)
    elif signal.ndim == 3:
        # Batch: (N, seq_len, n_leads)
        return filtfilt(b, a, signal, axis=1).astype(np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {signal.ndim}D")


def remove_baseline_wander(
    signal: np.ndarray,
    fs: float = 100.0,
    cutoff: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """
    Remove baseline wander using a high-pass Butterworth filter.

    Args:
        signal: ECG signal array of shape ``(seq_len, n_leads)`` or
                ``(N, seq_len, n_leads)``.
        fs: Sampling frequency in Hz.
        cutoff: High-pass cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Signal with baseline wander removed.
    """
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = butter(order, high, btype="high")

    axis = 0 if signal.ndim == 2 else 1
    return filtfilt(b, a, signal, axis=axis).astype(np.float32)


def clip_outliers(signal: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Clip signal values at the given percentile (symmetric).

    Args:
        signal: ECG signal array of any shape.
        percentile: Percentile threshold for clipping.

    Returns:
        Clipped signal.
    """
    high = np.percentile(np.abs(signal), percentile)
    return np.clip(signal, -high, high).astype(np.float32)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalization per lead.

    For a batch ``(N, seq_len, n_leads)``, normalization statistics are
    computed per-sample per-lead. For a single record ``(seq_len, n_leads)``,
    statistics are computed per-lead.

    Args:
        signal: ECG signal array.

    Returns:
        Normalized signal.
    """
    if signal.ndim == 2:
        # (seq_len, n_leads)
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True)
    elif signal.ndim == 3:
        # (N, seq_len, n_leads) — normalize per sample per lead
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {signal.ndim}D")

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    return ((signal - mean) / std).astype(np.float32)


def preprocess_pipeline(signals: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Run the full preprocessing pipeline on ECG signals.

    Pipeline order:
    1. Baseline wander removal
    2. Bandpass filter
    3. Outlier clipping
    4. Z-score normalization (optional)

    Args:
        signals: Raw ECG signals of shape ``(N, seq_len, n_leads)``.
        config: Preprocessing configuration dictionary with keys:
            ``bandpass``, ``baseline_wander``, ``outlier_clip_percentile``,
            ``normalize``.

    Returns:
        Preprocessed signals with the same shape.
    """
    preproc = config.get("preprocessing", config)
    fs = float(config.get("data", {}).get("sampling_rate", 100))

    logger.info("Starting preprocessing pipeline on %s signals", signals.shape)

    # 1. Baseline wander removal
    bw_cfg = preproc.get("baseline_wander", {})
    signals = remove_baseline_wander(
        signals, fs=fs,
        cutoff=bw_cfg.get("cutoff", 0.5),
        order=bw_cfg.get("order", 4),
    )
    logger.info("  ✓ Baseline wander removed")

    # 2. Bandpass filter
    bp_cfg = preproc.get("bandpass", {})
    signals = bandpass_filter(
        signals, fs=fs,
        lowcut=bp_cfg.get("lowcut", 0.5),
        highcut=bp_cfg.get("highcut", 40.0),
        order=bp_cfg.get("order", 4),
    )
    logger.info("  ✓ Bandpass filter applied")

    # 3. Outlier clipping
    clip_pct = preproc.get("outlier_clip_percentile", 99)
    signals = clip_outliers(signals, percentile=clip_pct)
    logger.info("  ✓ Outliers clipped at %sth percentile", clip_pct)

    # 4. Normalization
    if preproc.get("normalize", True):
        signals = normalize_signal(signals)
        logger.info("  ✓ Z-score normalization applied")

    logger.info("Preprocessing complete. Output shape: %s", signals.shape)
    return signals
