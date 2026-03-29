"""
Data loader for the PTB-XL ECG dataset.

Provides functions to load raw ECG signals, metadata, and SCP diagnostic
statements from the PTB-XL dataset.
"""

import ast
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import wfdb

logger = logging.getLogger(__name__)


def load_metadata(data_dir: str) -> pd.DataFrame:
    """
    Load PTB-XL metadata from ptbxl_database.csv.

    Args:
        data_dir: Path to the PTB-XL dataset root directory.

    Returns:
        DataFrame with parsed metadata. The ``scp_codes`` column is
        converted from string to dict via ``ast.literal_eval``.
    """
    csv_path = Path(data_dir) / "ptbxl_database.csv"
    logger.info("Loading metadata from %s", csv_path)

    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    logger.info("Loaded metadata for %d records", len(df))
    return df


def load_scp_statements(data_dir: str) -> pd.DataFrame:
    """
    Load SCP statement descriptions used for diagnostic aggregation.

    Args:
        data_dir: Path to the PTB-XL dataset root directory.

    Returns:
        DataFrame indexed by SCP code with columns for diagnostic,
        form, rhythm, diagnostic_class, and diagnostic_subclass.
    """
    csv_path = Path(data_dir) / "scp_statements.csv"
    logger.info("Loading SCP statements from %s", csv_path)

    scp_df = pd.read_csv(csv_path, index_col=0)
    return scp_df


def load_raw_signals(
    metadata_df: pd.DataFrame,
    data_dir: str,
    sampling_rate: int = 100,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Load raw ECG signal waveforms using wfdb.

    Args:
        metadata_df: DataFrame with ``filename_lr`` and ``filename_hr`` columns.
        data_dir: Path to the PTB-XL dataset root directory.
        sampling_rate: 100 (low-res) or 500 (high-res).
        max_samples: If set, only load this many records (useful for debugging).

    Returns:
        Numpy array of shape ``(N, seq_len, 12)`` where seq_len is
        1000 for 100 Hz or 5000 for 500 Hz.
    """
    data_path = str(Path(data_dir)) + "/"

    if sampling_rate == 100:
        filenames = metadata_df.filename_lr
    elif sampling_rate == 500:
        filenames = metadata_df.filename_hr
    else:
        raise ValueError(f"Unsupported sampling rate: {sampling_rate}. Use 100 or 500.")

    if max_samples is not None:
        filenames = filenames.iloc[:max_samples]

    logger.info(
        "Loading %d signals at %d Hz from %s",
        len(filenames), sampling_rate, data_path,
    )

    signals = []
    for fname in filenames:
        record = wfdb.rdsamp(data_path + fname)
        signals.append(record[0])

    signals = np.array(signals, dtype=np.float32)
    logger.info("Loaded signals with shape %s", signals.shape)
    return signals


def aggregate_diagnostics(
    metadata_df: pd.DataFrame,
    scp_df: pd.DataFrame,
    label_type: str = "diagnostic_superclass",
) -> pd.Series:
    """
    Map SCP codes to diagnostic superclass or subclass labels.

    Each record may map to multiple diagnostic classes, producing a
    multi-label setup.

    Args:
        metadata_df: DataFrame with ``scp_codes`` column (dict type).
        scp_df: SCP statements DataFrame from :func:`load_scp_statements`.
        label_type: ``"diagnostic_superclass"`` or ``"diagnostic_subclass"``.

    Returns:
        Series of lists, where each list contains the diagnostic
        class strings for that record.
    """
    # Filter to diagnostic statements only
    diag_df = scp_df[scp_df.diagnostic == 1.0]

    if label_type == "diagnostic_superclass":
        col = "diagnostic_class"
    elif label_type == "diagnostic_subclass":
        col = "diagnostic_subclass"
    else:
        raise ValueError(f"Unsupported label_type: {label_type}")

    def _aggregate(scp_dict):
        labels = []
        for code in scp_dict.keys():
            if code in diag_df.index:
                val = diag_df.loc[code][col]
                if pd.notna(val):
                    labels.append(val)
        return list(set(labels))

    logger.info("Aggregating diagnostics using '%s'", label_type)
    result = metadata_df.scp_codes.apply(_aggregate)
    return result
