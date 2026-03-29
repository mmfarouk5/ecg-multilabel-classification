"""
Multi-label encoding for ECG diagnostic labels.

Converts diagnostic class lists into binary label matrices and computes
class weights for handling class imbalance.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical ordering of PTB-XL diagnostic superclasses
DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Canonical ordering of PTB-XL diagnostic subclasses
DIAGNOSTIC_SUBCLASSES = [
    "NORM",
    "IMI", "AMI", "LMI", "PMI",           # MI subclasses
    "STTC", "NST_", "ISC_", "ISCA", "ISCI",  # STTC subclasses
    "LAFB/LPFB", "IRBBB", "_AVB", "IVCD",
    "CRBBB", "CLBBB", "WPW", "ILBBB",      # CD subclasses
    "LVH", "LAO/LAE", "RVH", "RAO/RAE", "SEHYP",  # HYP subclasses
]


def get_label_classes(label_type: str = "diagnostic_superclass") -> List[str]:
    """
    Get the canonical ordered list of label class names.

    Args:
        label_type: ``"diagnostic_superclass"`` or ``"diagnostic_subclass"``.

    Returns:
        Ordered list of class name strings.
    """
    if label_type == "diagnostic_superclass":
        return DIAGNOSTIC_SUPERCLASSES.copy()
    elif label_type == "diagnostic_subclass":
        return DIAGNOSTIC_SUBCLASSES.copy()
    else:
        raise ValueError(f"Unsupported label_type: {label_type}")


def encode_labels(
    diagnostic_labels: pd.Series,
    label_classes: Optional[List[str]] = None,
    label_type: str = "diagnostic_superclass",
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode diagnostic label lists into a binary multi-label matrix.

    Args:
        diagnostic_labels: Series of lists, e.g. ``[["NORM"], ["MI", "STTC"]]``.
        label_classes: Ordered list of class names. If None, uses default.
        label_type: Used to determine default classes if ``label_classes`` is None.

    Returns:
        Tuple of:
        - Binary label matrix of shape ``(N, num_classes)``, dtype float32.
        - List of class names in column order.
    """
    if label_classes is None:
        label_classes = get_label_classes(label_type)

    num_samples = len(diagnostic_labels)
    num_classes = len(label_classes)
    label_matrix = np.zeros((num_samples, num_classes), dtype=np.float32)

    class_to_idx = {cls: idx for idx, cls in enumerate(label_classes)}

    for i, labels in enumerate(diagnostic_labels):
        for label in labels:
            if label in class_to_idx:
                label_matrix[i, class_to_idx[label]] = 1.0

    logger.info(
        "Encoded %d samples into %d-class binary matrix",
        num_samples, num_classes,
    )
    return label_matrix, label_classes


def compute_class_weights(label_matrix: np.ndarray) -> np.ndarray:
    """
    Compute inverse frequency class weights for loss balancing.

    Uses the formula: ``weight_c = N / (num_classes * count_c)``
    where ``count_c`` is the number of positive samples for class c.

    Args:
        label_matrix: Binary label matrix of shape ``(N, num_classes)``.

    Returns:
        Class weights array of shape ``(num_classes,)``.
    """
    num_samples = label_matrix.shape[0]
    num_classes = label_matrix.shape[1]
    class_counts = label_matrix.sum(axis=0)

    # Prevent division by zero for classes with no samples
    class_counts = np.where(class_counts == 0, 1, class_counts)

    weights = num_samples / (num_classes * class_counts)
    weights = weights.astype(np.float32)

    logger.info("Class counts: %s", class_counts.astype(int))
    logger.info("Class weights: %s", np.round(weights, 3))
    return weights


def get_label_distribution(
    label_matrix: np.ndarray,
    label_classes: List[str],
) -> pd.DataFrame:
    """
    Compute and return the label distribution as a DataFrame.

    Args:
        label_matrix: Binary label matrix of shape ``(N, num_classes)``.
        label_classes: Ordered list of class names.

    Returns:
        DataFrame with columns: class, count, percentage.
    """
    counts = label_matrix.sum(axis=0).astype(int)
    total = label_matrix.shape[0]

    dist_df = pd.DataFrame({
        "class": label_classes,
        "count": counts,
        "percentage": np.round(counts / total * 100, 2),
    })
    return dist_df
