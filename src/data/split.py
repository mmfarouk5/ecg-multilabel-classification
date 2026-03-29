"""
Dataset splitting utilities for PTB-XL.

Supports PTB-XL's built-in stratified folds as well as multi-label
stratified K-fold splitting via iterstrat.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_val_test_split(
    metadata_df: pd.DataFrame,
    val_fold: int = 9,
    test_fold: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Split data using PTB-XL's built-in ``strat_fold`` column.

    Folds 1-8 → train, fold ``val_fold`` → validation, fold ``test_fold`` → test.

    Args:
        metadata_df: DataFrame with ``strat_fold`` column.
        val_fold: Fold number for validation set.
        test_fold: Fold number for test set.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``, each
        mapping to an array of integer indices (positional, 0-based).
    """
    folds = metadata_df.strat_fold.values

    train_mask = (folds != val_fold) & (folds != test_fold)
    val_mask = folds == val_fold
    test_mask = folds == test_fold

    splits = {
        "train": np.where(train_mask)[0],
        "val": np.where(val_mask)[0],
        "test": np.where(test_mask)[0],
    }

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return splits


def get_kfold_splits(
    metadata_df: pd.DataFrame,
    n_folds: int = 5,
    test_fold: int = 10,
    label_matrix: Optional[np.ndarray] = None,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate K-fold train/validation splits.

    If ``label_matrix`` is provided, uses multi-label stratified splitting
    via ``iterstrat.ml_stratifiers.MultilabelStratifiedKFold``.
    Otherwise, uses PTB-XL's built-in folds (folds 1 through n_folds).

    The test fold is always excluded.

    Args:
        metadata_df: DataFrame with ``strat_fold`` column.
        n_folds: Number of folds.
        test_fold: Fold number reserved for testing (excluded).
        label_matrix: Optional binary label matrix for stratification.
        seed: Random seed for reproducibility.

    Returns:
        List of ``(train_indices, val_indices)`` tuples (0-based positional).
    """
    # Exclude test fold
    non_test_mask = metadata_df.strat_fold.values != test_fold
    non_test_indices = np.where(non_test_mask)[0]

    if label_matrix is not None:
        # Use multi-label stratified K-fold
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        mskf = MultilabelStratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )
        non_test_labels = label_matrix[non_test_indices]
        X_dummy = np.zeros((len(non_test_indices), 1))

        folds = []
        for train_local, val_local in mskf.split(X_dummy, non_test_labels):
            train_idx = non_test_indices[train_local]
            val_idx = non_test_indices[val_local]
            folds.append((train_idx, val_idx))

        logger.info(
            "Created %d multi-label stratified folds (excl. test fold %d)",
            n_folds, test_fold,
        )
    else:
        # Use PTB-XL built-in folds
        strat_folds = metadata_df.strat_fold.values
        available_folds = sorted(
            set(strat_folds[non_test_mask].astype(int))
        )[:n_folds]

        folds = []
        for val_f in available_folds:
            val_mask = (strat_folds == val_f)
            train_mask = non_test_mask & ~val_mask
            folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))

        logger.info(
            "Created %d folds using PTB-XL strat_fold (excl. test fold %d)",
            len(folds), test_fold,
        )

    return folds


def save_split_indices(
    splits: Dict[str, np.ndarray],
    save_dir: str,
) -> None:
    """
    Save split indices to disk as ``.npy`` files.

    Args:
        splits: Dictionary mapping split names to index arrays.
        save_dir: Directory to save the files.
    """
    from pathlib import Path
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for name, indices in splits.items():
        filepath = save_path / f"{name}_indices.npy"
        np.save(filepath, indices)
        logger.info("Saved %s indices (%d) to %s", name, len(indices), filepath)
