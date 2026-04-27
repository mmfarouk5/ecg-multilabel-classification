"""
Utilities for cleaning and packaging training artifacts.
"""

from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional


DEFAULT_OUTPUT_DIR_KEYS = (
    "models_dir",
    "results_dir",
    "figures_dir",
    "logs_dir",
)


def prune_checkpoint_files(
    models_dir: str | Path,
    keep_patterns: Iterable[str] = ("best_*.pt",),
) -> list[Path]:
    """
    Delete checkpoint files while keeping files that match keep patterns.

    Args:
        models_dir: Directory containing model checkpoint files.
        keep_patterns: Glob-like patterns for checkpoint filenames to keep.

    Returns:
        List of removed checkpoint paths.
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return []

    keep_patterns = tuple(keep_patterns)
    removed: list[Path] = []

    for checkpoint in models_path.glob("*.pt"):
        if any(fnmatch.fnmatch(checkpoint.name, pattern) for pattern in keep_patterns):
            continue
        checkpoint.unlink()
        removed.append(checkpoint)

    return removed


def make_zip_archive(
    source_dir: str | Path,
    destination_zip: str | Path,
) -> Optional[Path]:
    """
    Create a zip archive for a directory.

    Args:
        source_dir: Directory to archive.
        destination_zip: Zip file path to create.

    Returns:
        Archive path when created, otherwise None.
    """
    source = Path(source_dir)
    if not source.exists() or not source.is_dir():
        return None

    destination = Path(destination_zip)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination.unlink()

    archive_path = shutil.make_archive(
        base_name=str(destination.with_suffix("")),
        format="zip",
        root_dir=str(source.parent),
        base_dir=source.name,
    )
    return Path(archive_path)


def zip_output_directories(
    output_config: Mapping[str, str],
    archive_dir: str | Path | None = None,
    output_dir_keys: Iterable[str] = DEFAULT_OUTPUT_DIR_KEYS,
) -> Dict[str, Path]:
    """
    Zip standard output directories (models, results, figures, logs).

    Args:
        output_config: Output section from config.
        archive_dir: Optional directory to write zip files into.
        output_dir_keys: Config keys that point to output directories to zip.

    Returns:
        Mapping from config key (e.g., "models_dir") to created zip path.
    """
    output_root = Path(output_config.get("dir", "outputs"))
    archive_root = Path(archive_dir) if archive_dir else output_root / "archives"

    created: Dict[str, Path] = {}
    for key in output_dir_keys:
        path_str = output_config.get(key)
        if not path_str:
            continue

        source = Path(path_str)
        if not source.exists() or not source.is_dir():
            continue
        if not any(source.iterdir()):
            continue

        archive_name = f"{source.name}.zip"
        archive_path = make_zip_archive(source, archive_root / archive_name)
        if archive_path is not None:
            created[key] = archive_path

    return created
