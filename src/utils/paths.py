"""
Path resolution helpers with Kaggle-aware defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PTBXL_DIRNAME = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"


def running_on_kaggle() -> bool:
    """Return True when running in a Kaggle runtime."""
    if not Path("/kaggle").exists():
        return False

    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
        or Path("/kaggle/input").exists()
    )


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen = set()
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _looks_like_ptbxl_dir(path: Path) -> bool:
    return (path / "ptbxl_database.csv").exists() and (path / "scp_statements.csv").exists()


def _discover_kaggle_ptbxl_dir() -> Optional[Path]:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None

    for dataset_dir in sorted(input_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if _looks_like_ptbxl_dir(dataset_dir):
            return dataset_dir

        for child in sorted(dataset_dir.iterdir()):
            if child.is_dir() and _looks_like_ptbxl_dir(child):
                return child

    return None


def resolve_raw_data_dir(raw_dir: str, project_root: Optional[Path] = None) -> Path:
    """
    Resolve raw PTB-XL directory, auto-detecting Kaggle input mounts when needed.
    """
    project_root = Path(project_root).resolve(
    ) if project_root else Path.cwd().resolve()
    raw_path = Path(raw_dir).expanduser()

    candidates = []

    env_data_dir = os.getenv("PTBXL_DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir).expanduser())

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            (project_root / raw_path),
            (Path.cwd() / raw_path),
        ])

    if running_on_kaggle():
        kaggle_input = Path("/kaggle/input")
        candidates.extend([
            kaggle_input / raw_path.name,
            kaggle_input / PTBXL_DIRNAME,
            kaggle_input / PTBXL_DIRNAME / PTBXL_DIRNAME,
        ])

    for candidate in _unique_paths(candidates):
        if candidate.exists():
            return candidate.resolve()

    if running_on_kaggle():
        discovered = _discover_kaggle_ptbxl_dir()
        if discovered is not None:
            return discovered.resolve()

    # Keep behavior stable when nothing is found yet (e.g., before download).
    fallback = candidates[0] if candidates else raw_path
    return fallback.resolve() if fallback.is_absolute() else (project_root / fallback).resolve()


def resolve_writable_dir(path_value: str, project_root: Optional[Path] = None) -> Path:
    """
    Resolve writable path. On Kaggle, relative paths are rooted at /kaggle/working.
    """
    project_root = Path(project_root).resolve(
    ) if project_root else Path.cwd().resolve()
    path = Path(path_value).expanduser()

    if path.is_absolute():
        if running_on_kaggle():
            kaggle_input = Path("/kaggle/input")
            if path.is_relative_to(kaggle_input):
                remapped = Path("/kaggle/working") / \
                    path.relative_to(kaggle_input)
                return remapped.resolve()
        return path.resolve()

    kaggle_working = Path("/kaggle/working")
    base = kaggle_working if running_on_kaggle() and kaggle_working.exists() else project_root
    return (base / path).resolve()


def resolve_runtime_paths(
    config: Dict[str, Any],
    project_root: Optional[Path] = None,
    logger: Optional[Any] = None,
    create_dirs: bool = True,
) -> Dict[str, Any]:
    """
    Normalize config paths for local and Kaggle execution.

    Mutates and returns the given config dictionary.
    """
    project_root = Path(project_root).resolve(
    ) if project_root else Path.cwd().resolve()

    data_cfg = config.get("data", {})
    if "raw_dir" in data_cfg:
        data_cfg["raw_dir"] = str(resolve_raw_data_dir(
            data_cfg["raw_dir"], project_root=project_root))
    if "processed_dir" in data_cfg:
        processed = resolve_writable_dir(
            data_cfg["processed_dir"], project_root=project_root)
        data_cfg["processed_dir"] = str(processed)
        if create_dirs:
            try:
                processed.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                if logger is not None:
                    logger.warning(
                        "Could not create directory %s: %s", processed, exc)

    out_cfg = config.get("output", {})
    for key in ["dir", "figures_dir", "logs_dir", "models_dir", "results_dir"]:
        if key in out_cfg:
            out_path = resolve_writable_dir(
                out_cfg[key], project_root=project_root)
            out_cfg[key] = str(out_path)
            if create_dirs:
                try:
                    out_path.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    if logger is not None:
                        logger.warning(
                            "Could not create directory %s: %s", out_path, exc)

    if logger is not None and running_on_kaggle():
        logger.info("Kaggle path resolution enabled")
        if "raw_dir" in data_cfg:
            logger.info("  data.raw_dir: %s", data_cfg["raw_dir"])
        if "processed_dir" in data_cfg:
            logger.info("  data.processed_dir: %s", data_cfg["processed_dir"])

    return config
