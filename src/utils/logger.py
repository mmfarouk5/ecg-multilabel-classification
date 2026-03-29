"""
Logging utilities for the ECG classification project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ecg",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a project logger.

    Args:
        name: Logger name.
        log_file: Optional file path to write logs. The parent directory
            is created automatically if it doesn't exist.
        level: Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
