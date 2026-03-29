"""
Data download utility for PTB-XL dataset.

Downloads the PTB-XL dataset from PhysioNet if not already present.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def download_ptbxl(data_dir: str = "data/raw") -> Path:
    """
    Download the PTB-XL dataset using wfdb.

    Args:
        data_dir: Directory to download the dataset into.

    Returns:
        Path to the downloaded dataset directory.
    """
    import wfdb

    data_path = Path(data_dir)
    # Check for the expected dataset folder
    expected = data_path / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

    if expected.exists():
        logger.info("PTB-XL dataset already exists at %s", expected)
        return expected

    logger.info("Downloading PTB-XL dataset to %s ...", data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("ptb-xl", dl_dir=str(data_path))
    logger.info("Download complete.")
    return expected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_ptbxl()