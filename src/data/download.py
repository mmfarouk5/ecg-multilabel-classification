from pathlib import Path
import wfdb


def download_ptbxl():
    data_path = Path("data/raw/ptb-xl")

    if not data_path.exists():
        wfdb.dl_database('ptb-xl', dl_dir='data/raw')
    else:
        print("Dataset already exists.")