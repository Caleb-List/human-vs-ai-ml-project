"""Download the project dataset from Kaggle."""

import shutil
from pathlib import Path

import kagglehub
import pandas as pd


DATASET_HANDLE = "datascikhan/ai-vs-human-text-classification-benchmark-50k"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CSV_PATH = RAW_DATA_DIR / "AI_vs_Human_Text_Classification_Benchmark.csv"
LABEL_NAMES = {0: "Human", 1: "AI"}


def download_dataset() -> str:
    """Download the dataset CSV into data/raw (once) and return that directory.

    The raw file is never re-downloaded or overwritten once present. Downloading
    goes through KaggleHub's cache and copies only the CSV, so other contents of
    data/raw (such as the M4 download) do not block it.
    """
    if not RAW_CSV_PATH.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
        shutil.copy2(cache_dir / RAW_CSV_PATH.name, RAW_CSV_PATH)
    return str(RAW_DATA_DIR)


def load_dataset() -> pd.DataFrame:
    """Load the downloaded CSV without modifying the raw file."""
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {RAW_CSV_PATH}. "
            "Download it first with: python src/prepare_data.py"
        )

    return pd.read_csv(RAW_CSV_PATH)
