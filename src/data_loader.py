"""Download the project dataset from Kaggle."""

from pathlib import Path

import kagglehub
import pandas as pd


DATASET_HANDLE = "datascikhan/ai-vs-human-text-classification-benchmark-50k"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CSV_PATH = RAW_DATA_DIR / "AI_vs_Human_Text_Classification_Benchmark.csv"
LABEL_NAMES = {0: "Human", 1: "AI"}


def download_dataset() -> str:
    """Download the latest dataset version into data/raw and return that path."""
    RAW_DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    return kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(RAW_DATA_DIR))


def load_dataset() -> pd.DataFrame:
    """Load the downloaded CSV without modifying the raw file."""
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {RAW_CSV_PATH}. "
            "Download it first with: python src/prepare_data.py"
        )

    return pd.read_csv(RAW_CSV_PATH)
