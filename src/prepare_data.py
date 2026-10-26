"""Download and write a lightly prepared copy of the dataset."""

from data_loader import PROJECT_ROOT, download_dataset, load_dataset
from preprocessing import preprocess_dataset


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "AI_vs_Human_Text_Classification_Benchmark_processed.csv"


def main() -> None:
    dataset_path = download_dataset()
    df = load_dataset()
    prepared = preprocess_dataset(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(PROCESSED_CSV_PATH, index=False)

    print(f"Path to raw dataset files: {dataset_path}")
    print(f"Processed dataset saved to: {PROCESSED_CSV_PATH}")


if __name__ == "__main__":
    main()
