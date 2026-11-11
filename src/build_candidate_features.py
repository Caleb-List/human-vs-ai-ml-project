"""Build a candidate CSV using only independently derived text features."""

import time

import pandas as pd

from data_loader import PROJECT_ROOT, load_dataset
from preprocessing import LABEL_NAMES
from text_features import extract_dataframe_features


OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "text_derived_features_candidate.csv"
CHUNK_SIZE = 1_000


def main() -> None:
    source = load_dataset()
    started = time.perf_counter()
    feature_chunks = []
    total_rows = len(source)
    for start in range(0, total_rows, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, total_rows)
        print(f"Extracting rows {start + 1:,}-{stop:,} of {total_rows:,}...", flush=True)
        feature_chunks.append(extract_dataframe_features(source.iloc[start:stop]))
        elapsed = time.perf_counter() - started
        rate = stop / elapsed if elapsed else 0
        remaining = (total_rows - stop) / rate if rate else 0
        print(
            f"Completed {stop:,}/{total_rows:,}; elapsed {elapsed:.1f}s; "
            f"estimated remaining {remaining:.1f}s",
            flush=True,
        )
    features = pd.concat(feature_chunks, ignore_index=True)
    candidate = source[["Text_ID", "Raw_Text", "Label"]].copy()
    candidate["Label_Name"] = candidate["Label"].map(LABEL_NAMES)
    candidate = candidate.join(features)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote candidate text-derived CSV: {OUTPUT_PATH}")
    print(f"Rows: {len(candidate):,}; columns: {len(candidate.columns)}")


if __name__ == "__main__":
    main()
