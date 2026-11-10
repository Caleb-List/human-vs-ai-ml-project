"""Write the reduced, independently derived preprocessing artifact."""

from data_loader import PROJECT_ROOT
import pandas as pd


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "text_derived_features_candidate.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "human_ai_text_processed.csv"
TRACEABILITY_COLUMNS = ["Text_ID", "Raw_Text", "Label", "Label_Name"]
FINAL_FEATURE_COLUMNS = [
    "Text_Word_Count",
    "Text_Average_Word_Length",
    "Text_Average_Sentence_Length_Words",
    "Text_Guiraud_Root_TTR",
    "Text_Punctuation_Density",
    "Text_Flesch_Kincaid_Grade_Level",
    "Text_Burstiness",
]


def validate_processed_data(df: pd.DataFrame) -> None:
    expected_columns = TRACEABILITY_COLUMNS + FINAL_FEATURE_COLUMNS
    if list(df.columns) != expected_columns:
        raise ValueError(f"Unexpected processed schema: {list(df.columns)}")
    if len(df) != 50_000:
        raise ValueError(f"Expected 50,000 rows, found {len(df)}")
    if df["Text_ID"].duplicated().any() or df["Raw_Text"].duplicated().any():
        raise ValueError("Processed data contains duplicate IDs or raw texts.")
    if df["Label"].isna().any() or set(df["Label"].unique()) != {0, 1}:
        raise ValueError("Label must be complete and contain exactly 0 and 1.")
    if df[FINAL_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Final model features contain missing values.")


def main() -> None:
    candidate = pd.read_csv(INPUT_PATH)
    processed = candidate[TRACEABILITY_COLUMNS + FINAL_FEATURE_COLUMNS].copy()
    validate_processed_data(processed)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUTPUT_PATH, index=False)
    print(f"Final reduced preprocessing artifact saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(processed):,}; model features: {len(FINAL_FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
