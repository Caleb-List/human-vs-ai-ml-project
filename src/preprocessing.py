"""Conservative, non-destructive preparation for exploratory analysis."""

import pandas as pd


LABEL_NAMES = {0: "Human", 1: "AI"}
REQUIRED_COLUMNS = {"Text_ID", "Raw_Text", "Label"}


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate key fields and add readable labels without altering raw text.

    Structural missing values in metadata (such as temperature for human
    samples) are preserved. Rows are never silently dropped or imputed.
    """
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    prepared = df.copy()
    if prepared[["Text_ID", "Raw_Text", "Label"]].isna().any().any():
        raise ValueError("Text_ID, Raw_Text, and Label must not contain missing values.")

    labels = pd.to_numeric(prepared["Label"], errors="raise").astype("int8")
    unexpected_labels = sorted(set(labels.unique()) - set(LABEL_NAMES))
    if unexpected_labels:
        raise ValueError(f"Unexpected Label values: {unexpected_labels}")

    prepared["Label"] = labels
    prepared["Label_Name"] = labels.map(LABEL_NAMES)
    return prepared
