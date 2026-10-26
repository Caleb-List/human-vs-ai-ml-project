"""Print a compact exploratory summary of the downloaded dataset."""

import pandas as pd

from data_loader import LABEL_NAMES, load_dataset


TEXT_PREVIEW_CHARS = 100


def _text_preview(value: object) -> str:
    """Return a single-line preview with a clear truncation marker."""
    text = " ".join(str(value).split())
    if len(text) <= TEXT_PREVIEW_CHARS:
        return text

    shortened = text[:TEXT_PREVIEW_CHARS].rsplit(" ", maxsplit=1)[0]
    return f"{shortened}..."


def main() -> None:
    df = load_dataset()

    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nFirst five rows (text preview):")
    for row_number, (_, row) in enumerate(df.head().iterrows(), start=1):
        label = LABEL_NAMES.get(row["Label"], str(row["Label"]))
        print(
            f"\n[{row_number}] Text_ID: {row['Text_ID']} | Label: {label} "
            f"| Source: {row['Source_Model']} | Words: {row['Word_Count']}"
        )
        print(f"    Text: {_text_preview(row['Raw_Text'])}")

    print("\nColumn types:")
    print(df.dtypes.to_string())

    print("\nLabel counts:")
    labels = df["Label"].map(LABEL_NAMES).fillna(df["Label"].astype(str))
    counts = labels.value_counts(dropna=False)
    print(counts.to_string())
    print("\nLabel proportions:")
    print((counts / len(df)).round(4).to_string())

    print("\nMissing values by column:")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False).to_string() or "None")

    numeric = df.select_dtypes(include="number").drop(columns=["Text_ID"], errors="ignore")
    print("\nNumeric summary:")
    print(numeric.describe().T.to_string())

    print("\nDuplicate Text_ID rows:", int(df["Text_ID"].duplicated().sum()))
    print("Duplicate Raw_Text rows:", int(df["Raw_Text"].duplicated().sum()))
    constant_columns = [column for column in df.columns if df[column].nunique(dropna=False) <= 1]
    print("Constant columns:", constant_columns or "None")

    if "Source_Model" in df:
        print("\nSource model counts by label:")
        print(pd.crosstab(df["Source_Model"], labels).to_string())

    if "Generation_Prompt_Type" in df:
        print("\nMost common generation/prompt categories:")
        print(df["Generation_Prompt_Type"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
