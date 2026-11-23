"""Validate approved M4 artifacts, provenance coverage and feature recomputation."""

import json

import numpy as np
import pandas as pd

from audit_m4 import REPORT_DIR, digest
from download_m4 import sha256
from prepare_final_data import FINAL_FEATURE_COLUMNS, TRACEABILITY_COLUMNS
from prepare_m4 import OUTPUT_DIR, PROCESSED_PATH
from text_features import extract_text_features


def main():
    summary = json.loads((REPORT_DIR / "preprocessing_summary.json").read_text())
    for name, checksum in summary["output_sha256"].items():
        if sha256(OUTPUT_DIR / name) != checksum:
            raise ValueError(f"Output checksum mismatch: {name}")
    data = pd.read_csv(PROCESSED_PATH, keep_default_na=False)
    provenance = pd.read_csv(OUTPUT_DIR / "m4_provenance.csv", keep_default_na=False, low_memory=False)
    groups = pd.read_csv(OUTPUT_DIR / "source_groups_candidate.csv")
    if list(data.columns) != TRACEABILITY_COLUMNS + FINAL_FEATURE_COLUMNS:
        raise ValueError("Unexpected schema")
    if len(data) != summary["operations"][-1]["after"] or len(provenance) != summary["provenance_rows"]:
        raise ValueError("Row counts differ from transformation log")
    if not data.Text_ID.is_unique or not data.Raw_Text.is_unique or not provenance.Observation_ID.is_unique:
        raise ValueError("Duplicate canonical/provenance identities")
    if not data.Text_ID.eq("m4-" + data.Raw_Text.map(digest)).all():
        raise ValueError("Content IDs no longer match original text")
    if set(data.Label) != {0, 1} or not data.Label_Name.eq(data.Label.map({0: "Human", 1: "AI"})).all():
        raise ValueError("Invalid target labels")
    if not np.isfinite(data[FINAL_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite features")
    if set(provenance.Text_ID) != set(data.Text_ID) or provenance.Missing_Or_Empty.any():
        raise ValueError("Incomplete provenance or retained empty entries")
    if not provenance.Text_ID.eq("m4-" + provenance.Exact_Text_SHA256).all():
        raise ValueError("Provenance hash differs from canonical ID")
    labels = data.set_index("Text_ID").Label
    if not provenance.Label.eq(provenance.Text_ID.map(labels)).all():
        raise ValueError("Provenance label mismatch")
    if not groups.Text_ID.is_unique or set(groups.Text_ID) != set(data.Text_ID):
        raise ValueError("Source groups do not cover every canonical text")
    origins = provenance.merge(groups, on="Text_ID", validate="many_to_one")
    for key in ["Source_Record_ID", "Source_Group_Candidate", "Normalized_Text_SHA256"]:
        valid = origins.loc[origins[key].ne("")]
        if (valid.groupby(key).Source_Group_ID.nunique() > 1).any():
            raise ValueError(f"Related origins were separated: {key}")
    # Deterministic coverage across the entire output, using the scalar extractor
    # to cross-check the vectorized build without selecting by target/performance.
    positions = np.linspace(0, len(data) - 1, min(100, len(data)), dtype=int)
    for index in positions:
        row = data.iloc[index]
        rebuilt = extract_text_features(row.Raw_Text)
        np.testing.assert_allclose([rebuilt[key] for key in FINAL_FEATURE_COLUMNS],
                                   row[FINAL_FEATURE_COLUMNS].to_numpy(dtype=float), rtol=1e-12, atol=1e-12)
    result = {"status": "passed", "rows": len(data), "provenance_rows": len(provenance),
              "class_counts": data.Label_Name.value_counts().to_dict(), "source_groups": groups.Source_Group_ID.nunique(),
              "scalar_feature_recomputation_rows": len(positions),
              "checks": ["output checksums", "exact schema and counts", "unique IDs/text", "content-derived IDs",
                         "binary target mapping", "finite features", "complete provenance and label mapping",
                         "all known source and normalized-equivalent relationships preserved", "scalar/vectorized feature agreement"],
              "scope": "No claim of semantic completeness or validation of an unreviewed similarity threshold"}
    (REPORT_DIR / "validation_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
