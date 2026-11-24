"""Apply owner-approved M4 empty-entry and exact-deduplication policy.

Source/normalized-equivalent groups are provisional until the near-duplicate
audit completes. Feature formulas remain unchanged. No model fitting occurs.
"""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from audit_m4 import DOMAINS, REPORT_DIR, flatten_record
from download_m4 import RAW_DIR, ROOT, sha256
from prepare_final_data import FINAL_FEATURE_COLUMNS, TRACEABILITY_COLUMNS
from text_features import extract_dataframe_features


OUTPUT_DIR = ROOT / "data" / "processed" / "m4"
PROCESSED_PATH = OUTPUT_DIR / "m4_processed.csv"
PROVENANCE_PATH = OUTPUT_DIR / "m4_provenance.csv"
GROUP_PATH = OUTPUT_DIR / "source_groups_candidate.csv"


def source_groups(provenance, canonical):
    """Conservatively connect source records, source IDs, and normalized text.

    Each canonical text may have many origins. Connecting every origin before
    splitting prevents deduplication from discarding a source relationship.
    """
    index = pd.Series(np.arange(len(canonical)), index=canonical.Text_ID)
    edges_a, edges_b = [], []
    for key in ["Source_Record_ID", "Source_Group_Candidate", "Normalized_Text_SHA256"]:
        valid = provenance.loc[provenance[key].notna() & provenance[key].ne("")]
        representatives = valid.groupby(key, sort=False).Text_ID.transform("first")
        edges_a.extend(valid.Text_ID.map(index).tolist())
        edges_b.extend(representatives.map(index).tolist())
    graph = coo_matrix((np.ones(len(edges_a), dtype=bool), (edges_a, edges_b)),
                       shape=(len(canonical), len(canonical))).tocsr()
    _, labels = connected_components(graph, directed=False)
    groups = pd.DataFrame({"Text_ID": canonical.Text_ID.to_numpy(), "component": labels})
    groups["Source_Group_ID"] = "m4-source-" + groups.groupby("component").Text_ID.transform("min").str.removeprefix("m4-")
    return groups[["Text_ID", "Source_Group_ID"]]


def canonicalize(observations):
    if observations.Text_ID.duplicated().any():
        raise ValueError("Duplicate observation IDs")
    retained = observations.loc[~observations.Missing_Or_Empty].copy()
    if (retained.groupby("Exact_Text_SHA256").Label.nunique() > 1).any():
        raise ValueError("An exact text has conflicting labels; owner review required")
    retained = retained.rename(columns={"Text_ID": "Observation_ID"})
    retained["Text_ID"] = "m4-" + retained.Exact_Text_SHA256
    retained["Source_Record_ID"] = retained.Raw_File + ":" + retained.Raw_Line.astype(str)
    canonical = retained.drop_duplicates("Text_ID", keep="first")[TRACEABILITY_COLUMNS].copy().reset_index(drop=True)
    provenance = retained.drop(columns=["Raw_Text", "Sample_Order"])
    return canonical, provenance, source_groups(provenance, canonical)


def main():
    paths = sorted(path for path in RAW_DIR.rglob("*.jsonl") if path.name.split("_", 1)[0].lower() in DOMAINS)
    if not paths:
        raise FileNotFoundError("Run src/download_m4.py first")
    manifest = json.loads((RAW_DIR / "download_manifest.json").read_text(encoding="utf-8"))
    expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    observations, file_counts = [], []
    for path in paths:
        name = path.relative_to(RAW_DIR).as_posix()
        if expected.get(name) != sha256(path):
            raise ValueError(f"Raw file differs from download manifest: {name}")
        records, before = 0, len(observations)
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    observations.extend(flatten_record(json.loads(line), name, line_number))
                    records += 1
        file_counts.append({"file": name, "records": records, "entries": len(observations) - before})
        print(f"Loaded {name}", flush=True)
    observations = pd.DataFrame(observations)
    canonical, provenance, groups = canonicalize(observations)
    initial, nonempty = len(observations), len(provenance)
    del observations
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance.sort_values("Observation_ID").to_csv(PROVENANCE_PATH, index=False)
    groups.to_csv(GROUP_PATH, index=False)
    canonical.to_csv(OUTPUT_DIR / "canonical_texts_intermediate.csv", index=False)
    print(f"{initial:,} entries -> {nonempty:,} nonempty -> {len(canonical):,} exact unique texts", flush=True)
    started = time.perf_counter()
    chunks = []
    for start in range(0, len(canonical), 1000):
        chunk = canonical.iloc[start:start + 1000]
        chunks.append(chunk.join(extract_dataframe_features(chunk)[FINAL_FEATURE_COLUMNS]))
        print(f"Features {min(start + 1000, len(canonical)):,}/{len(canonical):,}; {time.perf_counter() - started:.0f}s", flush=True)
    processed = pd.concat(chunks, ignore_index=True)
    if list(processed.columns) != TRACEABILITY_COLUMNS + FINAL_FEATURE_COLUMNS:
        raise ValueError("Unexpected final M4 schema")
    if not np.isfinite(processed[FINAL_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite features; owner review needed before any imputation/exclusion")
    if processed.Text_ID.duplicated().any() or processed.Raw_Text.duplicated().any():
        raise ValueError("Canonical IDs and exact texts must be unique")
    if set(processed.Label) != {0, 1}:
        raise ValueError("Both target classes must be present")
    temporary = PROCESSED_PATH.with_suffix(".csv.part")
    processed.to_csv(temporary, index=False)
    temporary.replace(PROCESSED_PATH)
    counts = groups.Source_Group_ID.value_counts()
    summary = {
        "status": "Approved preprocessing artifact; downstream grouping/split/evaluation are recorded separately",
        "policy": "All nonempty exact-unique texts; no length cuts/downsampling; original seven features",
        "files": file_counts, "raw_manifest_sha256": sha256(RAW_DIR / "download_manifest.json"),
        "operations": [
            {"operation": "Expand passages/reviews", "before": sum(x["records"] for x in file_counts), "after": initial},
            {"operation": "Exclude empty/null placeholders", "before": initial, "after": nonempty},
            {"operation": "Consolidate exact duplicates, preserving every retained origin", "before": nonempty, "after": len(processed)},
        ],
        "class_counts": processed.Label_Name.value_counts().to_dict(),
        "provenance_rows": len(provenance), "source_groups": len(counts), "largest_source_group": int(counts.max()),
        "columns": list(processed.columns), "features": FINAL_FEATURE_COLUMNS,
        "output_sha256": {path.name: sha256(path) for path in [PROCESSED_PATH, PROVENANCE_PATH, GROUP_PATH]},
        "validation": "Unique IDs/texts, both labels, exact schema, finite features; all retained origins mapped",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
