"""Exploratory M4 inventory and feature plots; no final preprocessing or splits."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_m4 import RAW_DIR, ROOT, sha256
from prepare_final_data import FINAL_FEATURE_COLUMNS
from text_features import extract_dataframe_features, normalize_text, word_tokens


DOMAINS = {"arxiv", "peerread", "reddit", "wikihow", "wikipedia"}
REPORT_DIR = ROOT / "reports" / "m4"
CANDIDATE_DIR = ROOT / "data" / "processed" / "m4"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text_entries(value):
    """Expose nested text entries without joining reviews or stringifying nulls."""
    if isinstance(value, list):
        if not value:
            yield "", None
        else:
            for index, item in enumerate(value):
                for subindex, text in text_entries(item):
                    yield f"{index}/{subindex}".rstrip("/"), text
    elif value is None or isinstance(value, str):
        yield "", value
    else:
        raise ValueError(f"Unexpected text type: {type(value).__name__}")


def flatten_record(record, filename, line_number):
    """Each text entry is an exploratory observation, retaining its source record."""
    domain = Path(filename).name.split("_", 1)[0].lower()
    source = record.get("source")
    source_id = next((record[key] for key in ["source_ID", "source_id", "question_id", "q_id", "id"]
                      if record.get(key) is not None), None)
    has_source_id = source_id is not None and str(source_id).strip() != ""
    canonical_id = str(int(source_id)) if isinstance(source_id, float) and source_id.is_integer() else str(source_id)
    # Source names are inconsistent across generators; only PeerRead needs a venue scope.
    scope = source if domain == "peerread" else domain
    group_key = json.dumps([domain, scope, canonical_id], sort_keys=True, ensure_ascii=True)
    group = digest(group_key) if has_source_id else None
    model = str(record.get("model") or Path(filename).stem.split("_", 1)[1])
    # Explicit exploratory adapters: BLOOMZ machine_text often contains the prompt.
    bloom_fields = {
        "arxiv": ("abstract", "machine_abstract"),
        "peerread": ("human_reviews", "bloom_reviews"),
        "reddit": ("text", "machine_answer"),
        "wikihow": ("text", "machine_abstract"),
        "wikipedia": ("text", "machine_abstract"),
    }
    human_field, machine_field = bloom_fields[domain] if "bloomz" in filename.lower() else ("human_text", "machine_text")
    raw_machine = record.get("machine_text")
    prompt = record.get("prompt")
    raw_prompt_prefix = isinstance(prompt, str) and bool(prompt.strip()) and isinstance(raw_machine, str) and raw_machine.startswith(prompt)
    for field, label, generator in [(human_field, 0, "human"), (machine_field, 1, model)]:
        if field not in record:
            raise ValueError(f"{filename}:{line_number} has no {field}")
        for entry, text in text_entries(record[field]):
            normalized = normalize_text(text).casefold() if text is not None else ""
            valid = bool(normalized)
            yield {
                "Text_ID": digest(f"{filename}:{line_number}:{field}:{entry}"),
                "Raw_Text": text, "Label": label, "Label_Name": "AI" if label else "Human",
                "Domain": domain, "Generator": generator, "Source": json.dumps(source),
                "Source_ID": json.dumps(source_id), "Source_Group_Candidate": group,
                "Raw_File": filename, "Raw_Line": line_number, "Entry_Index": entry,
                "Text_Field": field, "Raw_Machine_Text_Starts_With_Prompt": raw_prompt_prefix,
                "Missing_Or_Empty": not valid,
                "Exact_Text_SHA256": digest(text) if valid else None,
                "Normalized_Text_SHA256": digest(normalized) if valid else None,
                "Audit_Word_Count": len(word_tokens(text)) if valid else 0,
                "AI_Disclaimer_Flag": bool(re.search(r"as an? (?:ai|language model)", normalized)),
                "Sample_Order": digest("478:" + f"{filename}:{line_number}:{field}:{entry}"),
            }


def audit(raw_dir, per_stratum):
    paths = sorted(path for path in raw_dir.rglob("*.jsonl")
                   if path.name.split("_", 1)[0].lower() in DOMAINS)
    if not paths:
        raise FileNotFoundError(f"No English M4 JSONL files under {raw_dir}. Run download_m4.py first.")
    rows, files = [], []
    for path in paths:
        before = len(rows)
        shapes = Counter()
        count = 0
        blank_lines = 0
        relative = path.relative_to(raw_dir).as_posix()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    blank_lines += 1
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {relative}:{line_number}") from error
                count += 1
                for field in ["human_text", "machine_text", "prompt", "abstract", "text", "human_reviews", "bloom_reviews", "machine_abstract", "machine_answer"]:
                    shapes[f"{field}:{type(record.get(field)).__name__}"] += 1
                rows.extend(flatten_record(record, relative, line_number))
        files.append({"file": relative, "sha256": sha256(path), "records": count,
                      "blank_lines": blank_lines, "expanded_entries": len(rows) - before, "field_shapes": dict(shapes)})
        print(f"Audited {relative}: {count:,} records -> {len(rows) - before:,} text entries", flush=True)
    data = pd.DataFrame(rows)
    if data.Text_ID.duplicated().any():
        raise ValueError("Exploratory row IDs must be unique")
    coverage = data.groupby(["Domain", "Label_Name", "Generator"]).size().rename("Entries").reset_index()
    duplicates = []
    for column in ["Exact_Text_SHA256", "Normalized_Text_SHA256"]:
        nonempty = data.loc[~data.Missing_Or_Empty]
        groups = nonempty.groupby(column).agg(Rows=("Label", "size"), Classes=("Label", "nunique"))
        duplicates.append({"key": column, "nonempty_entries": len(nonempty),
                           "unique_texts": len(groups), "excess_entries": len(nonempty) - len(groups),
                           "repeated_groups": int((groups.Rows > 1).sum()),
                           "conflicting_label_groups": int((groups.Classes > 1).sum())})
    eligible = data.loc[~data.Missing_Or_Empty].sort_values("Sample_Order")
    sample = eligible.groupby(["Domain", "Label"], sort=True).head(per_stratum).copy()
    print(f"Extracting exploratory features for {len(sample):,} entries...", flush=True)
    features = extract_dataframe_features(sample)
    candidate = sample.join(features[FINAL_FEATURE_COLUMNS])
    summary = {
        "status": "EXPLORATORY: not a final feature matrix or evaluation split",
        "dataset": "serjhenrique/m4-multi-generator-domain", "version": 1,
        "source_records": sum(item["records"] for item in files), "expanded_text_entries": len(data),
        "expansion_note": "Lists expanded independently; human entries may repeat across generators. No deduplication applied.",
        "domains": sorted(data.Domain.unique()), "class_counts": data.Label_Name.value_counts().to_dict(),
        "missing_or_empty_entries": int(data.Missing_Or_Empty.sum()),
        "entries_without_source_id": int(data.Source_Group_Candidate.isna().sum()),
        "raw_machine_prompt_prefix_records": int(data.loc[data.Raw_Machine_Text_Starts_With_Prompt, ["Raw_File", "Raw_Line"]].drop_duplicates().shape[0]),
        "text_field_counts": data.Text_Field.value_counts().to_dict(),
        "candidate_source_groups": int(data.Source_Group_Candidate.nunique()),
        "duplicate_audit": duplicates, "files": files,
        "sample_policy": "Deterministic SHA256 ordering (seed 478), at most N nonempty entries per domain/class; audit only, repeats retained.",
        "per_domain_class_limit": per_stratum,
        "row_operations": [
            {"operation": "Expand English-file text fields (not row deletion)", "before": sum(x["records"] for x in files), "after": len(data)},
            {"operation": "Nonempty eligibility for exploratory feature sample ONLY", "before": len(data), "after": len(eligible)},
            {"operation": "Deterministic exploratory sampling ONLY", "before": len(eligible), "after": len(sample)},
        ],
        "sample_feature_missing_counts": candidate[FINAL_FEATURE_COLUMNS].isna().sum().to_dict(),
        "pending": ["Confirm prediction unit for list-valued reviews", "Review label mapping and repeated human passages",
                    "Approve sampling, exclusions, schema and output filenames", "Audit prompt remnants and near-duplicates",
                    "Validate source-ID consistency across generators", "Check overlap with original benchmark",
                    "Approve group-aware splits before evaluation"],
    }
    return data, coverage, candidate, summary


def write_outputs(data, coverage, candidate, summary):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(REPORT_DIR / "coverage.csv", index=False)
    # Full inventory contains hashes and provenance, rather than duplicating the raw text.
    data.drop(columns=["Raw_Text", "Sample_Order"]).to_csv(CANDIDATE_DIR / "inventory_exploratory.csv", index=False)
    candidate.drop(columns="Sample_Order").to_csv(CANDIDATE_DIR / "feature_sample_exploratory.csv", index=False)
    (REPORT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 5))
    data.groupby(["Domain", "Label_Name"]).size().unstack(fill_value=0).plot.bar(ax=ax)
    ax.set(title="Exploratory M4 coverage (repeated text entries retained)", ylabel="Text entries", xlabel="Domain")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "domain_class_coverage.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for feature, ax in zip(FINAL_FEATURE_COLUMNS, axes.flat):
        bins = np.histogram_bin_edges(candidate[feature].dropna().astype(float), bins=35)
        for label, part in candidate.groupby("Label_Name"):
            ax.hist(part[feature].dropna().astype(float), bins=bins, alpha=0.5, density=True, label=label)
        ax.set_title(feature.removeprefix("Text_").replace("_", " "))
        ax.legend()
    for ax in list(axes.flat)[len(FINAL_FEATURE_COLUMNS):]:
        ax.set_visible(False)
    fig.suptitle("Exploratory M4 feature sample: repeats retained; not an evaluation set")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "feature_distributions_exploratory.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 7))
    correlation = candidate[FINAL_FEATURE_COLUMNS].astype(float).corr()
    plot = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [column.removeprefix("Text_") for column in FINAL_FEATURE_COLUMNS]
    ax.set_xticks(range(7), labels, rotation=70, ha="right")
    ax.set_yticks(range(7), labels)
    for row in range(7):
        for col in range(7):
            ax.text(col, row, f"{correlation.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Exploratory M4 sample: feature correlations")
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "feature_correlation_exploratory.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    missing = data.groupby("Label_Name").Missing_Or_Empty.sum()
    missing.plot.bar(ax=axes[0], color=["#4C72B0", "#DD8452"])
    axes[0].set(title="Null/empty entries (including empty-list placeholders)", ylabel="Entries", xlabel="Class")
    audit = summary["duplicate_audit"][0]
    axes[1].bar(["Unique nonempty texts", "Excess repeated entries"],
                [audit["unique_texts"], audit["excess_entries"]], color=["#55A868", "#C44E52"])
    axes[1].set(title="Exact-text inventory: no deduplication applied", ylabel="Entries")
    fig.suptitle("Exploratory M4 data-quality audit")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "missingness_and_duplicates_exploratory.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--per-domain-class", type=int, default=500)
    args = parser.parse_args()
    if args.per_domain_class < 1:
        parser.error("--per-domain-class must be positive")
    data, coverage, candidate, summary = audit(args.raw_dir, args.per_domain_class)
    write_outputs(data, coverage, candidate, summary)
    print(json.dumps({key: summary[key] for key in ["source_records", "expanded_text_entries", "class_counts", "duplicate_audit"]}, indent=2))
    print(f"Exploratory reports: {REPORT_DIR}; review decisions before producing final data.")


if __name__ == "__main__":
    main()
