"""Create approved M4 partitions using an explicitly reviewed similarity threshold."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from audit_m4 import REPORT_DIR
from audit_m4_similarity import EDGE_PATH, FLOOR, combined_components
from download_m4 import sha256
from prepare_m4 import OUTPUT_DIR, PROCESSED_PATH


def validate_splits(splits):
    if splits.Text_ID.duplicated().any() or splits.isna().any().any():
        raise ValueError("Split IDs must be unique and all assignments complete")
    if set(splits.Split) != {"train", "test"}:
        raise ValueError("Both train and test must exist")
    if (splits.groupby("Group_ID").Split.nunique() > 1).any():
        raise ValueError("A related-text group crosses the train/test boundary")
    train = splits.loc[splits.Split == "train"]
    if set(train.CV_Fold) != set(range(5)):
        raise ValueError("Expected five complete training CV folds")
    if not splits.loc[splits.Split == "test", "CV_Fold"].eq(-1).all():
        raise ValueError("Test rows cannot belong to training CV folds")
    if (train.groupby("Group_ID").CV_Fold.nunique() > 1).any():
        raise ValueError("A related-text group crosses CV folds")
    for part in [splits.loc[splits.Split == "test"]] + [train.loc[train.CV_Fold == i] for i in range(5)]:
        if set(part.Label) != {0, 1}:
            raise ValueError("An evaluation partition is missing a class")


def assign_splits(data, group_ids):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=478)
    train, test = next(outer.split(np.zeros(len(data)), data.Label, group_ids))
    result = data[["Text_ID", "Label"]].copy()
    result["Group_ID"] = group_ids
    result["Split"] = "train"
    result.loc[test, "Split"] = "test"
    result["CV_Fold"] = -1
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=478)
    for fold, (_, valid) in enumerate(inner.split(np.zeros(len(train)), data.Label.iloc[train], np.asarray(group_ids)[train])):
        result.loc[train[valid], "CV_Fold"] = fold
    validate_splits(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, required=True, help="Owner-reviewed exact trigram cosine threshold")
    args = parser.parse_args()
    if not FLOOR <= args.threshold <= 1:
        parser.error(f"Threshold must be in [{FLOOR}, 1], covered by the similarity cache")
    data = pd.read_csv(PROCESSED_PATH, usecols=["Text_ID", "Label"])
    source = pd.read_csv(OUTPUT_DIR / "source_groups_candidate.csv")
    canonical = OUTPUT_DIR / "canonical_texts_intermediate.csv"
    meta = json.loads(EDGE_PATH.with_suffix(".json").read_text())
    if meta["text_sha256"] != sha256(canonical):
        raise ValueError("Similarity cache belongs to different canonical data")
    if data.Text_ID.tolist() != source.Text_ID.tolist():
        raise ValueError("Source assignments do not align with processed data")
    with np.load(EDGE_PATH) as saved:
        rows, cols, values = [saved[key] for key in ["rows", "cols", "values"]]
    components = combined_components(source.Source_Group_ID.to_numpy(), rows, cols, values, args.threshold)
    ids = data.Text_ID.groupby(components).transform("min").str.replace("m4-", "m4-group-", regex=False)
    splits = assign_splits(data, ids.to_numpy())
    # Explicitly validate every cached edge used for grouping, in addition to components.
    selected = values >= args.threshold
    if not np.array_equal(splits.Group_ID.to_numpy()[rows[selected]], splits.Group_ID.to_numpy()[cols[selected]]):
        raise ValueError("A reviewed near-duplicate edge was not grouped")
    path = OUTPUT_DIR / "m4_splits.csv"
    splits.to_csv(path, index=False)
    records = []
    for (part, fold), frame in splits.groupby(["Split", "CV_Fold"]):
        records.append({"split": part, "cv_fold": int(fold), "rows": len(frame), "groups": frame.Group_ID.nunique(),
                        "Human": int(frame.Label.eq(0).sum()), "AI": int(frame.Label.eq(1).sum()), "AI_proportion": float(frame.Label.mean())})
    summary = {"random_state": 478, "threshold": args.threshold, "rows_before": len(data), "rows_after": len(splits),
               "groups": splits.Group_ID.nunique(), "largest_group": int(splits.Group_ID.value_counts().max()),
               "test_proportion": float(splits.Split.eq("test").mean()), "partitions": records,
               "validation": "No source or selected near-duplicate group crosses train/test or CV folds; both classes in every evaluation fold",
               "processed_sha256": sha256(PROCESSED_PATH), "splits_sha256": sha256(path)}
    (REPORT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
