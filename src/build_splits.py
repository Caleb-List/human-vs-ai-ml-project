"""Build near-duplicate groups, template families, and group-aware splits.

This script does not change the approved feature schema. It reads the reduced
processed CSV only for `Text_ID`, `Raw_Text`, and `Label`, and writes a
separate split-assignment file keyed by `Text_ID`:

    data/processed/human_ai_text_splits.csv

Similarity definition
---------------------
Each text is represented by its set of lowercased word trigrams (shingles).
Two texts are compared with the cosine similarity of their binary trigram
indicator vectors:

    sim(a, b) = |T(a) & T(b)| / sqrt(|T(a)| * |T(b)|)

where `T(x)` is the set of word trigrams in `x`. This is an Ochiai/cosine
overlap coefficient: 1.0 means identical trigram sets and 0.0 means no shared
three-word phrase.

Two grouping levels are derived from the same similarity graph:

* Template families (`Template_Family_ID`): connected components of the graph
  whose edges have `sim >= FAMILY_THRESHOLD`. Singleton components are attached
  to the family of their single most similar text. These are the benchmark's
  generator templates, not near-duplicates.
* Near-duplicate groups (`NearDup_Group_ID`): connected components of the graph
  whose edges have `sim >= NEAR_DUP_THRESHOLD`. Every text appears in exactly
  one group, and a group never crosses the train/test boundary or a CV fold.

The thresholds are justified from the data; see
`docs/data_preprocessing_and_exploration.md` and the diagnostics
printed and plotted by this script.
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

from data_loader import PROJECT_ROOT


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "human_ai_text_processed.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "human_ai_text_splits.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
SUMMARY_PATH = METRICS_DIR / "split_summary.json"

RANDOM_STATE = 478
CHUNK_SIZE = 2_000
FAMILY_THRESHOLD = 0.60
NEAR_DUP_THRESHOLD = 0.82
DIAGNOSTIC_EDGE_FLOOR = 0.75
THRESHOLD_GRID = np.round(np.arange(0.75, 0.91, 0.01), 2)
TEST_FOLDS = 5  # one of five stratified group folds is held out -> 20% test
CV_FOLDS = 5
HIST_BINS = 101  # similarity histogram bins of width 0.01 over [0, 1]


def trigram_matrix(texts: pd.Series) -> sp.csr_matrix:
    """Return L2-normalized binary word-trigram vectors (one row per text)."""
    vectorizer = CountVectorizer(
        analyzer="word",
        ngram_range=(3, 3),
        lowercase=True,
        binary=True,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    counts = vectorizer.fit_transform(texts)
    return normalize(counts, norm="l2").tocsr()


def similarity_pass(X: sp.csr_matrix, labels: np.ndarray) -> dict:
    """Stream the all-pairs similarity matrix in row chunks.

    Collects nearest-neighbor similarity, edges for diagnostics, family
    components (merged incrementally to keep memory small), and pair-similarity
    histograms split by whether the two texts share a label.
    """
    n = X.shape[0]
    XT = X.T.tocsc()
    nn_sim = np.zeros(n, dtype=np.float32)
    nn_idx = np.zeros(n, dtype=np.int64)
    family = np.arange(n)
    edge_rows, edge_cols, edge_vals = [], [], []
    same_label_hist = np.zeros(HIST_BINS, dtype=np.int64)
    cross_label_hist = np.zeros(HIST_BINS, dtype=np.int64)

    started = time.perf_counter()
    for start in range(0, n, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, n)
        sims = (X[start:stop] @ XT).toarray()
        local = np.arange(stop - start)
        sims[local, start + local] = -1.0  # ignore self-similarity

        nn_idx[start:stop] = sims.argmax(axis=1)
        nn_sim[start:stop] = sims[local, nn_idx[start:stop]]

        # Upper triangle only, so each unordered pair is counted once.
        upper = np.arange(n)[None, :] > (start + local)[:, None]
        bins = np.clip((sims * 100).astype(np.int64), 0, HIST_BINS - 1)
        same = labels[start:stop, None] == labels[None, :]
        same_label_hist += np.bincount(bins[upper & same], minlength=HIST_BINS)
        cross_label_hist += np.bincount(bins[upper & ~same], minlength=HIST_BINS)

        rows, cols = np.nonzero(upper & (sims >= DIAGNOSTIC_EDGE_FLOOR))
        edge_rows.append(rows + start)
        edge_cols.append(cols)
        edge_vals.append(sims[rows, cols])

        rows, cols = np.nonzero(upper & (sims >= FAMILY_THRESHOLD))
        family = merge_components(family, rows + start, cols)

        print(
            f"Similarity rows {stop:,}/{n:,}; elapsed {time.perf_counter() - started:.0f}s",
            flush=True,
        )

    return {
        "nn_sim": nn_sim,
        "nn_idx": nn_idx,
        "family_raw": family,
        "edge_rows": np.concatenate(edge_rows),
        "edge_cols": np.concatenate(edge_cols),
        "edge_vals": np.concatenate(edge_vals),
        "same_label_hist": same_label_hist,
        "cross_label_hist": cross_label_hist,
    }


def merge_components(labels: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Merge new edges into an existing component labelling.

    Existing components are preserved by linking every node to its component's
    representative node (the first node carrying that label).
    """
    n = len(labels)
    nodes = np.arange(n)
    _, inverse = np.unique(labels, return_inverse=True)
    _, first_index = np.unique(inverse, return_index=True)
    representative = first_index[inverse]
    graph = sp.coo_matrix(
        (
            np.ones(len(rows) + n, dtype=np.int8),
            (np.concatenate([rows, nodes]), np.concatenate([cols, representative])),
        ),
        shape=(n, n),
    )
    _, merged = connected_components(graph, directed=False)
    return merged


def components_at(threshold: float, rows, cols, vals, n: int) -> np.ndarray:
    keep = vals >= threshold
    graph = sp.coo_matrix(
        (np.ones(keep.sum(), dtype=np.int8), (rows[keep], cols[keep])), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    return labels


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber groups 0..k-1 from largest to smallest for readable IDs."""
    sizes = np.bincount(labels)
    order = np.lexsort((np.arange(len(sizes)), -sizes))
    mapping = np.empty_like(order)
    mapping[order] = np.arange(len(order))
    return mapping[labels]


def attach_singleton_families(family: np.ndarray, nn_idx: np.ndarray) -> tuple[np.ndarray, int]:
    sizes = np.bincount(family)
    singleton = sizes[family] == 1
    attached = family.copy()
    attached[singleton] = family[nn_idx[singleton]]
    return attached, int(singleton.sum())


def within_family_quantiles(X: sp.csr_matrix, family: np.ndarray, quantiles) -> dict:
    """Exact pair-similarity quantiles over all within-family text pairs."""
    hist = np.zeros(10_001, dtype=np.int64)  # 0.0001 resolution
    for family_id in np.unique(family):
        members = np.flatnonzero(family == family_id)
        Xf = X[members]
        XfT = Xf.T.tocsc()
        for start in range(0, len(members), CHUNK_SIZE):
            sims = (Xf[start : start + CHUNK_SIZE] @ XfT).toarray()
            local = np.arange(sims.shape[0])
            upper = np.arange(len(members))[None, :] > (start + local)[:, None]
            bins = np.clip((sims[upper] * 10_000).astype(np.int64), 0, 10_000)
            hist += np.bincount(bins, minlength=10_001)
    cumulative = np.cumsum(hist) / hist.sum()
    return {str(q): float(np.searchsorted(cumulative, q) / 10_000) for q in quantiles}


def assign_splits(labels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hold out one stratified group fold as test; build CV folds on the rest."""
    outer = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(outer.split(np.zeros(len(labels)), labels, groups))
    split = np.full(len(labels), "train", dtype=object)
    split[test_idx] = "test"

    cv_fold = np.full(len(labels), -1, dtype=np.int64)
    inner = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, val_idx) in enumerate(
        inner.split(np.zeros(len(train_idx)), labels[train_idx], groups[train_idx])
    ):
        cv_fold[train_idx[val_idx]] = fold
    return split, cv_fold


def validate_splits(splits: pd.DataFrame) -> None:
    if splits["Text_ID"].duplicated().any():
        raise ValueError("Split file has duplicate Text_ID values.")
    for column in ["NearDup_Group_ID", "Template_Family_ID"]:
        if splits[column].isna().any():
            raise ValueError(f"{column} has missing values.")
    crossing = splits.groupby("NearDup_Group_ID")["Split"].nunique()
    if (crossing > 1).any():
        raise ValueError("A near-duplicate group crosses the train/test boundary.")
    train = splits[splits["Split"] == "train"]
    if (train["CV_Fold"] < 0).any() or (splits.loc[splits["Split"] == "test", "CV_Fold"] != -1).any():
        raise ValueError("CV folds must cover all train rows and no test rows.")
    if (train.groupby("NearDup_Group_ID")["CV_Fold"].nunique() > 1).any():
        raise ValueError("A near-duplicate group crosses CV folds.")


def class_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    table = pd.crosstab(frame[by], frame["Label_Name"])[["Human", "AI"]]
    table["Total"] = table.sum(axis=1)
    table["AI_Proportion"] = (table["AI"] / table["Total"]).round(4)
    return table


def plot_diagnostics(results: dict, grid: pd.DataFrame, families: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    centers = np.arange(HIST_BINS) / 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for key, name in [("same_label_hist", "Same-label pairs"), ("cross_label_hist", "Human-AI pairs")]:
        hist = results[key].astype(float)
        ax.plot(centers, np.where(hist > 0, hist, np.nan), label=name)
    ax.set_yscale("log")
    ax.axvline(FAMILY_THRESHOLD, color="gray", linestyle="--", label=f"Family threshold {FAMILY_THRESHOLD}")
    ax.axvline(NEAR_DUP_THRESHOLD, color="black", linestyle=":", label=f"Near-dup threshold {NEAR_DUP_THRESHOLD}")
    ax.set(
        title="All-pairs word-trigram similarity",
        xlabel="Cosine similarity of binary word-trigram sets",
        ylabel="Number of text pairs (log scale)",
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(grid["Threshold"], grid["Largest_Group"], marker="o")
    ax.axvline(NEAR_DUP_THRESHOLD, color="black", linestyle=":")
    ax.set_yscale("log")
    ax.set(
        title="Single-linkage chaining vs. threshold",
        xlabel="Similarity threshold",
        ylabel="Largest near-duplicate group (rows, log scale)",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "near_duplicate_similarity_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    families = families.sort_index()
    colors = ["#4C72B0" if ai == 0 else "#DD8452" for ai in families["AI"]]
    ax.bar(families.index.astype(str), families["Total"], color=colors)
    ax.set(
        title="Template families: every family is single-class (blue = Human, orange = AI)",
        xlabel="Template_Family_ID",
        ylabel="Rows",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "template_families_by_class.png", dpi=180)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_PATH, usecols=["Text_ID", "Raw_Text", "Label", "Label_Name"])
    labels = df["Label"].to_numpy()
    n = len(df)
    print(f"Loaded {n:,} rows from {INPUT_PATH}")

    X = trigram_matrix(df["Raw_Text"])
    print(f"Word-trigram vocabulary: {X.shape[1]:,}")
    results = similarity_pass(X, labels)
    rows, cols, vals = results["edge_rows"], results["edge_cols"], results["edge_vals"]

    family_raw = relabel_by_size(results["family_raw"])
    family, singleton_count = attach_singleton_families(family_raw, results["nn_idx"])
    family = relabel_by_size(family)

    grid_rows = []
    for threshold in THRESHOLD_GRID:
        groups = components_at(threshold, rows, cols, vals, n)
        sizes = np.bincount(groups)
        grid_rows.append(
            {
                "Threshold": float(threshold),
                "Groups": int(len(sizes)),
                "Largest_Group": int(sizes.max()),
                "Rows_In_Multi_Text_Groups": int((sizes[groups] > 1).sum()),
            }
        )
    grid = pd.DataFrame(grid_rows)

    near_dup = relabel_by_size(components_at(NEAR_DUP_THRESHOLD, rows, cols, vals, n))
    within_quantiles = within_family_quantiles(X, family, [0.5, 0.9, 0.99, 0.999, 0.9999])

    split, cv_fold = assign_splits(labels, near_dup)
    splits = pd.DataFrame(
        {
            "Text_ID": df["Text_ID"],
            "Label": labels,
            "Label_Name": df["Label_Name"],
            "NearDup_Group_ID": near_dup,
            "NearDup_Group_Size": np.bincount(near_dup)[near_dup],
            "Template_Family_ID": family,
            "Nearest_Neighbor_Similarity": results["nn_sim"].round(4),
            "Split": split,
            "CV_Fold": cv_fold,
        }
    )
    validate_splits(splits)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(OUTPUT_PATH, index=False)

    families = class_table(splits, "Template_Family_ID")
    mixed_families = int(((families["Human"] > 0) & (families["AI"] > 0)).sum())
    mixed_groups = int((splits.groupby("NearDup_Group_ID")["Label"].nunique() > 1).sum())
    plot_diagnostics(results, grid, families)

    cross_hist = results["cross_label_hist"]
    summary = {
        "rows": n,
        "similarity": "cosine of binary lowercased word-trigram sets",
        "trigram_vocabulary": int(X.shape[1]),
        "max_human_ai_pair_similarity": float(np.flatnonzero(cross_hist).max() / 100),
        "nearest_neighbor_similarity_quantiles": {
            str(q): float(np.quantile(results["nn_sim"], q)) for q in [0, 0.01, 0.5, 0.99, 1]
        },
        "nearest_neighbor_same_label_rate": float(
            (labels == labels[results["nn_idx"]]).mean()
        ),
        "family_threshold": FAMILY_THRESHOLD,
        "family_singletons_attached_to_nearest_neighbor": singleton_count,
        "template_families": int(families.shape[0]),
        "mixed_label_families": mixed_families,
        "within_family_pair_similarity_quantiles": within_quantiles,
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "near_dup_groups": int(splits["NearDup_Group_ID"].nunique()),
        "near_dup_rows_in_multi_text_groups": int((splits["NearDup_Group_Size"] > 1).sum()),
        "near_dup_largest_group": int(splits["NearDup_Group_Size"].max()),
        "near_dup_mixed_label_groups": mixed_groups,
        "threshold_grid": grid_rows,
        "random_state": RANDOM_STATE,
        "split_counts": class_table(splits, "Split").reset_index().to_dict(orient="records"),
        "cv_fold_counts": class_table(splits[splits["Split"] == "train"], "CV_Fold")
        .reset_index()
        .to_dict(orient="records"),
        "family_counts": families.reset_index().to_dict(orient="records"),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print()
    print(f"Max Human-AI pair similarity: {summary['max_human_ai_pair_similarity']:.2f}")
    print(f"Nearest neighbor has the same label for {summary['nearest_neighbor_same_label_rate']:.2%} of texts")
    print(f"Within-family pair similarity quantiles: {within_quantiles}")
    print("\nThreshold grid (single-linkage components):")
    print(grid.to_string(index=False))
    print(f"\nTemplate families (threshold {FAMILY_THRESHOLD}; {singleton_count} singletons attached):")
    print(families.to_string())
    print(
        f"\nNear-duplicate groups at {NEAR_DUP_THRESHOLD}: {summary['near_dup_groups']:,} groups, "
        f"{summary['near_dup_rows_in_multi_text_groups']:,} rows in multi-text groups, "
        f"largest {summary['near_dup_largest_group']}, mixed-label groups {mixed_groups}"
    )
    print("\nTrain/test split (rows by class):")
    print(class_table(splits, "Split").to_string())
    print("\nCV folds within the training partition:")
    print(class_table(splits[splits["Split"] == "train"], "CV_Fold").to_string())
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
