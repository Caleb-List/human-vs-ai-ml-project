"""Exact word-trigram near-duplicate diagnostics; no threshold is finalized."""

import argparse
import gc
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from audit_m4 import REPORT_DIR, digest
from download_m4 import ROOT, sha256
from prepare_m4 import OUTPUT_DIR
from text_features import normalize_text


FLOOR = 0.70
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
EDGE_PATH = OUTPUT_DIR / "similarity_edges_exploratory.npz"


def combined_components(source_labels, rows, cols, values, threshold):
    n = len(source_labels)
    _, first, inverse = np.unique(source_labels, return_index=True, return_inverse=True)
    keep = values >= threshold
    a = np.concatenate([np.arange(n), rows[keep]])
    b = np.concatenate([first[inverse], cols[keep]])
    graph = coo_matrix((np.ones(len(a), dtype=bool), (a, b)), shape=(n, n)).tocsr()
    return connected_components(graph, directed=False)[1]


def compute_edges(texts, chunk_size):
    vectorizer = CountVectorizer(ngram_range=(3, 3), binary=True, lowercase=True,
                                 token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    print("Building exact binary word-trigram vectors...", flush=True)
    matrix = normalize(vectorizer.fit_transform(texts), norm="l2").tocsr()
    zero_rows = int((matrix.getnnz(axis=1) == 0).sum())
    # Unique trigrams cannot contribute to cross-text dot products. Remove them
    # AFTER normalization to preserve exact full-vocabulary cosine similarities.
    frequencies = np.bincount(matrix.indices, minlength=matrix.shape[1])
    matrix = matrix[:, frequencies > 1].tocsr()
    del vectorizer, frequencies
    gc.collect()
    transpose = matrix.T.tocsc()
    n = len(texts)
    nearest = np.zeros(n, dtype=np.float32)
    rows, cols, vals = [], [], []
    started = time.perf_counter()
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        # Compare to this block and later texts only; every unordered pair is
        # evaluated once, then updates the nearest-neighbor score of BOTH texts.
        block = (matrix[start:stop] @ transpose[:, start:]).tocoo()
        global_rows = block.row + start
        global_cols = block.col + start
        upper = global_rows < global_cols
        np.maximum.at(nearest, global_rows[upper], block.data[upper])
        np.maximum.at(nearest, global_cols[upper], block.data[upper])
        keep = upper & (block.data >= FLOOR)
        rows.append(global_rows[keep])
        cols.append(global_cols[keep])
        vals.append(block.data[keep])
        if start == 0 or stop == n or stop // 5000 != start // 5000:
            print(f"Similarity rows {stop:,}/{n:,}; {time.perf_counter() - started:.0f}s", flush=True)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), nearest, zero_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    text_path = OUTPUT_DIR / "canonical_texts_intermediate.csv"
    text_hash = sha256(text_path)
    data = pd.read_csv(text_path, keep_default_na=False)
    groups = pd.read_csv(OUTPUT_DIR / "source_groups_candidate.csv")
    if data.Text_ID.tolist() != groups.Text_ID.tolist():
        raise ValueError("Source groups must align exactly with canonical texts")
    cache_meta = EDGE_PATH.with_suffix(".json")
    if EDGE_PATH.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())["text_sha256"] == text_hash:
        with np.load(EDGE_PATH) as saved:
            rows, cols, vals, nearest = [saved[key] for key in ["rows", "cols", "values", "nearest"]]
            zero_rows = int(saved["zero_rows"])
        print("Reusing checksum-matched exact similarity cache", flush=True)
    else:
        rows, cols, vals, nearest, zero_rows = compute_edges(data.Raw_Text, args.chunk_size)
        np.savez_compressed(EDGE_PATH, rows=rows, cols=cols, values=vals, nearest=nearest, zero_rows=zero_rows)
        cache_meta.write_text(json.dumps({"text_sha256": text_hash, "edge_floor": FLOOR}), encoding="utf-8")
    # Float32 accumulation can put identical-vector scores a few millionths
    # above one; clip roundoff to the mathematical cosine range.
    vals = np.clip(vals, 0.0, 1.0)
    nearest = np.clip(nearest, 0.0, 1.0)
    source = groups.Source_Group_ID.to_numpy()
    diagnostics = []
    for threshold in THRESHOLDS:
        components = combined_components(source, rows, cols, vals, threshold)
        sizes = np.bincount(components)
        diagnostics.append({"threshold": threshold, "near_duplicate_edges": int((vals >= threshold).sum()),
                            "combined_groups": len(sizes), "largest_group": int(sizes.max()),
                            "rows_in_multitext_groups": int(sizes[sizes > 1].sum())})
    # Deterministic label-blind review examples spanning similarity bands.
    examples = []
    for low, high in zip(THRESHOLDS, THRESHOLDS[1:] + [1.01]):
        indices = np.flatnonzero((vals >= low) & (vals < high))
        # Review both existing source relationships and edges that would merge
        # different source groups. Selection never uses labels or model scores.
        ordered = []
        for same_source in [False, True]:
            subset = [i for i in indices if bool(source[rows[i]] == source[cols[i]]) == same_source]
            ordered.extend(sorted(subset, key=lambda i: digest(data.Text_ID.iloc[rows[i]] + data.Text_ID.iloc[cols[i]]))[:4])
        for i in ordered:
            a, b = int(rows[i]), int(cols[i])
            examples.append({"Similarity": float(vals[i]), "Band": f"{low}-{high}",
                             "Text_ID_A": data.Text_ID.iloc[a], "Text_ID_B": data.Text_ID.iloc[b],
                             "Same_Source_Group": bool(source[a] == source[b]),
                             "Raw_Text_A": data.Raw_Text.iloc[a], "Raw_Text_B": data.Raw_Text.iloc[b]})
    pd.DataFrame(examples).to_csv(OUTPUT_DIR / "near_duplicate_pairs_review.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(REPORT_DIR / "similarity_threshold_review.csv", index=False)
    old_path = ROOT / "data" / "raw" / "AI_vs_Human_Text_Classification_Benchmark.csv"
    overlap = {"status": "Original benchmark not available locally"}
    if old_path.exists():
        old = pd.read_csv(old_path, usecols=["Raw_Text"])
        old_hashes = set(old.Raw_Text.map(lambda text: digest(normalize_text(text).casefold())))
        overlap = {"original_rows": len(old), "m4_normalized_exact_overlap_rows": int(data.Raw_Text.map(lambda text: digest(normalize_text(text).casefold())).isin(old_hashes).sum()),
                   "scope": "Normalized exact overlap only; no claim of semantic independence"}
    summary = {
        "status": "Similarity diagnostics; partition assignment and model evaluation are separate downstream steps",
        "similarity": "Exact cosine of lowercased binary word-trigram sets; same tokenizer as original split audit",
        "method": "All unordered text pairs evaluated via sparse block products; unique columns removed only after normalization",
        "precision": "Float32 sparse products; cosine roundoff clipped to [0,1] for diagnostics",
        "canonical_texts": len(data), "unordered_pairs": len(data) * (len(data) - 1) // 2,
        "zero_trigram_texts": zero_rows, "retained_edge_floor": FLOOR,
        "nearest_neighbor_quantiles": {str(q): float(np.quantile(nearest, q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]},
        "source_groups_before_near_duplicates": int(groups.Source_Group_ID.nunique()),
        "largest_source_group": int(groups.Source_Group_ID.value_counts().max()),
        "thresholds": diagnostics, "original_benchmark_overlap": overlap,
        "canonical_texts_sha256": text_hash,
        "threshold_selection": "Owner-approved active policy is recorded in AGENTS.md and split_summary.json; this diagnostic tool does not select by model performance",
    }
    (REPORT_DIR / "similarity_review.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(nearest, bins=40)
    axes[0].set(title="Nearest other text: exact trigram cosine", xlabel="Similarity", ylabel="Texts")
    axes[1].plot([x["threshold"] for x in diagnostics], [x["largest_group"] for x in diagnostics], marker="o")
    axes[1].set(title="Source + similarity grouping", xlabel="Proposed threshold", ylabel="Largest group (texts)")
    fig.suptitle("Exploratory M4 near-duplicate threshold review")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "similarity_thresholds_exploratory.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
