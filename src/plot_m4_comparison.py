"""Reproducible report figures comparing original and approved M4 features."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_m4 import REPORT_DIR
from download_m4 import ROOT
from prepare_final_data import FINAL_FEATURE_COLUMNS
from prepare_m4 import OUTPUT_DIR, PROCESSED_PATH


def main():
    original = pd.read_csv(ROOT / "data" / "processed" / "human_ai_text_processed.csv", usecols=["Label"] + FINAL_FEATURE_COLUMNS)
    m4 = pd.read_csv(PROCESSED_PATH, usecols=["Label"] + FINAL_FEATURE_COLUMNS)
    summary = json.loads((REPORT_DIR / "preprocessing_summary.json").read_text())
    groups = pd.read_csv(OUTPUT_DIR / "source_groups_candidate.csv")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    operations = summary["operations"][1:]
    heights = [operations[0]["before"], operations[0]["after"], operations[1]["after"]]
    bars = axes[0].bar(["Expanded\nentries", "Nonempty\noccurrences", "Exact unique\ntexts"], heights,
                       color=["#999999", "#4C72B0", "#55A868"])
    axes[0].bar_label(bars, labels=[f"{value:,}" for value in heights], padding=3)
    axes[0].set(title="M4: approved row operations", ylabel="Entries/texts", ylim=(0, max(heights) * 1.15))
    counts = pd.DataFrame({name: frame.Label.value_counts().reindex([0, 1], fill_value=0)
                           for name, frame in [("Original", original), ("M4", m4)]}).T
    proportions = counts.div(counts.sum(axis=1), axis=0) * 100
    proportions.columns = ["Human", "AI"]
    proportions.plot.bar(stacked=True, ax=axes[1], color=["#DD8452", "#4C72B0"], rot=0)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%", label_type="center")
    axes[1].set(title="Class balance after preprocessing", ylabel="Percent of texts", ylim=(0, 110))
    axes[1].legend(loc="upper center", ncol=2)
    sizes = groups.Source_Group_ID.value_counts()
    axes[2].hist(sizes, bins=np.arange(1, int(sizes.max()) + 2), color="#55A868")
    axes[2].set_yscale("log")
    axes[2].set(title=f"Known-source groups: {len(sizes):,}", xlabel="Texts per source group", ylabel="Groups (log scale)")
    figure.suptitle("M4 preprocessing checkpoint: all retained texts, before near-duplicate threshold review")
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "preprocessing_checkpoint.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(14, 12))
    colors = {("Original", 0): "#C66A1C", ("Original", 1): "#174F8A",
              ("M4", 0): "#E4A571", ("M4", 1): "#6EAED6"}
    ranges = {}
    for feature, axis in zip(FINAL_FEATURE_COLUMNS, axes.flat):
        combined = np.concatenate([original[feature].to_numpy(), m4[feature].to_numpy()])
        low, high = np.quantile(combined, [.005, .995])
        ranges[feature] = {"plot_x_min": float(low), "plot_x_max": float(high)}
        for name, frame in [("Original", original), ("M4", m4)]:
            for label, part in frame.groupby("Label"):
                values = np.sort(part[feature].to_numpy())
                axis.plot(values, np.arange(1, len(values) + 1) / len(values),
                          linestyle="--" if name == "Original" else "-", color=colors[(name, label)],
                          label=f"{name} {'AI' if label else 'Human'}", linewidth=1.7)
        axis.set(title=feature.removeprefix("Text_").replace("_", " "), ylabel="Fraction at or below value",
                 xlim=(low, high), ylim=(0, 1))
        axis.grid(alpha=.15)
    for axis in list(axes.flat)[len(FINAL_FEATURE_COLUMNS):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5, .015))
    figure.suptitle("Original vs. M4: identical seven feature formulas, all processed texts\n"
                   "ECDF curves use every row; x-axes zoom to pooled 0.5th-99.5th percentiles for readability")
    figure.tight_layout(rect=(0, .06, 1, .95))
    figure.savefig(REPORT_DIR / "original_vs_m4_feature_distributions.png", dpi=170)
    plt.close(figure)
    notes = {"original_rows": len(original), "m4_rows": len(m4),
             "plot": "Empirical cumulative distribution per dataset and class; no downsampling, fitting or row exclusion",
             "display_ranges": ranges,
             "interpretation": "Descriptive domain/style shift, not model performance or proof of dataset validity"}
    (REPORT_DIR / "comparison_figure_notes.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    print("Wrote preprocessing_checkpoint.png and original_vs_m4_feature_distributions.png")


if __name__ == "__main__":
    main()
