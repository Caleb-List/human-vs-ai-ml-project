"""Create exploratory plots and save them under reports/figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_loader import LABEL_NAMES, PROJECT_ROOT, load_dataset


FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SCATTER_SAMPLE_PER_CLASS = 5_000
CLASS_PALETTE = {"Human": "#4c72b0", "AI": "#dd8452"}
TEXT_DERIVED_NUMERIC_COLUMNS = [
    "Word_Count",
    "Character_Count_Total",
    "Sentence_Count",
    "Average_Word_Length",
    "Average_Sentence_Length_Words",
    "Unique_Words_Count",
    "Lexical_Diversity_Score",
    "Punctuation_Count",
    "Punctuation_Density",
    "Special_Char_Count",
    "Stopword_Count",
    "Stopword_Density",
    "Flesch_Kincaid_Grade_Level",
    "Smog_Index",
    "Average_Syllables_Per_Word",
    "Capital_Letter_Count",
    "Emotion_Tone_Intensity",
    "Coherence_Score",
    "Burstiness_Score",
]


def _label_names(df: pd.DataFrame) -> pd.Series:
    return df["Label"].map(LABEL_NAMES).fillna(df["Label"].astype(str))


def _save_scatter_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    filename: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue="Label_Name",
        hue_order=["Human", "AI"],
        palette=CLASS_PALETTE,
        alpha=0.35,
        s=14,
        linewidth=0,
        ax=ax,
    )
    ax.set(title=title, xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
    ax.legend(title="Class", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def main() -> None:
    df = load_dataset()
    plot_data = df.copy()
    plot_data["Label_Name"] = _label_names(plot_data)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=plot_data, x="Label_Name", order=["Human", "AI"], ax=ax)
    ax.set(title="Dataset class balance", xlabel="Class", ylabel="Number of samples")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=160)
    plt.close(fig)

    # 2. Word-count distributions by class
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=plot_data,
        x="Word_Count",
        hue="Label_Name",
        bins=50,
        element="step",
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set(title="Word-count distribution by class", xlabel="Word count", ylabel="Density")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "word_count_by_class.png", dpi=160)
    plt.close(fig)

    # 3. Benchmark detector score by class (descriptive dataset analysis)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=plot_data,
        x="Detection_Score_Benchmark_Model_A",
        hue="Label_Name",
        bins=40,
        element="step",
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set(
        title="Existing benchmark detector score by class",
        xlabel="Detector score",
        ylabel="Density",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "detector_score_by_class.png", dpi=160)
    plt.close(fig)

    # 4. Source-model counts show how the benchmark constructs its classes.
    fig, ax = plt.subplots(figsize=(11, 6))
    source_counts = pd.crosstab(plot_data["Source_Model"], plot_data["Label_Name"])
    source_counts.plot(kind="bar", ax=ax)
    ax.set(title="Samples by source model and class", xlabel="Source model", ylabel="Samples")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source_model_by_class.png", dpi=160)
    plt.close(fig)

    # Keep the scatter plots readable by sampling each class equally.
    scatter_data = pd.concat(
        [
            group.sample(n=min(len(group), SCATTER_SAMPLE_PER_CLASS), random_state=42)
            for _, group in plot_data.groupby("Label_Name", sort=True)
        ],
        ignore_index=True,
    )
    scatter_plots = [
        (
            "Word_Count",
            "Lexical_Diversity_Score",
            "scatter_word_count_vs_lexical_diversity.png",
            "Word count vs. lexical diversity",
        ),
        (
            "Average_Sentence_Length_Words",
            "Flesch_Kincaid_Grade_Level",
            "scatter_sentence_length_vs_readability.png",
            "Average sentence length vs. readability grade",
        ),
        (
            "Formal_Tone_Score",
            "Emotion_Tone_Intensity",
            "scatter_formal_tone_vs_emotion.png",
            "Formal tone vs. emotion intensity (score ceiling at 100)",
        ),
        (
            "Formal_Tone_Score",
            "Burstiness_Score",
            "scatter_formal_tone_vs_burstiness.png",
            "Formal tone vs. burstiness",
        ),
        (
            "Coherence_Score",
            "Burstiness_Score",
            "scatter_coherence_vs_burstiness.png",
            "Coherence vs. burstiness",
        ),
        (
            "Emotion_Tone_Intensity",
            "Burstiness_Score",
            "scatter_emotion_vs_burstiness.png",
            "Emotion intensity vs. burstiness",
        ),
    ]
    for x, y, filename, title in scatter_plots:
        _save_scatter_plot(scatter_data, x, y, filename, title)

    # Show the frequency of the formal-tone score's maximum value.
    ceiling_rates = (
        plot_data.groupby("Label_Name")["Formal_Tone_Score"]
        .apply(lambda values: values.eq(100).mean() * 100)
        .reindex(["Human", "AI"])
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ceiling_rates.index,
        ceiling_rates.values,
        color=[CLASS_PALETTE[label] for label in ceiling_rates.index],
    )
    for bar, rate in zip(bars, ceiling_rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 1, f"{rate:.1f}%", ha="center")
    ax.set(
        title="Samples at the formal-tone score ceiling",
        xlabel="Class",
        ylabel="Samples with score 100 (%)",
        ylim=(0, max(ceiling_rates.values) * 1.15),
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "formal_tone_score_ceiling.png", dpi=160)
    plt.close(fig)

    # Inspect whether the AI score values are approximately flat across their range.
    ai_formal_scores = plot_data.loc[
        plot_data["Label_Name"] == "AI", "Formal_Tone_Score"
    ].dropna()
    score_min = ai_formal_scores.min()
    score_max = ai_formal_scores.max()
    bins = 20
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        ai_formal_scores,
        bins=bins,
        range=(score_min, score_max),
        color=CLASS_PALETTE["AI"],
        edgecolor="white",
    )
    ax.axhline(
        len(ai_formal_scores) / bins,
        color="#333333",
        linestyle="--",
        label="Equal-count expectation",
    )
    ax.set(
        title="AI formal-tone score distribution",
        xlabel="Formal tone score",
        ylabel="Samples per bin",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ai_formal_tone_score_histogram.png", dpi=160)
    plt.close(fig)

    # 13. Numeric feature correlation with the target. This is an audit aid,
    # not a feature-selection rule: provenance and prediction-time availability
    # must be reviewed before any field is retained.
    numeric_columns = plot_data.select_dtypes(include="number").columns
    correlation = (
        plot_data[list(numeric_columns)]
        .corr(numeric_only=True)["Label"]
        .drop("Label")
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    correlation.plot(kind="barh", ax=ax, color="#4c72b0")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set(
        title="Numeric feature correlation with the target (audit only)",
        xlabel="Pearson correlation with Label",
        ylabel="Feature",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numeric_feature_target_correlations.png", dpi=160)
    plt.close(fig)

    # 14. Missingness by class makes structural missingness visible.
    missing_by_class = (
        plot_data.assign(Label_Name=_label_names(plot_data))
        .groupby("Label_Name", observed=True)
        .apply(lambda group: group.isna().mean() * 100, include_groups=False)
        .T
        .reindex(columns=["Human", "AI"])
    )
    missing_by_class = missing_by_class.loc[missing_by_class.sum(axis=1) > 0]
    fig, ax = plt.subplots(figsize=(9, 4))
    if missing_by_class.empty:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center")
        ax.set_axis_off()
    else:
        missing_by_class.plot(kind="bar", ax=ax, color=[CLASS_PALETTE["Human"], CLASS_PALETTE["AI"]])
        ax.set(
            title="Missingness by class (percentage of rows)",
            xlabel="Feature",
            ylabel="Missing (%)",
        )
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missingness_by_class.png", dpi=160)
    plt.close(fig)

    # 15. Metadata cross-tab audit: useful for deciding whether metadata is a
    # legitimate input or a proxy for how the benchmark was constructed.
    metadata = [
        column
        for column in ["Source_Model", "Model_Size", "Generation_Prompt_Type", "RAG_Context_Provided"]
        if column in plot_data
    ]
    fig, axes = plt.subplots(len(metadata), 1, figsize=(11, 3.2 * len(metadata)))
    axes = [axes] if len(metadata) == 1 else list(axes)
    for ax, column in zip(axes, metadata):
        counts = pd.crosstab(plot_data[column], plot_data["Label_Name"])
        counts.reindex(columns=["Human", "AI"], fill_value=0).plot(
            kind="bar", ax=ax, color=[CLASS_PALETTE["Human"], CLASS_PALETTE["AI"]]
        )
        ax.set(title=f"{column}: category counts by class", xlabel=column, ylabel="Rows")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metadata_associations_by_class.png", dpi=160)
    plt.close(fig)

    # 16. Correlation matrix for numeric fields that are plausibly derived from
    # text. This is a redundancy diagnostic, not an automatic keep/drop rule.
    text_columns = [column for column in TEXT_DERIVED_NUMERIC_COLUMNS if column in plot_data]
    text_correlation = plot_data[text_columns].corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        text_correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        square=True,
        linewidths=0.25,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set(title="Text-derived numeric feature correlation matrix (audit only)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "text_derived_feature_correlation_matrix.png", dpi=180)
    plt.close(fig)

    # 17. Exact and normalized duplicate audit. Normalization is deliberately
    # conservative: it lowercases and collapses whitespace only.
    normalized_text = (
        plot_data["Raw_Text"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    duplicate_counts = pd.Series(
        {
            "Exact raw text duplicates": int(plot_data["Raw_Text"].duplicated().sum()),
            "Normalized text duplicates": int(normalized_text.duplicated().sum()),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(duplicate_counts.index, duplicate_counts.values, color="#4c72b0")
    for bar, count in zip(bars, duplicate_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(count), ha="center")
    ax.set(
        title="Duplicate text audit",
        ylabel="Duplicate rows after first occurrence",
        ylim=(0, max(1, duplicate_counts.max() * 1.2)),
    )
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "duplicate_audit.png", dpi=160)
    plt.close(fig)

    print(f"Saved seventeen plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
