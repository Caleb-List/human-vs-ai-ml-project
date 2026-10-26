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

    print(f"Saved twelve plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
