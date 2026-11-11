"""Create review plots from independently derived text features."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_loader import PROJECT_ROOT


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "text_derived_features_candidate.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FEATURE_COLUMNS = [
    "Text_Character_Count",
    "Text_Word_Count",
    "Text_Sentence_Count",
    "Text_Unique_Word_Count",
    "Text_Type_Token_Ratio",
    "Text_Guiraud_Root_TTR",
    "Text_Average_Word_Length",
    "Text_Average_Sentence_Length_Words",
    "Text_Punctuation_Count",
    "Text_Punctuation_Density",
    "Text_Special_Character_Count",
    "Text_Capital_Letter_Count",
    "Text_Average_Syllables_Per_Word",
    "Text_Flesch_Kincaid_Grade_Level",
    "Text_Burstiness",
]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    features = df[FEATURE_COLUMNS]
    correlation = features.corr()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        correlation,
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
    ax.set(title="Independently derived text-feature correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_text_feature_correlation_matrix.png", dpi=180)
    plt.close(fig)

    selected = [
        "Text_Word_Count",
        "Text_Type_Token_Ratio",
        "Text_Guiraud_Root_TTR",
        "Text_Flesch_Kincaid_Grade_Level",
        "Text_Burstiness",
    ]
    long = df[["Label", *selected]].melt(id_vars="Label", var_name="Feature", value_name="Value")
    long["Class"] = long["Label"].map({0: "Human", 1: "AI"})
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 3 * len(selected)))
    for ax, feature in zip(axes, selected):
        sns.boxplot(
            data=long[long["Feature"] == feature],
            x="Class",
            y="Value",
            order=["Human", "AI"],
            ax=ax,
        )
        ax.set(title=f"{feature} by class", xlabel="Class", ylabel=feature)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_text_feature_distributions.png", dpi=180)
    plt.close(fig)
    print(f"Saved candidate feature figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
