"""Compare every trained model on the same metrics and the same test rows.

Reads the metrics and test predictions written by each train_*.py script, so
no model is retrained here. Add a model to MODELS once its script exists.

Outputs:
    reports/metrics/model_comparison.csv       one row per model x evaluation stage
    reports/metrics/model_comparison.json      test-error overlap and McNemar test
    reports/figures/model_comparison_error_rates.png
    reports/figures/model_comparison_roc.png
"""

from __future__ import annotations

import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_curve

from modeling import FIGURES_DIR, METRICS_DIR


# script name -> display name, in the order models were added.
MODELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
STAGES = [
    ("CV train", "cv_near_duplicate_groups", "train"),
    ("CV validation", "cv_near_duplicate_groups", "validation"),
    ("Held-out test", "test_metrics", None),
    ("Template-family hold-out", "template_family_holdout", "validation"),
]


def load_available() -> dict[str, dict]:
    available = {}
    for name in MODELS:
        metrics_path = METRICS_DIR / f"{name}_metrics.json"
        predictions_path = METRICS_DIR / f"{name}_test_predictions.csv"
        if metrics_path.exists() and predictions_path.exists():
            available[name] = {
                "metrics": json.loads(metrics_path.read_text()),
                "predictions": pd.read_csv(predictions_path),
            }
        else:
            print(f"Skipping {name}: run src/train_{name}.py first.")
    if len(available) < 2:
        raise SystemExit("Need at least two trained models to compare.")
    return available


def comparison_table(available: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, bundle in available.items():
        results = bundle["metrics"]
        for stage, key, part in STAGES:
            block = results[key]
            if key == "template_family_holdout":
                block = block["summary"]
            if part is not None:
                values = {m: block[part][m]["mean"] for m in METRICS}
                spread = {f"{m}_sd": block[part][m]["std"] for m in METRICS}
            else:
                values = {m: block[m] for m in METRICS}
                spread = {f"{m}_sd": np.nan for m in METRICS}
            rows.append({"Model": MODELS[name], "Stage": stage, **values, **spread})
    return pd.DataFrame(rows)


def error_overlap(available: dict[str, dict]) -> dict:
    """Which test rows each pair of models gets wrong, plus an exact McNemar test."""
    results = {}
    for a, b in combinations(available, 2):
        pa = available[a]["predictions"].set_index("Text_ID")
        pb = available[b]["predictions"].set_index("Text_ID").reindex(pa.index)
        wrong_a = pa["Predicted_Label"] != pa["Label"]
        wrong_b = pb["Predicted_Label"] != pb["Label"]
        only_a = int((wrong_a & ~wrong_b).sum())
        only_b = int((~wrong_a & wrong_b).sum())
        # Exact McNemar: under "equal error rates", disagreements split 50/50.
        p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        results[f"{a}_vs_{b}"] = {
            "test_rows": int(len(pa)),
            f"{a}_errors": int(wrong_a.sum()),
            f"{b}_errors": int(wrong_b.sum()),
            "both_wrong": int((wrong_a & wrong_b).sum()),
            f"only_{a}_wrong": only_a,
            f"only_{b}_wrong": only_b,
            "mcnemar_exact_p_value": float(p_value),
            "both_wrong_text_ids": sorted(pa.index[wrong_a & wrong_b].tolist()),
        }
    return results


def plot_error_rates(table: pd.DataFrame) -> None:
    stages = [stage for stage, _, _ in STAGES]
    models = table["Model"].unique().tolist()
    width = 0.8 / len(models)
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(models):
        sub = table[table["Model"] == model].set_index("Stage").reindex(stages)
        error = 100 * (1 - sub["accuracy"])
        sd = 100 * sub["accuracy_sd"].fillna(0)
        # Error rates cannot go below zero, so clip the lower bar there.
        yerr = [np.minimum(sd, error), sd]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, error, width, yerr=yerr,
                      capsize=3, label=model)
        ax.bar_label(bars, fmt="%.2f%%", fontsize=7, padding=2)
    ax.set_xticks(x, stages)
    ax.set(
        title="Error rate (100% − accuracy) at each evaluation stage\n"
              "Lower is better; error bars are SD across folds",
        ylabel="Error rate (%)",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_error_rates.png", dpi=180)
    plt.close(fig)


def plot_roc(available: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name, bundle in available.items():
        preds = bundle["predictions"]
        fpr, tpr, _ = roc_curve(preds["Label"], preds["Probability_AI"])
        auc = bundle["metrics"]["test_metrics"]["roc_auc"]
        for ax in axes:
            ax.plot(fpr, tpr, label=f"{MODELS[name]} (AUC {auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    axes[0].set(title="ROC on the held-out test set", xlim=(0, 1), ylim=(0, 1.01))
    axes[1].set(title="Zoomed: top-left corner", xlim=(0, 0.05), ylim=(0.95, 1.001))
    for ax in axes:
        ax.set(xlabel="False positive rate (Human called AI)",
               ylabel="True positive rate (AI caught)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_roc.png", dpi=180)
    plt.close(fig)


def main() -> None:
    available = load_available()
    table = comparison_table(available)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    overlap = error_overlap(available)
    (METRICS_DIR / "model_comparison.json").write_text(json.dumps(overlap, indent=2))
    plot_error_rates(table)
    plot_roc(available)

    pd.set_option("display.width", 140)
    print(table[["Model", "Stage", *METRICS]].round(4).to_string(index=False))
    for pair, info in overlap.items():
        summary = {k: v for k, v in info.items() if k != "both_wrong_text_ids"}
        print(f"\n{pair}: {summary}")
    print(f"\nWrote {METRICS_DIR / 'model_comparison.csv'} and comparison figures.")


if __name__ == "__main__":
    main()
