"""Compare transfer/retraining on identical M4 rows with source-group uncertainty."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve

from audit_m4 import REPORT_DIR
from prepare_m4 import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    if args.bootstrap < 20:
        parser.error("Use at least 20 group bootstrap replicates")
    transfer = json.loads((REPORT_DIR / "transfer_metrics.json").read_text())
    original = json.loads((REPORT_DIR / "original_reconstruction.json").read_text())
    split = pd.read_csv(OUTPUT_DIR / "m4_splits.csv")
    split_hash = json.loads((REPORT_DIR / "split_summary.json").read_text())["splits_sha256"]
    expected = split.loc[split.Split.eq("test")].sort_values("Text_ID").reset_index(drop=True)
    labels = expected.Label.to_numpy()
    rows, predictions, saved_metrics = [], {}, {}
    for name in ["logistic_regression", "random_forest"]:
        baseline = original[name]["reconstructed_original_test_metrics"]
        rows.append({"model": name, "stage": "Original benchmark test (reconstructed)", "test_rows": 10000,
                     **{key: value for key, value in baseline.items() if key != "confusion_matrix"}})
        stages = [("Transfer to M4", "transfer_", transfer[name])]
        trained_path = REPORT_DIR / f"{name}_metrics.json"
        if trained_path.exists():
            stages.append(("Retrained on M4", "", json.loads(trained_path.read_text())))
        for stage, prefix, result in stages:
            if result["splits_sha256"] != split_hash:
                raise ValueError("Transfer/retraining must use identical split hashes")
            frame = pd.read_csv(OUTPUT_DIR / f"{prefix}{name}_test_predictions.csv", float_precision="round_trip").sort_values("Text_ID").reset_index(drop=True)
            if not frame.Text_ID.is_unique or not frame.Text_ID.equals(expected.Text_ID) or not frame.Label.equals(expected.Label) or not frame.Group_ID.equals(expected.Group_ID):
                raise ValueError("Prediction rows/labels/groups differ from canonical held-out partition")
            key = f"{prefix}{name}"
            predictions[key] = (frame.Predicted_Label.to_numpy(), frame.Probability_AI.to_numpy())
            saved_metrics[key] = result["test_metrics"]
            if not np.isclose(roc_auc_score(labels, predictions[key][1]), saved_metrics[key]["roc_auc"], atol=1e-12, rtol=0):
                raise ValueError("Saved probabilities do not reproduce reported ROC-AUC")
            rows.append({"model": name, "stage": stage, "test_rows": len(frame),
                         **{field: value for field, value in result["test_metrics"].items() if field != "confusion_matrix"}})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    codes, names = pd.factorize(expected.Group_ID, sort=True)
    random = np.random.default_rng(478)
    boot = {key: {metric: [] for metric in ["accuracy", "balanced_accuracy", "f1", "roc_auc", "human_false_positive_rate"]} for key in predictions}
    for iteration in range(args.bootstrap):
        weights = np.bincount(random.integers(0, len(names), size=len(names)), minlength=len(names))[codes]
        human = labels == 0
        for key, (pred, score) in predictions.items():
            boot[key]["accuracy"].append(accuracy_score(labels, pred, sample_weight=weights))
            boot[key]["balanced_accuracy"].append(balanced_accuracy_score(labels, pred, sample_weight=weights))
            boot[key]["f1"].append(f1_score(labels, pred, sample_weight=weights, zero_division=0))
            boot[key]["roc_auc"].append(roc_auc_score(labels, score, sample_weight=weights))
            boot[key]["human_false_positive_rate"].append(float(np.sum(weights[human] * (pred[human] == 1)) / weights[human].sum()))
        if iteration in [0, args.bootstrap - 1] or (iteration + 1) % 50 == 0:
            print(f"Group bootstrap {iteration + 1}/{args.bootstrap}", flush=True)
    intervals = {key: {metric: {"estimate": saved_metrics[key][metric], "lower_95": float(np.quantile(values, .025)),
                               "upper_95": float(np.quantile(values, .975))} for metric, values in metrics.items()} for key, metrics in boot.items()}
    differences = {}
    for a, b, title in [("logistic_regression", "transfer_logistic_regression", "LR retrained minus transfer"),
                         ("random_forest", "transfer_random_forest", "RF retrained minus transfer"),
                         ("random_forest", "logistic_regression", "Retrained RF minus LR")]:
        if a not in boot or b not in boot:
            continue
        differences[title] = {}
        for metric in boot[a]:
            values = np.asarray(boot[a][metric]) - np.asarray(boot[b][metric])
            differences[title][metric] = {"estimate": saved_metrics[a][metric] - saved_metrics[b][metric],
                                          "lower_95": float(np.quantile(values, .025)), "upper_95": float(np.quantile(values, .975))}
    uncertainty = {"method": "Paired percentile bootstrap of whole held-out source/near-duplicate groups with replacement",
                   "replicates": args.bootstrap, "seed": 478, "test_groups": len(names),
                   "scope": "Conditional on fixed trained models/split; does not include retraining/tuning uncertainty or establish across-domain generalization",
                   "models": intervals, "paired_differences": differences}
    (REPORT_DIR / "group_bootstrap_intervals.json").write_text(json.dumps(uncertainty, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    phases = ["Original benchmark test (reconstructed)", "Transfer to M4", "Retrained on M4"]
    for axis, metric, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "roc_auc", "ROC-AUC")]:
        pivot = comparison.pivot(index="stage", columns="model", values=metric).reindex(phases)
        pivot.index = ["Original test", "M4 transfer", "M4 retrained"]
        pivot.rename(columns={"logistic_regression": "LR", "random_forest": "RF"}).plot.bar(ax=axis, rot=15)
        axis.set(title=title, ylim=(0, 1.07), xlabel="Evaluation")
        if metric == "accuracy":
            for location, baseline in enumerate([.5, float(expected.Label.mean()), float(expected.Label.mean())]):
                axis.hlines(baseline, location - .4, location + .4, color="gray", linestyle=":")
        else:
            axis.axhline(.5, color="gray", linestyle=":")
    for key, (_, score) in predictions.items():
        fpr, tpr, _ = roc_curve(labels, score)
        axes[2].plot(fpr, tpr, label=f"{key.replace('logistic_regression', 'LR').replace('random_forest', 'RF').replace('transfer_', 'Transfer ')} ({saved_metrics[key]['roc_auc']:.3f})")
    axes[2].plot([0, 1], [0, 1], color="gray", linestyle=":")
    axes[2].set(title="Same M4 test rows: ROC curves", xlabel="Human false-positive rate", ylabel="AI recall")
    axes[2].legend(fontsize=8)
    missing = [name for name in ["logistic_regression", "random_forest"] if name not in predictions]
    note = "; retraining not completed: " + ", ".join(missing) if missing else ""
    fig.suptitle("Original benchmark vs. M4 transfer and retraining\nDotted accuracy references = each dataset's majority baseline" + note)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "study_model_comparison.png", dpi=170)
    plt.close(fig)

    provenance = pd.read_csv(OUTPUT_DIR / "m4_provenance.csv", usecols=["Text_ID", "Domain"])
    domains = provenance.groupby("Text_ID").Domain.agg(lambda values: next(iter(set(values))) if len(set(values)) == 1 else "multiple")
    error_rows = []
    for key, (pred, score) in predictions.items():
        frame = expected[["Text_ID", "Label"]].copy()
        frame["Domain"] = frame.Text_ID.map(domains)
        frame["Prediction"] = pred
        frame["Score"] = score
        for domain, part in frame.groupby("Domain"):
            h = part.Label.eq(0)
            error_rows.append({"model": key, "domain": domain, "rows": len(part), "Human": int(h.sum()), "AI": int((~h).sum()),
                               "accuracy": float(part.Label.eq(part.Prediction).mean()),
                               "human_false_positive_rate": float(part.loc[h, "Prediction"].eq(1).mean()) if h.any() else None,
                               "roc_auc": float(roc_auc_score(part.Label, part.Score)) if part.Label.nunique() == 2 else None})
    pd.DataFrame(error_rows).to_csv(REPORT_DIR / "test_errors_by_domain.csv", index=False)
    print(comparison[["model", "stage", "accuracy", "recall", "f1", "roc_auc", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
