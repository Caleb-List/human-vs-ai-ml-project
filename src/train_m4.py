"""Train M4 LR/RF on reviewed groups; keep original benchmark artifacts intact."""

import argparse
import importlib.metadata
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV

from audit_m4 import REPORT_DIR
from build_m4_splits import validate_splits
from download_m4 import ROOT, sha256
from modeling import SCORING, classification_metrics
from prepare_final_data import FINAL_FEATURE_COLUMNS
from prepare_m4 import OUTPUT_DIR, PROCESSED_PATH


def metrics(y, predicted, scores):
    result = classification_metrics(y, predicted, scores)
    result["balanced_accuracy"] = float(balanced_accuracy_score(y, predicted))
    result["average_precision"] = float(average_precision_score(y, scores))
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    result["human_false_positive_rate"] = float(fp / (tn + fp))
    result["confusion_matrix"] = [[int(tn), int(fp)], [int(fn), int(tp)]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["logistic_regression", "random_forest"], default="logistic_regression")
    parser.add_argument("--jobs", type=int, default=2, help="Parallel tuning workers; default 2 limits CPU/RAM contention")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.model == "logistic_regression":
        from train_logistic_regression import build_pipeline, PARAM_GRID
    else:
        from train_random_forest import build_pipeline, PARAM_GRID
    splits_path = OUTPUT_DIR / "m4_splits.csv"
    splits = pd.read_csv(splits_path)
    validate_splits(splits)
    split_summary = json.loads((REPORT_DIR / "split_summary.json").read_text())
    if split_summary["processed_sha256"] != sha256(PROCESSED_PATH) or split_summary["splits_sha256"] != sha256(splits_path):
        raise ValueError("Data/splits differ from validated split manifest")
    data = pd.read_csv(PROCESSED_PATH, usecols=["Text_ID", "Label"] + FINAL_FEATURE_COLUMNS)
    data = data.merge(splits.drop(columns="Label"), on="Text_ID", validate="one_to_one", how="left")
    if data.isna().any().any():
        raise ValueError("Missing features or assignments")
    train = data.loc[data.Split == "train"].reset_index(drop=True)
    test = data.loc[data.Split == "test"].reset_index(drop=True)
    folds = [(np.flatnonzero(train.CV_Fold.to_numpy() != fold), np.flatnonzero(train.CV_Fold.to_numpy() == fold)) for fold in range(5)]
    started = time.perf_counter()
    search = GridSearchCV(build_pipeline(), PARAM_GRID, scoring=SCORING, refit="roc_auc", cv=folds,
                          return_train_score=True, n_jobs=args.jobs, pre_dispatch=args.jobs, error_score="raise", verbose=1)
    print(f"M4 {args.model}: {len(train):,} train / {len(test):,} test rows", flush=True)
    search.fit(train[FINAL_FEATURE_COLUMNS], train.Label)
    # Everything about configuration selection is complete before test scoring.
    model = search.best_estimator_
    predictions = model.predict(test[FINAL_FEATURE_COLUMNS])
    scores = model.predict_proba(test[FINAL_FEATURE_COLUMNS])[:, 1]
    train_predictions = model.predict(train[FINAL_FEATURE_COLUMNS])
    train_scores = model.predict_proba(train[FINAL_FEATURE_COLUMNS])[:, 1]
    selected = search.best_index_
    cv = {}
    for part, key in [("train", "train"), ("test", "validation")]:
        cv[key] = {metric: {"mean": float(search.cv_results_[f"mean_{part}_{metric}"][selected]),
                            "std": float(search.cv_results_[f"std_{part}_{metric}"][selected])} for metric in SCORING}
    best = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in search.best_params_.items()}
    results = {"dataset": "M4 Kaggle v1", "model": args.model, "features": FINAL_FEATURE_COLUMNS,
               "best_params": best, "selection_metric": "roc_auc", "decision_threshold": 0.5,
               "train_rows": len(train), "test_rows": len(test), "cv": cv,
               "train_metrics": metrics(train.Label, train_predictions, train_scores),
               "test_metrics": metrics(test.Label, predictions, scores),
               "majority_class_test_accuracy": float(test.Label.value_counts(normalize=True).max()),
               "fit_and_evaluation_seconds": time.perf_counter() - started,
               "processed_sha256": sha256(PROCESSED_PATH), "splits_sha256": sha256(splits_path),
               "environment": {package: importlib.metadata.version(package) for package in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]},
               "limitations": "In-corpus group-aware evaluation; domain/generator holdouts and independent transfer remain separate experiments"}
    if args.model == "logistic_regression":
        results["standardized_coefficients"] = dict(zip(FINAL_FEATURE_COLUMNS, model.named_steps["clf"].coef_[0].tolist()))
    else:
        results["training_impurity_importances"] = dict(zip(FINAL_FEATURE_COLUMNS, model.named_steps["clf"].feature_importances_.tolist()))
    model_dir = ROOT / "models" / "m4"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "feature_columns": FINAL_FEATURE_COLUMNS, "best_params": best,
                 "dataset": "M4", "splits_sha256": results["splits_sha256"]}, model_dir / f"{args.model}.joblib", compress=3)
    (REPORT_DIR / f"{args.model}_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(search.cv_results_).to_csv(REPORT_DIR / f"{args.model}_cv_results.csv", index=False)
    heldout = test[["Text_ID", "Label", "Group_ID"]].copy()
    heldout["Predicted_Label"], heldout["Probability_AI"] = predictions, scores
    heldout.to_csv(OUTPUT_DIR / f"{args.model}_test_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    matrix = np.array(results["test_metrics"]["confusion_matrix"])
    axes[0].imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            axes[0].text(col, row, f"{matrix[row, col]:,}", ha="center", va="center")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Human", "AI"], yticklabels=["Human", "AI"],
                xlabel="Predicted", ylabel="True", title="Held-out confusion matrix")
    fpr, tpr, _ = roc_curve(test.Label, scores)
    axes[1].plot(fpr, tpr, label=f"AUC {results['test_metrics']['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set(xlabel="Human false-positive rate", ylabel="AI recall", title="Held-out ROC curve")
    axes[1].legend()
    fig.suptitle(f"M4: {args.model}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"{args.model}_evaluation.png", dpi=160)
    plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
