"""Train, tune, and evaluate the logistic-regression baseline.

Model:      StandardScaler -> LogisticRegression (inside one Pipeline, so the
            scaler is refit on the training folds only).
Features:   the seven approved text-derived features (no metadata, no detector
            score, no Formal_Tone_Score).
Tuning:     grid search over regularization strength C and penalty type
            (L2 vs. L1) using the stored near-duplicate-group-aware CV folds on
            the training partition. Selection metric: ROC-AUC.
Evaluation: CV train/validation metrics, a single evaluation on the untouched
            test partition, and a template-family hold-out stress test run on
            the training partition only.

Outputs:
    reports/metrics/logistic_regression_cv_results.csv
    reports/metrics/logistic_regression_metrics.json
    reports/metrics/logistic_regression_test_predictions.csv
    reports/figures/logistic_regression_*.png
    models/logistic_regression.joblib
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    SCORING,
    TARGET_COLUMN,
    classification_metrics,
    confusion,
    detector_reference,
    format_cv_block,
    load_model_data,
    plot_confusion_matrix,
    plot_roc_curves,
    predefined_cv,
    save_test_predictions,
    summarize_cv,
    template_family_holdout,
    train_test_frames,
)


MODEL_NAME = "logistic_regression"
PARAM_GRID = {
    "clf__C": np.logspace(-4, 3, 8).tolist(),
    "clf__l1_ratio": [0.0, 1.0],  # 0.0 = L2 (ridge) penalty, 1.0 = L1 (lasso) penalty
}


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga", max_iter=5_000, tol=1e-4, random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def plot_tuning_curve(cv_results: pd.DataFrame, best_params: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for l1_ratio, name in [(0.0, "L2"), (1.0, "L1")]:
        rows = cv_results[cv_results["param_clf__l1_ratio"] == l1_ratio].sort_values("param_clf__C")
        C = rows["param_clf__C"].astype(float)
        ax.errorbar(C, rows["mean_train_roc_auc"], yerr=rows["std_train_roc_auc"], marker="o",
                    linestyle="--", capsize=3, label=f"{name} train")
        ax.errorbar(C, rows["mean_test_roc_auc"], yerr=rows["std_test_roc_auc"], marker="s",
                    capsize=3, label=f"{name} validation")
    ax.axvline(best_params["clf__C"], color="gray", linestyle=":", label="Selected C")
    ax.set_xscale("log")
    ax.set(
        title="Logistic regression tuning (5 group-aware CV folds)",
        xlabel="Inverse regularization strength C (log scale)",
        ylabel="ROC-AUC (mean ± SD)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{MODEL_NAME}_tuning_curve.png", dpi=180)
    plt.close(fig)


def plot_coefficients(coefficients: pd.Series) -> None:
    ordered = coefficients.reindex(coefficients.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#DD8452" if value > 0 else "#4C72B0" for value in ordered]
    ax.barh(ordered.index, ordered.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title="Standardized coefficients\n(orange pushes toward AI, blue toward Human)",
        xlabel="Change in log-odds of AI per 1 SD increase in feature",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{MODEL_NAME}_coefficients.png", dpi=180)
    plt.close(fig)


def main() -> None:
    data = load_model_data()
    train, test = train_test_frames(data)
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]
    folds = predefined_cv(train)
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}; CV folds: {len(folds)}")

    # 1. Hyperparameter tuning on the training partition only.
    search = GridSearchCV(
        build_pipeline(),
        PARAM_GRID,
        scoring=SCORING,
        refit="roc_auc",
        cv=folds,
        return_train_score=True,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    cv_results = pd.DataFrame(search.cv_results_)
    keep = ["param_clf__C", "param_clf__l1_ratio"] + [
        f"{stat}_{part}_{metric}"
        for metric in SCORING
        for part in ("train", "test")
        for stat in ("mean", "std")
    ] + ["rank_test_roc_auc"]
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cv_results[keep].to_csv(METRICS_DIR / f"{MODEL_NAME}_cv_results.csv", index=False)
    best_params = search.best_params_
    print(f"Best params: {best_params}; mean CV ROC-AUC {search.best_score_:.4f}")
    plot_tuning_curve(cv_results, best_params)

    # 2. Train vs. validation metrics for the selected configuration.
    best_pipeline = clone(build_pipeline()).set_params(**best_params)
    cv_scores = cross_validate(
        best_pipeline, X_train, y_train, cv=folds, scoring=SCORING, return_train_score=True
    )
    cv_summary = summarize_cv(cv_scores)

    # 3. Final fit on all training rows; single evaluation on the test partition.
    model = search.best_estimator_
    train_pred = model.predict(X_train)
    train_score = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict(X_test)
    test_score = model.predict_proba(X_test)[:, 1]
    train_metrics = classification_metrics(y_train, train_pred, train_score)
    test_metrics = classification_metrics(y_test, test_pred, test_score)
    test_confusion = confusion(y_test, test_pred)
    save_test_predictions(MODEL_NAME, test, test_pred, test_score)

    coefficients = pd.Series(model.named_steps["clf"].coef_[0], index=FEATURE_COLUMNS)
    plot_coefficients(coefficients)
    plot_confusion_matrix(
        y_test, test_pred, "Logistic regression: held-out test set",
        FIGURES_DIR / f"{MODEL_NAME}_confusion_matrix.png",
    )

    # 4. External reference: the benchmark's own detector score (never an input).
    detector_auc, detector = detector_reference(test)
    plot_roc_curves(
        [
            ("Logistic regression (7 features)", y_test, test_score),
            ("Benchmark detector score (reference)", y_test, detector),
        ],
        "ROC curve on the held-out test set",
        FIGURES_DIR / f"{MODEL_NAME}_roc_curve.png",
    )

    # 5. Stress test: hold out whole template families (training partition only).
    family = template_family_holdout(best_pipeline, train)
    family_summary, family_fold_members = family["summary"], family["folds"]

    # 6. Persist the fitted pipeline for predict_text.py.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": model, "feature_columns": FEATURE_COLUMNS, "best_params": best_params},
        MODELS_DIR / f"{MODEL_NAME}.joblib",
    )

    results = {
        "model": "StandardScaler + LogisticRegression (saga)",
        "features": FEATURE_COLUMNS,
        "param_grid": PARAM_GRID,
        "selection_metric": "roc_auc",
        "best_params": best_params,
        "cv_near_duplicate_groups": cv_summary,
        "full_train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_confusion,
        "standardized_coefficients": coefficients.to_dict(),
        "intercept": float(model.named_steps["clf"].intercept_[0]),
        "template_family_holdout": {"summary": family_summary, "folds": family_fold_members},
        "benchmark_detector_reference_on_test": detector_auc,
    }
    (METRICS_DIR / f"{MODEL_NAME}_metrics.json").write_text(json.dumps(results, indent=2))

    print("\nNear-duplicate-group CV (selected config):")
    print("  train:     ", format_cv_block(cv_summary["train"]))
    print("  validation:", format_cv_block(cv_summary["validation"]))
    print("\nFull training partition:", {k: round(v, 4) for k, v in train_metrics.items()})
    print("Held-out test:          ", {k: round(v, 4) for k, v in test_metrics.items()})
    print("Test confusion matrix:  ", test_confusion)
    print("\nTemplate-family hold-out (training partition only):")
    print("  train:     ", format_cv_block(family_summary["train"]))
    print("  validation:", format_cv_block(family_summary["validation"]))
    for fold in family_fold_members:
        print("   ", fold)
    print("\nBenchmark detector reference on test:", {k: round(v, 4) for k, v in detector_auc.items() if k != "note"})
    print("\nStandardized coefficients:")
    print(coefficients.sort_values().round(4).to_string())
    print("\nWrote metrics to reports/metrics/, figures to reports/figures/, model to models/")


if __name__ == "__main__":
    main()
