"""Train, tune, and evaluate the random-forest model.

Model:      RandomForestClassifier on the seven approved text-derived features.
            Trees split on feature thresholds, so no scaling is needed; the
            pipeline has a single step and nothing is fit outside it.
Features:   the same seven features as logistic regression (no metadata, no
            detector score, no Formal_Tone_Score).
Tuning:     grid search over tree depth, minimum leaf size, and the number of
            features tried at each split, using the same stored
            near-duplicate-group-aware CV folds as logistic regression.
            Selection metric: ROC-AUC (same as logistic regression).
Evaluation: identical to logistic regression: CV train/validation metrics, one
            evaluation on the untouched test partition, the template-family
            hold-out stress test, and the benchmark-detector reference.
            Permutation importance on the test partition explains the model.

Outputs:
    reports/metrics/random_forest_cv_results.csv
    reports/metrics/random_forest_metrics.json
    reports/metrics/random_forest_test_predictions.csv
    reports/figures/random_forest_*.png
    models/random_forest.joblib
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

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


MODEL_NAME = "random_forest"
N_TREES = 300
PARAM_GRID = {
    # None lets trees grow until leaves are pure (or hit min_samples_leaf).
    "clf__max_depth": [4, 8, 16, None],
    # Larger leaves average more texts per prediction and smooth the boundary.
    "clf__min_samples_leaf": [1, 10, 50],
    # Features tried per split: 2 (= floor(sqrt(7)), the usual default),
    # 4, or all 7 (plain bagged trees, least decorrelated).
    "clf__max_features": [2, 4, 7],
}


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=1
                ),
            ),
        ]
    )


def plot_tuning_curve(cv_results: pd.DataFrame, best_params: dict) -> None:
    """Train vs. validation score against depth, at the selected max_features."""
    rows = cv_results[cv_results["param_clf__max_features"] == best_params["clf__max_features"]]
    depth_labels = ["4", "8", "16", "None"]
    x = range(len(depth_labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, label in [
        (axes[0], "roc_auc", "ROC-AUC"),
        (axes[1], "accuracy", "Accuracy"),
    ]:
        for leaf in PARAM_GRID["clf__min_samples_leaf"]:
            subset = rows[rows["param_clf__min_samples_leaf"] == leaf].copy()
            subset["depth_label"] = subset["param_clf__max_depth"].map(
                lambda d: "None" if pd.isna(d) else str(int(d))
            )
            subset = subset.set_index("depth_label").reindex(depth_labels)
            line = ax.errorbar(
                x, subset[f"mean_test_{metric}"], yerr=subset[f"std_test_{metric}"],
                marker="s", capsize=3, label=f"leaf ≥ {leaf} validation",
            )
            ax.plot(
                x, subset[f"mean_train_{metric}"], marker="o", linestyle="--",
                color=line[0].get_color(), alpha=0.6, label=f"leaf ≥ {leaf} train",
            )
        ax.set_xticks(list(x), depth_labels)
        ax.set(xlabel="max_depth", ylabel=f"{label} (mean ± SD)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        f"Random forest tuning (5 group-aware CV folds, max_features = "
        f"{best_params['clf__max_features']}, {N_TREES} trees)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{MODEL_NAME}_tuning_curve.png", dpi=180)
    plt.close(fig)


def plot_importances(importances: pd.DataFrame) -> None:
    ordered = importances.sort_values("permutation_mean")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    axes[0].barh(
        ordered.index, ordered["permutation_mean"], xerr=ordered["permutation_std"],
        color="#4C72B0",
    )
    axes[0].set(
        title="Permutation importance (held-out test)",
        xlabel="Drop in ROC-AUC when the feature is shuffled",
    )
    axes[1].barh(ordered.index, ordered["impurity"], color="#8C8C8C")
    axes[1].set(
        title="Impurity importance (training fit)",
        xlabel="Mean decrease in Gini impurity",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{MODEL_NAME}_feature_importance.png", dpi=180)
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
    keep = ["param_clf__max_depth", "param_clf__min_samples_leaf", "param_clf__max_features"] + [
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
        best_pipeline, X_train, y_train, cv=folds, scoring=SCORING,
        return_train_score=True, n_jobs=-1,
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
    plot_confusion_matrix(
        y_test, test_pred, "Random forest: held-out test set",
        FIGURES_DIR / f"{MODEL_NAME}_confusion_matrix.png",
    )

    # 4. Feature importance: permutation on held-out data (reliable) and the
    #    built-in impurity measure (fast, but biased toward high-cardinality
    #    features and computed on training data).
    perm = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    importances = pd.DataFrame(
        {
            "permutation_mean": perm.importances_mean,
            "permutation_std": perm.importances_std,
            "impurity": model.named_steps["clf"].feature_importances_,
        },
        index=FEATURE_COLUMNS,
    )
    plot_importances(importances)

    # 5. External reference: the benchmark's own detector score (never an input).
    detector_auc, detector = detector_reference(test)
    plot_roc_curves(
        [
            ("Random forest (7 features)", y_test, test_score),
            ("Benchmark detector score (reference)", y_test, detector),
        ],
        "ROC curve on the held-out test set",
        FIGURES_DIR / f"{MODEL_NAME}_roc_curve.png",
    )

    # 6. Stress test: hold out whole template families (training partition only).
    family = template_family_holdout(best_pipeline, train)
    family_summary, family_fold_members = family["summary"], family["folds"]

    # 7. Persist the fitted pipeline for predict_text.py.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": model, "feature_columns": FEATURE_COLUMNS, "best_params": best_params},
        MODELS_DIR / f"{MODEL_NAME}.joblib",
        compress=3,
    )

    results = {
        "model": f"RandomForestClassifier ({N_TREES} trees)",
        "features": FEATURE_COLUMNS,
        "param_grid": {k: [("None" if v is None else v) for v in vals] for k, vals in PARAM_GRID.items()},
        "fixed_params": {"n_estimators": N_TREES, "random_state": RANDOM_STATE, "class_weight": None},
        "selection_metric": "roc_auc",
        "best_params": {k: ("None" if v is None else v) for k, v in best_params.items()},
        "cv_near_duplicate_groups": cv_summary,
        "full_train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_confusion,
        "feature_importance": importances.to_dict(orient="index"),
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
    print("\nFeature importance:")
    print(importances.sort_values("permutation_mean", ascending=False).round(4).to_string())
    print("\nWrote metrics to reports/metrics/, figures to reports/figures/, model to models/")


if __name__ == "__main__":
    main()
