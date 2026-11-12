"""Shared modeling framework: data loading, folds, metrics, and plots.

Every model in the comparison should use these helpers so that all models see
the same feature matrix, the same group-aware train/test split, the same CV
folds, and the same metric definitions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from data_loader import PROJECT_ROOT, RAW_CSV_PATH
from prepare_final_data import FINAL_FEATURE_COLUMNS


PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "human_ai_text_processed.csv"
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "human_ai_text_splits.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = list(FINAL_FEATURE_COLUMNS)
TARGET_COLUMN = "Label"
RANDOM_STATE = 478
FAMILY_CV_FOLDS = 5

# Scorer names understood by scikit-learn's model-selection utilities.
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def load_model_data() -> pd.DataFrame:
    """Join the approved feature CSV with the split assignments by Text_ID."""
    for path in (PROCESSED_PATH, SPLITS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run prepare_final_data.py and build_splits.py first."
            )
    processed = pd.read_csv(PROCESSED_PATH)
    splits = pd.read_csv(
        SPLITS_PATH,
        usecols=["Text_ID", "NearDup_Group_ID", "Template_Family_ID", "Split", "CV_Fold"],
    )
    data = processed.merge(splits, on="Text_ID", how="inner", validate="one_to_one")
    if len(data) != len(processed):
        raise ValueError("Split file does not cover every processed row.")
    if data[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Model features contain missing values.")
    return data


def train_test_frames(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[data["Split"] == "train"].reset_index(drop=True)
    test = data[data["Split"] == "test"].reset_index(drop=True)
    return train, test


def predefined_cv(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the stored near-duplicate-group-aware CV folds as index pairs."""
    folds = []
    for fold in sorted(train["CV_Fold"].unique()):
        is_val = (train["CV_Fold"] == fold).to_numpy()
        folds.append((np.flatnonzero(~is_val), np.flatnonzero(is_val)))
    return folds


def template_family_cv(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds that hold out whole template families (generalization stress test).

    Each validation fold contains template families that the model never saw
    during fitting. Because every family is single-class, stratification is
    applied at the family level so each fold still contains both classes.
    """
    splitter = StratifiedGroupKFold(
        n_splits=FAMILY_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    folds = list(
        splitter.split(
            np.zeros(len(train)), train[TARGET_COLUMN], train["Template_Family_ID"]
        )
    )
    for _, val_idx in folds:
        if train.loc[val_idx, TARGET_COLUMN].nunique() != 2:
            raise ValueError("A template-family fold is missing one class.")
    return folds


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def confusion(y_true, y_pred) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_human_pred_human": int(tn),
        "true_human_pred_ai": int(fp),
        "true_ai_pred_human": int(fn),
        "true_ai_pred_ai": int(tp),
    }


def load_detector_scores(text_ids: pd.Series) -> pd.Series:
    """Benchmark detector score, used ONLY as an external evaluation reference."""
    raw = pd.read_csv(RAW_CSV_PATH, usecols=["Text_ID", "Detection_Score_Benchmark_Model_A"])
    return text_ids.map(raw.set_index("Text_ID")["Detection_Score_Benchmark_Model_A"])


def plot_confusion_matrix(y_true, y_pred, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Human", "AI"], cmap="Blues", values_format=",", ax=ax
    )
    ax.set(title=title, xlabel="Predicted class", ylabel="True class")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_roc_curves(curves: list[tuple[str, np.ndarray, np.ndarray]], title: str, path) -> None:
    """`curves` is a list of (name, y_true, y_score)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, y_true, y_score in curves:
        RocCurveDisplay.from_predictions(y_true, y_score, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set(title=title, xlabel="False positive rate (Human called AI)", ylabel="True positive rate (AI caught)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
