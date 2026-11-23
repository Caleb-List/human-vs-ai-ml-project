"""Evaluate original benchmark models on M4 without fitting to M4.

Original model binaries are ignored by Git. --reconstruct-original rebuilds
them from saved selected parameters and original train membership, checks the
saved original test predictions, and stores separate explicitly labeled models.
"""

import argparse
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd

from audit_m4 import REPORT_DIR
from build_m4_splits import validate_splits
from download_m4 import ROOT, sha256
from prepare_final_data import FINAL_FEATURE_COLUMNS
from prepare_m4 import OUTPUT_DIR, PROCESSED_PATH
from train_m4 import metrics


NAMES = ["logistic_regression", "random_forest"]
REBUILT_DIR = ROOT / "models" / "original_reconstructed"


def reconstruct():
    original_path = ROOT / "data" / "processed" / "human_ai_text_processed.csv"
    original = pd.read_csv(original_path, usecols=["Text_ID", "Label"] + FINAL_FEATURE_COLUMNS)
    if len(original) != 50000 or not original.Text_ID.is_unique:
        raise ValueError("Original benchmark must contain 50,000 unique IDs")
    reports = {}
    for name in NAMES:
        source = ROOT / "reports" / "metrics" / f"{name}_metrics.json"
        prediction_path = source.with_name(f"{name}_test_predictions.csv")
        saved = json.loads(source.read_text())
        if saved["features"] != FINAL_FEATURE_COLUMNS:
            raise ValueError("Original feature policy differs")
        old_test = pd.read_csv(prediction_path)
        if len(old_test) != 10000 or not old_test.Text_ID.is_unique:
            raise ValueError("Saved original test IDs must contain 10,000 unique rows")
        test = old_test.merge(original, on=["Text_ID", "Label"], validate="one_to_one")
        train = original.loc[~original.Text_ID.isin(old_test.Text_ID)]
        if len(test) != 10000 or len(train) != 40000:
            raise ValueError("Saved original membership/labels do not match local processed data")
        if name == "logistic_regression":
            from train_logistic_regression import build_pipeline
        else:
            from train_random_forest import build_pipeline
        params = {key: (None if value == "None" else value) for key, value in saved["best_params"].items()}
        model = build_pipeline().set_params(**params)
        model.fit(train[FINAL_FEATURE_COLUMNS], train.Label)
        pred = model.predict(test[FINAL_FEATURE_COLUMNS])
        score = model.predict_proba(test[FINAL_FEATURE_COLUMNS])[:, 1]
        report = {
            "status": "Reconstructed from saved selected configuration, not the original binary; no retuning",
            "best_params": params, "original_train_rows": len(train), "original_test_rows": len(test),
            "prediction_disagreements_with_saved_original": int((pred != test.Predicted_Label).sum()),
            "max_probability_difference_from_saved_original": float(np.abs(score - test.Probability_AI).max()),
            "reconstructed_original_test_metrics": metrics(test.Label, pred, score),
            "saved_original_test_metrics": saved["test_metrics"],
            "original_processed_sha256": sha256(original_path), "saved_test_predictions_sha256": sha256(prediction_path),
            "sklearn_version": importlib.metadata.version("scikit-learn"),
        }
        REBUILT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": model, "feature_columns": FINAL_FEATURE_COLUMNS, "best_params": params,
                     "reconstruction": report}, REBUILT_DIR / f"{name}.joblib", compress=3)
        reports[name] = report
        print(f"Reconstructed {name}: {report['prediction_disagreements_with_saved_original']} disagreements with saved original test labels", flush=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "original_reconstruction.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruct-original", action="store_true")
    parser.add_argument("--reconstruct-only", action="store_true")
    args = parser.parse_args()
    if args.reconstruct_only and not args.reconstruct_original:
        parser.error("--reconstruct-only requires --reconstruct-original")
    if args.reconstruct_original:
        reconstruct()
    if args.reconstruct_only:
        return
    split_path = OUTPUT_DIR / "m4_splits.csv"
    splits = pd.read_csv(split_path)
    validate_splits(splits)
    manifest = json.loads((REPORT_DIR / "split_summary.json").read_text())
    if manifest["processed_sha256"] != sha256(PROCESSED_PATH) or manifest["splits_sha256"] != sha256(split_path):
        raise ValueError("M4 data/splits differ from validated manifest")
    data = pd.read_csv(PROCESSED_PATH, usecols=["Text_ID", "Label"] + FINAL_FEATURE_COLUMNS)
    test = splits.loc[splits.Split == "test"].merge(data, on=["Text_ID", "Label"], validate="one_to_one")
    if len(test) != int(splits.Split.eq("test").sum()):
        raise ValueError("Not every test row has matching features/target")
    results = {}
    for name in NAMES:
        path = ROOT / "models" / f"{name}.joblib"
        if not path.exists():
            path = REBUILT_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError("Original model binaries absent; use --reconstruct-original explicitly")
        bundle = joblib.load(path)
        if bundle["feature_columns"] != FINAL_FEATURE_COLUMNS:
            raise ValueError("Model feature columns do not match")
        model = bundle["pipeline"]
        pred = model.predict(test[FINAL_FEATURE_COLUMNS])
        score = model.predict_proba(test[FINAL_FEATURE_COLUMNS])[:, 1]
        results[name] = {"stage": "Original-benchmark model -> M4 test, no refitting/scaling/tuning on M4",
                         "model_path": str(path.relative_to(ROOT)), "model_sha256": sha256(path),
                         "reconstructed": "reconstruction" in bundle,
                         "test_rows": len(test), "test_metrics": metrics(test.Label, pred, score),
                         "processed_sha256": sha256(PROCESSED_PATH), "splits_sha256": sha256(split_path)}
        predictions = test[["Text_ID", "Label", "Group_ID"]].copy()
        predictions["Predicted_Label"], predictions["Probability_AI"] = pred, score
        predictions.to_csv(OUTPUT_DIR / f"transfer_{name}_test_predictions.csv", index=False)
    (REPORT_DIR / "transfer_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
