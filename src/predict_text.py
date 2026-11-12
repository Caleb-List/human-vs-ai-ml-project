"""Classify an arbitrary text snippet as Human or AI with a trained model.

Usage:
    python src/predict_text.py "Some passage of text to classify."
    python src/predict_text.py --file path/to/passage.txt
    echo "Some text" | python src/predict_text.py

The model only sees the seven approved features derived from the text itself,
so any text can be scored. Results describe this benchmark's notion of Human vs.
AI writing and should not be treated as a general-purpose AI detector.
"""

from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from modeling import MODELS_DIR
from text_features import extract_text_features


DEFAULT_MODEL = MODELS_DIR / "logistic_regression.joblib"


def predict(text: str, model_path=DEFAULT_MODEL) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Train it first: python src/train_logistic_regression.py"
        )
    bundle = joblib.load(model_path)
    features = extract_text_features(text)
    row = pd.DataFrame([{name: features[name] for name in bundle["feature_columns"]}])
    if row.isna().any().any():
        missing = row.columns[row.isna().any()].tolist()
        raise ValueError(f"Text is too short to compute: {missing}")
    probability_ai = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return {
        "prediction": "AI" if probability_ai >= 0.5 else "Human",
        "probability_ai": probability_ai,
        "confidence": max(probability_ai, 1 - probability_ai),
        "features": row.iloc[0].to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="Text to classify (or use --file / stdin)")
    parser.add_argument("--file", help="Read the text from this file")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            text = handle.read()
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    result = predict(text)
    print(f"Prediction: {result['prediction']}  (P(AI) = {result['probability_ai']:.3f}, "
          f"confidence {result['confidence']:.1%})")
    print("Features used:")
    for name, value in result["features"].items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
