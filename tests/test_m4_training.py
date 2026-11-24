"""Exercise training artifact output and training-only scaling end to end."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_m4_splits import assign_splits
from download_m4 import sha256
from prepare_final_data import FINAL_FEATURE_COLUMNS
import train_logistic_regression
import train_m4


class M4TrainingTest(unittest.TestCase):
    def test_saved_scaler_uses_only_training_rows_despite_shifted_test_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, reports = root / "data", root / "reports"
            output.mkdir()
            reports.mkdir()
            random = np.random.default_rng(478)
            data = pd.DataFrame(random.normal(size=(120, 7)), columns=FINAL_FEATURE_COLUMNS)
            data["Text_ID"] = [f"row-{i}" for i in range(120)]
            data["Label"] = [0, 1] * 60
            data[FINAL_FEATURE_COLUMNS[1]] += data.Label * 2
            splits = assign_splits(data, np.repeat(np.arange(60), 2))
            heldout = splits.Split.eq("test")
            data.loc[heldout, FINAL_FEATURE_COLUMNS[0]] += 1000
            processed = output / "m4_processed.csv"
            split_path = output / "m4_splits.csv"
            data.to_csv(processed, index=False)
            splits.to_csv(split_path, index=False)
            (reports / "split_summary.json").write_text(json.dumps({"processed_sha256": sha256(processed),
                                                                   "splits_sha256": sha256(split_path)}))
            with patch.object(train_m4, "ROOT", root), patch.object(train_m4, "OUTPUT_DIR", output), \
                 patch.object(train_m4, "REPORT_DIR", reports), patch.object(train_m4, "PROCESSED_PATH", processed), \
                 patch.object(train_logistic_regression, "PARAM_GRID", {"clf__C": [1.0], "clf__l1_ratio": [0.0]}), \
                 patch.object(sys, "argv", ["train_m4.py", "--jobs", "1"]), contextlib.redirect_stdout(io.StringIO()):
                train_m4.main()
            model = joblib.load(root / "models" / "m4" / "logistic_regression.joblib")["pipeline"]
            np.testing.assert_allclose(model.named_steps["scale"].mean_,
                                       data.loc[~heldout, FINAL_FEATURE_COLUMNS].mean().to_numpy())
            self.assertGreater(abs(model.named_steps["scale"].mean_[0] - data[FINAL_FEATURE_COLUMNS[0]].mean()), 100)
            predictions = pd.read_csv(output / "logistic_regression_test_predictions.csv")
            self.assertEqual(set(predictions.Text_ID), set(data.loc[heldout, "Text_ID"]))
            report = json.loads((reports / "logistic_regression_metrics.json").read_text())
            self.assertEqual(report["test_rows"], int(heldout.sum()))
            self.assertEqual(report["splits_sha256"], sha256(split_path))


if __name__ == "__main__":
    unittest.main()
