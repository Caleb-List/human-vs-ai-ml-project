"""Verify that provenance bridges and split boundaries survive deduplication."""

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_m4 import flatten_record
from prepare_m4 import canonicalize
from audit_m4_similarity import combined_components, compute_edges
from build_m4_splits import assign_splits, validate_splits


class M4PipelineTests(unittest.TestCase):
    def test_duplicate_human_bridges_generated_versions_with_missing_source_ids(self):
        rows = []
        for filename, machine in [("reddit_chatGPT.jsonl", "AI version A"), ("reddit_dolly.jsonl", "AI version B")]:
            rows.extend(flatten_record({"human_text": "Shared human text", "machine_text": machine}, filename, 1))
        rows.extend(flatten_record({"human_text": None, "machine_text": ""}, "reddit_cohere.jsonl", 1))
        canonical, provenance, groups = canonicalize(pd.DataFrame(rows))
        self.assertEqual(len(canonical), 3)
        self.assertEqual(len(provenance), 4)
        self.assertEqual(groups.Source_Group_ID.nunique(), 1)
        self.assertEqual(provenance.Observation_ID.nunique(), 4)

    def test_normalized_equivalent_texts_group_but_are_not_deleted(self):
        rows = []
        for line, human in [(1, "Hello World"), (2, " hello   WORLD ")]:
            rows.extend(flatten_record({"human_text": human, "machine_text": f"Response {line}"}, "reddit_dolly.jsonl", line))
        canonical, _, groups = canonicalize(pd.DataFrame(rows))
        self.assertEqual(len(canonical), 4)
        self.assertEqual(groups.Source_Group_ID.nunique(), 1)

    def test_conflicting_exact_labels_fail(self):
        rows = list(flatten_record({"human_text": "same", "machine_text": "same"}, "reddit_dolly.jsonl", 1))
        with self.assertRaises(ValueError):
            canonicalize(pd.DataFrame(rows))

    def test_exact_similarity_retains_original_norm_after_unique_column_removal(self):
        texts = pd.Series(["a b c d e", "a b c d f", "x y z", "hi"])
        rows, cols, vals, nearest, zero = compute_edges(texts, 2)
        # Shared trigrams a-b-c and b-c-d, out of three in each text.
        np.testing.assert_allclose(nearest[:2], [2 / 3, 2 / 3], rtol=1e-6)
        self.assertEqual(zero, 1)
        self.assertEqual(len(rows), 0)  # 2/3 is below cache floor 0.70

    def test_components_and_splits_preserve_bridging_edges(self):
        source = np.array(["a", "a", "b", "b", "c"])
        components = combined_components(source, np.array([1]), np.array([2]), np.array([.9]), .85)
        self.assertEqual(len(set(components[:4])), 1)
        self.assertNotEqual(components[0], components[4])
        data = pd.DataFrame({"Text_ID": [f"t{i}" for i in range(120)], "Label": [0, 1] * 60})
        splits = assign_splits(data, np.repeat(np.arange(60), 2))
        validate_splits(splits)
        pd.testing.assert_frame_equal(splits, assign_splits(data, np.repeat(np.arange(60), 2)))
        broken = splits.copy()
        group = broken.Group_ID.iloc[0]
        members = broken.index[broken.Group_ID == group]
        broken.loc[members[0], "Split"] = "test" if broken.loc[members[1], "Split"] == "train" else "train"
        with self.assertRaises(ValueError):
            validate_splits(broken)


if __name__ == "__main__":
    unittest.main()
