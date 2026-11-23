"""Check provenance-sensitive M4 parsing without downloading data."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_m4 import flatten_record, text_entries


class M4ParsingTests(unittest.TestCase):
    def test_list_reviews_remain_separate_and_share_source_group(self):
        row = {"human_text": ["Review A", "Review B"], "machine_text": ["AI review"],
               "source": "PeerRead/acl_2017/", "source_ID": "104", "model": "chatgpt"}
        texts = list(flatten_record(row, "peerread_chatgpt.jsonl", 1))
        self.assertEqual([x["Raw_Text"] for x in texts], ["Review A", "Review B", "AI review"])
        self.assertEqual(len({x["Text_ID"] for x in texts}), 3)
        self.assertEqual(len({x["Source_Group_Candidate"] for x in texts}), 1)
        row["source_ID"] = 104.0
        other = list(flatten_record(row, "peerread_dolly.jsonl", 1))
        self.assertEqual(texts[0]["Source_Group_Candidate"], other[0]["Source_Group_Candidate"])
        row["source"] = "PeerRead/nips_2017/"
        scoped = list(flatten_record(row, "peerread_dolly.jsonl", 1))
        self.assertNotEqual(texts[0]["Source_Group_Candidate"], scoped[0]["Source_Group_Candidate"])

    def test_bloom_response_selected_instead_of_prompt_bearing_field(self):
        row = {"abstract": "Human abstract", "machine_abstract": "Generated abstract",
               "machine_text": "Write an abstract. Generated abstract", "prompt": "Write an abstract.",
               "source_id": 704.0007, "model": "bigscience/bloomz"}
        texts = list(flatten_record(row, "arxiv_bloomz.jsonl", 1))
        self.assertEqual(texts[1]["Raw_Text"], "Generated abstract")
        self.assertEqual(texts[1]["Text_Field"], "machine_abstract")
        self.assertTrue(texts[1]["Raw_Machine_Text_Starts_With_Prompt"])
        self.assertEqual([x["Label"] for x in texts], [0, 1])

    def test_missing_and_empty_are_flagged_without_stringification(self):
        row = {"human_text": None, "machine_text": "   "}
        texts = list(flatten_record(row, "reddit_dolly.jsonl", 1))
        self.assertIsNone(texts[0]["Raw_Text"])
        self.assertTrue(all(x["Missing_Or_Empty"] for x in texts))
        self.assertTrue(all(x["Source_Group_Candidate"] is None for x in texts))
        self.assertEqual(list(text_entries([])), [("", None)])

    def test_unknown_schema_fails_instead_of_silently_losing_rows(self):
        with self.assertRaises(ValueError):
            list(flatten_record({"machine_text": "Generated"}, "reddit_chatGPT.jsonl", 1))
        with self.assertRaises(ValueError):
            list(text_entries({"unexpected": "dictionary"}))


if __name__ == "__main__":
    unittest.main()
