from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from injury_demo.data import (  # noqa: E402
    EXPECTED_COLUMNS,
    FEATURES,
    DatasetValidationError,
    load_dataset,
    validate_dataset,
)


class DatasetValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / "data" / "injury_data.csv")

    def test_checked_in_dataset_schema_and_rows(self):
        duplicate_count = validate_dataset(self.frame)
        self.assertEqual(list(self.frame.columns), EXPECTED_COLUMNS)
        self.assertEqual(self.frame.shape, (1000, 7))
        self.assertEqual(duplicate_count, 0)

    def test_duplicate_rows_are_reported_and_retained(self):
        frame = pd.concat([self.frame.iloc[:20], self.frame.iloc[[0]]], ignore_index=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            duplicate_count = validate_dataset(frame)
        self.assertEqual(duplicate_count, 1)
        self.assertEqual(len(frame), 21)
        self.assertTrue(any("not assumed to represent duplicate athletes" in str(w.message) for w in caught))

    def test_unexpected_or_missing_columns_fail_loudly(self):
        with self.assertRaises(DatasetValidationError):
            validate_dataset(self.frame.drop(columns=["Recovery_Time"]))
        extra = self.frame.assign(hidden_id=range(len(self.frame)))
        with self.assertRaises(DatasetValidationError):
            validate_dataset(extra)

    def test_invalid_target_fails_loudly(self):
        frame = self.frame.copy()
        frame.loc[0, "Likelihood_of_Injury"] = 2
        with self.assertRaises(DatasetValidationError):
            validate_dataset(frame)

    def test_missing_feature_can_be_imputed_but_entirely_missing_fails(self):
        frame = self.frame.copy()
        frame.loc[0, FEATURES[0]] = float("nan")
        validate_dataset(frame)
        frame[FEATURES[0]] = float("nan")
        with self.assertRaises(DatasetValidationError):
            validate_dataset(frame)

    def test_missing_file_has_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(ROOT / "data" / "not-a-real-dataset.csv")


if __name__ == "__main__":
    unittest.main()