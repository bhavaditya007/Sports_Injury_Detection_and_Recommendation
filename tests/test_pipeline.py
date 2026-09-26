from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from injury_demo.data import FEATURES  # noqa: E402
from injury_demo.evaluation import stratified_holdout  # noqa: E402
from injury_demo.pipeline import build_candidate_pipelines  # noqa: E402


class PipelineConstructionTests(unittest.TestCase):
    def test_candidates_include_requested_estimators_and_nested_preprocessor(self):
        candidates = build_candidate_pipelines(seed=42)
        self.assertEqual(
            set(candidates),
            {
                "DummyClassifier",
                "LogisticRegression",
                "RandomForestClassifier",
                "ExtraTreesClassifier",
                "HistGradientBoostingClassifier",
            },
        )
        for candidate in candidates.values():
            self.assertIsInstance(candidate, Pipeline)
            self.assertIsInstance(candidate.named_steps["preprocessor"], ColumnTransformer)
            self.assertEqual(
                list(candidate.named_steps["preprocessor"].transformers[0][2]), FEATURES
            )
            self.assertNotIn(
                "Recovery_Time", candidate.named_steps["preprocessor"].transformers[0][2]
            )

    def test_holdout_is_fixed_stratified_and_disjoint(self):
        labels = np.array([0, 1] * 50)
        frame = pd.DataFrame(
            {
                "Player_Age": np.arange(100),
                "Player_Weight": np.arange(100, 200),
                "Player_Height": np.arange(150, 250),
                "Previous_Injuries": labels,
                "Training_Intensity": np.linspace(0, 1, 100),
                "Recovery_Time": np.ones(100),
                "Likelihood_of_Injury": labels,
            }
        )
        first = stratified_holdout(frame, seed=42)
        second = stratified_holdout(frame, seed=42)
        X_train, X_test, y_train, y_test = first
        self.assertEqual(len(X_test), 20)
        self.assertEqual(set(X_train.index).intersection(X_test.index), set())
        self.assertEqual(y_train.value_counts(normalize=True).to_dict(), {0: 0.5, 1: 0.5})
        self.assertEqual(y_test.value_counts(normalize=True).to_dict(), {0: 0.5, 1: 0.5})
        self.assertTrue(X_train.equals(second[0]))
        self.assertTrue(X_test.equals(second[1]))


if __name__ == "__main__":
    unittest.main()