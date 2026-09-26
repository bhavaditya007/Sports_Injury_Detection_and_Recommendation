from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_model as train_cli  # noqa: E402
from injury_demo.data import FEATURES  # noqa: E402
from injury_demo.inference import (  # noqa: E402
    ArtifactLoadError,
    InferenceInputError,
    load_pipeline,
    predict_synthetic_label,
    validate_inference_frame,
)
def tiny_synthetic_fixture() -> pd.DataFrame:
    """Small made-up inference fixture; contains no source or holdout rows."""
    return pd.DataFrame(
        {
            "Player_Age": [22, 31, 27],
            "Player_Weight": [68.5, 81.2, 74.0],
            "Player_Height": [174.0, 183.0, 179.0],
            "Previous_Injuries": [0, 1, 0],
            "Training_Intensity": [0.35, 0.72, 0.51],
        }
    )[FEATURES]


class InferenceVerificationTests(unittest.TestCase):
    def test_predict_on_explicit_synthetic_fixture_with_mock_pipeline(self):
        pipeline = Mock()
        pipeline.classes_ = np.asarray([0, 1])
        pipeline.predict_proba.return_value = np.asarray(
            [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]
        )
        fixture = tiny_synthetic_fixture()
        result = predict_synthetic_label(pipeline, fixture)

        self.assertEqual(list(result.columns), ["predicted_synthetic_label", "uncalibrated_class_1_score"])
        self.assertEqual(len(result), len(fixture))
        self.assertTrue(result["predicted_synthetic_label"].isin([0, 1]).all())
        self.assertTrue(result["uncalibrated_class_1_score"].between(0, 1).all())
        pipeline.predict_proba.assert_called_once()

    def test_input_schema_rejects_missing_extra_reordered_and_non_numeric_columns(self):
        fixture = tiny_synthetic_fixture()
        with self.assertRaises(InferenceInputError):
            validate_inference_frame(fixture.drop(columns=["Player_Height"]))
        with self.assertRaises(InferenceInputError):
            validate_inference_frame(fixture.assign(Recovery_Time=2))
        with self.assertRaises(InferenceInputError):
            validate_inference_frame(fixture[FEATURES[::-1]])
        bad = fixture.copy()
        bad["Player_Age"] = "twenty-two"
        with self.assertRaises(InferenceInputError):
            validate_inference_frame(bad)

    def test_input_schema_allows_partial_missing_values_for_pipeline_imputation(self):
        fixture = tiny_synthetic_fixture()
        fixture.loc[0, "Player_Age"] = np.nan
        self.assertTrue(pd.isna(validate_inference_frame(fixture).loc[0, "Player_Age"]))

    def test_unreadable_or_wrong_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "wrong.joblib"
            joblib.dump({"not": "a pipeline"}, wrong)
            with self.assertRaises(ArtifactLoadError):
                load_pipeline(wrong)


class CliSummaryVerificationTests(unittest.TestCase):
    def test_success_summary_with_mocked_workflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / name for name in [
                "injury_classifier.joblib", "metrics.json", "predictions.csv", "metadata.json"
            ]]
            for path in paths[:3]:
                path.write_text("mock artifact", encoding="utf-8")
            paths[3].write_text(
                json.dumps({"dataset": {"exact_duplicate_rows_retained": 7}}),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch.object(sys, "argv", ["train_model.py"]), patch.object(
                train_cli, "run", return_value=paths
            ) as mocked_run, contextlib.redirect_stdout(output):
                exit_code = train_cli.main()

        mocked_run.assert_called_once()
        self.assertEqual(exit_code, 0)
        self.assertIn("Completed synthetic-label evaluation", output.getvalue())
        self.assertIn("Exact duplicate rows retained: 7", output.getvalue())
        self.assertIn(str(paths[3]), output.getvalue())


if __name__ == "__main__":
    unittest.main()