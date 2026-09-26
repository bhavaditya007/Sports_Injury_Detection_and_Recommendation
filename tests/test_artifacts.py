from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from injury_demo.artifacts import build_metadata, write_artifacts  # noqa: E402
from injury_demo.pipeline import build_pipeline  # noqa: E402


class ArtifactWriterTests(unittest.TestCase):
    def test_generated_metadata_uses_portable_dataset_path(self):
        pipeline = build_pipeline(LogisticRegression(random_state=42))
        metadata = build_metadata(
            dataset_path=ROOT / "data" / "injury_data.csv",
            dataset_sha256="example-digest",
            duplicate_rows=0,
            row_count=1000,
            selected_name="LogisticRegression",
            selected_pipeline=pipeline,
        )
        self.assertEqual(metadata["dataset"]["path_at_training"], "data\\injury_data.csv")
        self.assertEqual(metadata["dataset"]["path_reference_scope"], "repository-relative")
        self.assertNotIn(str(Path.home()), metadata["dataset"]["path_at_training"])

    def test_artifacts_are_written_as_a_complete_set(self):
        pipeline = build_pipeline(LogisticRegression(random_state=42))
        predictions = pd.DataFrame(
            {"source_row_index": [3], "y_true": [1], "y_pred": [1], "uncalibrated_class_1_score": [0.8]}
        )
        metadata = {
            "feature_list_ordered": [
                "Player_Age",
                "Player_Weight",
                "Player_Height",
                "Previous_Injuries",
                "Training_Intensity",
            ],
            "random_seed": 42,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifacts"
            paths = write_artifacts(
                output,
                fitted_pipeline=pipeline,
                metrics={"holdout": {"recall": 1.0}},
                predictions=predictions,
                metadata_payload=metadata,
            )
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))
            self.assertEqual(json.loads((output / "metrics.json").read_text())["holdout"]["recall"], 1.0)
            self.assertEqual(json.loads((output / "metadata.json").read_text())["random_seed"], 42)
            self.assertEqual(pd.read_csv(output / "predictions.csv").loc[0, "y_true"], 1)
            self.assertFalse(any(path.name.startswith(".phase3a-staging-") for path in output.iterdir()))


if __name__ == "__main__":
    unittest.main()