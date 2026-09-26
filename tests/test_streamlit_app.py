from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import injury_demo.streamlit_app as app  # noqa: E402
from injury_demo.data import FEATURES  # noqa: E402
from injury_demo.inference import ArtifactLoadError, InferenceInputError  # noqa: E402


def compatible_metadata() -> dict:
    return {
        "artifact_schema_version": 1,
        "selected_model": "ExtraTreesClassifier",
        "target": "Likelihood_of_Injury",
        "positive_class": 1,
        "feature_list_ordered": list(FEATURES),
        "excluded_features": {"Recovery_Time": "unclear timing"},
        "python_version": "3.13.0",
        "dependency_versions": {
            "scikit-learn": "1.8.0",
            "numpy": "2.1.3",
            "joblib": "1.5.3",
        },
    }


class FakeSidebar:
    def __init__(self):
        self.calls = []

    def header(self, *args, **kwargs):
        self.calls.append(("header", args, kwargs))

    def number_input(self, label, **kwargs):
        self.calls.append(("number_input", (label,), kwargs))
        return {"Age": 28, "Weight": 75.0, "Height": 180.0}[label.split(" ")[0]]

    def selectbox(self, label, **kwargs):
        self.calls.append(("selectbox", (label,), kwargs))
        return 0

    def slider(self, label, **kwargs):
        self.calls.append(("slider", (label,), kwargs))
        return 0.5


class FakeUI:
    def __init__(self, *, button=True):
        self.sidebar = FakeSidebar()
        self.button_value = button
        self.events = []

    def __getattr__(self, name):
        if name in {"set_page_config", "markdown", "title", "error", "warning", "caption", "subheader", "metric"}:
            return lambda *args, **kwargs: self.events.append((name, args, kwargs))
        raise AttributeError(name)

    def button(self, *args, **kwargs):
        self.events.append(("button", args, kwargs))
        return self.button_value

    def event_text(self, event_name):
        return " ".join(
            str(part)
            for name, args, kwargs in self.events
            if name == event_name
            for part in args
        )


class AppFacingInputTests(unittest.TestCase):
    def test_values_map_in_exact_pipeline_order(self):
        frame = app.build_prediction_input(28, 75.0, 180.0, 1, 0.65)
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(frame.iloc[0].tolist(), [28, 75.0, 180.0, 1, 0.65])
        self.assertNotIn("Recovery_Time", frame.columns)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(InferenceInputError):
            app.build_prediction_input(28.5, 75, 180, 0, 0.5)
        with self.assertRaises(InferenceInputError):
            app.build_prediction_input(28, 75, 180, 2, 0.5)
        with self.assertRaises(InferenceInputError):
            app.build_prediction_input(28, 75, 180, 0, 1.01)
        with self.assertRaises(InferenceInputError):
            app.build_prediction_input(28, float("inf"), 180, 0, 0.5)

    def test_dataset_bounds_are_warnings_not_medical_validation(self):
        frame = app.build_prediction_input(45, 120, 220, 0, 0.5)
        self.assertEqual(app.out_of_observed_range(frame), ["Player_Age", "Player_Weight", "Player_Height"])


class ArtifactMetadataValidationTests(unittest.TestCase):
    def _write_files(self, root: Path, metadata_payload: dict):
        model = root / "model.joblib"
        metadata = root / "metadata.json"
        model.write_bytes(b"not deserialized in metadata failure tests")
        metadata.write_text(json.dumps(metadata_payload), encoding="utf-8")
        return model, metadata

    def test_compatible_metadata_reaches_trusted_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, metadata_path = self._write_files(Path(temporary), compatible_metadata())
            expected_model = Mock()
            expected_model.named_steps = {"classifier": type("ExtraTreesClassifier", (), {})()}
            expected_model.classes_ = [0, 1]
            with patch.object(app, "_installed_version", side_effect=["1.8.0", "2.1.3", "1.5.3"]), patch.object(
                app, "load_pipeline", return_value=expected_model
            ) as loader:
                result = app.load_verified_artifact(model, metadata_path)
        self.assertIs(result, expected_model)
        loader.assert_called_once_with(model)

    def test_version_or_schema_mismatch_fails_before_joblib_load(self):
        payload = compatible_metadata()
        payload["dependency_versions"]["scikit-learn"] = "1.7.0"
        with tempfile.TemporaryDirectory() as temporary:
            model, metadata_path = self._write_files(Path(temporary), payload)
            with patch.object(app, "_installed_version", return_value="1.8.0"), patch.object(
                app, "load_pipeline"
            ) as loader:
                with self.assertRaisesRegex(ArtifactLoadError, "version mismatch"):
                    app.load_verified_artifact(model, metadata_path)
                loader.assert_not_called()

        payload = compatible_metadata()
        payload["feature_list_ordered"] = ["Player_Weight"]
        with tempfile.TemporaryDirectory() as temporary:
            model, metadata_path = self._write_files(Path(temporary), payload)
            with patch.object(app, "load_pipeline") as loader:
                with self.assertRaisesRegex(ArtifactLoadError, "feature list/order"):
                    app.load_verified_artifact(model, metadata_path)
                loader.assert_not_called()

    def test_missing_artifact_and_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                app.load_verified_artifact(root / "absent.joblib", root / "metadata.json")
            model = root / "model.joblib"
            model.write_bytes(b"unused")
            with self.assertRaisesRegex(ArtifactLoadError, "metadata is missing"):
                app.load_verified_artifact(model, root / "absent.json")

    def test_malformed_metadata_fails_before_model_deserialization(self):
        for content in ("{", "[]", "null"):
            with self.subTest(metadata_content=content), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model = root / "model.joblib"
                metadata_path = root / "metadata.json"
                model.write_bytes(b"must not be deserialized")
                metadata_path.write_text(content, encoding="utf-8")
                with patch.object(app, "load_pipeline") as loader:
                    with self.assertRaises(ArtifactLoadError):
                        app.load_verified_artifact(model, metadata_path)
                    loader.assert_not_called()


class MockedStreamlitBehaviorTests(unittest.TestCase):
    def test_success_ui_displays_label_and_uncalibrated_0_to_1_score(self):
        ui = FakeUI(button=True)
        result = pd.DataFrame(
            {"predicted_synthetic_label": [1], "uncalibrated_class_1_score": [0.731]}
        )
        predictor = Mock(return_value=(pd.DataFrame(columns=FEATURES), result, []))
        app.render_app(streamlit_module=ui, model_loader=Mock(return_value=object()), predictor=predictor)
        metric_events = [event for event in ui.events if event[0] == "metric"]
        self.assertIn("Predicted synthetic label", str(metric_events))
        self.assertIn("Uncalibrated class-1 model score (0–1)", str(metric_events))
        self.assertIn("0.731", str(metric_events))
        self.assertIn("SYNTHETIC DATA", ui.event_text("error"))
        self.assertNotIn("Recommended Rest Days", str(ui.events))
        self.assertNotIn("%", str(metric_events))
        predictor.assert_called_once()

    def test_ui_displays_descriptive_observed_range_warning(self):
        ui = FakeUI(button=True)
        result = pd.DataFrame(
            {"predicted_synthetic_label": [0], "uncalibrated_class_1_score": [0.4]}
        )
        predictor = Mock(
            return_value=(pd.DataFrame(columns=FEATURES), result, ["Player_Age"])
        )
        app.render_app(
            streamlit_module=ui,
            model_loader=Mock(return_value=object()),
            predictor=predictor,
        )
        warning = ui.event_text("warning")
        self.assertIn("outside observed dataset bounds", warning)
        self.assertIn("descriptive only", warning)

    def test_missing_artifact_and_inference_error_are_user_facing(self):
        missing_ui = FakeUI()
        app.render_app(
            streamlit_module=missing_ui,
            model_loader=Mock(side_effect=FileNotFoundError("artifact missing")),
        )
        self.assertIn("saved model is unavailable", missing_ui.event_text("error"))
        self.assertFalse(any(call[0] == "number_input" for call in missing_ui.sidebar.calls))

        inference_ui = FakeUI()
        app.render_app(
            streamlit_module=inference_ui,
            model_loader=Mock(return_value=object()),
            predictor=Mock(side_effect=InferenceInputError("invalid fixture input")),
        )
        self.assertIn("Please correct the input values", inference_ui.event_text("error"))

        runtime_ui = FakeUI()
        with self.assertLogs(app.LOGGER, level="ERROR"):
            app.render_app(
                streamlit_module=runtime_ui,
                model_loader=Mock(return_value=object()),
                predictor=Mock(side_effect=RuntimeError("mocked estimator failure")),
            )
        self.assertIn("Prediction could not be completed", runtime_ui.event_text("error"))
        self.assertNotIn("Uncalibrated class-1 model score", str(runtime_ui.events))


if __name__ == "__main__":
    unittest.main()