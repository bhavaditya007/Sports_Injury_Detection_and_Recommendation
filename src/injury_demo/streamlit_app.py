"""Streamlit UI and app-facing helpers for the synthetic-label model demo."""

from __future__ import annotations

import json
import logging
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .data import FEATURES
from .inference import (
    ArtifactLoadError,
    InferenceInputError,
    load_pipeline,
    predict_synthetic_label,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "injury_classifier.joblib"
EXPECTED_ARTIFACT_SCHEMA_VERSION = 1
EXPECTED_TARGET = "Likelihood_of_Injury"
EXPECTED_MODEL = "ExtraTreesClassifier"

# Observed min/max values in Kaggle version 1. These are demonstration-data
# bounds, not medically valid, safe, or population-reference ranges.
OBSERVED_RANGES = {
    "Player_Age": (18.0, 39.0),
    "Player_Weight": (40.19191171239916, 104.6501035172072),
    "Player_Height": (145.28570108061618, 207.3086723322357),
    "Previous_Injuries": (0.0, 1.0),
    "Training_Intensity": (0.00003071884538241587, 0.9977493890102006),
}


class ArtifactCompatibilityError(ArtifactLoadError):
    """Raised when artifact metadata is absent, inconsistent, or version-incompatible."""


def build_prediction_input(
    age: int | float,
    weight: int | float,
    height: int | float,
    previous_injuries: int,
    training_intensity: int | float,
) -> pd.DataFrame:
    """Map UI values to the model's exact five-feature schema and validate basics.

    Age, weight, and height units are assumptions inherited from the existing UI,
    not definitions provided by the Kaggle data dictionary. Training intensity is
    only represented as a numeric 0-to-1 demonstration scale.
    """
    values = {
        "Player_Age": age,
        "Player_Weight": weight,
        "Player_Height": height,
        "Previous_Injuries": previous_injuries,
        "Training_Intensity": training_intensity,
    }
    numeric = {}
    for name in FEATURES:
        try:
            value = float(values[name])
        except (TypeError, ValueError) as exc:
            raise InferenceInputError(f"{name} must be numeric.") from exc
        if not np.isfinite(value):
            raise InferenceInputError(f"{name} must be a finite number.")
        numeric[name] = value

    if numeric["Player_Age"] <= 0 or not numeric["Player_Age"].is_integer():
        raise InferenceInputError("Age must be a positive whole number in the demo input.")
    if numeric["Player_Weight"] <= 0:
        raise InferenceInputError("Weight must be greater than zero.")
    if numeric["Player_Height"] <= 0:
        raise InferenceInputError("Height must be greater than zero.")
    if numeric["Previous_Injuries"] not in (0.0, 1.0):
        raise InferenceInputError("Previous injuries must be 0 or 1.")
    if not 0.0 <= numeric["Training_Intensity"] <= 1.0:
        raise InferenceInputError(
            "Training intensity must be between 0 and 1 on the demonstration input scale."
        )

    ordered = {name: numeric[name] for name in FEATURES}
    ordered["Player_Age"] = int(ordered["Player_Age"])
    ordered["Previous_Injuries"] = int(ordered["Previous_Injuries"])
    return pd.DataFrame([ordered], columns=FEATURES)


def out_of_observed_range(frame: pd.DataFrame) -> list[str]:
    """Return feature names outside source-data bounds; these bounds are descriptive only."""
    outside = []
    for feature, (minimum, maximum) in OBSERVED_RANGES.items():
        value = float(frame.iloc[0][feature])
        if value < minimum or value > maximum:
            outside.append(feature)
    return outside


def predict_app_input(
    pipeline,
    age: int | float,
    weight: int | float,
    height: int | float,
    previous_injuries: int,
    training_intensity: int | float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Pure app-facing mapping and prediction; returns input, result, range warnings."""
    input_frame = build_prediction_input(
        age, weight, height, previous_injuries, training_intensity
    )
    warnings = out_of_observed_range(input_frame)
    result = predict_synthetic_label(pipeline, input_frame)
    return input_frame, result, warnings


def _installed_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise ArtifactCompatibilityError(
            f"Required runtime dependency {distribution!r} is not installed."
        ) from exc


def load_verified_artifact(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path | None = None,
):
    """Validate trusted artifact metadata and runtime compatibility before deserialization."""
    model_file = Path(model_path)
    metadata_file = Path(metadata_path) if metadata_path else model_file.with_name("metadata.json")
    if not model_file.is_file():
        raise FileNotFoundError(f"Saved model artifact is missing: {model_file}")
    if not metadata_file.is_file():
        raise ArtifactCompatibilityError(f"Saved model metadata is missing: {metadata_file}")

    try:
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"Could not read model metadata: {exc}") from exc

    if not isinstance(payload, dict):
        raise ArtifactCompatibilityError("Model metadata root must be a JSON object.")
    if payload.get("artifact_schema_version") != EXPECTED_ARTIFACT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("Unsupported or missing artifact schema version.")
    if payload.get("selected_model") != EXPECTED_MODEL:
        raise ArtifactCompatibilityError(
            f"Expected {EXPECTED_MODEL} artifact, found {payload.get('selected_model')!r}."
        )
    if payload.get("target") != EXPECTED_TARGET or payload.get("positive_class") != 1:
        raise ArtifactCompatibilityError("Artifact target or positive class does not match this app.")
    if payload.get("feature_list_ordered") != FEATURES:
        raise ArtifactCompatibilityError("Artifact feature list/order does not match this app.")
    excluded_features = payload.get("excluded_features")
    if not isinstance(excluded_features, dict) or "Recovery_Time" not in excluded_features:
        raise ArtifactCompatibilityError("Metadata does not confirm Recovery_Time was excluded.")

    versions = payload.get("dependency_versions")
    if not isinstance(versions, dict):
        raise ArtifactCompatibilityError("Artifact dependency-version metadata is missing.")
    python_saved = str(payload.get("python_version", ""))
    python_current = platform.python_version()
    if not python_saved or python_saved.split(".")[:2] != python_current.split(".")[:2]:
        raise ArtifactCompatibilityError(
            f"Python compatibility mismatch: artifact uses {python_saved or 'unknown'}, "
            f"runtime is {python_current}; use the recorded Python major/minor version."
        )
    for package in ("scikit-learn", "numpy", "joblib"):
        saved = versions.get(package)
        if not saved:
            raise ArtifactCompatibilityError(f"Metadata does not record {package} version.")
        current = _installed_version(package)
        if current != saved:
            raise ArtifactCompatibilityError(
                f"{package} version mismatch: artifact uses {saved}, runtime has {current}. "
                "Install the compatible versions listed in README.md and restart the app."
            )

    try:
        pipeline = load_pipeline(model_file)
    except (FileNotFoundError, ArtifactLoadError):
        raise
    except Exception as exc:
        raise ArtifactLoadError(f"Could not load the trusted saved model: {exc}") from exc
    if type(pipeline.named_steps["classifier"]).__name__ != EXPECTED_MODEL:
        raise ArtifactCompatibilityError("Loaded estimator class does not match saved metadata.")
    if list(pipeline.classes_) != [0, 1]:
        raise ArtifactCompatibilityError("Loaded model class order must be exactly [0, 1].")
    return pipeline


@st.cache_resource(show_spinner=False)
def _cached_verified_artifact(model_path: str):
    """Cache one trusted local pipeline for the Streamlit session/process."""
    return load_verified_artifact(model_path)


def render_app(
    *,
    streamlit_module=st,
    model_loader=_cached_verified_artifact,
    predictor=predict_app_input,
) -> None:
    """Render the Streamlit UI; dependencies are injectable for UI tests."""
    ui = streamlit_module
    ui.set_page_config(page_title="Synthetic Injury-Label Demo", page_icon="🏃", layout="centered")
    ui.markdown(
        """<style>
        .stApp { background-color: #080b12; color: #f4f6fb; }
        h1, h2, h3, label { color: #f4f6fb !important; }
        [data-testid="stSidebar"] { background-color: #111827; }
        .stButton > button { background: #f97316; color: white; border: 0; font-weight: 700; }
        </style>""",
        unsafe_allow_html=True,
    )
    ui.title("Athlete Injury-Label Classification Demo")
    ui.error(
        "SYNTHETIC DATA — NOT MEDICAL ADVICE. This model classifies labels in a synthetic "
        "Kaggle dataset. It does not estimate real injury probability, provide clinical "
        "assessment, or recommend participation, treatment, or recovery."
    )
    ui.caption(
        "The target-generation rule and feature measurement definitions are undocumented. "
        "The displayed class-1 score is uncalibrated and applies only to this synthetic-label demo."
    )

    try:
        model = model_loader(str(DEFAULT_MODEL_PATH))
    except FileNotFoundError:
        ui.error(
            "The saved model is unavailable. Provision the trusted Phase 3A artifacts/ files "
            "including injury_classifier.joblib and metadata.json, then restart the app."
        )
        return
    except ArtifactLoadError as exc:
        LOGGER.exception("Model artifact could not be loaded or validated.")
        ui.error(f"The saved model is incompatible or invalid: {exc}")
        return
    except Exception:
        LOGGER.exception("Unexpected failure loading the saved model.")
        ui.error("The saved model could not be loaded. Check server logs and compatible versions.")
        return

    ui.subheader("Synthetic-demo inputs")
    ui.caption(
        "Observed dataset ranges are descriptive only—not medically valid ranges. Age is "
        "assumed years; weight kg and height cm follow the old UI labels but source units are "
        "unverified. Training intensity is an undefined numeric dataset scale represented here as 0–1."
    )
    ui.sidebar.header("Model input features")
    age = ui.sidebar.number_input(
        "Age (assumed years; observed dataset range 18–39)", min_value=1, max_value=120, value=28, step=1
    )
    weight = ui.sidebar.number_input(
        "Weight (assumed kg; observed dataset range 40.19–104.65)",
        min_value=0.1,
        value=75.0,
        step=0.1,
        format="%.2f",
    )
    height = ui.sidebar.number_input(
        "Height (assumed cm; observed dataset range 145.29–207.31)",
        min_value=0.1,
        value=180.0,
        step=0.1,
        format="%.2f",
    )
    previous_injuries = ui.sidebar.selectbox(
        "Previous injuries (dataset coding)", options=[0, 1], index=0
    )
    training_intensity = ui.sidebar.slider(
        "Training intensity (undefined demo scale: 0–1)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    if not ui.button("Classify synthetic label"):
        return
    try:
        input_frame, result, range_warnings = predictor(
            model, age, weight, height, previous_injuries, training_intensity
        )
        if range_warnings:
            ui.warning(
                "Input outside observed dataset bounds for: "
                + ", ".join(range_warnings)
                + ". The bounds are descriptive only; prediction may be extrapolative."
            )
        label = int(result.iloc[0]["predicted_synthetic_label"])
        score = float(result.iloc[0]["uncalibrated_class_1_score"])
        ui.subheader("Synthetic-label model output")
        ui.metric("Predicted synthetic label", str(label))
        ui.metric("Uncalibrated class-1 model score (0–1)", f"{score:.3f}")
        ui.caption(
            "This score is not a calibrated probability of injury. Output represents the "
            "model's class-1 score for a synthetic label only."
        )
    except InferenceInputError as exc:
        ui.error(f"Please correct the input values: {exc}")
    except Exception:
        LOGGER.exception("Synthetic-label inference failed.")
        ui.error("Prediction could not be completed. Check server logs; no heuristic fallback was used.")