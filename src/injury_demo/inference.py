"""Safe schema checks and inference helpers for the saved synthetic-label pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .data import FEATURES


class InferenceInputError(ValueError):
    """Raised when prediction input does not match the documented feature schema."""


class ArtifactLoadError(RuntimeError):
    """Raised when a serialized estimator is unreadable or has an unexpected structure."""


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a trusted local joblib artifact and check its fitted pipeline structure.

    Joblib files can execute code while unpickling. Only load artifacts from trusted sources.
    """
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {artifact_path}")
    try:
        estimator = joblib.load(artifact_path)
    except Exception as exc:
        raise ArtifactLoadError(f"Could not load model artifact {artifact_path}: {exc}") from exc

    if not isinstance(estimator, Pipeline):
        raise ArtifactLoadError("Model artifact must contain a scikit-learn Pipeline.")
    if list(estimator.named_steps) != ["preprocessor", "classifier"]:
        raise ArtifactLoadError("Pipeline must contain preprocessor then classifier steps.")
    preprocessor = estimator.named_steps["preprocessor"]
    if not isinstance(preprocessor, ColumnTransformer):
        raise ArtifactLoadError("Pipeline preprocessor must be a ColumnTransformer.")

    configured_features = []
    for _, _, columns in preprocessor.transformers:
        if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
            configured_features.extend(list(columns))
    if configured_features != FEATURES:
        raise ArtifactLoadError(
            f"Pipeline feature order/schema mismatch: expected {FEATURES}, found {configured_features}."
        )
    try:
        check_is_fitted(preprocessor)
        check_is_fitted(estimator.named_steps["classifier"])
    except Exception as exc:
        raise ArtifactLoadError("Model artifact contains an unfitted pipeline.") from exc
    return estimator


def validate_inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate exact ordered feature names and numeric finite-or-missing inputs."""
    if not isinstance(frame, pd.DataFrame):
        raise InferenceInputError("Inference input must be a pandas DataFrame.")
    if frame.empty:
        raise InferenceInputError("Inference input contains no rows.")
    if list(frame.columns) != FEATURES:
        missing = [name for name in FEATURES if name not in frame.columns]
        unexpected = [name for name in frame.columns if name not in FEATURES]
        raise InferenceInputError(
            f"Expected exactly these columns in this order: {FEATURES}; "
            f"missing={missing}, unexpected={unexpected}, received={list(frame.columns)}."
        )

    validated = frame.copy()
    for column in FEATURES:
        if not pd.api.types.is_numeric_dtype(validated[column]):
            raise InferenceInputError(f"Feature {column!r} must be numeric.")
        values = validated[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise InferenceInputError(f"Feature {column!r} contains an infinite value.")
        if np.isnan(values).all():
            raise InferenceInputError(f"Feature {column!r} is entirely missing.")
    return validated


def predict_synthetic_label(pipeline: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Return labels and explicitly uncalibrated class-1 scores for supplied fixture/input rows."""
    validated = validate_inference_frame(frame)
    classes = np.asarray(pipeline.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if len(positive_positions) != 1:
        raise ArtifactLoadError("Fitted pipeline must expose positive class label 1.")
    if not hasattr(pipeline, "predict_proba"):
        raise ArtifactLoadError("Fitted pipeline does not expose predict_proba.")

    scores = np.asarray(pipeline.predict_proba(validated))
    predictions = classes[np.argmax(scores, axis=1)]
    positive_scores = scores[:, int(positive_positions[0])]
    return pd.DataFrame(
        {
            "predicted_synthetic_label": predictions,
            "uncalibrated_class_1_score": positive_scores,
        },
        index=validated.index,
    )