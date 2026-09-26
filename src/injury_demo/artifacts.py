"""Metadata collection and staged, atomically replaced model artifact writes."""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .data import FEATURES, TARGET
from .evaluation import CV_FOLDS, SEED, SELECTION_RULE, TEST_SIZE

DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "joblib": "joblib",
    "streamlit": "streamlit",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "lightgbm": "lightgbm",
}


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def build_metadata(
    *,
    dataset_path: Path,
    dataset_sha256: str,
    duplicate_rows: int,
    row_count: int,
    selected_name: str,
    selected_pipeline,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
    cv_folds: int = CV_FOLDS,
) -> dict:
    resolved_dataset_path = Path(dataset_path).resolve()
    repository_root = Path(__file__).resolve().parents[2]
    try:
        recorded_dataset_path = str(resolved_dataset_path.relative_to(repository_root))
        dataset_path_scope = "repository-relative"
    except ValueError:
        recorded_dataset_path = resolved_dataset_path.name
        dataset_path_scope = "external-basename-only"

    versions = {}
    for import_name, distribution_name in DEPENDENCIES.items():
        try:
            versions[import_name] = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            versions[import_name] = None

    return {
        "artifact_schema_version": 1,
        "project_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "dependency_versions": versions,
        "dataset": {
            "path_at_training": recorded_dataset_path,
            "path_reference_scope": dataset_path_scope,
            "source": "Kaggle Injury Prediction Dataset by mrsimple07, version 1",
            "sha256": dataset_sha256,
            "rows": row_count,
            "exact_duplicate_rows_retained": duplicate_rows,
        },
        "task": "classification of the supplied synthetic Likelihood_of_Injury label",
        "target": TARGET,
        "positive_class": 1,
        "feature_list_ordered": list(FEATURES),
        "excluded_features": {"Recovery_Time": "Unclear measurement timing; possible post-outcome leakage."},
        "random_seed": seed,
        "split": {"method": "train_test_split with stratify=y", "test_size": test_size},
        "cross_validation": {
            "method": "StratifiedKFold(shuffle=True)",
            "folds": cv_folds,
            "seed": seed,
            "selection_rule": SELECTION_RULE,
        },
        "selected_model": selected_name,
        "selected_estimator_parameters": _json_safe(
            selected_pipeline.named_steps["classifier"].get_params(deep=False)
        ),
        "safety_limitation": "Synthetic-label demonstration only; not observed injury prediction, clinical validity, or calibrated probability.",
    }


def write_artifacts(
    output_dir: str | Path,
    *,
    fitted_pipeline,
    metrics: dict,
    predictions: pd.DataFrame,
    metadata_payload: dict,
) -> list[Path]:
    """Stage all outputs, then atomically replace each destination after staging succeeds."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".phase3a-staging-", dir=destination))
    names = ["injury_classifier.joblib", "metrics.json", "predictions.csv", "metadata.json"]
    try:
        joblib.dump(fitted_pipeline, staging / names[0])
        for filename, payload in ((names[1], metrics), (names[3], metadata_payload)):
            with (staging / filename).open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
        predictions.to_csv(staging / names[2], index=False, lineterminator="\n")
        for filename in names:
            if not (staging / filename).is_file() or (staging / filename).stat().st_size == 0:
                raise OSError(f"Artifact staging produced an empty or missing file: {filename}")

        for filename in names:
            os.replace(staging / filename, destination / filename)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return [destination / filename for filename in names]