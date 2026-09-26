"""Dataset loading and validation for the synthetic Kaggle classification demo."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Likelihood_of_Injury"
FEATURES = [
    "Player_Age",
    "Player_Weight",
    "Player_Height",
    "Previous_Injuries",
    "Training_Intensity",
]
EXCLUDED_FEATURES = ["Recovery_Time"]
EXPECTED_COLUMNS = [
    "Player_Age",
    "Player_Weight",
    "Player_Height",
    "Previous_Injuries",
    "Training_Intensity",
    "Recovery_Time",
    TARGET,
]


class DatasetValidationError(ValueError):
    """Raised when the source table does not meet the documented schema."""


def validate_dataset(frame: pd.DataFrame) -> int:
    """Validate the expected schema and return (but never remove) exact duplicates."""
    if not isinstance(frame, pd.DataFrame):
        raise DatasetValidationError("Dataset must be loaded as a pandas DataFrame.")
    if frame.empty:
        raise DatasetValidationError("Dataset contains no rows.")

    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise DatasetValidationError("Invalid dataset schema (" + "; ".join(details) + ").")

    if frame[TARGET].isna().any():
        raise DatasetValidationError(f"Target {TARGET!r} contains missing values.")
    if not pd.api.types.is_numeric_dtype(frame[TARGET]):
        raise DatasetValidationError(f"Target {TARGET!r} must be numeric with labels 0 and 1.")
    target_values = set(frame[TARGET].unique().tolist())
    if not target_values.issubset({0, 1}) or len(target_values) != 2:
        raise DatasetValidationError(
            f"Target {TARGET!r} must contain both binary labels 0 and 1; found {sorted(target_values)}."
        )

    for column in FEATURES + EXCLUDED_FEATURES:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise DatasetValidationError(f"Feature column {column!r} must be numeric.")
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise DatasetValidationError(f"Feature column {column!r} contains infinite values.")
        if np.isnan(values).all():
            raise DatasetValidationError(f"Feature column {column!r} is entirely missing.")

    duplicates = int(frame.duplicated().sum())
    if duplicates:
        warnings.warn(
            f"Found {duplicates} exact duplicate rows. They are retained; without athlete IDs, "
            "they are not assumed to represent duplicate athletes.",
            UserWarning,
            stacklevel=2,
        )
    return duplicates


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, int]:
    """Load the CSV, validate its schema, and report exact duplicate-row count."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {source}")
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise DatasetValidationError(f"Could not read CSV dataset {source}: {exc}") from exc
    duplicate_count = validate_dataset(frame)
    return frame, duplicate_count


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's original bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()