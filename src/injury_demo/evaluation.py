"""Train-only cross-validation comparison and single final holdout evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .data import FEATURES, TARGET
from .pipeline import build_candidate_pipelines

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SELECTION_RULE = (
    "Choose the candidate with the highest mean training-only cross-validated average precision; "
    "break exact ties by higher mean ROC-AUC, then lexicographically ascending model name. "
    "The held-out test set is not used for model selection."
)


def stratified_holdout(frame: pd.DataFrame, seed: int = SEED) -> tuple:
    """Create the one fixed stratified holdout split from validated rows."""
    X = frame[FEATURES].copy()
    y = frame[TARGET].copy()
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )


def _scoring() -> dict[str, Any]:
    return {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "balanced_accuracy": "balanced_accuracy",
    }


def compare_candidates(X_train, y_train, seed: int = SEED, folds: int = CV_FOLDS) -> dict:
    """Compare fixed pipelines using stratified CV on training rows only."""
    if folds < 2:
        raise ValueError("Cross-validation requires at least two folds.")
    class_counts = pd.Series(y_train).value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < folds:
        raise ValueError(
            f"Each binary class must have at least {folds} training examples for StratifiedKFold."
        )

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = _scoring()
    results: dict[str, dict] = {}
    for name, estimator in build_candidate_pipelines(seed).items():
        scores = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
            n_jobs=1,
        )
        fold_metrics = []
        for fold_index in range(folds):
            fold_metrics.append(
                {metric: float(scores[f"test_{metric}"][fold_index]) for metric in scoring}
            )
        summary = {}
        for metric in scoring:
            values = np.asarray(scores[f"test_{metric}"], dtype=float)
            summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
        results[name] = {"fold_metrics": fold_metrics, "summary": summary}

    selected = sorted(
        results,
        key=lambda name: (
            -results[name]["summary"]["average_precision"]["mean"],
            -results[name]["summary"]["roc_auc"]["mean"],
            name,
        ),
    )[0]
    return {
        "candidates": results,
        "selected_model": selected,
        "selection_rule": SELECTION_RULE,
        "cv_folds": folds,
        "cv_seed": seed,
    }


def evaluate_holdout_once(estimator, X_test, y_test) -> tuple[dict, pd.DataFrame]:
    """Fit the selected pipeline on training data before calling; score holdout once."""
    classes = np.asarray(estimator.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if len(positive_positions) != 1:
        raise ValueError("Fitted classifier does not expose the expected positive class label 1.")
    positive_position = int(positive_positions[0])

    if not hasattr(estimator, "predict_proba"):
        raise ValueError("Selected estimator must provide class scores for average precision and ROC-AUC.")
    class_scores = np.asarray(estimator.predict_proba(X_test))
    positive_scores = class_scores[:, positive_position]
    predictions = classes[np.argmax(class_scores, axis=1)]
    score_kind = "uncalibrated_class_1_score"

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics = {
        "average_precision": float(average_precision_score(y_test, positive_scores)),
        "roc_auc": float(roc_auc_score(y_test, positive_scores)),
        "recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "positive_class_prevalence": float(np.mean(np.asarray(y_test) == 1)),
        "confusion_matrix_labels_0_then_1": [[tn, fp], [fn, tp]],
        "test_rows": int(len(y_test)),
        "positive_score_field": score_kind,
        "interpretation": "Performance against synthetic labels in the held-out rows only; not real-world injury performance or calibrated risk.",
    }
    holdout_predictions = pd.DataFrame(
        {
            "source_row_index": list(X_test.index),
            "y_true": np.asarray(y_test),
            "y_pred": predictions,
            score_kind: positive_scores,
        }
    )
    return metrics, holdout_predictions