"""Leakage-safe preprocessing and candidate classifier construction."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

from .data import FEATURES


def build_pipeline(classifier, feature_names: list[str] | None = None) -> Pipeline:
    """Build a complete estimator with all learned preprocessing inside the pipeline."""
    columns = list(feature_names or FEATURES)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_candidate_pipelines(seed: int) -> dict[str, Pipeline]:
    """Return fixed candidate configurations; all stochastic estimators are seeded."""
    candidates = {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=seed),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=1
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=300, random_state=seed, n_jobs=1
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=100, random_state=seed
        ),
    }
    return {name: build_pipeline(model) for name, model in candidates.items()}