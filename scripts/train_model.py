"""Run the reproducible Phase 3A synthetic-label classification workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from injury_demo.artifacts import build_metadata, write_artifacts  # noqa: E402
from injury_demo.data import FEATURES, TARGET, load_dataset, sha256_file  # noqa: E402
from injury_demo.evaluation import (  # noqa: E402
    CV_FOLDS,
    SEED,
    compare_candidates,
    evaluate_holdout_once,
    stratified_holdout,
)
from injury_demo.pipeline import build_candidate_pipelines  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare classifiers for the supplied synthetic injury label; not medical advice."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "injury_data.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def run(data_path: Path, output_dir: Path, seed: int = SEED) -> list[Path]:
    frame, duplicate_count = load_dataset(data_path)
    if frame[TARGET].value_counts().min() < 2:
        raise ValueError("Need at least two rows in each target class for a stratified holdout split.")

    X_train, X_test, y_train, y_test = stratified_holdout(frame, seed=seed)
    if y_train.value_counts().min() < CV_FOLDS:
        raise ValueError(
            f"Each training class needs at least {CV_FOLDS} rows for the configured stratified CV."
        )

    cv_results = compare_candidates(X_train[FEATURES], y_train, seed=seed, folds=CV_FOLDS)
    selected_name = cv_results["selected_model"]
    selected_pipeline = build_candidate_pipelines(seed)[selected_name]
    selected_pipeline.fit(X_train[FEATURES], y_train)

    # This is the workflow's sole access to held-out labels for model scoring.
    holdout_metrics, predictions = evaluate_holdout_once(
        selected_pipeline, X_test[FEATURES], y_test
    )
    report = {
        "evaluation_scope": "synthetic labels only",
        "selection_rule": cv_results["selection_rule"],
        "selected_model": selected_name,
        "cross_validation": {
            "method": "StratifiedKFold",
            "folds": cv_results["cv_folds"],
            "seed": cv_results["cv_seed"],
            "selection_metric": "mean average_precision; ties by mean ROC-AUC then model name",
            "candidates": cv_results["candidates"],
        },
        "holdout": holdout_metrics,
    }
    metadata_payload = build_metadata(
        dataset_path=data_path,
        dataset_sha256=sha256_file(data_path),
        duplicate_rows=duplicate_count,
        row_count=len(frame),
        selected_name=selected_name,
        selected_pipeline=selected_pipeline,
        seed=seed,
    )
    return write_artifacts(
        output_dir,
        fitted_pipeline=selected_pipeline,
        metrics=report,
        predictions=predictions,
        metadata_payload=metadata_payload,
    )


def main() -> int:
    args = parse_args()
    try:
        created = run(args.data, args.output_dir, seed=args.seed)
    except Exception as exc:
        print(f"Training workflow failed; no new artifacts were committed: {exc}", file=sys.stderr)
        return 1
    print("Completed synthetic-label evaluation. This is not real-world injury prediction.")
    metadata_payload = json.loads(created[3].read_text(encoding="utf-8"))
    print(
        "Exact duplicate rows retained: "
        f"{metadata_payload['dataset']['exact_duplicate_rows_retained']}"
    )
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())