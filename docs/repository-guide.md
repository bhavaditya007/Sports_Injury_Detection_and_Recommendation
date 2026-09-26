# Repository guide

## Top-level files

- `README.md` — project overview, setup, source attribution, run commands, and documentation links.
- `app.py` — root-level Streamlit launch entry point for the canonical UI implementation.
- `requirements.txt` — pinned serving/model dependencies plus an unpinned legacy-notebook-only LightGBM dependency.
- `.gitignore` — excludes local environments, caches, secrets, and generated model/evaluation artifacts while allowing source datasets and `artifacts/.gitkeep`.
- `.gitattributes` — specifies LF checkout normalization for the root app launcher.

## Directories and canonical files

- `src/injury_demo/` — canonical Python package. `data.py` validates source tables; `pipeline.py` builds the estimator pipelines; `evaluation.py` performs CV/model selection and final evaluation; `artifacts.py` writes training outputs and metadata; `inference.py` validates/loads a fitted pipeline and predicts the supplied synthetic label; `streamlit_app.py` contains the one canonical UI and app-facing helpers.
- `scripts/train_model.py` — explicit training CLI. Running it starts a new model-comparison and evaluation run and writes new artifacts; it is not needed to launch the app.
- `tests/` — `unittest` suite for schema handling, pipeline construction, artifact operations, inference, and mocked UI behavior.
- `data/injury_data.csv` — canonical Kaggle version-1 synthetic injury-label dataset. `data/README.md` documents provenance and schema.
- `data/archive/updated_bmi_workouts.csv` — supporting CSV for the archived workout experiment. Its original basename is retained because the archived notebook expects that Colab upload name. It is not joined to the injury data.
- `notebooks/archive/` — preserved legacy notebooks. `injury_classification_legacy_eda.ipynb` contains earlier exploratory modeling; `workout_regression_experiment.ipynb` contains a separate demonstration using random regression data. Neither is canonical pipeline code.
- `docs/` — focused pages for [methodology](methodology.md), [model evaluation](model-evaluation.md), [limitations](limitations.md), this guide, and screenshot instructions.
- `artifacts/` — local generated model and evaluation outputs. Only `.gitkeep` is intended for source control; model, metadata, metrics, and prediction files are ignored.

New training metadata records the dataset path relative to the repository when possible. For data supplied from outside the repository, it records only the basename to avoid embedding a machine-specific absolute path. The existing local Phase 3A `metadata.json` is preserved unchanged and ignored by Git; a separate ignored `artifacts/metadata.sanitized.json` copy has its personal path removed and passes the loader checks. In a separate trusted runtime directory, that sanitized copy can be provisioned under the expected filename `metadata.json` alongside its matching model. Do not commit generated artifacts.

## App launch

Install the pinned dependencies and provision a trusted `injury_classifier.joblib` with its matching `metadata.json` from the same authorized Phase 3A run under `artifacts/`. Then, from the repository root, launch with:

```powershell
python -m streamlit run app.py
```

A clean clone does not include the generated model. The app validates metadata/runtime compatibility and fails closed when model files are missing or incompatible. Never load arbitrary user-provided joblib files because deserialization can execute code.

## Tests

From the repository root, run:

```powershell
python -m unittest discover -s tests -v
```

The suite does not establish external or clinical validity. Launching the app requires the trusted artifact; most UI behaviors are tested with mocks.

The archived notebooks retain their original Colab `/content/...` dataset paths so their saved notebook contents remain unchanged. They may need manual path or environment adjustments before interactive local execution; the canonical locally supported training path is documented above.

## Training (separate, consequential operation)

`python scripts/train_model.py` creates a new model and evaluation artifacts. Do not run it just to launch the demo or when preserving the existing Phase 3A artifact/results is required. It is the only canonical training entry point.

## Archived data and license notes

The Excel workout copy is preserved in a local backup outside the repository; it is not an active project dataset. The Kaggle dataset's Apache 2.0 listing applies to that dataset only. Original project code is All Rights Reserved under the root `LICENSE`; third-party materials retain their own terms.
