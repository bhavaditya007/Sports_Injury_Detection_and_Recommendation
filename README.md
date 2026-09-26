# Synthetic Athlete Injury-Label Classification Demo

> **Educational demonstration using synthetic data only.** This project classifies a synthetic Kaggle label. It does not estimate real injury probability, have clinical validity, provide medical advice, or support athlete participation or recovery decisions.

## Overview

This repository demonstrates dataset validation, leakage-conscious preprocessing, stratified model comparison, artifact metadata, and a Streamlit interface for the Kaggle **Injury Prediction Dataset**. The saved pipeline predicts the supplied binary `Likelihood_of_Injury` label. The label-generation formula and its outcome horizon are undocumented, so reported performance is only agreement with synthetic labels from this dataset.

The app displays a predicted synthetic label and an **uncalibrated class-1 model score on a 0–1 scale**. That score is not a calibrated probability of injury.

## Features and architecture

- Schema validation and exact duplicate-row reporting (duplicates are retained; no athlete identity is inferred).
- Scikit-learn `Pipeline` and `ColumnTransformer` keep preprocessing within training folds.
- Stratified train/test split with seed 42; 5-fold stratified CV compares a dummy baseline and four classifiers using training data only.
- The pre-outcome feature set excludes `Recovery_Time` because its timing and meaning are unclear.
- A single canonical Streamlit UI lives in `src/injury_demo/streamlit_app.py`; root `app.py` is the thin launch entry point.
- Model, metadata, metrics, and predictions are generated locally and ignored by Git.

See [methodology](docs/methodology.md), [model evaluation](docs/model-evaluation.md), [limitations](docs/limitations.md), and the [repository guide](docs/repository-guide.md).

## Dataset and attribution

- Source: [Kaggle Injury Prediction Dataset](https://www.kaggle.com/datasets/mrsimple07/injury-prediction-dataset)
- Author: **MrSimple** (`mrsimple07`)
- Version: **1 — Initial release, 2024-02-24**
- Kaggle-listed dataset license: Apache 2.0. This applies to the dataset only; see [LICENSE](LICENSE) for the original project code notice.
- The dataset author describes the data as synthetic and generated with NumPy and Pandas.
- `data/injury_data.csv` is the verified version-1 dataset (1,000 rows, 7 columns).

See [data/README.md](data/README.md) for the schema and unresolved field definitions. The supporting workout CSV is retained locally at `data/archive/updated_bmi_workouts.csv` for the archived experiment, but it is excluded from Git because its redistribution rights and provenance have not been verified. It has no documented join key with the injury dataset.

## License

The original project source code is **All Rights Reserved** by Bhavaditya Lellapalli, copyright © 2026. Public visibility of this repository does not grant a general reuse license. No permission is granted to copy, modify, distribute, sublicense, or commercially reuse the original project code without prior written permission, except where applicable law provides otherwise. See [LICENSE](LICENSE).

This notice applies only to original project code. The Kaggle dataset, Python libraries, and any other third-party material are not claimed as original project code and remain subject to their respective owners' licenses and terms. In particular, Kaggle lists Apache 2.0 for the dataset; consult the dataset source and third-party package documentation for the applicable terms.

## Setup

The existing serialized model was created with Python **3.13.0**, scikit-learn **1.8.0**, NumPy **2.1.3**, and joblib **1.5.3**. The app checks Python major/minor and exact scikit-learn, NumPy, and joblib versions before loading the artifact. Use the pinned serving dependencies below for compatibility.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

`lightgbm` remains listed for the archived legacy notebook import only; its historical version was not recorded. The current pipeline and Streamlit app do not require LightGBM. The archived notebooks may require additional packages or adjustments to run in other environments.

The archived notebooks keep their original Colab `/content/...` dataset paths and saved outputs unchanged. The workout experiment requires an authorized local copy named `updated_bmi_workouts.csv` in its expected Colab location; that dataset is not supplied through GitHub while its redistribution rights remain unverified. The notebooks are historical references; the supported local training workflow is `scripts/train_model.py`.

## Provision the trusted model artifact

The model is intentionally not committed. `artifacts/` is Git-ignored apart from `.gitkeep`. A clean clone cannot run predictions until a trusted `injury_classifier.joblib` and its matching `metadata.json` are provisioned under `artifacts/` from the original Phase 3A run or another explicitly authorized trusted build. Keep both files from the same run. Do not load arbitrary or user-uploaded joblib files: deserialization can execute code. The app fails closed if the artifact, metadata, schema, or compatible runtime is unavailable.

A separate ignored `artifacts/metadata.sanitized.json` copy removes the local absolute dataset path while preserving the loader compatibility fields. The original local `metadata.json` was left unchanged. When provisioning a separate clean runtime, place the sanitized copy there under the expected name `metadata.json` beside the exact matching trusted model; do not overwrite the preserved original or commit either generated file. Do not distribute these files through an unapproved channel.

## Run the Streamlit demo

From the repository root, after provisioning the trusted artifact:

```powershell
python -m streamlit run app.py
```

The app uses the five fitted features in pipeline order: age, weight, height, previous-injury flag, and training intensity. Age/weight/height units are assumptions, and the training-intensity scale is undefined. Displayed data ranges are descriptive, not medically valid boundaries. No recovery-time input or rest-day recommendation is used.

## Screenshots

No screenshot is currently included: the available browser capture produced a cropped image unsuitable for a portfolio. See [docs/screenshots/README.md](docs/screenshots/README.md) for two genuine screenshots to capture manually from the running app: the input/disclaimer view and a demo output view. Use only synthetic demo inputs and do not present the score as an injury probability.

## Tests

Using the configured Python 3.13 interpreter, run:

```powershell
python -m unittest discover -s tests -v
```

The tests cover validation, pipeline construction, artifact writing/loading guards, app inputs, and mocked UI behavior. They do not establish external or clinical validity.

## Train a new model — not needed to launch the demo

**Do not run this command merely to start the app.** It creates a new model and new evaluation artifacts and performs model comparison plus one final holdout evaluation. The existing Phase 3A model and results are not reproduced by the launch command.

Only when intentionally starting a new training run:

```powershell
python scripts/train_model.py
```

The command reads `data/injury_data.csv` and writes the pipeline, metrics, predictions, and metadata under ignored `artifacts/`. It does not modify source data. A new run creates new evaluation results and should be treated as a distinct model version.

## Repository layout

```text
.
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
├── .gitattributes
├── src/
│   └── injury_demo/
│       ├── __init__.py
│       ├── data.py
│       ├── pipeline.py
│       ├── evaluation.py
│       ├── artifacts.py
│       ├── inference.py
│       └── streamlit_app.py
├── scripts/
│   └── train_model.py
├── tests/
│   ├── test_data.py
│   ├── test_pipeline.py
│   ├── test_artifacts.py
│   ├── test_inference_and_cli.py
│   └── test_streamlit_app.py
├── data/
│   ├── README.md
│   ├── injury_data.csv
│   └── archive/
│       └── updated_bmi_workouts.csv
├── notebooks/
│   └── archive/
│       ├── injury_classification_legacy_eda.ipynb
│       └── workout_regression_experiment.ipynb
├── docs/
│   ├── methodology.md
│   ├── model-evaluation.md
│   ├── limitations.md
│   ├── repository-guide.md
│   └── screenshots/
│       └── README.md
└── artifacts/
    └── .gitkeep
```
