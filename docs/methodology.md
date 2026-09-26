# Methodology

## Dataset provenance and schema

The project uses version 1 (Initial release, 2024-02-24) of the [Kaggle Injury Prediction Dataset](https://www.kaggle.com/datasets/mrsimple07/injury-prediction-dataset), authored by MrSimple (`mrsimple07`). The source author identifies it as synthetic and generated with NumPy and Pandas. The local data contains 1,000 rows and seven fields:

- `Player_Age`
- `Player_Weight`
- `Player_Height`
- `Previous_Injuries`
- `Training_Intensity`
- `Recovery_Time`
- `Likelihood_of_Injury` (binary synthetic label)

The exact label-generation rule, outcome horizon, and operational definitions/measurement timing of several fields are undocumented. Dataset details are in [data/README.md](../data/README.md).

## Target and features

The current task is strictly classification of the supplied synthetic `Likelihood_of_Injury` label. The model uses `Player_Age`, `Player_Weight`, `Player_Height`, `Previous_Injuries`, and `Training_Intensity`, in that order. `Recovery_Time` is excluded because its timing is unresolved and it could be post-outcome information. `Previous_Injuries` and `Training_Intensity` are retained as supplied but their semantics and time windows are not established. Age, weight, and height units are assumptions, not confirmed dataset metadata.

## Validation and preprocessing

`src/injury_demo/data.py` validates the expected columns, numeric fields, finite values, and presence of both binary target classes. It reports exact duplicate rows and warns while retaining them; without athlete identifiers, duplicates are not interpreted as duplicate athletes. Partially missing feature values are allowed and handled by the model pipeline; missing target values and fully missing features are rejected.

`src/injury_demo/pipeline.py` constructs each estimator as a scikit-learn `Pipeline` containing a `ColumnTransformer`. Numeric features are median-imputed and standardized. All learned preprocessing is fit inside each training fold and later on the full training partition for the chosen estimator. No preprocessing is fit on the complete dataset before validation.

## Split, model comparison, and final fit

`src/injury_demo/evaluation.py` uses a stratified 80/20 train/test split with random seed 42. The held-out partition is reserved until model selection is complete. On the training partition, five-fold shuffled `StratifiedKFold` cross-validation compares `DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`, `ExtraTreesClassifier`, and `HistGradientBoostingClassifier`.

The selection rule is fixed: highest mean cross-validated average precision; ties use higher mean ROC-AUC and then lexicographically ascending model name. The selected full pipeline is fit on the training partition. The test set is used once for final reporting; it is not used to tune a model or threshold. The previously recorded Phase 3A values are documented in [model-evaluation.md](model-evaluation.md); this document does not recompute them.

## Reproducibility and artifacts

The training script is `scripts/train_model.py`. It records the dataset byte-level SHA-256, runtime/dependency versions, seed, split/CV settings, feature list, excluded fields, selection rule, and selected estimator configuration in metadata. Generated files are written under `artifacts/`, which is ignored by Git except for `.gitkeep`. Training is an explicit operation and is not needed merely to launch the app.

## Historical notebook methodology

`notebooks/archive/injury_classification_legacy_eda.ipynb` preserves an earlier exploratory notebook. It engineered BMI and age groups, encoded categories before its single random 90/10 split, trained earlier model candidates, and selected using the same test set. That workflow is historical and not the canonical leakage-conscious pipeline described above.

`notebooks/archive/workout_regression_experiment.ipynb` is an unrelated experiment: it plots the workout table but fits its linear regression to 50 randomly generated rows. It is not part of injury-label model training. Both original notebooks are archived byte-for-byte with their saved outputs preserved. Their original Colab `/content/...` data paths are intentionally retained; they are historical archives, not the canonical local workflow. The active training CLI uses the current `data/` path.
