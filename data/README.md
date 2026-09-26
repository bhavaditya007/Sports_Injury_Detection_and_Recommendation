# Data documentation

## Canonical classification dataset

- **File:** `injury_data.csv`
- **Source:** [Kaggle Injury Prediction Dataset](https://www.kaggle.com/datasets/mrsimple07/injury-prediction-dataset)
- **Author:** MrSimple (`mrsimple07`)
- **Version:** 1, Initial release, 2024-02-24
- **Kaggle-listed license:** Apache 2.0 (dataset license only; not this project's source-code license)
- **Shape:** 1,000 rows × 7 columns
- **Source characterization:** The author describes it as synthetic and generated with NumPy and Pandas. It is not verified observed athlete or injury-event data.

The local data rows match Kaggle version 1; raw line endings differ. The exact target-generation formula and outcome horizon are undocumented.

## Schema

Types shown are as read from the CSV. Field names indicate apparent roles, but the source provides no complete measurement dictionary.

| Column | Type | Observed values / apparent role | Definition status |
|---|---|---|---|
| `Player_Age` | integer | 18–39; candidate feature | Units/collection protocol not documented |
| `Player_Weight` | float | approximately 40.19–104.65; candidate feature | Units/collection protocol not documented |
| `Player_Height` | float | approximately 145.29–207.31; candidate feature | Units/collection protocol not documented |
| `Previous_Injuries` | integer | 0/1; candidate feature | History definition and provenance not documented |
| `Training_Intensity` | float | approximately 0–1; candidate feature | Scale and measurement window not documented |
| `Recovery_Time` | integer | 1–6; excluded from current model | Units and timing relative to an outcome are unresolved; possible post-outcome leakage |
| `Likelihood_of_Injury` | integer | 0/1; 500 records per value in the source file | Synthetic target; generation formula, interpretation, and horizon undocumented |

The dataset has no explicit athlete ID, timestamp, sport, or observed injury-event field. Age in years, weight in kilograms, and height in centimeters are assumptions used for the demo UI, not verified source definitions. Observed feature ranges are descriptive only, not medically valid boundaries.

## Archived supporting workout data

`archive/updated_bmi_workouts.csv` is retained locally solely for `../notebooks/archive/workout_regression_experiment.ipynb` and is explicitly excluded from Git. Its redistribution rights and provenance have not been verified, so do not publish or redistribute it until those rights are confirmed. It has 1,000 rows and five columns: `Player_Weight`, `Player_Height`, `Player BMI`, `Workout_Days`, and `Training_Intensity`. It has no documented shared record key with `injury_data.csv`; do not join the tables by row position or similar measurements. The redundant Excel representation is preserved in a local backup outside the repository and is not an active project dataset.

The archived notebook retains its original `/content/updated_bmi_workouts.csv` path. To reproduce that historical experiment, obtain an authorized copy and supply it locally under the expected name; the repository intentionally does not provide the CSV to GitHub.

## Modeling caution

A classifier trained on `Likelihood_of_Injury` models the supplied synthetic labels only. These data do not establish prediction of observed future injuries. Do not use the dataset, model, or app for diagnosis, medical risk assessment, player selection, treatment, or recovery decisions. See [limitations](../docs/limitations.md).
