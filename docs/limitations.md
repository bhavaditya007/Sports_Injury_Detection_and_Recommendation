# Limitations and appropriate use

## Synthetic data and synthetic target

The source is the version-1 Kaggle Injury Prediction Dataset by MrSimple (`mrsimple07`). Its author describes it as synthetic and generated with NumPy and Pandas. The 0/1 `Likelihood_of_Injury` field is a synthetic label; the exact formula, threshold, meaning, and outcome horizon are unknown. A model trained here classifies that supplied label, not observed injury events.

## Unclear feature semantics and timing

- `Previous_Injuries` has no documented definition, provenance, or observation window.
- `Training_Intensity` is numeric and approximately 0–1 in the file, but its units, scale, and measurement period are undefined.
- `Recovery_Time` has unresolved units and timing and may be post-outcome; it is excluded from the current model.
- The source does not formally define age, weight, and height units or measurement protocols. Age in years, weight in kg, and height in cm are UI assumptions, not verified data facts.

## Model scores and performance

The class-1 output is an uncalibrated model score on a 0–1 scale. It is not a calibrated probability of injury. Previously reported test metrics are close to chance on synthetic labels; see [model evaluation](model-evaluation.md). No score threshold has been clinically or operationally validated.

## Generalization and external validation

The dataset contains 1,000 rows and has no athlete identifier, timestamps, sport, injury-event record, or documented sampling procedure. It cannot support player-grouped or temporal validation as supplied. The feature set is small, provenance is synthetic, and no external dataset or real-world validation exists. Observed sample ranges are descriptive, not medically valid or population-representative limits.

## Explicit non-use statement

This project has no demonstrated clinical validity and is not a medical prediction, diagnostic, treatment, training, participation, or recovery recommendation system. Do not use its outputs to assess an individual's real injury risk or make decisions about athletes. The Streamlit demo is an educational interface for a synthetic-label classifier only.
