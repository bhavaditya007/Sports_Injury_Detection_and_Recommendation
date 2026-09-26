# Model evaluation

## Scope and interpretation

All figures below are the **previously recorded Phase 3A results** for the Kaggle synthetic `Likelihood_of_Injury` labels. They are transcribed from the already reported run, not newly computed in this documentation phase. No training, cross-validation, holdout scoring, or inspection of saved prediction/metric artifacts was performed while preparing this page.

These metrics measure agreement with a synthetic label in a small dataset. They do not demonstrate real-world injury prediction, clinical validity, useful athlete risk ranking, or calibrated probabilities. The displayed class-1 model score is uncalibrated.

## Training-only cross-validation comparison

Five-fold stratified cross-validation on the 80% training partition; each value is mean ± standard deviation across folds. Candidate selection used highest mean average precision, with ties broken by higher mean ROC-AUC and then lexicographically ascending model name.

| Candidate | Average precision | ROC-AUC | Recall | Precision | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| DummyClassifier | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| LogisticRegression | 0.5002 ± 0.0349 | 0.4892 ± 0.0322 | 0.4775 ± 0.0185 | 0.4861 ± 0.0344 | 0.4814 ± 0.0230 | 0.4850 ± 0.0335 |
| RandomForestClassifier | 0.5025 ± 0.0183 | 0.4863 ± 0.0242 | 0.5025 ± 0.0324 | 0.4950 ± 0.0159 | 0.4984 ± 0.0216 | 0.4950 ± 0.0162 |
| ExtraTreesClassifier | **0.5305 ± 0.0316** | **0.5207 ± 0.0319** | 0.5100 ± 0.0634 | 0.5073 ± 0.0401 | 0.5078 ± 0.0468 | 0.5075 ± 0.0401 |
| HistGradientBoostingClassifier | 0.4913 ± 0.0240 | 0.4661 ± 0.0345 | 0.4550 ± 0.0512 | 0.4617 ± 0.0406 | 0.4580 ± 0.0442 | 0.4625 ± 0.0403 |

Under the predeclared training-only rule, `ExtraTreesClassifier` was selected. Its cross-validation performance is only modestly above the dummy average-precision baseline and has fold variability.

## Previously recorded final holdout results

The holdout was a stratified 20% partition of 200 rows and was evaluated once after model selection. These are the previously recorded values, **not a new evaluation**.

| Metric | Previously recorded result |
|---|---:|
| Average precision | 0.4992 |
| ROC-AUC | 0.5015 |
| Recall | 0.5700 |
| Precision | 0.5089 |
| F1 | 0.5377 |
| Balanced accuracy | 0.5100 |
| Specificity | 0.4500 |
| Positive-class prevalence | 0.5000 |
| Confusion matrix (rows true 0/1, columns predicted 0/1) | `[[45, 55], [43, 57]]` |

ROC-AUC and average precision are approximately 0.50, and balanced accuracy is 0.51. These holdout results are close to chance-level discrimination on this synthetic label. The class-1 score was not calibrated and must not be interpreted as an injury probability.

## Metric definitions

- **Average precision (PR-AUC summary):** summarizes precision/recall across score thresholds, weighting precision by recall increments.
- **ROC-AUC:** ranks positive labels above negative labels across thresholds; 0.5 is chance ranking in the binary setting.
- **Recall:** fraction of class-1 labels predicted as class 1.
- **Precision:** fraction of class-1 predictions that are class 1.
- **F1:** harmonic mean of precision and recall.
- **Balanced accuracy:** average recall over both classes.
- **Specificity:** fraction of class-0 labels predicted as class 0.
- **Positive-class prevalence:** share of rows with label 1.
- **Confusion matrix:** counts of true/predicted class combinations, with class order stated explicitly.

These are descriptive evaluation statistics for synthetic labels, not evidence for a safety-critical or medical use case.
