# Reproducible OULAD Early-Risk Prediction Study

**Manuscript:** *Predicting At Risk Student Performance at the First Quarter Semester Boundary Using Ensemble Learning*

This repository reconstructs the analysis around a strict temporal prediction protocol using the public **Open University Learning Analytics Dataset (OULAD)**. OULAD contains anonymised information on students, course presentations, assessments and VLE interactions across seven selected modules. The official OULAD page provides the dataset and documentation; the dataset is also archived on Figshare under CC BY 4.0. 

Official dataset:
- https://analyse.kmi.open.ac.uk/open_dataset
- https://doi.org/10.6084/m9.figshare.5081998

Citation:
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171

## Study design

The primary estimand is predictive performance at the **first-quarter boundary** of each module presentation:

`T25 = floor(0.25 × module_presentation_length)`

Only information available on or before `T25` is eligible as a predictor. In particular:

- VLE events must have `date <= T25`.
- An assessment is eligible only when its scheduled date is `<= T25`.
- A student assessment score/submission is eligible only when the student's `date_submitted <= T25` and the associated assessment is eligible.
- The final outcome is never used as a predictor.
- Student IDs are used only for grouping and leakage auditing, not as predictive features.

### Target

`at_risk = 1`: `fail` or `withdraw`

`at_risk = 0`: `pass` or `distinction`

Unmapped outcomes are excluded and reported rather than silently recoded.

### Leakage control

The experiment uses **StratifiedGroupKFold**. The group is `id_student`, preventing the same learner from appearing in both training and validation folds. All imputation, encoding, scaling and SMOTE operations are fitted within each training fold. Hyperparameters are selected using an inner grouped cross-validation loop.

### Models

- Majority-class baseline
- Logistic regression
- Decision tree
- Random forest
- XGBoost
- LightGBM

Primary performance metrics:
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion-matrix counts
- Runtime

F1 is the primary optimization metric because the study focuses on identifying at-risk learners while retaining precision.

## Ablation design

Four controlled configurations are evaluated:

A. Static features + default random forest + no SMOTE  
B. A + early dynamic/engineered features  
C. B + fold-local SMOTE  
D. C + nested hyperparameter optimisation

This allows the manuscript to separate the effects of feature engineering, class balancing and optimisation.

## Statistical analysis

Outer-fold results are paired by fold. Pairwise model and ablation comparisons use two-sided Wilcoxon signed-rank tests, with Holm correction for multiple comparisons. With five outer folds, inferential results are treated as limited evidence rather than definitive population-level claims.

## Reproducibility

Run scripts in this order:

```text
python src/01_data_loading.py
python src/02_temporal_filtering.py
python src/03_feature_engineering.py
python src/04_target_construction.py
python src/05_preprocessing.py
python src/06_smote_pipeline.py
python src/07_model_training.py
python src/08_ablation.py
python src/09_statistical_tests.py
python src/10_figures_tables.py
python src/11_interpretability.py
```

Install:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Place the seven OULAD CSV files in `data/raw/`. The raw data are intentionally excluded from version control.

## Important interpretation rule

Feature importance is interpreted as **predictive contribution/association**, not a causal determinant. No live institutional deployment or user evaluation is claimed by this repository.

## Expected outputs

The `results/` directory is populated with:

- data inventory and temporal audits
- first-quarter cohort and feature files
- target distribution
- feature dictionary
- nested-CV fold results
- out-of-fold predictions
- model summary with mean ± SD
- ablation results
- corrected statistical comparisons
- publication tables
- figures

The repository does not contain fabricated performance values. Results in the manuscript should be updated only after the pipeline has been executed on the public OULAD data.

## Reproducibility note

The OULAD raw files are public and anonymised. Do not commit the raw files if repository policy or data terms do not permit redistribution. The repository should contain code, configuration, documentation and derived non-sensitive summaries rather than a second copy of the raw dataset.
