"""08. Ablation study: isolate feature engineering, SMOTE and tuning.

Four configurations:
A Baseline/static features, no SMOTE, default model
B + engineered early behavioural/assessment features
C B + fold-local SMOTE
D C + nested hyperparameter optimisation

The ablation is run with grouped 5-fold CV and produces paired fold results.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from common import load_config, set_seed, save_csv

from sklearn.pipeline import Pipeline as SKPipeline

def make_pre(X):
    cat = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", SKPipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", SKPipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat)
    ])

def main():
    cfg = load_config()
    seed = cfg["random_seed"]
    set_seed(seed)
    df = pd.read_csv("results/04_modeling_dataset.csv")

    keys = ["code_module","code_presentation","id_student","final_result",
            "at_risk","target_exclusion_reason","feature_boundary_day"]
    y = df["at_risk"].astype(int)
    groups = df["id_student"].astype(str)

    # Fallback baseline features available in the dataset
    baseline_cols = ["module_presentation_length", "boundary_day"]
    existing_baseline = [c for c in baseline_cols if c in df.columns]
    if not existing_baseline:
        existing_baseline = [df.columns[3]] # fallback to first available numeric column

    X_all = df.drop(columns=keys, errors="ignore")
    X_static = df[existing_baseline].copy()

    configurations = {
        "A_static_default_no_smote": (X_static, False, False),
        "B_engineered_default_no_smote": (X_all, False, False),
        "C_engineered_default_smote": (X_all, True, False),
        "D_engineered_tuned_smote": (X_all, True, True)
    }

    outer = StratifiedGroupKFold(n_splits=cfg["outer_folds"], shuffle=True,
                                 random_state=seed)
    rows = []

    for name, (X, use_smote, tuned) in configurations.items():
        for fold, (tr, va) in enumerate(outer.split(X, y, groups), 1):
            model = RandomForestClassifier(
                n_estimators=400, random_state=seed, n_jobs=cfg["n_jobs"]
            )
            steps = [("preprocess", make_pre(X))]
            if use_smote:
                steps.append(("smote", SMOTE(
                    sampling_strategy="auto", k_neighbors=5,
                    random_state=seed + fold
                )))
            steps.append(("model", model))
            pipe = Pipeline(steps)

            if tuned:
                dist = {
                    "model__n_estimators": [300, 500, 700],
                    "model__max_depth": [None, 10, 20, 30],
                    "model__min_samples_leaf": [1, 2, 5],
                    "model__max_features": ["sqrt","log2"]
                }
                inner = StratifiedGroupKFold(
                    n_splits=cfg["inner_folds"], shuffle=True,
                    random_state=seed + fold
                )
                estimator = RandomizedSearchCV(
                    pipe, dist, n_iter=6, scoring="f1",
                    cv=list(inner.split(X.iloc[tr], y.iloc[tr], groups.iloc[tr])),
                    n_jobs=cfg["n_jobs"], random_state=seed + fold
                )
            else:
                estimator = pipe

            t0 = time.perf_counter()
            estimator.fit(X.iloc[tr], y.iloc[tr])
            elapsed = time.perf_counter() - t0
            pred = estimator.predict(X.iloc[va])
            prob = estimator.predict_proba(X.iloc[va])[:,1]

            rows.append({
                "configuration": name,
                "fold": fold,
                "precision": precision_score(y.iloc[va], pred, zero_division=0),
                "recall": recall_score(y.iloc[va], pred, zero_division=0),
                "f1": f1_score(y.iloc[va], pred, zero_division=0),
                "roc_auc": roc_auc_score(y.iloc[va], prob),
                "fit_seconds": elapsed
            })

    result = pd.DataFrame(rows)
    save_csv(result, "08_ablation_fold_results.csv")
    summary = result.groupby("configuration").agg(
        precision_mean=("precision","mean"), precision_sd=("precision","std"),
        recall_mean=("recall","mean"), recall_sd=("recall","std"),
        f1_mean=("f1","mean"), f1_sd=("f1","std"),
        roc_auc_mean=("roc_auc","mean"), roc_auc_sd=("roc_auc","std"),
        fit_seconds_mean=("fit_seconds","mean")
    ).reset_index()
    save_csv(summary, "08_ablation_summary.csv")
    print(summary.to_string(index=False))
    print("Ablation study completed successfully!")

if __name__ == "__main__":
    main()