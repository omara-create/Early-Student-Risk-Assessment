"""07. Nested, grouped, leakage-safe model evaluation.

Models:
- Majority-class baseline
- Logistic regression
- Decision tree
- Random forest
- XGBoost
- LightGBM

Outer evaluation uses StratifiedGroupKFold so a learner cannot appear in both
training and validation folds. Hyperparameters are selected only inside the
training portion using an inner StratifiedGroupKFold.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from common import load_config, save_csv, set_seed

ROOT = Path(__file__).resolve().parents[1]

def build_preprocessor(X):
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat)
    ], remainder="drop")

def model_specs(cfg, seed):
    specs = {
        "majority_baseline": (
            DummyClassifier(strategy="most_frequent"),
            {}
        ),
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=seed),
            {
                "model__C": cfg["models"]["logistic_regression"]["C"],
                "model__class_weight": cfg["models"]["logistic_regression"]["class_weight"]
            }
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=seed),
            {
                "model__max_depth": cfg["models"]["decision_tree"]["max_depth"],
                "model__min_samples_leaf": cfg["models"]["decision_tree"]["min_samples_leaf"],
                "model__class_weight": cfg["models"]["decision_tree"]["class_weight"]
            }
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=cfg["n_jobs"]),
            {
                "model__n_estimators": cfg["models"]["random_forest"]["n_estimators"],
                "model__max_depth": cfg["models"]["random_forest"]["max_depth"],
                "model__min_samples_leaf": cfg["models"]["random_forest"]["min_samples_leaf"],
                "model__max_features": cfg["models"]["random_forest"]["max_features"],
                "model__class_weight": cfg["models"]["random_forest"]["class_weight"]
            }
        )
    }
    if XGBClassifier is not None:
        specs["xgboost"] = (
            XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=seed,
                n_jobs=cfg["n_jobs"]
            ),
            {f"model__{k}": v for k, v in cfg["models"]["xgboost"].items()}
        )
    if LGBMClassifier is not None:
        specs["lightgbm"] = (
            LGBMClassifier(
                objective="binary", random_state=seed,
                n_jobs=cfg["n_jobs"], verbosity=-1
            ),
            {f"model__{k}": v for k, v in cfg["models"]["lightgbm"].items()}
        )
    return specs

def metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": len(y)
    }

def main():
    cfg = load_config()
    seed = cfg["random_seed"]
    set_seed(seed)

    df = pd.read_csv("results/04_modeling_dataset.csv")
    drop = ["code_module", "code_presentation", "id_student",
            "final_result", "at_risk", "target_exclusion_reason",
            "feature_boundary_day"]
    X = df.drop(columns=drop, errors="ignore")
    y = df["at_risk"].astype(int)
    groups = df["id_student"].astype(str)

    outer = StratifiedGroupKFold(
        n_splits=cfg["outer_folds"], shuffle=True, random_state=seed
    )

    all_rows = []
    fold_predictions = []
    specs = model_specs(cfg, seed)

    for model_name, (estimator, param_dist) in specs.items():
        for fold, (tr, va) in enumerate(outer.split(X, y, groups), start=1):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            gtr = groups.iloc[tr]

            prep = build_preprocessor(Xtr)
            steps = [("preprocess", prep)]

            if cfg["smote"]["enabled"]:
                steps.append(("smote", SMOTE(
                    sampling_strategy=cfg["smote"]["sampling_strategy"],
                    k_neighbors=cfg["smote"]["k_neighbors"],
                    random_state=seed + fold
                )))
            steps.append(("model", estimator))
            pipe = Pipeline(steps)

            if model_name == "majority_baseline":
                search = SkPipeline([
                    ("preprocess", prep),
                    ("model", estimator)
                ])
            else:
                inner = StratifiedGroupKFold(
                    n_splits=cfg["inner_folds"], shuffle=True,
                    random_state=seed + fold
                )
                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=param_dist,
                    n_iter=min(
                        12,
                        max(1, int(np.prod([len(v) for v in param_dist.values()])))
                    ),
                    scoring=cfg["scoring"],
                    cv=list(inner.split(Xtr, ytr, gtr)),  # FIXED: Converted generator to a picklable list of folds
                    random_state=seed + fold,
                    n_jobs=cfg["n_jobs"],
                    refit=True
                )

            t0 = time.perf_counter()
            search.fit(Xtr, ytr)
            elapsed = time.perf_counter() - t0

            pred = search.predict(Xva)
            prob = search.predict_proba(Xva)[:, 1]
            row = metrics(yva, pred, prob)
            best_params = getattr(search, "best_params_", {})
            row.update({
                "model": model_name,
                "fold": fold,
                "fit_seconds": elapsed,
                "best_params": json.dumps(best_params, default=str)
            })
            all_rows.append(row)

            fold_predictions.append(pd.DataFrame({
                "row_index": va,
                "fold": fold,
                "model": model_name,
                "y_true": yva.to_numpy(),
                "y_pred": pred,
                "y_prob": prob
            }))

    result = pd.DataFrame(all_rows)
    save_csv(result, "07_nested_cv_fold_results.csv")
    save_csv(pd.concat(fold_predictions, ignore_index=True),
             "07_oof_predictions.csv")

    summary = result.groupby("model").agg(
        precision_mean=("precision", "mean"),
        precision_sd=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_sd=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_sd=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_sd=("pr_auc", "std"),
        fit_seconds_mean=("fit_seconds", "mean"),
        fit_seconds_sd=("fit_seconds", "std")
    ).reset_index()
    save_csv(summary, "07_model_summary.csv")
    print(summary.to_string(index=False))
    print("Model training completed successfully!")

if __name__ == "__main__":
    main()