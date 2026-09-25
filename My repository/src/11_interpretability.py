"""11. Fold-local permutation importance and domain aggregation.

Permutation importance is computed on held-out outer folds using the fitted
pipeline. Because the permutation occurs on the raw feature matrix, the
importance is attributable to original manuscript-level features rather than
one-hot encoded columns. The result is predictive importance, not causation.
"""
from __future__ import annotations

import ast
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from common import load_config, save_csv, set_seed

DROP = ["code_module", "code_presentation", "id_student",
        "final_result", "at_risk", "target_exclusion_reason",
        "feature_boundary_day"]

def make_preprocessor(X):
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", SKPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat)
    ])

def main():
    cfg = load_config()
    seed = cfg["random_seed"]
    set_seed(seed)

    df = pd.read_csv("results/04_modeling_dataset.csv")
    X = df.drop(columns=DROP, errors="ignore")
    y = df["at_risk"].astype(int)
    groups = df["id_student"].astype(str)

    cv_results = pd.read_csv("results/07_nested_cv_fold_results.csv")
    lgbm_results = cv_results[cv_results["model"] == "lightgbm"].copy()
    if lgbm_results.empty:
        raise RuntimeError("No LightGBM results found. Run 07_model_training.py first.")

    outer = StratifiedGroupKFold(
        n_splits=cfg["outer_folds"], shuffle=True, random_state=seed
    )
    rows = []

    for fold, (tr, va) in enumerate(outer.split(X, y, groups), 1):
        model_row = lgbm_results.loc[lgbm_results["fold"] == fold]
        if model_row.empty:
            continue
        params_text = model_row.iloc[0]["best_params"]
        params = json.loads(params_text)
        # Search parameters are stored with "model__" prefixes.
        model_params = {
            k.replace("model__", ""): v for k, v in params.items()
            if k.startswith("model__")
        }

        pre = make_preprocessor(X.iloc[tr])
        model = LGBMClassifier(
            objective="binary", random_state=seed + fold,
            n_jobs=cfg["n_jobs"], verbosity=-1, **model_params
        )
        steps = [("preprocess", pre)]
        if cfg["smote"]["enabled"]:
            steps.append(("smote", SMOTE(
                sampling_strategy=cfg["smote"]["sampling_strategy"],
                k_neighbors=cfg["smote"]["k_neighbors"],
                random_state=seed + fold
            )))
        steps.append(("model", model))
        pipe = Pipeline(steps)
        pipe.fit(X.iloc[tr], y.iloc[tr])

        perm = permutation_importance(
            pipe, X.iloc[va], y.iloc[va],
            scoring="f1", n_repeats=10,
            random_state=seed + fold, n_jobs=cfg["n_jobs"]
        )
        for feature, mean_imp, sd_imp in zip(
            X.columns, perm.importances_mean, perm.importances_std
        ):
            rows.append({
                "model": "lightgbm",
                "fold": fold,
                "feature": feature,
                "permutation_f1_mean": mean_imp,
                "permutation_f1_sd": sd_imp
            })

    out = pd.DataFrame(rows)
    save_csv(out, "11_fold_permutation_importance.csv")

    summary = out.groupby("feature").agg(
        mean_permutation_f1=("permutation_f1_mean", "mean"),
        sd_permutation_f1=("permutation_f1_mean", "std")
    ).reset_index().sort_values("mean_permutation_f1", ascending=False)

    summary["domain"] = np.where(
        summary["feature"].str.contains(
            "click|vle|assessment|submission|score|delay",
            case=False, regex=True
        ),
        "early_dynamic",
        "static"
    )
    save_csv(summary, "11_permutation_importance_summary.csv")

    domain = summary.groupby("domain")["mean_permutation_f1"].agg(
        ["count", "sum", "mean"]
    ).reset_index()
    domain.columns = ["domain", "feature_count", "sum_importance", "mean_importance"]
    total = domain["sum_importance"].sum()
    domain["share_of_positive_sum"] = np.where(
        total > 0, domain["sum_importance"] / total, np.nan
    )
    save_csv(domain, "11_permutation_importance_by_domain.csv")

    print(summary.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
