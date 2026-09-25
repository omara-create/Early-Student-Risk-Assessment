"""05. Define leakage-safe preprocessing metadata.

Actual fitting of imputers/encoders occurs inside each CV fold in 07_model_training.py.
This script prepares the modeling matrix specification and a deterministic audit.
"""
from common import save_csv
import pandas as pd

IDENTIFIERS = ["code_module", "code_presentation", "id_student", "final_result",
               "at_risk", "target_exclusion_reason", "feature_boundary_day"]

def main():
    df = pd.read_csv("results/04_modeling_dataset.csv")
    feature_cols = [c for c in df.columns if c not in IDENTIFIERS]
    audit = []
    for c in feature_cols:
        audit.append({
            "feature": c,
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "missing_pct": float(df[c].isna().mean() * 100),
            "unique": int(df[c].nunique(dropna=True))
        })
    save_csv(pd.DataFrame(audit), "05_feature_preprocessing_audit.csv")
    pd.DataFrame({"feature": feature_cols}).to_csv(
        "results/05_feature_columns.csv", index=False
    )
    print(f"Prepared {len(feature_cols)} candidate features.")

if __name__ == "__main__":
    main()
