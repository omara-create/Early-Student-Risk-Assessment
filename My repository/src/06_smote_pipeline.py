"""06. Document and validate the fold-local SMOTE rule.

SMOTE is NOT fitted here globally. The definitive leakage-safe implementation
is in 07_model_training.py using imblearn.pipeline.Pipeline, so synthetic
examples can only be generated from each training partition.
"""
from common import load_config, save_csv, set_seed
import pandas as pd

def main():
    cfg = load_config()
    set_seed(cfg["random_seed"])
    rule = pd.DataFrame([{
        "smote_enabled": cfg["smote"]["enabled"],
        "sampling_strategy": cfg["smote"]["sampling_strategy"],
        "k_neighbors": cfg["smote"]["k_neighbors"],
        "application": "training folds only",
        "validation_or_test_smote": False
    }])
    save_csv(rule, "06_smote_protocol.csv")
    print(rule.to_string(index=False))

if __name__ == "__main__":
    main()
