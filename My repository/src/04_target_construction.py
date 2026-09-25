"""04. Construct the binary early-risk target.

Primary protocol:
    at_risk = 1 for final_result in {Fail, Withdrawn}
    at_risk = 0 for final_result in {Pass, Distinction}

No ambiguous/other outcomes are silently mapped.
"""
from common import load_config, set_seed, save_csv
import pandas as pd

def main():
    cfg = load_config()
    set_seed(cfg["random_seed"])

    df = pd.read_csv("results/03_engineered_student_module_features.csv")
    
    # FIXED: Match exact casing and string values from OULAD dataset ("Withdrawn", "Fail", "Pass", "Distinction")
    mapping = {
        "Fail": 1, "Withdrawn": 1,
        "Pass": 0, "Distinction": 0
    }
    df["at_risk"] = df["final_result"].map(mapping)
    df["target_exclusion_reason"] = df["at_risk"].isna().map(
        {True: "unmapped_final_result", False: ""}
    )

    counts = (
        df.loc[df["at_risk"].notna(), "at_risk"]
        .value_counts(dropna=False)
        .rename_axis("at_risk").reset_index(name="n")
    )
    counts["proportion"] = counts["n"] / counts["n"].sum()
    save_csv(counts, "04_class_distribution_preprocessing.csv")

    exclusions = df.loc[df["at_risk"].isna(),
                        ["code_module", "code_presentation", "id_student",
                         "final_result", "target_exclusion_reason"]]
    save_csv(exclusions, "04_target_exclusions.csv")

    df = df.loc[df["at_risk"].notna()].copy()
    df["at_risk"] = df["at_risk"].astype(int)
    save_csv(df, "04_modeling_dataset.csv")

    print(counts.to_string(index=False))
    print("Target construction completed successfully!")

if __name__ == "__main__":
    main()