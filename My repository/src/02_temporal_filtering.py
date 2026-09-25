"""02. Establish the first-quarter information boundary.

The boundary is defined per module presentation as:
T25 = 0.25 * module_presentation_length.

Only observations whose event date is <= T25 are eligible as predictors.
Final outcome is never used as a predictor.
"""
from common import load_oulad, save_csv, load_config, set_seed
import pandas as pd
import numpy as np

def main():
    cfg = load_config()
    set_seed(cfg["random_seed"])
    d = load_oulad()

    courses = d["courses"].copy()
    courses["boundary_day"] = np.floor(
        courses["module_presentation_length"] * cfg["boundary_fraction"]
    ).astype(int)

    # Student registration is retained only for cohort definition/metadata.
    si = d["studentInfo"][[
        "code_module", "code_presentation", "id_student", "final_result"
    ]].copy()
    cohort = si.merge(
        courses[["code_module", "code_presentation",
                 "module_presentation_length", "boundary_day"]],
        on=["code_module", "code_presentation"], how="inner"
    )

    print("Processing studentVle memory-efficiently...")
    # OPTIMIZATION: Drop NaNs and filter studentVle *before* merging to save RAM
    sv_raw = d["studentVle"].dropna(subset=["date"]).copy()
    max_boundary = courses["boundary_day"].max()
    sv_raw = sv_raw.loc[sv_raw["date"] <= max_boundary]

    # VLE events available by the boundary
    sv = sv_raw.merge(
        courses[["code_module", "code_presentation", "boundary_day"]],
        on=["code_module", "code_presentation"], how="inner"
    )
    sv_q = sv.loc[sv["date"] <= sv["boundary_day"]].copy()
    
    del sv_raw, sv

    # Assessment metadata: only assessments with due date on/before boundary
    ass = d["assessments"].merge(
        courses[["code_module", "code_presentation", "boundary_day"]],
        on=["code_module", "code_presentation"], how="inner"
    )
    ass_q = ass.loc[
        ass["date"].notna() & (ass["date"] <= ass["boundary_day"])
    ].copy()

    # Student assessment records with boundary_day included
    sa = d["studentAssessment"].merge(
        ass_q[["id_assessment", "code_module", "code_presentation", "date", "boundary_day"]],
        on="id_assessment", how="inner", suffixes=("", "_assessment")
    )
    sa_q = sa.loc[
        sa["date_submitted"].notna()
        & (sa["date_submitted"] <= sa["boundary_day"])
    ].copy()

    cohort.to_csv("results/02_cohort_with_boundary.csv", index=False)
    sv_q.to_csv("results/02_vle_first_quarter.csv", index=False)
    ass_q.to_csv("results/02_assessments_first_quarter.csv", index=False)
    sa_q.to_csv("results/02_student_assessments_first_quarter.csv", index=False)

    audit = pd.DataFrame([
        {"dataset": "cohort", "rows": len(cohort)},
        {"dataset": "vle_first_quarter", "rows": len(sv_q)},
        {"dataset": "assessments_first_quarter", "rows": len(ass_q)},
        {"dataset": "student_assessments_first_quarter", "rows": len(sa_q)}
    ])
    save_csv(audit, "02_temporal_audit.csv")
    print("Temporal filtering completed successfully!")

if __name__ == "__main__":
    main()