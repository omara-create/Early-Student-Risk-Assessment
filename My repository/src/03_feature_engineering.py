"""03. Build reproducible student-module features using only first-quarter data."""
from common import load_oulad, load_config, set_seed, save_csv
import pandas as pd
import numpy as np

def main():
    cfg = load_config()
    set_seed(cfg["random_seed"])
    d = load_oulad()

    cohort = pd.read_csv("results/02_cohort_with_boundary.csv")
    sv = pd.read_csv("results/02_vle_first_quarter.csv")
    sa = pd.read_csv("results/02_student_assessments_first_quarter.csv")
    ass = pd.read_csv("results/02_assessments_first_quarter.csv")

    keys = ["code_module", "code_presentation", "id_student"]

    # Static learner attributes. final_result is excluded here and becomes y.
    static_cols = [
        "gender", "region", "highest_education", "imd_band",
        "age_band", "num_of_prev_attempts", "studied_credits",
        "disability"
    ]
    static = cohort[keys + [c for c in static_cols if c in cohort.columns]].drop_duplicates(keys)

    # VLE aggregates available by boundary.
    if len(sv):
        vle = sv.groupby(keys).agg(
            total_clicks=("sum_click", "sum"),
            active_vle_days=("date", "nunique"),
            vle_event_count=("id_site", "size"),
            distinct_vle_sites=("id_site", "nunique"),
            first_vle_day=("date", "min"),
            last_vle_day=("date", "max")
        ).reset_index()
        vle["vle_span_days"] = vle["last_vle_day"] - vle["first_vle_day"]
    else:
        vle = pd.DataFrame(columns=keys)

    # Assessment aggregates: scores and submission timing only when both
    # assessment due date and actual submission are within the boundary.
    if len(sa):
        sa["submission_delay"] = sa["date_submitted"] - sa["date"]
        sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
        assess = sa.groupby(keys).agg(
            early_assessments_submitted=("id_assessment", "nunique"),
            mean_early_score=("score", "mean"),
            median_early_score=("score", "median"),
            max_early_score=("score", "max"),
            mean_submission_delay=("submission_delay", "mean"),
            late_early_submissions=("submission_delay", lambda x: int((x > 0).sum())),
            early_assessment_attempts=("id_assessment", "size")
        ).reset_index()
    else:
        assess = pd.DataFrame(columns=keys)

    # Number of scheduled assessments that were available by the boundary.
    if len(ass):
        sched = ass.groupby(
            ["code_module", "code_presentation"]
        ).agg(early_assessments_available=("id_assessment", "nunique")).reset_index()
    else:
        sched = pd.DataFrame(columns=["code_module", "code_presentation",
                                      "early_assessments_available"])

    # FIXED: Avoid duplicating columns already present in `keys`
    base = cohort[keys + [
        "module_presentation_length",
        "boundary_day"
    ]].drop_duplicates(keys)

    base = base.merge(static, on=keys, how="left", suffixes=("", "_static"))
    base = base.merge(vle, on=keys, how="left")
    base = base.merge(assess, on=keys, how="left")
    base = base.merge(sched, on=["code_module", "code_presentation"], how="left")

    # A useful explicit availability ratio.
    base["assessment_submission_rate"] = np.where(
        base["early_assessments_available"].fillna(0) > 0,
        base["early_assessments_submitted"].fillna(0)
        / base["early_assessments_available"].fillna(0),
        0.0
    )

    # Preserve identifiers and target for later target construction.
    target = cohort[keys + ["final_result"]].drop_duplicates(keys)
    base = base.merge(target, on=keys, how="left")

    # Audit that every dynamic feature is first-quarter bounded.
    base["feature_boundary_day"] = base["boundary_day"]

    save_csv(base, "03_engineered_student_module_features.csv")

    feature_dictionary = pd.DataFrame([
        ["gender", "studentInfo", "static", "entire record", "categorical"],
        ["region", "studentInfo", "static", "entire record", "categorical"],
        ["highest_education", "studentInfo", "static", "entire record", "categorical"],
        ["imd_band", "studentInfo", "static", "entire record", "categorical"],
        ["age_band", "studentInfo", "static", "entire record", "categorical"],
        ["num_of_prev_attempts", "studentInfo", "static", "entire record", "numeric"],
        ["studied_credits", "studentInfo", "static", "entire record", "numeric"],
        ["disability", "studentInfo", "static", "entire record", "categorical"],
        ["total_clicks", "studentVle", "dynamic", "date <= T25", "numeric"],
        ["active_vle_days", "studentVle", "dynamic", "date <= T25", "numeric"],
        ["distinct_vle_sites", "studentVle", "dynamic", "date <= T25", "numeric"],
        ["vle_span_days", "studentVle", "dynamic", "date <= T25", "numeric"],
        ["early_assessments_submitted", "studentAssessment", "dynamic", "submission <= T25", "numeric"],
        ["mean_early_score", "studentAssessment", "dynamic", "submission <= T25", "numeric"],
        ["median_early_score", "studentAssessment", "dynamic", "submission <= T25", "numeric"],
        ["max_early_score", "studentAssessment", "dynamic", "submission <= T25", "numeric"],
        ["mean_submission_delay", "studentAssessment", "dynamic", "submission <= T25", "numeric"],
        ["late_early_submissions", "studentAssessment", "dynamic", "submission <= T25", "numeric"],
        ["assessment_submission_rate", "studentAssessment/assessments", "dynamic", "all included records <= T25", "numeric"]
    ], columns=["feature", "source_table", "type", "availability_rule", "data_type"])
    save_csv(feature_dictionary, "03_feature_dictionary.csv")
    print("Feature engineering completed successfully!")

if __name__ == "__main__":
    main()