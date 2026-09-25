"""09. Paired statistical comparisons with Holm correction.

Primary comparison uses paired outer-fold F1 values. Because only five outer
folds are available, results are treated as inferentially limited; the script
reports paired Wilcoxon tests rather than treating folds as independent
subjects. Holm correction controls family-wise error across the tested pairs.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from common import save_csv

def main():
    df = pd.read_csv("results/07_nested_cv_fold_results.csv")
    pivot = df.pivot(index="fold", columns="model", values="f1")

    rows = []
    for a, b in itertools.combinations(pivot.columns, 2):
        x = pivot[a].dropna()
        y = pivot[b].reindex(x.index).dropna()
        common = x.index.intersection(y.index)
        x, y = x.loc[common], y.loc[common]
        try:
            stat, p = wilcoxon(x, y, zero_method="wilcox",
                                alternative="two-sided", method="auto")
        except ValueError:
            stat, p = np.nan, 1.0
        rows.append({
            "model_a": a, "model_b": b,
            "n_pairs": len(common),
            "wilcoxon_stat": stat,
            "p_raw": p,
            "median_f1_difference_a_minus_b": float(np.median(x-y))
        })

    out = pd.DataFrame(rows)
    if len(out):
        reject, p_adj, _, _ = multipletests(out["p_raw"], method="holm")
        out["p_holm"] = p_adj
        out["holm_significant_0_05"] = reject
    save_csv(out, "09_model_pairwise_statistics.csv")

    # Ablation comparisons: pair each configuration by fold.
    ab = pd.read_csv("results/08_ablation_fold_results.csv")
    pv = ab.pivot(index="fold", columns="configuration", values="f1")
    rows = []
    for a, b in itertools.combinations(pv.columns, 2):
        z = pv[[a,b]].dropna()
        try:
            stat, p = wilcoxon(z[a], z[b], zero_method="wilcox",
                                alternative="two-sided", method="auto")
        except ValueError:
            stat, p = np.nan, 1.0
        rows.append({
            "configuration_a": a, "configuration_b": b,
            "n_pairs": len(z), "wilcoxon_stat": stat, "p_raw": p,
            "median_f1_difference_a_minus_b": float(np.median(z[a]-z[b]))
        })
    ab_out = pd.DataFrame(rows)
    if len(ab_out):
        reject, p_adj, _, _ = multipletests(ab_out["p_raw"], method="holm")
        ab_out["p_holm"] = p_adj
        ab_out["holm_significant_0_05"] = reject
    save_csv(ab_out, "09_ablation_pairwise_statistics.csv")

if __name__ == "__main__":
    main()
