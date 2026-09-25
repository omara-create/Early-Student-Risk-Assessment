"""10. Generate publication-ready tables and diagnostic figures."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common import RESULTS, save_csv

def main():
    RESULTS.mkdir(exist_ok=True)
    fold = pd.read_csv(RESULTS/"07_nested_cv_fold_results.csv")
    summary = pd.read_csv(RESULTS/"07_model_summary.csv")
    ab_fold = pd.read_csv(RESULTS/"08_ablation_fold_results.csv") # FIXED: load fold-level results for boxplot

    # Publication table with mean ± SD.
    rows = []
    for _, r in summary.iterrows():
        rows.append({
            "Model": r["model"],
            "Precision": f'{r["precision_mean"]:.3f} ± {r["precision_sd"]:.3f}',
            "Recall": f'{r["recall_mean"]:.3f} ± {r["recall_sd"]:.3f}',
            "F1": f'{r["f1_mean"]:.3f} ± {r["f1_sd"]:.3f}',
            "ROC-AUC": f'{r["roc_auc_mean"]:.3f} ± {r["roc_auc_sd"]:.3f}',
            "PR-AUC": f'{r["pr_auc_mean"]:.3f} ± {r["pr_auc_sd"]:.3f}',
            "Fit time (s)": f'{r["fit_seconds_mean"]:.2f}'
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS/"10_table_model_performance.csv", index=False)

    # Box/strip style fold plot.
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=fold, x="model", y="f1")
    sns.stripplot(data=fold, x="model", y="f1", color="black", size=4)
    plt.ylabel("F1 score")
    plt.xlabel("Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS/"10_figure_f1_across_outer_folds.png", dpi=300)
    plt.close()

    # Ablation fold plot.
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=ab_fold, x="configuration", y="f1")
    sns.stripplot(data=ab_fold, x="configuration", y="f1", color="black", size=4)
    plt.ylabel("F1 score")
    plt.xlabel("Ablation configuration")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS/"10_figure_ablation_f1.png", dpi=300)
    plt.close()
    print("Figures and tables generated successfully!")

if __name__ == "__main__":
    main()