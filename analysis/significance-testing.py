import sys
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def significance_test(file_a, file_b, metric="map"):

    # Read tab-separated files
    a = pd.read_csv(
        file_a,
        sep="\t",
        header=None,
        names=["metric", "qid", "score"]
    )

    b = pd.read_csv(
        file_b,
        sep="\t",
        header=None,
        names=["metric", "qid", "score"]
    )
    a = a[a["qid"].str.lower() != "all"]
    b = b[b["qid"].str.lower() != "all"]

    # Select metric
    a = a[a["metric"].str.lower() == metric.lower()]
    b = b[b["metric"].str.lower() == metric.lower()]

    # Match the same queries
    df = pd.merge(
        a[["qid", "score"]],
        b[["qid", "score"]],
        on="qid",
        suffixes=("_A", "_B")
    )

    scores_a = df["score_A"].values
    scores_b = df["score_B"].values

    print(f"Metric: {metric}")
    print(f"Number of paired queries: {len(df)}")
    print()

    print(f"Method A mean: {np.mean(scores_a):.6f}")
    print(f"Method B mean: {np.mean(scores_b):.6f}")
    print(f"Mean difference (A - B): {np.mean(scores_a - scores_b):.6f}")
    print()

    # Paired t-test
    t_stat, t_p = ttest_rel(scores_a, scores_b)

    print("Paired t-test")
    print(f"  t-statistic = {t_stat:.6f}")
    print(f"  p-value     = {t_p:.6g}")
    print()

    # Wilcoxon signed-rank test
    try:
        w_stat, w_p = wilcoxon(scores_a, scores_b)

        print("Wilcoxon signed-rank test")
        print(f"  statistic = {w_stat:.6f}")
        print(f"  p-value   = {w_p:.6g}")

    except ValueError as e:
        print("Wilcoxon test could not be computed:", e)


if __name__ == "__main__":

    file_a = sys.argv[1]
    file_b = sys.argv[2]

    significance_test(file_a, file_b, metric="map")
