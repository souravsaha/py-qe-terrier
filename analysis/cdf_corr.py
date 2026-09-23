import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

collection = "trec678rb"


def plot_cdf_correlations(correlations, num_points=500):
    r = np.asarray(correlations)
    assert np.all((-1 <= r) & (r <= 1)), "Correlations must be in [-1, 1]"

    x_vals = np.linspace(-1, 1, num_points)
    cdf_vals = [(r < x).mean() for x in x_vals]

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, cdf_vals, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("P(r < x)")
    plt.title("Empirical CDF of Correlations")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.show()


df = pd.read_csv(
    f"./correlation-computations/{collection}/lda_idealq_k300/query-wise-correlation-l2.csv",
    header=None,
)
df.columns = ["qid", "pearson", "kendall", "spearman"]

plot_cdf_correlations(list(df.pearson))
