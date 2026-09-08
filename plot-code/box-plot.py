import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Correlation files
# ------------------------------------------------------------------
files = {
    "Robust": "/Users/sourav/Downloads/py-qe-terrier/py-qe-terrier/corr-synthetic/trec678rb/leastsq_0.1_200/correlations.csv",
    "DL19-20 Passage": "/Users/sourav/Downloads/py-qe-terrier/py-qe-terrier/corr-synthetic/msmarco_passage/leastsq_0.1_200/correlations.csv",
    "DL19-20 Document": "/Users/sourav/Downloads/py-qe-terrier/py-qe-terrier/corr-synthetic/msmarco_document/leastsq_0.1_200/correlations.csv",
    "DL21-22 Passage": "/Users/sourav/Downloads/py-qe-terrier/py-qe-terrier/corr-synthetic/msmarcov2_passage/leastsq_1_200/correlations.csv",
}

# ------------------------------------------------------------------
# Read all files
# ------------------------------------------------------------------
dfs = []

for dataset, path in files.items():
    df = pd.read_csv(
        path,
        header=None,
        names=["qid", "pearson", "kendall", "spearman"],
    )

    temp = pd.DataFrame({
        "Dataset": dataset,
        "Pearson": df["pearson"],
        "Spearman": df["spearman"],
    })

    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

# Convert to long format
plot_df = df.melt(
    id_vars="Dataset",
    value_vars=["Pearson", "Spearman"],
    var_name="Metric",
    value_name="Correlation",
)
print(plot_df)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 4.8))

ax = sns.boxplot(
    data=plot_df,
    x="Dataset",
    y="Correlation",
    hue="Metric",
    width=0.65,
    showfliers=False,
)

# Optional: show individual query values
# sns.stripplot(
#     data=plot_df,
#     x="Dataset",
#     y="Correlation",
#     hue="Metric",
#     dodge=True,
#     alpha=0.4,
#     size=2,
#     color="black",
# )

ax.set_xlabel("Dataset", fontsize=14)
ax.set_ylabel("Correlation", fontsize=14)

ax.tick_params(axis="both", labelsize=12)

plt.legend(
    title="",
    fontsize=12,
)

plt.tight_layout()

plt.savefig(
    "correlation_boxplot.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
)

plt.show()