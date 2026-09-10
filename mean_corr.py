import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

file = sys.argv[1]
print("Reading file: ", file)
# Load CSV without headers
df = pd.read_csv(file, header=None)
df.columns = ["qid", "pearson", "kendall", "spearman"]

print(f"Number of initial rows: {len(df)}")
df.dropna()

# ---- Column-wise stats ----
means = df[["pearson", "kendall", "spearman"]].mean()
stds = df[["pearson", "kendall", "spearman"]].std()

print(f"Num Rows: {len(df)}")
print("Column-wise statistics:")
for col in ["pearson", "kendall", "spearman"]:
    print(f"{col}: mean={means[col]:.4f}, std={stds[col]:.4f}")

# ---------------------------------------------------------
# qid vs pearson (simple scatter plot, no legend)
# ---------------------------------------------------------
qid_vals = df["qid"].unique()
qid_map = {q: i for i, q in enumerate(qid_vals)}
df["_qid_int"] = df["qid"].map(qid_map)
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="_qid_int", y="pearson", s=40)
plt.axhline(0.7)
plt.xticks(ticks=np.arange(len(qid_vals)), labels=qid_vals, rotation=90)
plt.title("qid vs Pearson")
plt.xlabel("qid")
plt.ylabel("Pearson")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# KDE plot with histogram of pearson
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.histplot(df["pearson"], stat="density", bins=30, alpha=0.4)
sns.kdeplot(df["pearson"], fill=True, alpha=0.6)
plt.title("Pearson Distribution (Histogram + KDE)")
plt.xlabel("Pearson")
plt.tight_layout()
plt.show()
