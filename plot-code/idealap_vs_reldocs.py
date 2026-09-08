import sys
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os
import importlib.util

collection = sys.argv[1]
method_1 = sys.argv[2]
method_2 = sys.argv[3]
method_3 = sys.argv[4]
ideal_ap_1_file = f"correlation-computations/{collection}/{method_1}/idealq-ap.txt"
ideal_ap_2_file = f"correlation-computations/{collection}/{method_2}/idealq-ap.txt"
ideal_ap_3_file = f"correlation-computations/{collection}/{method_3}/idealq-ap.txt"

# file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"definitions/{collection}.py"))
file_path = f"definitions/{collection}.py"
spec = importlib.util.spec_from_file_location(collection, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# module = importlib.import_module(f"definitions.{collection}")
qid_range = getattr(module, "qid_range")
num_rel_non_rel = getattr(module, "num_rel_non_rel")

df = num_rel_non_rel

# Read AP file
ap_1_df = pd.read_csv(
    ideal_ap_1_file,
    sep="\t",
    header=None,
    names=["qid", "ap"]
)

ap_2_df = pd.read_csv(
    ideal_ap_2_file,
    sep="\t",
    header=None,
    names=["qid", "ap"]
)
ap_3_df = pd.read_csv(
    ideal_ap_3_file,
    sep="\t",
    header=None,
    names=["qid", "ap"]
)


print("MAP:", ap_1_df.loc[ap_1_df["qid"] == "all", "ap"].iloc[0])

ap_1_df = ap_1_df[ap_1_df["qid"].astype(str) != "all"]

print("MAP:", ap_2_df.loc[ap_2_df["qid"] == "all", "ap"].iloc[0])

ap_2_df = ap_2_df[ap_2_df["qid"].astype(str) != "all"]

print("MAP:", ap_3_df.loc[ap_3_df["qid"] == "all", "ap"].iloc[0])

ap_3_df = ap_3_df[ap_3_df["qid"].astype(str) != "all"]


# Ensure same dtype for comparison
ap_1_df["qid"] = ap_1_df["qid"].astype(str)
ap_2_df["qid"] = ap_2_df["qid"].astype(str)
ap_3_df["qid"] = ap_3_df["qid"].astype(str)

df["qid"] = df["qid"].astype(str)


# Check that qids coincide exactly
ap_1_qids = set(ap_1_df["qid"])
ap_2_qids = set(ap_2_df["qid"])
ap_3_qids = set(ap_3_df["qid"])


df_qids = set(df["qid"])

if ap_1_qids != df_qids:
    missing_in_df = ap_1_qids - df_qids
    missing_in_ap = df_qids - ap_1_qids

    raise ValueError(
        f"QID mismatch!\n"
        f"Missing in dataframe: {sorted(missing_in_df)}\n"
        f"Missing in AP file: {sorted(missing_in_ap)}"
    )

if ap_2_qids != df_qids:
    missing_in_df = ap_2_qids - df_qids
    missing_in_ap = df_qids - ap_2_qids

    raise ValueError(
        f"QID mismatch!\n"
        f"Missing in dataframe: {sorted(missing_in_df)}\n"
        f"Missing in AP file: {sorted(missing_in_ap)}"
    )

if ap_3_qids != df_qids:
    missing_in_df = ap_3_qids - df_qids
    missing_in_ap = df_qids - ap_3_qids

    raise ValueError(
        f"QID mismatch!\n"
        f"Missing in dataframe: {sorted(missing_in_df)}\n"
        f"Missing in AP file: {sorted(missing_in_ap)}"
    )

# Merge
# merged = df.merge(ap_1_df, on="qid", how="inner")

merged = (
    df.merge(ap_1_df.rename(columns={"ap": method_1}), on="qid")
      .merge(ap_2_df.rename(columns={"ap": method_2}), on="qid")
      .merge(ap_3_df.rename(columns={"ap": method_3}), on="qid")
)

# Scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(np.log(merged["num_relevant"]), merged["ap"], alpha=0.7)

# plt.xlabel("log(Number of Relevant Documents)")
# plt.ylabel("AP of IEQ")
# plt.title("AP of IEQ vs Number of Relevant Documents")
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig(f"ap_vs_num_relevant_{collection}_{method_1}.png", dpi=300, bbox_inches="tight")
# plt.close()

# pearson, _ = pearsonr(merged["num_relevant"], merged["ap"])
# spearman, _ = spearmanr(merged["num_relevant"], merged["ap"])

# print(f"Pearson correlation:  {pearson:.4f}")
# print(f"Spearman correlation: {spearman:.4f}")

# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

colors = [
    "#4C72B0",  # muted blue
    "#DD8452",  # muted orange
    "#55A868",  # muted green
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

methods = [method_1, method_2, method_3]
x_axis_names = ["$\mathbf{Q}_{\mathrm{IEQ}}^{\mathrm{DFO}}$", "$\mathbf{Q}_{\mathrm{IEQ}}^{\mathrm{LS}}(\lambda=0.1)$", "$\mathbf{Q}_{\mathrm{IEQ}}^{\mathrm{LS}}(\lambda=1)$"]

# markers = ['o', 's', '^']
markers = ['o', 'o', 'o']
for ax, method, marker, color, x_axis_name in zip(axes, methods, markers, colors, x_axis_names):
    ax.scatter(
        np.log(merged["num_relevant"]),
        merged[method],
        alpha=1,
        marker=marker,
        color=color
    )
    spearman, _ = spearmanr(merged["num_relevant"], merged[method])
    print(f"Spearman correlation of method: {spearman:.4f}")
    # axis_name = x_axis_name + " Spearman $r_s$ ("+ str(f"spearman:.4f") + ")"
    # axis_name = rf"{x_axis_name} $r_s={spearman:.4f}$"
    axis_name = rf"{x_axis_name}"
    ax.set_title(axis_name, fontsize=14)
    #ax.set_xlabel(r'#relevant documents in log scale')
    # ax.set_xlabel(rf'$r_s={spearman:.4f}$ #relevant documents in natural log scale')
    ax.set_xlabel(rf'$r_s={spearman:.4f}$; #relevant docs (ln)', fontsize=14)
    ax.grid(alpha=0.3)

axes[0].set_ylabel('AP', fontsize=14)
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

plt.tight_layout()
plt.savefig(
    f"ap_vs_num_relevant_{collection}_3panel.pdf",
    bbox_inches="tight"
)
