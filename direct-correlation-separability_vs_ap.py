import os
import numpy as np
from itertools import product
from scipy.stats import (
    pearsonr,
    kendalltau,
    spearmanr,
)
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import importlib

from iqg_learn import IdealQueryGeneration
from io_utils import parse_ap, parse_queries

parser = argparse.ArgumentParser(description="Run different IQG methods with options")
parser.add_argument("--dataset", required=True, help="dataset name")
parser.add_argument("--name", required=True, help="trial name")
parser.add_argument("--idealq_name", required=False, default=None, help="Existing ideal query name")
parser.add_argument("--skip_upto", type=str, default=None, required=False, help="Skip previous queries")
args = parser.parse_args()

module = importlib.import_module(f"definitions.{args.dataset}")
index = getattr(module, "index")
qrels_bin = getattr(module, "qrels_bin")
qid_range = getattr(module, "qid_range")
collection = getattr(module, "collection")


def cosine(qarr1, qarr2):
    return np.dot(qarr1, qarr2) / (np.linalg.norm(qarr1) * np.linalg.norm(qarr2))


def separability(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    group1 = x[y == 1]
    group0 = x[y == 0]

    if len(group1) < 2 or len(group0) < 2:
        raise ValueError("Each group must contain at least two samples.")

    mean1 = np.mean(group1)
    mean0 = np.mean(group0)

    var1 = np.var(group1, ddof=1)
    var0 = np.var(group0, ddof=1)

    n1 = len(group1)
    n0 = len(group0)

    pooled_std = np.sqrt(
        ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)
    )

    return (mean1 - mean0) / pooled_std


sns.set_theme(style="whitegrid")

iqg = IdealQueryGeneration(index, qrels_bin, collection)
name = args.name

expansion_methods = ["rm3", "bo1", "kl", "ceqe", "hyde", "hyderocchio"]
num_exp_terms = [15, 25, 35, 45, 55]
num_top_docs = {
        "rm3":  [10, 20, 30, 40],
        "bo1":  [10, 20, 30, 40],
        "kl":   [10, 20, 30, 40],
        "ceqe": [10, 20, 30, 40],
}
method_params = {"rm3": "0.6", "ceqe": "max"}


corr_dir = f"./correlation-computations-separability-with-ap/{collection}/{name}"
corr_outfile = (
    f"{corr_dir}/query-wise-correlation-l2.csv"
)
os.makedirs(corr_dir, exist_ok=True)

if args.idealq_name:
    idealq_dir = f"./ideal_queries/{collection}/{args.idealq_name}"
else:
    idealq_dir = f"./ideal_queries/{collection}/{name}"


for qid in qid_range:

    print(f"Processing qid {qid}...")

    if args.skip_upto is not None and qid < args.skip_upto:
        continue

    X, y, term_list = iqg.get_data(qid)

    print(f"Using Stored Ideal Query in {idealq_dir}/{qid}...")
    qideal_vec = parse_queries(f"{idealq_dir}/{qid}")[qid]
    qideal_arr = iqg.qvec_to_array(qideal_vec, term_list)
    qideal_arr = qideal_arr / np.linalg.norm(qideal_arr)

    aps_all,sep_all = [], []
    fig, ax = plt.subplots(figsize=(7, 5))

    for exp_method, color in zip(
        expansion_methods, sns.color_palette("tab10", len(expansion_methods))
    ):
        aps_mth, sep_mth = [], []

        for num_docs, num_terms in product(num_top_docs.get(exp_method, ["na"]), num_exp_terms):
            exp_runid = f"{exp_method}-{num_terms}-{num_docs}-{method_params.get(exp_method, 'na')}"
            exp_query_file = (
                f"./expanded-queries/{collection}/weights/{exp_runid}.term_weights"
            )
            exp_ap_file = f"./expanded-queries/{collection}/aps/{exp_runid}.ap"

            if not os.path.exists(exp_query_file) or not os.path.exists(
                exp_ap_file
            ):
                continue

            exp_queries = parse_queries(exp_query_file)
            exp_query_aps = parse_ap(exp_ap_file)

            if qid not in exp_query_aps or qid not in exp_queries:
                continue

            qvec = exp_queries[qid]
            qarr = iqg.qvec_to_array(qvec, term_list)
            qarr = qarr / np.linalg.norm(qarr)

            # curr_score = cosine(qideal_arr, qarr)
            # Replaced cosine similarity with the ap of qaa
            curr_ap = exp_query_aps[qid]

            curr_sep = separability(X @ qarr, y)

            aps_mth.append(curr_ap)
            sep_mth.append(curr_sep)
            aps_all.append(curr_ap)
            sep_all.append(curr_sep)

            print(f"{qid}-{exp_runid}: Ap={curr_ap:.4f}, Separability={curr_sep:.4f}")

        # print(f"length of {len(aps_mth)} and length of {len(sep_mth)}")
        if len(sep_mth) > 0:
            sns.scatterplot(
                x=aps_mth,
                y=sep_mth,
                ax=ax,
                label=exp_method,
                color=color,
                s=40,
                alpha=0.8,
            )

    # Compute and save correlations
    pearson_corr, _ = pearsonr(sep_all, aps_all)
    kendall_corr, _ = kendalltau(sep_all, aps_all)
    spearman_corr, _ = spearmanr(sep_all, aps_all)

    print(
        f"QID {qid}: Pearson={pearson_corr:.3f}, Kendall={kendall_corr:.3f}, Spearman={spearman_corr:.3f}"
    )
    with open(corr_outfile, "a") as f:
        print(f"{qid},{pearson_corr},{kendall_corr},{spearman_corr}", file=f)

    ax.set_title(
        f"Query {qid} -- Corr(P={pearson_corr:.2f}, K={kendall_corr:.2f}, S={spearman_corr:.2f})"
    )
    ax.set_xlabel("Separability")
    ax.set_ylabel("AP")
    ax.legend()
    plt.tight_layout()

    out_path = f"{corr_dir}/{qid}-scatter.png"
    plt.savefig(out_path)
    print(f"Saved scatterplot -> {out_path}")
    plt.close(fig)

print(f"\nDone! Correlations saved in {corr_outfile}, plots in {corr_dir}/")