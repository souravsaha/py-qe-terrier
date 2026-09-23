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


parser = argparse.ArgumentParser(description="Run options")
parser.add_argument("--dataset", required=True, help="dataset name")
parser.add_argument("--skip_upto", type=str, default=None, required=False, help="Skip previous queries")
args = parser.parse_args()


module = importlib.import_module(f"definitions.{args.dataset}")
index = getattr(module, "index")
qrels_bin = getattr(module, "qrels_bin")
qid_range = getattr(module, "qid_range")
collection = getattr(module, "collection")

sns.set_theme(style="whitegrid")


def average_precision(scores, labels):
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    order = np.argsort(scores)[::-1]
    labels = labels[order]

    rel_count = labels.sum()
    if rel_count == 0:
        return 0.0

    cum_rel = np.cumsum(labels)
    ranks = np.arange(1, len(labels) + 1)

    precision_at_k = cum_rel / ranks
    return (precision_at_k * labels).sum() / rel_count


expansion_methods = ["rm3", "bo1", "kl", "ceqe", "hyde", "hyderocchio"]
num_exp_terms = [15, 25, 35, 45, 55]
num_top_docs = {
        "rm3":  [10, 20, 30, 40],
        "bo1":  [10, 20, 30, 40],
        "kl":   [10, 20, 30, 40],
        "ceqe": [10, 20, 30, 40],
}
method_params = {"rm3": 0.6, "ceqe": "max"}


name = "ap-vs-ap-FINAL"
corr_dir = f"./correlation-computations/{collection}/{name}"
corr_outfile = (
    f"{corr_dir}/query-wise-correlation.csv"
)
os.makedirs(corr_dir, exist_ok=True)

iqg = IdealQueryGeneration(index, qrels_bin, collection)

for qid in qid_range:
    print(f"Processing qid {qid}...")
    if args.skip_upto is not None and  qid <= args.skip_upto: 
        continue
    X, y, term_list = iqg.get_data(qid)

    scores_all, aps_all = [], []
    fig, ax = plt.subplots(figsize=(7, 5))

    for exp_method, color in zip(
        expansion_methods, sns.color_palette("tab10", len(expansion_methods))
    ):
        scores_mth, aps_mth = [], []

        for num_docs, num_terms in product(num_top_docs.get(exp_method, ["na"]), num_exp_terms):
            exp_runid = f"{exp_method}-{num_terms}-{num_docs}-{method_params.get(exp_method, 'na')}"
            exp_query_file = f"./expanded-queries/{collection}/weights/{exp_runid}.term_weights"
            exp_ap_file = f"./expanded-queries/{collection}/aps/{exp_runid}.ap"

            if not os.path.exists(exp_query_file) or not os.path.exists(exp_ap_file):
                print(f"Not found file {exp_query_file} or {exp_ap_file}")

            exp_queries = parse_queries(exp_query_file)
            exp_query_aps = parse_ap(exp_ap_file)

            if qid not in exp_query_aps or qid not in exp_queries:
                continue

            qvec = exp_queries[qid]

            qarr = iqg.qvec_to_array(qvec, term_list)
            qarr = qarr / np.linalg.norm(qarr)

            curr_score = average_precision(X @ qarr, y)
            curr_ap = exp_query_aps[qid]

            scores_mth.append(curr_score)
            aps_mth.append(curr_ap)
            scores_all.append(curr_score)
            aps_all.append(curr_ap)

            print(f"{qid}-{exp_runid}: Restricted AP={curr_score:.4f}, AP={curr_ap:.4f}")

        if len(aps_mth) > 0:
            sns.scatterplot(
                x=aps_mth,
                y=scores_mth,
                ax=ax,
                label=exp_method,
                color=color,
                s=40,
                alpha=0.8,
            )

    # Compute and save correlations
    pearson_corr, _ = pearsonr(aps_all, scores_all)
    kendall_corr, _ = kendalltau(aps_all, scores_all)
    spearman_corr, _ = spearmanr(aps_all, scores_all)

    print(
        f"QID {qid}: Pearson={pearson_corr:.3f}, Kendall={kendall_corr:.3f}, Spearman={spearman_corr:.3f}"
    )
    with open(corr_outfile, "a") as f:
        print(f"{qid},{pearson_corr},{kendall_corr},{spearman_corr}", file=f)

    ax.set_title(
        f"Query {qid} -- Corr(P={pearson_corr:.2f}, K={kendall_corr:.2f}, S={spearman_corr:.2f})"
    )
    ax.set_xlabel("AP")
    ax.set_ylabel("Restricted AP")
    ax.legend()
    plt.tight_layout()

    out_path = f"{corr_dir}/{qid}-scatter.png"
    plt.savefig(out_path)
    print(f"Saved scatterplot -> {out_path}")
    plt.close(fig)

print(f"\nDone! Correlations saved in {corr_outfile}, plots in {corr_dir}/")
