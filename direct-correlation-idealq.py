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
import torch

from iqg_learn import IdealQueryGeneration
from iqg_lda import IdealQueryGenerationLDA
from iqg_leastsq_graddesc import IdealQueryGenerationLeastSq
from iqg_dfo import IdealQueryGenerationDFO
from iqg_rocchio import IdealQueryGenerationRocchio
from io_utils import parse_ap, parse_queries, store_mean_ap

parser = argparse.ArgumentParser(description="Run different IQG methods with options")
parser.add_argument("--dataset", required=True, help="dataset name")
parser.add_argument("--name", required=True, help="trial name")
parser.add_argument("--idealq_name", required=False, default=None, help="Existing ideal query name")
parser.add_argument("--method", required=False, help="method name")
parser.add_argument("--k", type=int, default=None, required=False, help="SVD truncation upto")
parser.add_argument("--l2_lambda", type=float, default=0.1, required=False, help="regularization constant")
parser.add_argument("--skip_upto", type=str, default=None, required=False, help="Skip previous queries")
args = parser.parse_args()

module = importlib.import_module(f"definitions.{args.dataset}")
index = getattr(module, "index")
qrels_bin = getattr(module, "qrels_bin")
qid_range = getattr(module, "qid_range")
collection = getattr(module, "collection")


def cosine(qarr1, qarr2):
    return np.dot(qarr1, qarr2) / (np.linalg.norm(qarr1) * np.linalg.norm(qarr2))


def torch_svd(X):
    print("Computing SVD of X...")
    X_t = torch.tensor(X, dtype=torch.float32)
    U, S, Vt = torch.linalg.svd(X_t, full_matrices=False)

    U = U.cpu().numpy()
    S = S.cpu().numpy()
    Vt = Vt.cpu().numpy()

    return U, S, Vt

sns.set_theme(style="whitegrid")


if args.method == "lda":
    iqg = IdealQueryGenerationLDA(index, qrels_bin, collection)
elif args.method == "leastsq":
    iqg = IdealQueryGenerationLeastSq(index, qrels_bin, collection)
elif args.method == "dfo":
    iqg = IdealQueryGenerationDFO(index, qrels_bin, collection)
elif args.method == "rocchio":
    iqg = IdealQueryGenerationRocchio(index, qrels_bin, collection)
else:
    iqg = IdealQueryGeneration(index, qrels_bin, collection)
    if args.idealq_name is None:
        raise ValueError("Incorrect method! Must be one of lda, leastsq, dfo, rocchio")
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


corr_dir = f"./correlation-computations/{collection}/{name}"
if args.idealq_name:
    idealq_dir = f"./ideal_queries/{collection}/{args.idealq_name}"
else:
    idealq_dir = f"./ideal_queries/{collection}/{name}"
corr_outfile = f"{corr_dir}/query-wise-correlation-l2.csv"
os.makedirs(corr_dir, exist_ok=True)
os.makedirs(idealq_dir, exist_ok=True)


qideal_num_terms = 1000
for qid in qid_range:

    print(f"Processing qid {qid}...")
    if args.skip_upto is not None and  qid <= args.skip_upto: 
        continue
    X, y, term_list = iqg.get_data(qid)

    if os.path.exists(f"{idealq_dir}/{qid}"):
        print(f"Using Stored Ideal Query in {idealq_dir}/{qid}...")
        qideal_vec = parse_queries(f"{idealq_dir}/{qid}")[qid]
        qideal_arr = IdealQueryGenerationLeastSq.qvec_to_array(qideal_vec, term_list)
        qideal_arr = qideal_arr / np.linalg.norm(qideal_arr)
    else:
        if args.method == "lda":
            U, S, V = torch_svd(X)
            model = iqg.train_model(X, y, S=S, V=V, k=args.k)
        elif args.method == "leastsq":
            model = iqg.train_model(X, y, l2_lambda=args.l2_lambda)
        elif args.method == "dfo":
            model = iqg.train_model(X, y, term_list=term_list, qid=qid)
            term_list = model.term_list
        elif args.method == "rocchio":
            model = iqg.train_model(X, y)

        qideal_arr = model.coef_.flatten()
        qideal_arr = qideal_arr / np.linalg.norm(qideal_arr)

        qideal_vec = iqg.array_to_qvec(qideal_arr, term_list)
        qideal_vec.store_txt(qid, f"{idealq_dir}/{qid}", append=False)
        qideal_vec.remove_non_positive_weights()
        qideal_vec.trim(qideal_num_terms)
        qideal_ap = iqg.computeAP(qid, qideal_vec)
        with open(f"{idealq_dir}/idealq-ap_{qideal_num_terms}.txt", "a") as f:
            print(f"map\t{qid}\t{qideal_ap}", file=f)
        print(f"AP: {qideal_ap}")

    scores_all, aps_all = [], []
    fig, ax = plt.subplots(figsize=(7, 5))

    for exp_method, color in zip(
        expansion_methods, sns.color_palette("tab10", len(expansion_methods))
    ):
        scores_mth, aps_mth = [], []

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

            curr_score = cosine(qideal_arr, qarr)
            curr_ap = exp_query_aps[qid]

            scores_mth.append(curr_score)
            aps_mth.append(curr_ap)
            scores_all.append(curr_score)
            aps_all.append(curr_ap)

            print(f"{qid}-{exp_runid}: Similarity={curr_score:.4f}, AP={curr_ap:.4f}")

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
    ax.set_xlabel("Average Precision (AP)")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()

    out_path = f"{corr_dir}/{qid}-scatter.png"
    plt.savefig(out_path)
    print(f"Saved scatterplot -> {out_path}")
    plt.close(fig)

store_mean_ap(f"{idealq_dir}/idealq-ap_{qideal_num_terms}.txt")

print(f"\nDone! Correlations saved in {corr_outfile}, plots in {corr_dir}/")
