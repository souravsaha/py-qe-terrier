import os
import numpy as np
import argparse
import importlib
import torch

from io_utils import store_mean_ap

parser = argparse.ArgumentParser(description="Run different IQG methods with options")
parser.add_argument("--dataset", required=True, help="dataset name")
parser.add_argument("--name", required=True, help="trial name")
parser.add_argument("--method", required=True, help="method name")
parser.add_argument("--k", type=int, default=None, required=False, help="SVD truncation upto")
parser.add_argument("--l2_lambda", type=float, default=0.1, required=False, help="regularization constant")
parser.add_argument("--skip_upto", type=str, default=None, required=False, help="Skip previous queries")
args = parser.parse_args()

module = importlib.import_module(f"definitions.{args.dataset}")
index = getattr(module, "index")
qrels_bin = getattr(module, "qrels_bin")
qid_range = getattr(module, "qid_range")
collection = getattr(module, "collection")

from iqg_lda import IdealQueryGenerationLDA
from iqg_leastsq_graddesc import IdealQueryGenerationLeastSq
from iqg_leastsq_closedform import IdealQueryGenerationLeastSqDirect
from iqg_dfo import IdealQueryGenerationDFO
from iqg_rocchio import IdealQueryGenerationRocchio

if args.method == "lda":
    iqg = IdealQueryGenerationLDA(index, qrels_bin, collection)
elif args.method == "leastsq":
    iqg = IdealQueryGenerationLeastSq(index, qrels_bin, collection)
elif args.method == "dfo":
    iqg = IdealQueryGenerationDFO(index, qrels_bin, collection)
elif args.method == "rocchio":
    iqg = IdealQueryGenerationRocchio(index, qrels_bin, collection)
else:
    raise ValueError("Incorrect method! Must be one of lda, leastsq, dfo, rocchio")
name = args.name

def torch_svd(X):
    print("Computing SVD of X...")
    X_t = torch.tensor(X, dtype=torch.float32)
    U, S, Vt = torch.linalg.svd(X_t, full_matrices=False)

    U = U.cpu().numpy()
    S = S.cpu().numpy()
    Vt = Vt.cpu().numpy()

    return U, S, Vt

idealq_dir = f"./ideal_queries/{collection}/{name}"
os.makedirs(idealq_dir, exist_ok=True)

for qid in qid_range:

    print(f"Processing qid {qid}...")
    if args.skip_upto is not None and  qid <= args.skip_upto: 
        continue
    X, y, term_list = iqg.get_data(qid)

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
    qideal_vec.trim(1000)
    qideal_ap = iqg.computeAP(qid, qideal_vec)
    with open(f"{idealq_dir}/idealq-ap.txt", "a") as f:
        print(f"map\t{qid}\t{qideal_ap}", file=f)
    print(f"AP: {qideal_ap}")


store_mean_ap(f"{idealq_dir}/idealq-ap.txt")

print("Done!")
