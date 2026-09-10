import os
import numpy as np
from itertools import product
import importlib
import argparse

from io_utils import parse_queries, parse_ap
from iqg_learn import IdealQueryGeneration


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="dataset name")
parser.add_argument("--name", required=True, help="trial name")
args = parser.parse_args()


collection = args.dataset
name = args.name


module = importlib.import_module(f"definitions.{collection}")
num_rel_non_rel = getattr(module, "num_rel_non_rel")


expansion_methods = ["rm3", "bo1", "kl", "ceqe", "hyde", "hyderocchio"]
num_exp_terms = [15, 25, 35, 45, 55]
num_top_docs = {
        "rm3":  [10, 20, 30, 40],
        "bo1":  [10, 20, 30, 40],
        "kl":   [10, 20, 30, 40],
        "ceqe": [10, 20, 30, 40],
}
method_params = {"rm3": "0.6", "ceqe": "max"}


angle_range_dir = f"./angle_range/{collection}"
os.makedirs(angle_range_dir, exist_ok=True)
angle_range_file = f"{angle_range_dir}/{name}"
ideal_query_dir = f"./ideal_queries/{collection}/{name}/"


def get_angle(x, y):
    cos = cosine(x, y)
    return np.arccos(cos)


def cosine(x, y):
    return max(min(np.dot(x, y), 1), -1)


for qid in os.listdir(ideal_query_dir):
    qideal_vec = parse_queries(f"./{ideal_query_dir}/{qid}")[qid]

    term_list = sorted(list(qideal_vec.keys()))

    qideal_arr = IdealQueryGeneration.qvec_to_array(qideal_vec, term_list)
    qideal_arr = qideal_arr / np.linalg.norm(qideal_arr)

    angles = []
    aps = []
    sims = []
    for exp_method in expansion_methods:
        for num_docs, num_terms in product(num_top_docs.get(exp_method, ["na"]), num_exp_terms):
            exp_runid = f"{exp_method}-{num_terms}-{num_docs}-{method_params.get(exp_method, 'na')}"
            exp_query_file = (
                f"./expanded-queries/{collection}/weights/{exp_runid}.term_weights"
            )
            exp_ap_file = f"./expanded-queries/{collection}/aps/{exp_runid}.ap"

            if not os.path.exists(exp_query_file) or not os.path.exists(exp_ap_file):
                continue

            exp_queries = parse_queries(exp_query_file)
            exp_query_aps = parse_ap(exp_ap_file)

            if qid not in exp_query_aps or qid not in exp_queries:
                continue

            qvec = exp_queries[qid]
            qarr = IdealQueryGeneration.qvec_to_array(qvec, term_list)
            if np.linalg.norm(qarr) != 0:
                qarr = qarr / np.linalg.norm(qarr)

            curr_angle = get_angle(qideal_arr, qarr)
            curr_ap = exp_query_aps[qid]
            curr_sim = cosine(qideal_arr, qarr)

            print(f"{qid}-{exp_runid}: Angle = {curr_angle}, AP = {curr_ap}")
            angles.append(float(curr_angle))
            aps.append(float(curr_ap))
            sims.append(float(curr_sim))

    with open(angle_range_file, "a") as f:
        num_relevant = num_rel_non_rel.loc[num_rel_non_rel["qid"] == qid, "num_relevant"].iloc[0]
        print(f"{qid}\t{angles}\t{sims}\t{aps}\t{num_relevant}", file=f)
