import importlib

from io_utils import parse_queries, store_mean_ap
from iqg_learn import IdealQueryGeneration

# dataset = "msmarco_passage"
dataset = "msmarco_passage_v2"
name = "leastsq_1"
ideal_query_dir = f"./ideal_queries/{dataset}/{name}/"

module = importlib.import_module(f"definitions.{dataset}")
index = getattr(module, "index")
qrels_bin = getattr(module, "qrels_bin")
qid_range = getattr(module, "qid_range")
collection = getattr(module, "collection")

num_trim = 200

iqg = IdealQueryGeneration(index, qrels_bin, collection)
for qid in qid_range:
    qvec = parse_queries(f"{ideal_query_dir}/{qid}")[qid]
    qvec.remove_non_positive_weights()
    qvec.sort_by_weight()
    qvec.trim(num_trim)
    ap = iqg.computeAP(qid, qvec)
    with open(f"./idealq_ap_{dataset}_{name}.ap", "a") as f:
        print(f"map\t{qid}\t{ap}", file=f)
    print(f"{qid}\t{ap}")

store_mean_ap(f"./idealq_ap_{dataset}_{name}.ap")
