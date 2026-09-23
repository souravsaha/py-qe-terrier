import importlib

collection = "msmarco_passage_v2"
method = "hyderocchio"

from io_utils import parse_queries, store_ap, store_mean_ap
from iqg_learn import IdealQueryGeneration

module = importlib.import_module(f"definitions.{collection}")
index = getattr(module, "index")
qrels_bin = getattr(module, "qrels_bin")
qid_range = getattr(module, "qid_range")
collection = getattr(module, "collection")

iqg = IdealQueryGeneration(index, qrels_bin, collection)

param = {
        "hyde": "na", "hyderocchio": "na", "ceqe": "max"
}

num_prf = {
        "ceqe": [10, 20, 30, 40],
        "hyde": ["na"],
        "hyderocchio": ["na"]
}

for num_prf in num_prf[method]:
    for num_terms in [55, 45, 35, 25, 15]:
        expanded_query_path = f"./expanded-queries/{collection}/weights/{method}-all-{num_prf}-{param[method]}.term_weights"
        exp_output = f"./expanded-queries/{collection}/weights/{method}-{num_terms}-{num_prf}-{param[method]}.term_weights"
        ap_output = (
            exp_output
            .replace("/weights/", "/aps/")
            .replace(".term_weights", ".ap")
        )
        qvecs = parse_queries(expanded_query_path)

        for qid, qvec in qvecs.items():
            qvec.sort_by_weight()

        aps = {}
        for qid in qid_range:
            qvec = qvecs[qid]
            qvec.trim(num_terms)
            qvec.store_txt(qid, exp_output, append=True)
            ap = iqg.computeAP(qid, qvec)
            aps[qid] = ap
            print(f"{method}\t{num_prf}\t{num_terms}\t{qid}\t{ap}")
        store_ap(aps, ap_output)
