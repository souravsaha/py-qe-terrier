import sys
import pyterrier as pt
import os
from itertools import product
import pandas as pd
import importlib

from io_utils import store_ap, parse_queries
from iqg_learn import IdealQueryGeneration

dataset = sys.argv[1]
module = importlib.import_module(f"definitions.{dataset}")
index = getattr(module, "index")
qrels_bin = getattr(module, "qrels_bin")
topics = getattr(module, "topics")
collection = getattr(module, "collection")


def store_expq(expq, runid):
    os.makedirs(f"./expanded-queries/{collection}/weights", exist_ok=True)
    output_path = f"./expanded-queries/{collection}/weights/{runid}.term_weights"
    with open(output_path, "w") as f:
        for row in expq.itertuples(index=False):
            qid = row.qid
            query = row.query

            term_weights = {}
            for token in query.split():
                if token == "applypipeline:off":
                    continue
                if "^" in token:
                    term, weight = token.rsplit("^", 1)
                    term_weights[term] = weight
                else:
                    term_weights[token] = "1.0"

            sorted_terms = sorted(
                term_weights.items(), key=lambda x: x[1], reverse=True
            )
            for term, weight in sorted_terms:
                f.write(f"{qid}\t{term}\t{weight}\n")


def store_result(res, runid):
    os.makedirs(f"./expanded-queries/{collection}/aps", exist_ok=True)
    output_path = f"./expanded-queries/{collection}/aps/{runid}.ap"
    ap_dict = dict()
    qids = sorted(list(res.keys()))
    for qid in qids:
        ap_dict[qid] = res[qid]["map"]
    store_ap(ap_dict, output_path)


expansion_methods = ["rm3", "bo1", "kl"]
num_exp_terms = [15, 25, 35, 45, 55]
num_top_docs = [10, 20, 30, 40]
method_params = {"rm3": 0.6}

bm25 = pt.terrier.Retriever(index.index_ref(), wmodel="BM25", num_results=1000)
initial_retrieved = bm25.transform(topics)

iqg = IdealQueryGeneration(index, qrels_bin, collection)

for exp_method, num_terms, num_docs in product(
    expansion_methods, num_exp_terms, num_top_docs
):
    runid = f"{exp_method}-{num_terms}-{num_docs}-{method_params.get(exp_method, 'na')}"
    print(f"Generating for {runid}...", end=" ", flush=True)
    if exp_method == "rm3":
        expq = pt.rewrite.RM3(
            index.index_ref(),
            fb_terms=num_terms,
            fb_docs=num_docs,
            fb_lambda=method_params["rm3"],
        ).transform(initial_retrieved)
    elif exp_method == "bo1":
        expq = pt.rewrite.Bo1QueryExpansion(
            index.index_ref(), fb_terms=num_terms, fb_docs=num_docs
        ).transform(initial_retrieved)
    elif exp_method == "kl":
        expq = pt.rewrite.KLQueryExpansion(
            index.index_ref(), fb_terms=num_terms, fb_docs=num_docs
        ).transform(initial_retrieved)
    else:
        raise ValueError
    store_expq(expq, runid)
    ret = bm25.transform(expq)
    # expqvec = parse_queries(
    #     f"./expanded-queries/msmarco_passage/weights/{runid}.term_weights"
    # )
    # ret = iqg.search_with_qvecs(expqvec)
    res = pt.Evaluate(ret, qrels_bin, metrics=["map"], perquery=True)
    store_result(res, runid)
    print("Done!")
