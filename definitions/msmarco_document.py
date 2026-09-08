import pyterrier as pt
import pandas as pd

collection = "msmarco_document"
ds = pt.get_dataset(collection)

indexref = ds.get_index("terrier_stemmed_text")
index = pt.terrier.TerrierIndex(indexref)

topics_2019 = ds.get_topics("test")
topics_2020 = ds.get_topics("test-2020")
topics = pd.concat([topics_2019, topics_2020], ignore_index=True)

qrels_2019 = ds.get_qrels("test")
qrels_2020 = ds.get_qrels("test-2020")
qrels = pd.concat([qrels_2019, qrels_2020], ignore_index=True)

# Binarize qrels
qrels_bin = qrels.copy()
qrels_bin["label"] = (qrels_bin["label"] >= 1).astype(int)

print(len(qrels_2019.qid.unique()))
print(len(qrels_2020.qid.unique()))

# Pick out qids that are judged in qrels
qid_range = sorted(list(set(qrels.qid)))

# filter out those topics which are not in qrels
topics = topics[topics["qid"].isin(qid_range)]

# important, otherwise returns error because of queries like "define: geon"
topics = pt.rewrite.tokenise().transform(topics)

print("Number of topics:", len(topics))

# bm25 = pt.terrier.Retriever(index.index_ref(), wmodel="BM25", num_results=1000)
# ret = bm25.transform(topics)
#
# final_qids = []
# final_docs = []
# final_labels = []
# for qid in qid_range:
#     judge_docs = list(qrels_bin[qrels_bin["qid"] == qid].docno)
#     judge_labels = list(qrels_bin[qrels_bin["qid"] == qid].label)
#     qid_ret = list(ret[ret["qid"] == qid].docno)
#     for d in qid_ret:
#         if d not in judge_docs:
#             judge_docs.append(d)
#             judge_labels.append(0)
#     final_qids.extend([qid] * len(judge_docs))
#     final_docs.extend(judge_docs)
#     final_labels.extend(judge_labels)
#
# qrels_bin = pd.DataFrame(
#     {"qid": final_qids, "docno": final_docs, "label": final_labels}
# )

num_rel_non_rel = qrels_bin.assign(
    relevant=qrels_bin["label"] == 1, non_relevant=qrels_bin["label"] == 0
    ).groupby("qid").agg(
    num_relevant=("relevant", "sum"),
    num_non_relevant=("non_relevant", "sum"),
    total_judged=("docno", "count"),
    ).reset_index()

with pd.option_context("display.max_rows", None, "display.max_columns", None):
     print(num_rel_non_rel)
