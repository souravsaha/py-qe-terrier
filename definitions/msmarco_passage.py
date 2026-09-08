import pyterrier as pt
import pandas as pd

collection = "msmarco_passage"
ds = pt.get_dataset(collection)

indexref = ds.get_index("terrier_stemmed_text")
index = pt.terrier.TerrierIndex(indexref)

topics_2019 = ds.get_topics("test-2019")
topics_2020 = ds.get_topics("test-2020")
topics = pd.concat([topics_2019, topics_2020], ignore_index=True)

qrels_2019 = ds.get_qrels("test-2019")
qrels_2020 = ds.get_qrels("test-2020")
qrels = pd.concat([qrels_2019, qrels_2020], ignore_index=True)

print(len(qrels_2019.qid.unique()))
print(len(qrels_2020.qid.unique()))

# Binarize qrels
# TREC DL 2019, 2020 0,1 -> non-relevant; 2,3 -> relevant
qrels_bin = qrels.copy()
qrels_bin["label"] = (qrels_bin["label"] >= 2).astype(int)

# Pick out qids that are judged in qrels
qid_range = sorted(list(set(qrels.qid)))

# filter out those topics which are not in qrels
topics = topics[topics["qid"].isin(qid_range)]

# important, otherwise returns error because of queries like "define: geon"
topics = pt.rewrite.tokenise().transform(topics)

print("Number of topics:", len(topics))

num_rel_non_rel = qrels_bin.assign(
    relevant=qrels_bin["label"] == 1, non_relevant=qrels_bin["label"] == 0
    ).groupby("qid").agg(
    num_relevant=("relevant", "sum"),
    num_non_relevant=("non_relevant", "sum"),
    total_judged=("docno", "count"),
    ).reset_index()

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(num_rel_non_rel)
