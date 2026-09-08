import pyterrier as pt
import pandas as pd

# TODO
# collection = "msmarco_passage"
# collection = "irds:msmarco-passage-v2"
collection = "msmarco_passage_v2"
ds = pt.get_dataset("msmarcov2_passage")

# indexref = ds.get_index("terrier_stemmed_text")
indexref = ds.get_index("terrier_stemmed")
index = pt.terrier.TerrierIndex(indexref)

# TODO
# topics_2019 = ds.get_topics("test-2019")
# topics_2020 = ds.get_topics("test-2020")
ds21 = pt.get_dataset("irds:msmarco-passage-v2/trec-dl-2021")
ds22 = pt.get_dataset("irds:msmarco-passage-v2/trec-dl-2022")

topics_2021 = ds21.get_topics()
topics_2022 = ds22.get_topics()

# topics_2021 = ds.get_topics("trec_2021")
# topics_2022 = ds.get_topics("trec_2022")
topics = pd.concat([topics_2021, topics_2022], ignore_index=True)

qrels_2021 = ds21.get_qrels()
qrels_2022 = ds22.get_qrels()

# qrels_2021 = ds.get_qrels("test-2021")
# qrels_2022 = ds.get_qrels("test-2022")
qrels = pd.concat([qrels_2021, qrels_2022], ignore_index=True)

print(len(qrels_2021.qid.unique()))
print(len(qrels_2022.qid.unique()))

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


num_rel_non_rel = qrels_bin.assign(
    relevant=qrels_bin["label"] == 1, non_relevant=qrels_bin["label"] == 0
    ).groupby("qid").agg(
    num_relevant=("relevant", "sum"),
    num_non_relevant=("non_relevant", "sum"),
    total_judged=("docno", "count"),
    ).reset_index()

print("Number of topics:", len(topics))

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(num_rel_non_rel)

# with pd.option_context("display.max_rows", None, "display.max_columns", None):
#     print(
#         qrels_bin.assign(
#             relevant=qrels_bin["label"] == 1, non_relevant=qrels_bin["label"] == 0
#         )
#         .groupby("qid")
#         .agg(
#             num_relevant=("relevant", "sum"),
#             num_non_relevant=("non_relevant", "sum"),
#             total_judged=("docno", "count"),
#         )
#         .reset_index()
#     )
