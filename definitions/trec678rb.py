import pyterrier as pt
import pandas as pd

collection = "trec678rb"

collection_documents_dir = "./collections/trec678rb/documents/"
index_loc = "./indexed/trec678rb/"
index = pt.terrier.TerrierIndex(index_loc)

topics_301_450 = pt.io.read_topics("./collections/trec678rb/topics/trec678.xml")
topics_601_700 = pt.io.read_topics("./collections/trec678rb/topics/robust.xml")
topics = pd.concat([topics_301_450, topics_601_700], ignore_index=True)


qrels_301_450 = pt.io.read_qrels("./collections/trec678rb/qrels/trec678_301-450.qrel")
qrels_601_700 = pt.io.read_qrels("./collections/trec678rb/qrels/robust_601-700.qrel")
qrels = pd.concat([qrels_301_450, qrels_601_700], ignore_index=True)

qrels_bin = qrels.copy()
qrels_bin["label"] = (qrels_bin["label"] >= 1).astype(int)

qid_range = sorted(list(set(qrels.qid)))

# filter out those topics which are not in qrels
topics = topics[topics["qid"].isin(qid_range)]

print("Number of topics:", len(topics))
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(topics)

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(
        qrels_bin.assign(
            relevant=qrels_bin["label"] == 1, non_relevant=qrels_bin["label"] == 0
        )
        .groupby("qid")
        .agg(
            num_relevant=("relevant", "sum"),
            num_non_relevant=("non_relevant", "sum"),
            total_judged=("docno", "count"),
        )
        .reset_index()
    )
