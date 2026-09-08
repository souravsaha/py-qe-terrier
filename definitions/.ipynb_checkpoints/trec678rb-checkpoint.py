import pyterrier as pt
import pandas as pd

collection_documents_dir = "./collections/trec678rb/documents/"
index_loc = "./indexed/trec678rb/"
index = pt.terrier.TerrierIndex(index_loc)

topics_301_450 = pt.io.read_topics("./collections/trec678rb/topics/trec678.xml")
topics_601_700 = pt.io.read_topics("./collections/trec678rb/topics/robust.xml")
topics = pd.concat([topics_301_450, topics_601_700])


qrels_301_450 = pt.io.read_qrels("./collections/trec678rb/qrels/trec678_301-450.qrel")
qrels_601_700 = pt.io.read_qrels("./collections/trec678rb/qrels/robust_601-700.qrel")
qrels = pd.concat([qrels_301_450, qrels_601_700])

qid_range = sorted(list(set(qrels.qid)))
