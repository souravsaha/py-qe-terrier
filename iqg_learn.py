import pandas as pd
import os
import math
import numpy as np
import pickle
import pyterrier as pt
from pyterrier.terrier import TerrierIndex
from typing import Tuple

from queryvector import QueryVector

DocumentVector = dict[str, float]


class IdealQueryGeneration:
    def __init__(
        self,
        index: TerrierIndex,
        qrels_bin: pd.DataFrame,
        collection_name: str,
        k1: float = 1.2,
        b: float = 0.75,
        num_results: int = 1000,
    ):

        self.index = index
        self.index_ref = self.index.index_ref()
        self.qrels_bin = qrels_bin
        self.k1 = k1
        self.b = b
        self.collection_name = collection_name
        self.num_results = num_results
        self.bm25_retriever = pt.terrier.Retriever(
            self.index_ref, wmodel="BM25", num_results=self.num_results
        )

        # Terrier internals
        self.di = self.index.direct_index()
        self.doi = self.index.document_index()
        self.lexicon = self.index.lexicon()
        self.meta = self.index.meta_index()

        # Collection statistics
        stats = self.index.collection_statistics()
        self.N = self.doi.getNumberOfDocuments()
        self.avg_dl = stats.averageDocumentLength

    def docno_to_docid(self, docno: str) -> int:
        return self.meta.getDocument("docno", docno)

    def bm25(self, tf: int, df: int, dl: int) -> float:
        idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
        denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
        return idf * tf / denom

    def get_document_vector(self, docid: int) -> dict:
        vec = {}
        dl = self.doi.getDocumentLength(docid)

        for posting in self.di.getPostings(self.doi.getDocumentEntry(docid)):
            termid = posting.getId()
            tf = posting.getFrequency()
            lex_entry = self.lexicon.getLexiconEntry(termid)

            term = lex_entry.getKey()
            df = self.lexicon[term].getDocumentFrequency()

            vec[term] = self.bm25(tf, df, dl)

        return vec

    def _get_rel_nonrel_docids(self, qid: str) -> Tuple[list[int], list[int]]:
        q = self.qrels_bin[self.qrels_bin.qid == qid]
        rel = q[q.label >= 1].docno
        nonrel = q[q.label == 0].docno

        rel_ids = []
        nonrel_ids = []
        for d in rel:
            if self.docno_to_docid(d) != -1:
                rel_ids.append(self.docno_to_docid(d))
        for d in nonrel:
            if self.docno_to_docid(d) != -1:
                nonrel_ids.append(self.docno_to_docid(d))

        return rel_ids, nonrel_ids

    @staticmethod
    def docvecs_to_matrix(
        docvecs: list[DocumentVector], vocab: list[str]
    ) -> np.ndarray:
        term_to_idx = {t: i for i, t in enumerate(vocab)}
        X = np.zeros((len(docvecs), len(vocab)), dtype=float)
        for i, d in enumerate(docvecs):
            for term, w in d.items():
                j = term_to_idx.get(term)
                if j is not None:
                    X[i, j] = w
        return X

    @staticmethod
    def qvec_to_array(qvec: QueryVector, term_list: list[str]) -> np.ndarray:
        final_vec = np.zeros(len(term_list))
        for i in range(len(term_list)):
            term = term_list[i]
            if term in qvec:
                final_vec[i] = qvec[term]
            else:
                final_vec[i] = 0.0
        return final_vec

    @staticmethod
    def array_to_qvec(qarray: np.ndarray, term_list: list[str]) -> QueryVector:
        qvec = QueryVector()
        for term, weight in zip(term_list, qarray):
            qvec[term] = weight
        return qvec

    @staticmethod
    def qvec_to_str(qvec: QueryVector, applypipeline=False):
        final = " ".join([f"{term}^{weight:.10f}" for term, weight in qvec.items()])
        if not applypipeline:
            final = "applypipeline:off " + final
        return final

    def search_with_qvecs(self, qvecs: dict[str, QueryVector], num_results=1000, applypipeline=False):
        qids, queries = [], []
        for qid, qvec in qvecs.items():
            qids.append(qid)
            queries.append(self.qvec_to_str(qvec, applypipeline=applypipeline))
        topics = pd.DataFrame({"qid": qids, "query": queries})
        if num_results != self.num_results:
            self.num_results = num_results
            self.bm25_retriever = pt.terrier.Retriever(
                self.index_ref, wmodel="BM25", num_results=self.num_results
            )
        ret = self.bm25_retriever.transform(topics)
        return ret

    def computeAP(self, qid: str, qvec: QueryVector):
        qvec_dict = {qid: qvec}
        ret = self.search_with_qvecs(qvec_dict)
        res = pt.Evaluate(ret, self.qrels_bin, metrics=["map"], perquery=True)
        return res[qid]["map"]

    def get_data(self, qid: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        cache_dir = getattr(self, "cache_dir", f"./cache/{self.collection_name}/")
        os.makedirs(cache_dir, exist_ok=True)

        cache_path = os.path.join(cache_dir, f"get_data_{qid}.pkl")

        if os.path.exists(cache_path):
            print(f"QID {qid}: Loading cached doc vectors.", flush=True)
            with open(cache_path, "rb") as f:
                rel_vecs, nonrel_vecs, vocab = pickle.load(f)

        else:
            rel_ids, nonrel_ids = self._get_rel_nonrel_docids(qid)

            rel_vecs, nonrel_vecs = [], []
            vocab = set()

            print(f"QID {qid}: Forming document vectors.", end=" ", flush=True)
            for d in rel_ids:
                v = self.get_document_vector(d)
                rel_vecs.append(v)
                vocab |= v.keys()

            for d in nonrel_ids:
                v = self.get_document_vector(d)
                nonrel_vecs.append(v)
                vocab |= v.keys()
            print("Done!")

            vocab = list(vocab)

            print(f"QID {qid}: Caching sparse doc vectors.", flush=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    (rel_vecs, nonrel_vecs, vocab),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        print("#rel-docs:", len(rel_vecs), "#nonrel-docs", len(nonrel_vecs))
        print("#terms:", len(vocab))

        print(f"QID {qid}: Forming design matrix.", end=" ", flush=True)
        X_rel = self.docvecs_to_matrix(rel_vecs, vocab)
        X_nonrel = self.docvecs_to_matrix(nonrel_vecs, vocab)

        X = np.vstack([X_rel, X_nonrel])
        y = np.hstack(
            [
                np.ones(len(rel_vecs), dtype=np.float32),
                np.zeros(len(nonrel_vecs), dtype=np.float32),
            ]
        )
        print("Done!")

        return X, y, vocab
