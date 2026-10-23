"""Keyword (BM25), semantic (dense embeddings) and hybrid (reciprocal rank fusion) retrieval."""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from config import DENSE_MODEL

RRF_K = 60

TOKEN = r"(?u)\b[\w#+.\-]{2,}\b"


class BM25:
    """BM25 as a sparse matrix product: score(q, d) = sum_t idf(t) * tf(t,d)*(k1+1)/(tf + k1*(1-b+b*|d|/avgdl))."""

    def __init__(self, k1=1.2, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, docs):
        self.vec = CountVectorizer(token_pattern=TOKEN, lowercase=True, min_df=1)
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.W = sp.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape).T.tocsr()  # terms x docs
        return self

    def scores(self, queries):
        Q = self.vec.transform(queries)
        Q.data[:] = 1.0
        return (Q @ self.W).toarray()


class Dense:
    """Sentence-transformer embeddings; falls back to LSA (TF-IDF + SVD) if the model is unavailable."""

    def __init__(self, model_name=DENSE_MODEL, dim=256):
        self.model_name, self.dim, self.backend = model_name, dim, None

    def _st(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_name)

    def fit(self, docs):
        try:
            self.model = self._st()
            self.backend = self.model_name
            self.E = self.model.encode(docs, batch_size=128, normalize_embeddings=True,
                                       show_progress_bar=True).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"  dense model unavailable ({type(e).__name__}); using LSA fallback")
            self.backend = "lsa_tfidf_svd"
            self.tfidf = TfidfVectorizer(token_pattern=TOKEN, sublinear_tf=True, min_df=2)
            X = self.tfidf.fit_transform(docs)
            self.svd = TruncatedSVD(n_components=min(self.dim, X.shape[1] - 1), random_state=0)
            self.E = normalize(self.svd.fit_transform(X)).astype(np.float32)
        return self

    def encode(self, queries):
        if self.backend == "lsa_tfidf_svd":
            return normalize(self.svd.transform(self.tfidf.transform(queries))).astype(np.float32)
        if not hasattr(self, "model"):
            self.model = self._st()
        return self.model.encode(queries, normalize_embeddings=True).astype(np.float32)

    def scores(self, queries):
        return self.encode(queries) @ self.E.T

    def __getstate__(self):
        s = self.__dict__.copy()
        s.pop("model", None)          # reload the transformer lazily instead of pickling it
        return s


def top_k(scores_row, k, exclude=None):
    if exclude is not None:
        scores_row = scores_row.copy()
        scores_row[exclude] = -np.inf
    k = min(k, len(scores_row))
    idx = np.argpartition(-scores_row, k - 1)[:k]
    return idx[np.argsort(-scores_row[idx])]


def rrf(rank_lists, k=RRF_K, top=10):
    score = {}
    for ranks in rank_lists:
        for r, d in enumerate(ranks):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r + 1)
    return [d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])[:top]]


class HybridRetriever:
    def __init__(self, bm25: BM25, dense: Dense, corpus_ids):
        self.bm25, self.dense = bm25, dense
        self.ids = np.asarray(corpus_ids)

    def search(self, queries, k=10, depth=100, exclude_ids=None, mode="hybrid"):
        """Returns, for each query, a list of corpus row indices (best first)."""
        out = []
        sb = self.bm25.scores(queries) if mode in ("bm25", "hybrid") else None
        sd = self.dense.scores(queries) if mode in ("dense", "hybrid") else None
        for i in range(len(queries)):
            excl = None
            if exclude_ids is not None and exclude_ids[i] is not None:
                excl = np.where(np.isin(self.ids, exclude_ids[i]))[0]
            if mode == "bm25":
                out.append(list(top_k(sb[i], k, excl)))
            elif mode == "dense":
                out.append(list(top_k(sd[i], k, excl)))
            else:
                out.append(rrf([top_k(sb[i], depth, excl), top_k(sd[i], depth, excl)], top=k))
        return out
