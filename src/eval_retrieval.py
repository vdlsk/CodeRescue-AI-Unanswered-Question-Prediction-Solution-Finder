"""Evaluate retrieval with real ground truth: Stack Overflow moderators' duplicate links.

For a question posted AFTER the training period that was closed as a duplicate of an older
question in the corpus, a good retriever should return that older question near the top.

Usage:  python src/eval_retrieval.py
"""
import json

import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, MODEL_DIR, OUT_DIR, SEED, TOP_K
from features import strip_html
from retrieval import HybridRetriever

MAX_QUERIES = 3000


def main():
    links_path = DATA_DIR / "duplicate_links.parquet"
    if not links_path.exists():
        raise SystemExit("No PostLinks.xml was parsed; duplicate-based evaluation needs it.")
    links = pd.read_parquet(links_path)
    q = pd.read_parquet(DATA_DIR / "questions.parquet")
    corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
    splits = pd.read_parquet(DATA_DIR / "splits.parquet")
    train_end = splits.loc[splits.split == "train", "created"].max()
    R = joblib.load(MODEL_DIR / "retriever.joblib")
    retr = HybridRetriever(R["bm25"], R["dense"], R["ids"])

    corpus_ids = set(corpus.id)
    pairs = links[links.related_post_id.isin(corpus_ids)].merge(q, left_on="post_id", right_on="id")
    pairs = pairs[pairs.created > train_end]
    gold = pairs.groupby("post_id")["related_post_id"].apply(set)
    qs = q.set_index("id").loc[gold.index]
    if len(qs) > MAX_QUERIES:
        qs = qs.sample(MAX_QUERIES, random_state=SEED)
    print(f"{len(qs):,} duplicate questions with an original in the corpus")
    queries = [f"{t} . {strip_html(b)[:2000]}" for t, b in zip(qs.title, qs.body)]
    ids = R["ids"]

    res = {"n_queries": int(len(qs)), "dense_backend": R["dense"].backend, "modes": {}}
    rng = np.random.default_rng(SEED)
    for mode in ("bm25", "dense", "hybrid"):
        hits_rank = []
        for start in range(0, len(queries), 256):
            batch = retr.search(queries[start:start + 256], k=TOP_K, mode=mode)
            for qi, ranked in zip(qs.index[start:start + 256], batch):
                g = gold[qi]
                r = next((i for i, d in enumerate(ranked) if ids[d] in g), None)
                hits_rank.append(r)
        ranks = np.array([np.inf if r is None else r for r in hits_rank])
        m = {f"recall@{k}": float((ranks < k).mean()) for k in (1, 5, 10)}
        m["mrr@10"] = float(np.where(ranks < 10, 1 / (ranks + 1), 0).mean())
        boots = [(ranks[rng.integers(0, len(ranks), len(ranks))] < 10).mean() for _ in range(500)]
        m["recall@10_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        res["modes"][mode] = m
        print(mode, {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})

    res["as_of_posting"] = as_of_posting_eval(links, q, rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval_results.json").write_text(json.dumps(res, indent=2))


def as_of_posting_eval(links, q, rng):
    """Uses ALL duplicate pairs. For each duplicate question, search only solved questions whose question AND best
    answer existed before the duplicate was posted - what the app could have shown at posting time."""
    from build_index import index_text
    from retrieval import BM25, Dense
    a = pd.read_parquet(DATA_DIR / "answers.parquet")
    acc = a.merge(q[["id", "accepted_answer_id"]], left_on="parent_id", right_on="id", suffixes=("", "_q"))
    acc = acc[acc["id"] == acc["accepted_answer_id"]]
    top = a[a.score >= 1].sort_values("score", ascending=False).drop_duplicates("parent_id")
    best = pd.concat([acc[top.columns], top]).drop_duplicates("parent_id").rename(
        columns={"id": "answer_id", "created": "answer_created"})[["parent_id", "answer_created"]]
    corpus = q.merge(best, left_on="id", right_on="parent_id").sort_values("id").reset_index(drop=True)
    docs = [index_text(r) for r in corpus.itertuples()]
    retr = HybridRetriever(BM25().fit(docs), Dense().fit(docs), corpus.id.values)
    pairs = links.merge(q[["id", "created", "title", "body"]], left_on="post_id", right_on="id")
    gold = pairs.groupby("post_id")["related_post_id"].apply(set)
    qs = q.set_index("id").loc[gold.index]
    ready = np.maximum(corpus.created.values, corpus.answer_created.values)
    excl = [corpus.id.values[ready >= t].tolist() + [qid] for qid, t in zip(qs.index, qs.created)]
    keep = [i for i, qid in enumerate(qs.index) if gold[qid] - set(excl[i])]   # original must be searchable
    qs, excl = qs.iloc[keep], [excl[i] for i in keep]
    queries = [f"{t} . {strip_html(b)[:2000]}" for t, b in zip(qs.title, qs.body)]
    out = {"n_queries": int(len(qs)), "corpus": int(len(corpus)), "dense_backend": retr.dense.backend}
    for mode in ("bm25", "dense", "hybrid"):
        ranked = retr.search(queries, k=TOP_K, mode=mode, exclude_ids=excl)
        ranks = np.array([next((i for i, d in enumerate(r) if corpus.id.values[d] in gold[qid]), np.inf)
                          for r, qid in zip(ranked, qs.index)])
        m = {f"recall@{k}": float((ranks < k).mean()) for k in (1, 5, 10)}
        m["mrr@10"] = float(np.where(ranks < 10, 1 / (ranks + 1), 0).mean())
        boots = [(ranks[rng.integers(0, len(ranks), len(ranks))] < 10).mean() for _ in range(500)]
        m["recall@10_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        out[mode] = m
        print("as-of-posting", mode, {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
    print(f"as-of-posting: {len(qs)} duplicate questions, corpus {len(corpus):,}")
    return out


if __name__ == "__main__":
    main()
