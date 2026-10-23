"""Build the retrieval corpus (solved questions from the TRAIN period only) and its indexes.

Usage:  python src/build_index.py
"""
import joblib
import pandas as pd

from config import CORPUS_MIN_ANSWER_SCORE, DATA_DIR, MODEL_DIR
from features import strip_html
from retrieval import BM25, Dense


def index_text(r):
    return f"{r.title} . {r.title} . {strip_html(r.body)[:2000]} . {r.tags.replace('|', ' ')}"


def main():
    q = pd.read_parquet(DATA_DIR / "questions.parquet")
    a = pd.read_parquet(DATA_DIR / "answers.parquet")
    splits = pd.read_parquet(DATA_DIR / "splits.parquet")
    train_end = splits.loc[splits.split == "train", "created"].max()

    # Best answer per question: the accepted one, else highest-scoring with score >= threshold
    acc = a.merge(q[["id", "accepted_answer_id"]], left_on="parent_id", right_on="id", suffixes=("", "_q"))
    acc = acc[acc["id"] == acc["accepted_answer_id"]].assign(is_accepted=True)
    top = (a[a.score >= CORPUS_MIN_ANSWER_SCORE].sort_values("score", ascending=False)
           .drop_duplicates("parent_id").assign(is_accepted=False))
    best = (pd.concat([acc[top.columns], top]).drop_duplicates("parent_id", keep="first")
            .rename(columns={"id": "answer_id", "body": "answer_body", "score": "answer_score",
                             "created": "answer_created"}))

    corpus = q[q.created <= train_end].merge(best, left_on="id", right_on="parent_id")
    corpus = corpus.sort_values("id").reset_index(drop=True)
    print(f"Corpus: {len(corpus):,} solved questions posted on/before {train_end.date()}")

    docs = [index_text(r) for r in corpus.itertuples()]
    bm25 = BM25().fit(docs)
    dense = Dense().fit(docs)
    print(f"Dense backend: {dense.backend}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(DATA_DIR / "corpus.parquet", index=False)
    joblib.dump({"bm25": bm25, "dense": dense, "ids": corpus["id"].values}, MODEL_DIR / "retriever.joblib")


if __name__ == "__main__":
    main()
