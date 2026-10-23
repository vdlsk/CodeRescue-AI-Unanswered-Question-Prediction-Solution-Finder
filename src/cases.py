"""Case-based retrieval: an inspector's free-text observation -> similar past observations ->
voted violation number -> that section's code requirements.

Build:  python src/cases.py [--until 2013-12-31]   (writes data/case_index.joblib)
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from build_code_corpus import SPLIT, is_code_sentence, iter_items
from config import CSV_NAME, INDEX_PATH, RAW_DIR, SEED
from retrieval import BM25

CASE_PATH = INDEX_PATH.parent / "case_index.joblib"
MAX_CASES = 200_000


def observation_text(comments):
    return " ".join(s for s in SPLIT.split(comments) if s.strip() and not is_code_sentence(s.strip()))


def build(until):
    df = pd.read_csv(RAW_DIR / CSV_NAME, usecols=["Inspection ID", "Inspection Date", "Violations"]).dropna()
    df["Inspection Date"] = pd.to_datetime(df["Inspection Date"], format="%m/%d/%Y")
    df = df[df["Inspection Date"] <= until]
    rows = [(iid, no, title, obs[:600]) for iid, _, no, title, c in iter_items(df)
            if len((obs := observation_text(c)).split()) >= 5]
    cases = pd.DataFrame(rows, columns=["inspection_id", "violation_no", "violation_title", "text"])
    if len(cases) > MAX_CASES:
        cases = cases.sample(MAX_CASES, random_state=SEED).reset_index(drop=True)
    bm = BM25().fit(cases.text.tolist())
    joblib.dump({"cases": cases, "bm25": bm, "until": str(until)}, CASE_PATH)
    print(f"Indexed {len(cases):,} past observations (through {until}) -> {CASE_PATH}")


class CaseRetriever:
    def __init__(self):
        d = joblib.load(CASE_PATH)
        self.cases, self.bm25 = d["cases"], d["bm25"]

    def suggest(self, observation, k_neighbors=50, top=3):
        row = self.bm25.scores([observation])[0]
        idx = np.argpartition(-row, k_neighbors)[:k_neighbors]
        idx = idx[np.argsort(-row[idx])]
        nb = self.cases.iloc[idx].assign(score=row[idx])
        votes = nb.groupby(["violation_no", "violation_title"])["score"].sum().sort_values(ascending=False)
        share = votes / votes.sum()
        out = [{"violation_no": int(no), "violation_title": t, "vote_share": float(share[(no, t)])}
               for no, t in votes.index[:top]]
        return out, nb.head(5)[["inspection_id", "violation_no", "text", "score"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="2013-12-31")
    build(ap.parse_args().until)
