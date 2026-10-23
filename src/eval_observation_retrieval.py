"""Retrieval evaluation with real ground truth.

Query  = an inspector's observation text from a HELD-OUT year (code-language sentences removed),
         e.g. "OBSERVED MOUSE DROPPINGS UNDER PREP TABLE".
Target = the provision section for the violation number the inspector assigned to it.
Hit@k  = a top-k retrieved chunk belongs to that violation section.

Usage:  python src/eval_observation_retrieval.py [--test-from 2014-01-01] [--n 3000]
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from build_code_corpus import SPLIT, is_code_sentence, iter_items
from config import CSV_NAME, OUT_DIR, RAW_DIR, SEED
from regs import load_index
from retrieval import BM25

SEC = re.compile(r"Section\s+(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-from", default="2014-01-01")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    chunks, retr = load_index()
    sec_no = chunks.section.map(lambda s: int(SEC.search(s).group(1)) if SEC.search(s) else -1).values
    covered = set(sec_no[sec_no >= 0])

    df = pd.read_csv(RAW_DIR / CSV_NAME, usecols=["Inspection ID", "Inspection Date", "Violations"]).dropna()
    df["Inspection Date"] = pd.to_datetime(df["Inspection Date"], format="%m/%d/%Y")
    df = df[df["Inspection Date"] >= args.test_from]
    rows = []
    for _, _, no, _, comments in iter_items(df):
        if no not in covered:
            continue
        obs = " ".join(s for s in SPLIT.split(comments) if s.strip() and not is_code_sentence(s.strip()))
        if len(obs.split()) >= 5:
            rows.append((no, obs[:600]))
    q = pd.DataFrame(rows, columns=["violation_no", "query"])
    n_pool = len(q)
    q = q.sample(min(args.n, len(q)), random_state=SEED)
    # majority-class baseline: always guess the most common violation number
    base_top = q.violation_no.value_counts().index[:5]
    res = {"n_queries": int(len(q)), "pool": int(n_pool), "n_sections": len(covered),
           "dense_backend": retr.dense.backend, "test_from": args.test_from,
           "baseline_most_frequent": {f"hit@{k}": float(q.violation_no.isin(base_top[:k]).mean()) for k in (1, 3, 5)}}
    for mode in ("bm25", "dense", "hybrid"):
        hits = retr.search(q["query"].tolist(), k=15, mode=mode)
        ranks = []
        for target, h in zip(q.violation_no, hits):
            secs = []
            for i in h:                     # rank by distinct sections, not chunks
                if sec_no[i] not in secs:
                    secs.append(sec_no[i])
            ranks.append(secs.index(target) if target in secs else np.inf)
        r = np.array(ranks)
        res[mode] = {f"hit@{k}": float((r < k).mean()) for k in (1, 3, 5)}
        res[mode]["mrr@5"] = float(np.where(r < 5, 1 / (r + 1), 0).mean())
        print(mode, {k: round(v, 3) for k, v in res[mode].items()})
    # Case-based retrieval: find similar TRAINING-period observations, let their violation numbers vote,
    # then return that section's provisions. Only data before --test-from is indexed.
    tr = pd.read_csv(RAW_DIR / CSV_NAME, usecols=["Inspection ID", "Inspection Date", "Violations"]).dropna()
    tr["Inspection Date"] = pd.to_datetime(tr["Inspection Date"], format="%m/%d/%Y")
    tr = tr[tr["Inspection Date"] < args.test_from]
    cases = []
    for _, _, no, _, comments in iter_items(tr):
        obs = " ".join(s for s in SPLIT.split(comments) if s.strip() and not is_code_sentence(s.strip()))
        if len(obs.split()) >= 5:
            cases.append((no, obs[:600]))
    cases = pd.DataFrame(cases, columns=["violation_no", "text"])
    if len(cases) > 200_000:
        cases = cases.sample(200_000, random_state=SEED)
    cbm = BM25().fit(cases.text.tolist())
    labels = cases.violation_no.values
    ranks = []
    queries = q["query"].tolist()
    for start in range(0, len(queries), 100):
        S = cbm.scores(queries[start:start + 100])
        for row, target in zip(S, q.violation_no.values[start:start + 100]):
            top = np.argpartition(-row, 50)[:50]
            votes = pd.Series(row[top], index=labels[top]).groupby(level=0).sum().sort_values(ascending=False)
            order = list(votes.index)
            ranks.append(order.index(target) if target in order else np.inf)
    r = np.array(ranks)
    # Robustness: queries that quote a municipal-code section (e.g. 7-38-005) could give the answer away
    cites = q["query"].str.contains(r"\b7-\d{2}-\d{3}", regex=True).values
    res["case_based_without_code_citations"] = {
        "n": int((~cites).sum()), **{f"hit@{k}": float((r[~cites] < k).mean()) for k in (1, 3, 5)}}
    res["share_queries_with_code_citation"] = float(cites.mean())
    print("case_based (no code citations)", res["case_based_without_code_citations"])
    res["case_based"] = {f"hit@{k}": float((r < k).mean()) for k in (1, 3, 5)}
    res["case_based"]["mrr@5"] = float(np.where(r < 5, 1 / (r + 1), 0).mean())
    res["case_based"]["n_training_cases"] = int(len(cases))
    print("case_based", {k: round(v, 3) for k, v in res["case_based"].items()})
    print("baseline", res["baseline_most_frequent"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "observation_retrieval_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
