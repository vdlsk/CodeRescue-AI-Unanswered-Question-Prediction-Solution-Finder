"""Evaluate the RAG layer.

1) Retrieval: eval/reg_gold.csv holds questions with keywords that a relevant provision must contain
   (e.g. '41°F' for cold holding). Hit@k = a top-k chunk contains any keyword. Edit/extend the gold set
   to match the documents you ingested.
2) Answer grounding (only with ANTHROPIC_API_KEY): for N establishments with failures, ask a standard
   question and measure how many numbers in the answer appear in the context and how many citations
   point to real records / retrieved provisions.

Usage:  python src/eval_rag.py
"""
import json
import os

import numpy as np
import pandas as pd

from config import OUT_DIR, ROOT, SEED
from regs import load_index

N_ANSWERS = int(os.environ.get("RAG_EVAL_N", "30"))
QUESTION = "Why does this establishment keep failing inspections, and which regulations are relevant?"


def retrieval_eval():
    chunks, retr = load_index()
    gold = pd.read_csv(ROOT / "eval" / "reg_gold.csv")
    out = {"n_queries": int(len(gold)), "dense_backend": retr.dense.backend}
    for mode in ("bm25", "dense", "hybrid"):
        hits = retr.search(gold["query"].tolist(), k=5, mode=mode)
        first = []
        for kws, h in zip(gold.any_of_keywords, hits):
            keys = [k.lower() for k in kws.split("|")]
            texts = [(chunks.iloc[i].section + " " + chunks.iloc[i].text).lower() for i in h]
            first.append(next((r for r, t in enumerate(texts) if any(k in t for k in keys)), None))
        r = np.array([np.inf if x is None else x for x in first])
        out[mode] = {f"hit@{k}": float((r < k).mean()) for k in (1, 3, 5)}
        out[mode]["mrr@5"] = float(np.where(r < 5, 1 / (r + 1), 0).mean())
        print(mode, out[mode])
    return out


def answer_eval():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set: skipping generated-answer grounding evaluation")
        return None
    from assistant import FoodSafe
    fs = FoodSafe()
    ids = fs.con.execute("SELECT license_no FROM establishments WHERE n_fails >= 2").df().license_no
    sample = ids.sample(min(N_ANSWERS, len(ids)), random_state=SEED)
    rows = []
    for lic in sample:
        g = fs.ask(int(lic), QUESTION)["grounding"]
        rows.append(g)
        print(lic, g)
    df = pd.DataFrame(rows)
    return {"n_answers": int(len(df)),
            "number_support_rate": float(df.numbers_supported.sum() / max(1, df.numbers.sum())),
            "reg_citation_valid_rate": float(df.reg_citations_valid.sum() / max(1, df.reg_citations.sum())),
            "record_citation_valid_rate": float(df.record_citations_valid.sum() / max(1, df.record_citations.sum())),
            "answers_citing_a_regulation": float((df.reg_citations > 0).mean())}


def main():
    res = {"retrieval": retrieval_eval(), "answers": answer_eval()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rag_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
