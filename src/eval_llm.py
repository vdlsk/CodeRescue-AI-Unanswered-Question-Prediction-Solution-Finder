"""Does retrieval improve LLM troubleshooting answers? Blind pairwise evaluation.

For N held-out (test-period) questions that have an accepted answer:
  A) LLM answers without retrieval
  B) LLM answers with the top-5 retrieved threads (training-period corpus only)
A separate judge model sees the question, the human accepted answer as a reference, and the two
candidates in random order, and returns a winner plus 1-5 correctness scores.

Requires ANTHROPIC_API_KEY.  Usage:  python src/eval_llm.py
"""
import json
import random
import re

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_DIR, JUDGE_MODEL, LLM_EVAL_N, LLM_MODEL, OUT_DIR, SEED
from features import strip_html
from rag import CodeRescue, call_llm, llm_available

JUDGE = """You are grading two answers to a programming question. A reference answer (the asker's
accepted answer) is given. Judge which candidate would better solve the asker's actual problem:
technical correctness and agreement with the reference first, then actionability. Ignore length
and style. Return ONLY JSON: {"winner": "1" | "2" | "tie", "score_1": 1-5, "score_2": 1-5}"""


def main():
    if not llm_available():
        raise SystemExit("Set ANTHROPIC_API_KEY to run the LLM evaluation.")
    cr = CodeRescue()
    q = pd.read_parquet(DATA_DIR / "questions.parquet")
    a = pd.read_parquet(DATA_DIR / "answers.parquet")
    sp = pd.read_parquet(DATA_DIR / "splits.parquet")
    test = q[q.id.isin(sp.loc[sp.split == "test", "id"]) & (q.accepted_answer_id > 0)]
    test = test.merge(a[["id", "body"]].rename(columns={"id": "accepted_answer_id", "body": "ref"}),
                      on="accepted_answer_id").sample(min(LLM_EVAL_N, len(test)), random_state=SEED)

    rng = random.Random(SEED)
    rows = []
    for i, r in enumerate(test.itertuples(), 1):
        base = cr.answer(r.title, r.body, use_retrieval=False)["answer"]
        rag = cr.answer(r.title, r.body, use_retrieval=True)
        flip = rng.random() < 0.5
        c1, c2 = (rag["answer"], base) if flip else (base, rag["answer"])
        prompt = (f"Question:\n{r.title}\n{strip_html(r.body, keep_code=True)[:2500]}\n\n"
                  f"Reference (accepted) answer:\n{strip_html(r.ref, keep_code=True)[:2500]}\n\n"
                  f"Candidate 1:\n{c1}\n\nCandidate 2:\n{c2}")
        raw = call_llm(JUDGE, prompt, model=JUDGE_MODEL, max_tokens=200)
        try:
            j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except Exception:  # noqa: BLE001
            continue
        rag_slot = "1" if flip else "2"
        winner = "tie" if j["winner"] == "tie" else ("rag" if j["winner"] == rag_slot else "no_rag")
        rows.append({"id": r.id, "winner": winner,
                     "score_rag": j[f"score_{rag_slot}"], "score_no_rag": j[f"score_{'2' if flip else '1'}"],
                     "rag_has_citation": rag["citations"]["has_citation"],
                     "rag_citation_valid_rate": rag["citations"]["valid_rate"]})
        print(f"{i}/{len(test)} {winner}")

    df = pd.DataFrame(rows)
    wins, losses, ties = (df.winner == "rag").sum(), (df.winner == "no_rag").sum(), (df.winner == "tie").sum()
    sign_p = stats.binomtest(int(wins), int(wins + losses), 0.5).pvalue if wins + losses else None
    wil = stats.wilcoxon(df.score_rag, df.score_no_rag) if (df.score_rag != df.score_no_rag).any() else None
    res = {
        "n_evaluated": int(len(df)), "generator_model": LLM_MODEL, "judge_model": JUDGE_MODEL,
        "rag_wins": int(wins), "no_rag_wins": int(losses), "ties": int(ties),
        "rag_win_rate_excl_ties": float(wins / (wins + losses)) if wins + losses else None,
        "sign_test_p": float(sign_p) if sign_p is not None else None,
        "mean_score_rag": float(df.score_rag.mean()), "mean_score_no_rag": float(df.score_no_rag.mean()),
        "wilcoxon_p": float(wil.pvalue) if wil else None,
        "rag_answers_with_citation": float(df.rag_has_citation.mean()),
        "rag_citation_valid_rate": float(df.rag_citation_valid_rate.dropna().mean()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "llm_eval_results.json").write_text(json.dumps(res, indent=2))
    df.to_csv(OUT_DIR / "llm_eval_rows.csv", index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
