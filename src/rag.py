"""CodeRescue core: risk scoring + retrieval + evidence-grounded answer generation."""
import os
import re

import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, LLM_MODEL, MODEL_DIR, TOP_K
from features import TagRateEncoder, content_features, strip_html  # noqa: F401  (unpickling)
from retrieval import HybridRetriever

SYSTEM_RAG = """You are CodeRescue, a troubleshooting assistant for developers.
Answer ONLY from the numbered Stack Overflow threads provided. Cite every technical claim with
the thread number in square brackets, e.g. [2]. If the threads do not contain a fix for this
problem, say so plainly and list what extra information the asker should add. Be concise:
a short diagnosis, then concrete steps or code."""

SYSTEM_NO_RAG = """You are a troubleshooting assistant for developers. Give a concise diagnosis,
then concrete steps or code."""


def llm_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_llm(system, user, model=LLM_MODEL, max_tokens=900):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class CodeRescue:
    def __init__(self):
        self.m = joblib.load(MODEL_DIR / "unanswered_model.joblib")
        R = joblib.load(MODEL_DIR / "retriever.joblib")
        self.retr = HybridRetriever(R["bm25"], R["dense"], R["ids"])
        self.corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")

    # ---------- risk ----------
    def risk(self, title, body_html, tags):
        q = pd.DataFrame([{"title": title, "body": body_html, "tags": "|".join(tags),
                           "created": pd.Timestamp.utcnow().tz_localize(None)}])
        X = content_features(q)
        # A new asker with no history (the app does not know who is asking)
        X["user_prior_q"], X["user_known_outcomes"] = 0, 0
        X["user_prior_unans_rate"], X["account_age_days"] = 0.5, -1
        X = X.join(self.m["enc"].transform(q["tags"]))[self.m["feature_names"]]
        p_gbm = self.m["gbm"].predict_proba(X)[:, 1][0]
        p_lr = self.m["lr"].predict_proba(self.m["tfidf"].transform([f"{title} . {strip_html(body_html)}"]))[:, 1][0]
        return float(self.m["w"] * p_gbm + (1 - self.m["w"]) * p_lr), X.iloc[0].to_dict()

    # ---------- retrieval ----------
    def retrieve(self, title, body_html, k=5, exclude_ids=None):
        query = f"{title} . {strip_html(body_html)[:2000]}"
        rows = self.retr.search([query], k=k, exclude_ids=[exclude_ids] if exclude_ids else None)[0]
        hits = self.corpus.iloc[rows].copy()
        hits["rank"] = np.arange(1, len(hits) + 1)
        hits["url"] = "https://stackoverflow.com/q/" + hits["id"].astype(str)
        return hits

    @staticmethod
    def context(hits, max_chars=1800):
        blocks = []
        for i, r in enumerate(hits.itertuples(), 1):
            blocks.append(f"[{i}] {r.title}\nQuestion: {strip_html(r.body)[:600]}\n"
                          f"{'Accepted' if r.is_accepted else 'Top'} answer (score {r.answer_score}): "
                          f"{strip_html(r.answer_body, keep_code=True)[:max_chars]}")
        return "\n\n".join(blocks)

    # ---------- generation ----------
    def answer(self, title, body_html, use_retrieval=True, k=5, exclude_ids=None):
        question = f"Title: {title}\n\n{strip_html(body_html, keep_code=True)[:3000]}"
        if not use_retrieval:
            return {"answer": call_llm(SYSTEM_NO_RAG, question), "hits": None}
        hits = self.retrieve(title, body_html, k=k, exclude_ids=exclude_ids)
        user = f"Stack Overflow threads:\n\n{self.context(hits)}\n\n---\nNew question:\n{question}"
        if llm_available():
            text = call_llm(SYSTEM_RAG, user)
        else:  # extractive fallback so the app still works without an API key
            top = hits.iloc[0]
            text = (f"Closest solved thread [1]: **{top.title}**\n\n"
                    f"{strip_html(top.answer_body, keep_code=True)[:900]} [1]\n\n"
                    "(Set ANTHROPIC_API_KEY for a synthesized, cited answer across all retrieved threads.)")
        return {"answer": text, "hits": hits, "citations": citation_check(text, len(hits))}


def citation_check(text, k):
    cited = [int(c) for c in re.findall(r"\[(\d+)\]", text)]
    valid = [c for c in cited if 1 <= c <= k]
    return {"n_citations": len(cited), "n_valid": len(valid),
            "valid_rate": (len(valid) / len(cited)) if cited else None,
            "has_citation": bool(cited)}
