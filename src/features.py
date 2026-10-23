"""Leak-free features: only information available at the moment a question is posted.

Excluded on purpose (they are only known later): score, view count, answer count, comments,
edits, close status, the asker's CURRENT reputation.
"""
import html
import re

import numpy as np
import pandas as pd

from config import UNANSWERED_DAYS

CODE_BLOCK = re.compile(r"<pre[^>]*>.*?</pre>", re.S | re.I)
INLINE_CODE = re.compile(r"<code>.*?</code>", re.S | re.I)
TAGS_RE = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def strip_html(body: str, keep_code=False) -> str:
    if not keep_code:
        body = CODE_BLOCK.sub(" [CODE] ", body)
    text = TAGS_RE.sub(" ", body)
    return WS.sub(" ", html.unescape(text)).strip()


def code_text(body: str) -> str:
    blocks = CODE_BLOCK.findall(body)
    return html.unescape(TAGS_RE.sub(" ", " ".join(blocks)))


def text_for_model(title: str, body: str) -> str:
    return f"{title} . {strip_html(body)}"


def content_features(q: pd.DataFrame) -> pd.DataFrame:
    b = q["body"].fillna("")
    t = q["title"].fillna("")
    prose = b.map(strip_html)
    f = pd.DataFrame(index=q.index)
    f["title_chars"] = t.str.len()
    f["title_words"] = t.str.split().str.len()
    f["title_has_qmark"] = t.str.contains(r"\?").astype(int)
    f["title_starts_wh"] = t.str.lower().str.match(r"^(how|why|what|when|which|is|can|does)\b").astype(int)
    f["body_chars"] = prose.str.len()
    f["body_words"] = prose.str.split().str.len()
    f["n_code_blocks"] = b.str.count(r"<pre")
    f["code_chars"] = b.map(code_text).str.len()
    f["n_inline_code"] = b.str.count(r"<code>") - f["n_code_blocks"]
    f["n_links"] = b.str.count(r"<a ")
    f["n_images"] = b.str.count(r"<img ")
    f["n_paragraphs"] = b.str.count(r"<p>")
    f["mentions_error"] = prose.str.contains(r"error|exception|traceback|fails?|crash", case=False).astype(int)
    f["n_tags"] = q["tags"].fillna("").str.split("|").map(lambda x: len([y for y in x if y]))
    created = pd.to_datetime(q["created"])
    f["hour_utc"] = created.dt.hour
    f["weekday"] = created.dt.weekday
    f["is_weekend"] = (f["weekday"] >= 5).astype(int)
    return f


def make_labels(q: pd.DataFrame, a: pd.DataFrame, dump_end: pd.Timestamp):
    """unanswered_7d = 1 if no answer was posted within 7 days. Questions newer than
    dump_end - 7 days are dropped because their outcome is not yet observable (censoring)."""
    window = pd.Timedelta(days=UNANSWERED_DAYS)
    first = a.groupby("parent_id")["created"].min().rename("first_answer_at")
    q = q.merge(first, left_on="id", right_index=True, how="left")
    q = q[q["created"] <= dump_end - window].copy()
    answered = q["first_answer_at"].notna() & (q["first_answer_at"] <= q["created"] + window)
    q["unanswered_7d"] = (~answered).astype(int)
    q["hours_to_first_answer"] = (q["first_answer_at"] - q["created"]).dt.total_seconds() / 3600
    return q


def user_history_features(q: pd.DataFrame, users: pd.DataFrame | None) -> pd.DataFrame:
    """Asker history computed strictly from the past.
    user_prior_q:           questions the asker posted before this one
    user_prior_unans_rate:  share of the asker's questions whose 7-day outcome was already known
                            (posted >= 7 days earlier) that went unanswered (smoothed)
    account_age_days:       days since the account was created (known at post time)"""
    q = q.sort_values(["created", "id"]).copy()
    out = pd.DataFrame(index=q.index)
    known = q["owner_user_id"] >= 0
    q["_one"] = 1
    q["user_prior_q"] = q.groupby("owner_user_id")["_one"].cumsum() - 1
    q.loc[~known, "user_prior_q"] = 0

    r = q.loc[known, ["owner_user_id", "created", "unanswered_7d"]].copy()
    r["cum_n"] = r.groupby("owner_user_id").cumcount() + 1
    r["cum_unans"] = r.groupby("owner_user_id")["unanswered_7d"].cumsum()
    left = q.loc[known, ["owner_user_id", "created"]].copy()
    left["cutoff"] = left["created"] - pd.Timedelta(days=UNANSWERED_DAYS)
    left = left.reset_index().sort_values("cutoff")
    right = r[["owner_user_id", "created", "cum_n", "cum_unans"]].sort_values("created").rename(
        columns={"created": "r_created"})
    m = pd.merge_asof(left, right, left_on="cutoff", right_on="r_created", by="owner_user_id",
                      direction="backward").set_index("index")
    prior_n = m["cum_n"].fillna(0)
    prior_unans = m["cum_unans"].fillna(0)
    q["user_known_outcomes"] = 0
    q.loc[m.index, "user_known_outcomes"] = prior_n
    q["user_prior_unans_rate"] = np.nan
    q.loc[m.index, "user_prior_unans_rate"] = (prior_unans + 1) / (prior_n + 2)   # Laplace smoothing
    q["user_prior_unans_rate"] = q["user_prior_unans_rate"].fillna(0.5)

    if users is not None and len(users):
        q = q.merge(users, left_on="owner_user_id", right_on="user_id", how="left").set_index(q.index)
        q["account_age_days"] = ((q["created"] - q["user_created"]).dt.total_seconds() / 86400).clip(lower=0)
    else:
        q["account_age_days"] = np.nan
    q["account_age_days"] = q["account_age_days"].fillna(-1)

    for c in ("user_prior_q", "user_known_outcomes", "user_prior_unans_rate", "account_age_days"):
        out[c] = q[c]
    return out


class TagRateEncoder:
    """Smoothed historical unanswered rate per tag, fitted on the TRAIN split only."""

    def __init__(self, alpha=20):
        self.alpha = alpha

    def fit(self, tags: pd.Series, y: pd.Series):
        ex = pd.DataFrame({"tag": tags.str.split("|"), "y": y.values}).explode("tag")
        ex = ex[ex.tag.astype(bool)]
        self.prior_ = float(y.mean())
        g = ex.groupby("tag")["y"].agg(["sum", "count"])
        self.rate_ = ((g["sum"] + self.alpha * self.prior_) / (g["count"] + self.alpha)).to_dict()
        self.count_ = g["count"].to_dict()
        return self

    def transform(self, tags: pd.Series) -> pd.DataFrame:
        rows = []
        for s in tags.fillna(""):
            ts = [t for t in s.split("|") if t]
            rates = [self.rate_.get(t, self.prior_) for t in ts] or [self.prior_]
            counts = [self.count_.get(t, 0) for t in ts] or [0]
            rows.append({"tag_rate_mean": np.mean(rates), "tag_rate_max": np.max(rates),
                         "tag_rate_min": np.min(rates), "tag_min_popularity": np.log1p(np.min(counts)),
                         "n_unseen_tags": sum(t not in self.rate_ for t in ts)})
        return pd.DataFrame(rows, index=tags.index)
