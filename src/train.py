"""Train and evaluate the 'will this question be unanswered after 7 days?' classifier.

Temporal split (oldest 70% train, next 15% validation, newest 15% test) so the model is always
evaluated on questions posted after everything it learned from.

Usage:  python src/train.py
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import DATA_DIR, MODEL_DIR, OUT_DIR, SEED, TRAIN_FRAC, VALID_FRAC
from features import (TagRateEncoder, content_features, make_labels, text_for_model,
                      user_history_features)


def precision_at(y, p, frac=0.10):
    k = max(1, int(len(p) * frac))
    idx = np.argsort(-p)[:k]
    return float(y[idx].mean()), float(y[idx].sum() / max(1, y.sum()))


def bootstrap_ci(y, p, fn, n=500, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metrics(y, p):
    prec10, rec10 = precision_at(y, p, 0.10)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "roc_auc_ci95": bootstrap_ci(y, p, roc_auc_score),
        "pr_auc": float(average_precision_score(y, p)),
        "pr_auc_ci95": bootstrap_ci(y, p, average_precision_score),
        "brier": float(brier_score_loss(y, p)),
        "precision_at_top10pct": prec10,
        "recall_at_top10pct": rec10,
        "lift_at_top10pct": prec10 / float(y.mean()),
    }


def main():
    q = pd.read_parquet(DATA_DIR / "questions.parquet")
    a = pd.read_parquet(DATA_DIR / "answers.parquet")
    users = pd.read_parquet(DATA_DIR / "users.parquet") if (DATA_DIR / "users.parquet").exists() else None
    dump_end = max(q.created.max(), a.created.max() if len(a) else q.created.max())

    q = make_labels(q, a, dump_end)
    q = q.sort_values(["created", "id"]).reset_index(drop=True)
    print(f"{len(q):,} labelled questions, unanswered-7d rate {q.unanswered_7d.mean():.3f}")

    feats = content_features(q).join(user_history_features(q, users))
    n = len(q)
    i_tr, i_va = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VALID_FRAC))
    split = np.array(["train"] * i_tr + ["valid"] * (i_va - i_tr) + ["test"] * (n - i_va))
    q["split"] = split
    tr, va, te = (split == "train"), (split == "valid"), (split == "test")
    y = q["unanswered_7d"].values

    enc = TagRateEncoder().fit(q.loc[tr, "tags"], q.loc[tr, "unanswered_7d"])
    X = feats.join(enc.transform(q["tags"]))
    feature_names = list(X.columns)

    # 1) Metadata model: LightGBM
    gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                             verbose=-1)
    gbm.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)])
    p_gbm = gbm.predict_proba(X)[:, 1]

    # 2) Text model: TF-IDF + logistic regression (C tuned on validation)
    text = [text_for_model(t, b) for t, b in zip(q.title, q.body)]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True,
                            token_pattern=r"(?u)\b[\w#+.\-]{2,}\b")
    T_tr = tfidf.fit_transform([text[i] for i in np.where(tr)[0]])
    T_all = tfidf.transform(text)
    best = None
    for C in (0.1, 0.3, 1.0, 3.0):
        lr = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
        lr.fit(T_tr, y[tr])
        auc = roc_auc_score(y[va], lr.predict_proba(T_all[va])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, lr)
    lr = best[2]
    p_lr = lr.predict_proba(T_all)[:, 1]

    # 3) Blend, weight chosen on validation only
    weights = np.linspace(0, 1, 11)
    w = max(weights, key=lambda w: roc_auc_score(y[va], w * p_gbm[va] + (1 - w) * p_lr[va]))
    p_blend = w * p_gbm + (1 - w) * p_lr

    results = {
        "data": {
            "questions_labelled": int(n),
            "date_range": [str(q.created.min().date()), str(q.created.max().date())],
            "unanswered_rate_overall": float(y.mean()),
            "split_sizes": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
            "split_dates": {"train_end": str(q.created[i_tr - 1].date()), "test_start": str(q.created[i_va].date())},
            "unanswered_rate_test": float(y[te].mean()),
        },
        "models": {
            "baseline_constant": {"roc_auc": 0.5, "pr_auc": float(y[te].mean())},
            "lightgbm_metadata": metrics(y[te], p_gbm[te]),
            "tfidf_logreg_text": {**metrics(y[te], p_lr[te]), "C": best[1]},
            "blend": {**metrics(y[te], p_blend[te]), "weight_metadata": float(w)},
        },
        "top_features_gain": dict(sorted(zip(feature_names, gbm.booster_.feature_importance("gain").tolist()),
                                         key=lambda kv: -kv[1])[:12]),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "model_results.json").write_text(json.dumps(results, indent=2))
    joblib.dump({"gbm": gbm, "tfidf": tfidf, "lr": lr, "enc": enc, "w": float(w),
                 "feature_names": feature_names}, MODEL_DIR / "unanswered_model.joblib")
    q["p_unanswered"] = p_blend
    q.drop(columns=["body"]).to_parquet(DATA_DIR / "questions_scored.parquet", index=False)
    q[["id", "created", "split", "unanswered_7d"]].to_parquet(DATA_DIR / "splits.parquet", index=False)

    print(json.dumps(results["models"], indent=2))


if __name__ == "__main__":
    main()
