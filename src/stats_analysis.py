"""Repeat-failure statistics.

Q1  Which establishments fail repeatedly, beyond what their peers would predict?
    Expected failures = sum of peer fail rates (same facility type x risk level x year x inspection class).
    One-sided Poisson test of observed vs expected, Benjamini-Hochberg FDR across establishments.
Q2  Is a failed inspection associated with failing the NEXT routine inspection?
    Consecutive canvass (routine) inspections only. Re-inspections are excluded because they are
    triggered by failures. Logistic regression with facility type, risk, year and time-gap controls,
    clustered by establishment; then adds the establishment's failure rate on its OTHER inspections
    to separate 'persistently weak establishment' from 'recent failure'.
Q1b Which violations recur? P(cited again at next inspection | cited now) vs P(... | not cited now).

Usage:  python src/stats_analysis.py
"""
import json

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from config import DB_PATH, FDR_Q, MIN_INSPECTIONS_FOR_FLAG, OUT_DIR


def risk_ratio(a, b, c, d):
    """a = exposed & event, b = exposed & no event, c = unexposed & event, d = unexposed & no event."""
    r1, r0 = a / (a + b), c / (c + d)
    rr = r1 / r0
    se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    orr = (a * d) / (b * c)
    se_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    chi2, p, _, _ = stats.chi2_contingency([[a, b], [c, d]])
    return {"p_fail_after_fail": r1, "p_fail_after_pass": r0, "risk_ratio": rr,
            "rr_ci95": [rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)],
            "odds_ratio": orr, "or_ci95": [orr * np.exp(-1.96 * se_or), orr * np.exp(1.96 * se_or)],
            "chi2": chi2, "p_value": p, "n_pairs": int(a + b + c + d)}


def ame(fit, df, var):
    d1, d0 = df.copy(), df.copy()
    d1[var], d0[var] = 1, 0
    return float((fit.predict(d1) - fit.predict(d0)).mean())


def sequence_pairs(insp, classes):
    s = insp[insp.inspection_class.isin(classes)].sort_values(["license_no", "inspection_date", "inspection_id"]).copy()
    g = s.groupby("license_no")
    s["prev_failed"] = g["failed"].shift(1)
    s["prev2_failed"] = g["failed"].shift(2)
    s["days_since_prev"] = (s["inspection_date"] - g["inspection_date"].shift(1)).dt.days
    s["n_prior"] = g.cumcount()
    tot = g["failed"].transform("sum")
    cnt = g["failed"].transform("count")
    # establishment fail rate on its OTHER inspections (excluding this one and the previous one)
    other_n = cnt - 2
    s["other_fail_rate"] = np.where(other_n > 0, (tot - s["failed"] - s["prev_failed"].fillna(0)) / other_n.clip(lower=1), np.nan)
    s = s.dropna(subset=["prev_failed"])
    s = s[s.days_since_prev > 0]
    s["prev_failed"] = s["prev_failed"].astype(int)
    s["year"] = s["inspection_date"].dt.year.astype(str)
    s["log_days"] = np.log(s["days_since_prev"])
    s["risk"] = s["risk_level"].fillna(0).astype(int).astype(str)
    return s


def q2_state_dependence(insp):
    out = {}
    for name, classes in {"canvass_to_canvass": ["canvass"],
                          "any_regular": ["canvass", "complaint", "license", "food poisoning", "other"]}.items():
        s = sequence_pairs(insp, classes)
        a = int(((s.prev_failed == 1) & (s.failed == 1)).sum()); b = int(((s.prev_failed == 1) & (s.failed == 0)).sum())
        c = int(((s.prev_failed == 0) & (s.failed == 1)).sum()); d = int(((s.prev_failed == 0) & (s.failed == 0)).sum())
        res = {"crude": risk_ratio(a, b, c, d), "n_establishments": int(s.license_no.nunique())}

        f1 = "failed ~ prev_failed + C(facility_type) + C(risk) + C(year) + log_days + np.log1p(n_prior)"
        m1 = smf.glm(f1, s, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": s["license_no"]})
        res["adjusted"] = {"odds_ratio": float(np.exp(m1.params["prev_failed"])),
                           "or_ci95": np.exp(m1.conf_int().loc["prev_failed"]).tolist(),
                           "p_value": float(m1.pvalues["prev_failed"]),
                           "avg_marginal_effect_pp": 100 * ame(m1, s, "prev_failed"), "n": int(m1.nobs)}

        s2 = s.dropna(subset=["other_fail_rate"])
        m2 = smf.glm(f1 + " + other_fail_rate", s2,
                     family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": s2["license_no"]})
        res["adjusted_for_establishment_history"] = {
            "odds_ratio": float(np.exp(m2.params["prev_failed"])),
            "or_ci95": np.exp(m2.conf_int().loc["prev_failed"]).tolist(),
            "p_value": float(m2.pvalues["prev_failed"]),
            "avg_marginal_effect_pp": 100 * ame(m2, s2, "prev_failed"), "n": int(m2.nobs),
            "other_fail_rate_coef": float(m2.params["other_fail_rate"])}

        dose = []
        s3 = s.dropna(subset=["prev2_failed"])
        for k in (0, 1, 2):
            sub = s3[(s3.prev_failed + s3.prev2_failed) == k]
            if len(sub):
                lo, hi = proportion_confint(sub.failed.sum(), len(sub), method="wilson")
                dose.append({"prior_fails_of_last_2": k, "n": int(len(sub)), "fail_rate": float(sub.failed.mean()),
                             "ci95": [float(lo), float(hi)]})
        res["dose_response"] = dose
        out[name] = res
    return out


def q1_unusual_establishments(con, insp):
    reg = insp[insp.inspection_class != "reinspection"].copy()
    reg["year"] = reg.inspection_date.dt.year
    reg["risk"] = reg.risk_level.fillna(0)
    keys = ["facility_type", "risk", "year", "inspection_class"]
    reg["p_peer"] = reg.groupby(keys)["failed"].transform("mean")
    est = reg.groupby("license_no").agg(n=("failed", "size"), observed=("failed", "sum"), expected=("p_peer", "sum"),
                                        name=("dba_name", "last"), facility_type=("facility_type", "last"),
                                        address=("address", "last")).reset_index()
    est = est[est.n >= MIN_INSPECTIONS_FOR_FLAG].copy()
    est["ratio"] = est.observed / est.expected
    est["p_value"] = stats.poisson.sf(est.observed - 1, est.expected)
    est["q_value"] = multipletests(est.p_value, method="fdr_bh")[1]
    est["flagged"] = est.q_value < FDR_Q
    con.execute("CREATE OR REPLACE TABLE establishment_flags AS SELECT * FROM est")
    top = est[est.flagged].sort_values(["q_value", "ratio"], ascending=[True, False]).head(15)
    return {"establishments_tested": int(len(est)), "flagged_fdr05": int(est.flagged.sum()),
            "flagged_pct": float(100 * est.flagged.mean()),
            "uncorrected_p05": int((est.p_value < 0.05).sum()),
            "expected_false_positives_at_p05": float(0.05 * len(est)),
            "min_p_value": float(est.p_value.min()),
            "share_of_all_fails_from_flagged": float(est.loc[est.flagged, "observed"].sum() / est.observed.sum()),
            "median_ratio_flagged": float(est.loc[est.flagged, "ratio"].median()) if est.flagged.any() else None,
            "top_flagged": top[["license_no", "name", "facility_type", "n", "observed", "expected", "ratio",
                                "q_value"]].to_dict(orient="records")}


def q1b_recurring_violations(con):
    df = con.execute("""
        WITH seq AS (
            SELECT inspection_id, license_no, inspection_date,
                   lead(inspection_id) OVER (PARTITION BY license_no ORDER BY inspection_date, inspection_id) AS next_id
            FROM inspections
            WHERE new_code_era AND has_outcome AND inspection_class <> 'reinspection' AND license_no > 0
        ),
        pairs AS (SELECT * FROM seq WHERE next_id IS NOT NULL),
        codes AS (SELECT violation_no FROM violation_codes WHERE citations >= 200),
        cited AS (SELECT DISTINCT inspection_id, violation_no FROM violations WHERE violation_no IS NOT NULL),
        grid AS (
            SELECT p.inspection_id, p.next_id, c.violation_no,
                   (n.inspection_id IS NOT NULL) AS now, (x.inspection_id IS NOT NULL) AS nxt
            FROM pairs p CROSS JOIN codes c
            LEFT JOIN cited n ON n.inspection_id = p.inspection_id AND n.violation_no = c.violation_no
            LEFT JOIN cited x ON x.inspection_id = p.next_id     AND x.violation_no = c.violation_no
        )
        SELECT g.violation_no, vc.violation_title,
               sum(CASE WHEN now THEN 1 ELSE 0 END)                       AS times_cited,
               avg(CASE WHEN nxt THEN 1.0 ELSE 0 END) FILTER (WHERE now)     AS p_recur,
               avg(CASE WHEN nxt THEN 1.0 ELSE 0 END) FILTER (WHERE NOT now) AS p_new
        FROM grid g JOIN violation_codes vc USING (violation_no)
        GROUP BY ALL
    """).df()
    df["recurrence_lift"] = df.p_recur / df.p_new
    est_repeat = con.execute("""SELECT violation_no, count(*) AS establishments_with_repeat
                                FROM repeat_violations GROUP BY 1""").df()
    df = df.merge(est_repeat, on="violation_no", how="left").fillna({"establishments_with_repeat": 0})
    df = df.sort_values("establishments_with_repeat", ascending=False)
    con.execute("CREATE OR REPLACE TABLE violation_recurrence AS SELECT * FROM df")
    return df.head(10).to_dict(orient="records")


def main():
    con = duckdb.connect(str(DB_PATH))
    insp = con.execute("""
        SELECT inspection_id, license_no, dba_name, address, facility_type, risk_level, inspection_date,
               inspection_class, failed
        FROM inspections WHERE has_outcome AND license_no > 0
    """).df()
    insp["inspection_date"] = pd.to_datetime(insp["inspection_date"])
    print("Q2: state dependence ...")
    q2 = q2_state_dependence(insp)
    print("Q1: unusual establishments ...")
    q1 = q1_unusual_establishments(con, insp)
    print("Q1b: recurring violations ...")
    q1b = q1b_recurring_violations(con)
    top_repeat_est = con.execute("""
        SELECT e.license_no, e.name, e.facility_type, e.n_inspections, e.n_fails
        FROM establishments e WHERE e.n_inspections >= 5 ORDER BY e.n_fails DESC LIMIT 15""").df()

    res = {"data": json.loads((OUT_DIR / "build_stats.json").read_text()),
           "q1_most_fails": top_repeat_est.to_dict(orient="records"),
           "q1_unusual_establishments": q1, "q1_recurring_violations": q1b, "q2_state_dependence": q2}
    (OUT_DIR / "stats_results.json").write_text(json.dumps(res, indent=2, default=float))
    c = q2["canvass_to_canvass"]
    print(f"\nCanvass->canvass: P(fail|prev fail)={c['crude']['p_fail_after_fail']:.3f} vs "
          f"P(fail|prev pass)={c['crude']['p_fail_after_pass']:.3f}, RR={c['crude']['risk_ratio']:.2f}")
    print(f"Adjusted OR={c['adjusted']['odds_ratio']:.2f} {c['adjusted']['or_ci95']}, "
          f"+ establishment history OR={c['adjusted_for_establishment_history']['odds_ratio']:.2f}")
    print(f"Flagged {q1['flagged_fdr05']} of {q1['establishments_tested']} establishments at FDR {FDR_Q}")


if __name__ == "__main__":
    main()
