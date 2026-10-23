"""FoodSafe assistant: answers questions about an establishment from (1) its SQL inspection history,
(2) its statistical comparison with peers, and (3) retrieved regulatory provisions, with citations."""
import os
import re

import duckdb
import pandas as pd

from config import DB_PATH, LLM_MODEL
from regs import load_index

SYSTEM = """You are FoodSafe, an assistant for food-safety inspectors in Chicago.
Use ONLY the context provided. Cite inspection records as [I-<inspection id>] and regulations as [R<n>].
Rules:
- Violations are inspector findings on a given date, not proof of current conditions.
- Statistics show associations and peer comparisons; they do not prove causes or predict a specific outcome.
- Do not make legal determinations; say which provisions appear relevant and that an inspector must confirm.
- If the context does not answer the question, say so.
Be concise and concrete: what has been found, how unusual it is, which provisions apply, what to check next."""


class FoodSafe:
    def __init__(self):
        self.con = duckdb.connect(str(DB_PATH), read_only=True)
        self.chunks, self.retr = load_index()

    def find(self, text, limit=20):
        if str(text).strip().isdigit():
            return self.con.execute("SELECT * FROM establishments WHERE license_no = ?", [int(text)]).df()
        return self.con.execute("""SELECT * FROM establishments WHERE name ILIKE ? OR address ILIKE ?
                                   ORDER BY n_inspections DESC LIMIT ?""", [f"%{text}%", f"%{text}%", limit]).df()

    def history(self, license_no, n=10):
        return self.con.execute("""SELECT inspection_id, inspection_date, inspection_type_raw AS type, result
                                   FROM inspections WHERE license_no = ? ORDER BY inspection_date DESC LIMIT ?""",
                                [license_no, n]).df()

    def violations(self, license_no, last_n_inspections=3):
        return self.con.execute("""
            WITH recent AS (SELECT inspection_id FROM inspections WHERE license_no = ? AND has_outcome
                            ORDER BY inspection_date DESC LIMIT ?)
            SELECT v.inspection_id, v.inspection_date, v.violation_no, v.violation_title, left(v.comments, 300) AS comments
            FROM violations v JOIN recent USING (inspection_id) ORDER BY v.inspection_date DESC, v.violation_no""",
                                [license_no, last_n_inspections]).df()

    def repeat(self, license_no):
        return self.con.execute("""SELECT r.violation_no, vc.violation_title, r.times_cited, r.last_cited
                                   FROM repeat_violations r LEFT JOIN violation_codes vc USING (violation_no)
                                   WHERE license_no = ? ORDER BY times_cited DESC""", [license_no]).df()

    def peer_stats(self, license_no):
        df = self.con.execute("SELECT * FROM establishment_flags WHERE license_no = ?", [license_no]).df()
        return None if df.empty else df.iloc[0].to_dict()

    def regulations(self, queries, k=6):
        seen, out = set(), []
        for hits in self.retr.search(queries, k=3):
            for i in hits:
                if i not in seen:
                    seen.add(i)
                    out.append(self.chunks.iloc[i])
        return pd.DataFrame(out).head(k).reset_index(drop=True)

    def build_context(self, license_no, question):
        est = self.find(str(license_no)).iloc[0]
        hist, viol, rep, peer = (self.history(license_no), self.violations(license_no), self.repeat(license_no),
                                 self.peer_stats(license_no))
        # the establishment's own cited violations first, then the free-text question
        queries = viol.violation_title.dropna().drop_duplicates().head(5).tolist() + [question]
        regs = self.regulations(queries)
        lines = [f"ESTABLISHMENT: {est['name']} (license {license_no}), {est.facility_type}, risk {est.risk_level}, "
                 f"{est.address}. {est.n_inspections} inspections with outcome, {est.n_fails} failed. "
                 f"Last result: {est.last_result}."]
        if peer:
            lines.append(f"PEER COMPARISON: {int(peer['observed'])} regular-inspection failures vs "
                         f"{peer['expected']:.1f} expected from peers of the same type, risk level and year "
                         f"(ratio {peer['ratio']:.2f}, FDR q-value {peer['q_value']:.3g}, "
                         f"{'statistically unusual' if peer['flagged'] else 'not statistically unusual'}).")
        lines.append("INSPECTION HISTORY:")
        lines += [f"[I-{r.inspection_id}] {r.inspection_date} {r.type}: {r.result}" for r in hist.itertuples()]
        lines.append("VIOLATIONS CITED IN THE LAST 3 INSPECTIONS:")
        lines += [f"[I-{r.inspection_id}] #{r.violation_no} {r.violation_title}. Inspector comment: {r.comments}"
                  for r in viol.itertuples()]
        if len(rep):
            lines.append("REPEATED VIOLATIONS (same violation number in 2+ inspections):")
            lines += [f"#{r.violation_no} {r.violation_title}: cited {r.times_cited} times, last {r.last_cited}"
                      for r in rep.head(8).itertuples()]
        lines.append("REGULATORY PROVISIONS:")
        lines += [f"[R{i + 1}] {r.doc}, {r.section}: {r.text[:1200]}" for i, r in regs.iterrows()]
        return "\n".join(lines), {"history": hist, "violations": viol, "repeat": rep, "peer": peer, "regs": regs}

    def ask(self, license_no, question):
        ctx, parts = self.build_context(license_no, question)
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            msg = anthropic.Anthropic().messages.create(
                model=LLM_MODEL, max_tokens=900, system=SYSTEM,
                messages=[{"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"}])
            answer = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        else:
            answer = template_answer(parts)
        return {"answer": answer, "context": ctx, "parts": parts, "grounding": grounding(answer, ctx, len(parts["regs"]))}


def template_answer(p):
    """Deterministic answer used when no LLM key is configured."""
    out = []
    if p["peer"]:
        out.append(f"Failures on regular inspections: {int(p['peer']['observed'])} vs {p['peer']['expected']:.1f} "
                   f"expected for comparable establishments ({'statistically unusual' if p['peer']['flagged'] else 'within the expected range'}).")
    if len(p["repeat"]):
        r = p["repeat"].iloc[0]
        out.append(f"Most repeated violation: #{r.violation_no} {r.violation_title} ({r.times_cited} inspections).")
    if len(p["violations"]):
        v = p["violations"].iloc[0]
        out.append(f"Most recent cited violation: #{v.violation_no} {v.violation_title} [I-{v.inspection_id}].")
    if len(p["regs"]):
        out.append("Possibly relevant provisions: " + "; ".join(
            f"{r.section} ({r.doc}) [R{i + 1}]" for i, r in p["regs"].head(3).iterrows()) + ". An inspector must confirm applicability.")
    return "\n\n".join(out) or "No inspection records with violations were found."


NUM = re.compile(r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])")


def grounding(answer, ctx, n_regs):
    nums = [n for n in NUM.findall(answer) if len(n) > 1 or n not in "0123456789"]
    supported = [n for n in nums if n in ctx]
    rcites = [int(x) for x in re.findall(r"\[R(\d+)\]", answer)]
    icites = re.findall(r"\[I-(\d+)\]", answer)
    return {"numbers": len(nums), "numbers_supported": len(supported),
            "number_support_rate": len(supported) / len(nums) if nums else None,
            "reg_citations": len(rcites), "reg_citations_valid": sum(1 <= r <= n_regs for r in rcites),
            "record_citations": len(icites), "record_citations_valid": sum(f"[I-{i}]" in ctx for i in icites)}
