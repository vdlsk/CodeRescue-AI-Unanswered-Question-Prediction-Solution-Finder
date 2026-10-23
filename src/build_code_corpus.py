"""Build a regulatory corpus from the standard code-requirement language that Chicago inspectors
quote at the start of their comments (e.g. "All cold food shall be stored at a temperature of 40F or less.").

Why: the official code PDFs could not be downloaded in the environment used for this run, but the
2010-2014 inspection reports embed the City's own requirement wording for every violation number.

Rules for a sentence to count as code language (not an observation):
  - appears at least MIN_COUNT times under the same violation number in TRAINING years,
  - is not upper-case (observations are typed in capitals),
  - contains normative wording (shall / must / should / prohibited / required).
Output: data/regulations/chicago_code_language.md with one "Section <violation no>" per violation.
Usage:  python src/build_code_corpus.py [--train-until 2013-12-31]
"""
import argparse
import collections
import re

import pandas as pd

from config import CSV_NAME, RAW_DIR, REG_DIR

MIN_COUNT = 20
NORMATIVE = re.compile(r"\b(shall|must|should|prohibited|required|is not permitted)\b", re.I)
ITEM = re.compile(r"^(\d+)\.\s*(.*?)\s+-\s+Comments:\s*(.*)$", re.S)
SPLIT = re.compile(r"(?<=[.!?])\s+")


def iter_items(df):
    for iid, date, v in zip(df["Inspection ID"], df["Inspection Date"], df["Violations"]):
        for item in str(v).split(" | "):
            m = ITEM.match(item.strip())
            if m:
                comments = re.sub(r"\.(?=[A-Z]{2})", ". ", m.group(3).strip())   # "facilities.VIOLATION CORRECTED"
                yield iid, date, int(m.group(1)), m.group(2).strip(), comments


def is_code_sentence(s):
    letters = [c for c in s if c.isalpha()]
    upper_share = sum(c.isupper() for c in letters) / max(1, len(letters))
    return upper_share < 0.5 and NORMATIVE.search(s) and len(s.split()) >= 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-until", default="2013-12-31")
    args = ap.parse_args()
    df = pd.read_csv(RAW_DIR / CSV_NAME, usecols=["Inspection ID", "Inspection Date", "Violations"]).dropna()
    df["Inspection Date"] = pd.to_datetime(df["Inspection Date"], format="%m/%d/%Y")
    df = df[df["Inspection Date"] <= args.train_until]

    titles, counts = {}, collections.defaultdict(collections.Counter)
    for _, _, no, title, comments in iter_items(df):
        titles.setdefault(no, collections.Counter())[title] += 1
        for s in SPLIT.split(comments):
            s = s.strip()
            if is_code_sentence(s):
                counts[no][s] += 1

    REG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Chicago food establishment code requirements (standard language from inspection reports)",
             f"Source: City of Chicago food inspection reports, inspections on or before {args.train_until}. "
             f"Sentences quoted by inspectors at least {MIN_COUNT} times for the same violation number.", ""]
    n_sent = 0
    for no in sorted(counts):
        sents = [s for s, c in counts[no].most_common() if c >= MIN_COUNT][:12]
        if not sents:
            continue
        title = titles[no].most_common(1)[0][0]
        lines.append(f"Section {no} - {title}")
        lines += [f"{s}" for s in sents]
        lines.append("")
        n_sent += len(sents)
    out = REG_DIR / "chicago_code_language.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out} : {len([n for n in counts if counts[n]])} violation sections, {n_sent} requirement sentences")


if __name__ == "__main__":
    main()
