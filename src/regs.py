"""Ingest regulatory documents, split them into sections, index them, and map each violation
code to its most relevant provisions (for human verification).

Usage:  python src/regs.py
"""
import re

import duckdb
import joblib
import pandas as pd

from config import DB_PATH, INDEX_PATH, OUT_DIR, REG_DIR
from retrieval import BM25, Dense, HybridRetriever

# Section headers: FDA Food Code "3-501.16", Chicago Municipal Code "7-38-005", generic "Section 12" / "§ 4"
HEADER = re.compile(r"^\s*(?:§\s*)?((?:\d-\d{3}\.\d{1,3})|(?:\d{1,2}-\d{1,3}-\d{3})|(?:Section\s+\d+[\w.]*))\b(.*)$",
                    re.M | re.I)
MAX_CHARS, MIN_CHARS = 1800, 200


def read_doc(path):
    suf = path.suffix.lower()
    if suf == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    if suf in (".html", ".htm"):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
        for t in soup(["script", "style", "nav", "header", "footer"]):
            t.decompose()
        return soup.get_text("\n")
    return path.read_text(errors="ignore")


def chunk(doc_name, text):
    text = re.sub(r"[ \t]+", " ", text)
    heads = list(HEADER.finditer(text))
    pieces = []
    if heads:
        for h, nxt in zip(heads, heads[1:] + [None]):
            pieces.append((f"{h.group(1)} {h.group(2).strip()[:80]}".strip(), text[h.start(): nxt.start() if nxt else len(text)]))
    else:
        pieces = [("", text)]
    chunks = []
    for section, body in pieces:
        body = re.sub(r"\n{2,}", "\n", body).strip()
        if len(body) < 40:
            continue
        for i in range(0, len(body), MAX_CHARS - 200):     # 200-char overlap inside long sections
            part = body[i: i + MAX_CHARS]
            if i and len(part) < MIN_CHARS:
                break
            chunks.append({"doc": doc_name, "section": section or f"part {len(chunks) + 1}",
                           "text": part if i == 0 else f"[{section} cont.] {part}"})
    return chunks


def build():
    files = [p for p in sorted(REG_DIR.iterdir()) if p.suffix.lower() in (".pdf", ".html", ".htm", ".txt", ".md")]
    if not files:
        raise SystemExit(f"No regulation files in {REG_DIR}. See src/download.py for what to save there.")
    rows = []
    for f in files:
        c = chunk(f.name, read_doc(f))
        print(f"  {f.name}: {len(c)} chunks")
        rows += c
    chunks = pd.DataFrame(rows)
    chunks["chunk_id"] = range(len(chunks))
    docs = (chunks.section + " " + chunks.text).tolist()
    bm25, dense = BM25().fit(docs), Dense().fit(docs)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"chunks": chunks, "bm25": bm25, "dense": dense}, INDEX_PATH)
    print(f"Indexed {len(chunks)} chunks from {len(files)} files (dense backend: {dense.backend})")
    return chunks, HybridRetriever(bm25, dense, chunks.chunk_id.values)


def load_index():
    d = joblib.load(INDEX_PATH)
    return d["chunks"], HybridRetriever(d["bm25"], d["dense"], d["chunks"].chunk_id.values)


def map_violations(chunks, retr, k=3):
    if not DB_PATH.exists():
        return
    con = duckdb.connect(str(DB_PATH), read_only=True)
    codes = con.execute("SELECT violation_no, violation_title, citations FROM violation_codes").df()
    con.close()
    hits = retr.search(codes.violation_title.tolist(), k=k)
    rows = []
    for (_, c), h in zip(codes.iterrows(), hits):
        for rank, i in enumerate(h, 1):
            ch = chunks.iloc[i]
            rows.append({"violation_no": c.violation_no, "violation_title": c.violation_title,
                         "citations": c.citations, "rank": rank, "doc": ch.doc, "section": ch.section,
                         "excerpt": ch.text[:300].replace("\n", " "), "human_verified_relevant": ""})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "violation_regulation_map.csv", index=False)
    print(f"Wrote violation -> regulation map for {len(codes)} codes (verify the 'human_verified_relevant' column)")


if __name__ == "__main__":
    ch, r = build()
    map_violations(ch, r)
