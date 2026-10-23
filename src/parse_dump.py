"""Stream a Stack Exchange data dump into parquet files.

Expects extracted XML files in data/raw/: Posts.xml (required), Users.xml, PostLinks.xml (optional).

Usage:
    python src/parse_dump.py                                   # whole site
    python src/parse_dump.py --tags python --since 2021-01-01  # filter (recommended for stackoverflow.com)
"""
import argparse
import re
import xml.etree.ElementTree as ET

import pandas as pd

from config import DATA_DIR, RAW_DIR

TAG_RE = re.compile(r"<([^>]+)>|\|([^|]+)")


def parse_tags(s: str):
    if not s:
        return []
    return [a or b for a, b in TAG_RE.findall(s) if (a or b)]


def iter_rows(path):
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "row":
            yield el.attrib
        el.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None, help="keep questions with any of these tags")
    ap.add_argument("--since", default=None, help="keep questions created on/after this date")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--max-body-chars", type=int, default=6000)
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tagset = set(args.tags) if args.tags else None

    qs, ans, kept = [], [], set()
    n = 0
    for r in iter_rows(RAW_DIR / "Posts.xml"):
        n += 1
        if n % 500_000 == 0:
            print(f"  {n:,} posts read, {len(qs):,} questions kept")
        pt = r.get("PostTypeId")
        if pt == "1":
            created = r["CreationDate"]
            if args.since and created < args.since:
                continue
            tags = parse_tags(r.get("Tags", ""))
            if tagset and not tagset.intersection(tags):
                continue
            if args.max_questions and len(qs) >= args.max_questions:
                continue
            qid = int(r["Id"])
            kept.add(qid)
            qs.append({
                "id": qid, "created": created, "title": r.get("Title", ""),
                "body": r.get("Body", "")[: args.max_body_chars],
                "tags": "|".join(tags), "owner_user_id": int(r["OwnerUserId"]) if r.get("OwnerUserId") else -1,
                "accepted_answer_id": int(r["AcceptedAnswerId"]) if r.get("AcceptedAnswerId") else -1,
                # end-of-dump fields: NEVER used as features (leakage), kept for analysis only
                "final_score": int(r.get("Score", 0)), "final_answer_count": int(r.get("AnswerCount", 0)),
                "closed_date": r.get("ClosedDate"),
            })
        elif pt == "2":
            pid = int(r.get("ParentId", -1))
            if pid in kept:
                ans.append({"id": int(r["Id"]), "parent_id": pid, "created": r["CreationDate"],
                            "score": int(r.get("Score", 0)), "body": r.get("Body", "")[: args.max_body_chars]})

    q = pd.DataFrame(qs)
    a = pd.DataFrame(ans)
    for d in (q, a):
        d["created"] = pd.to_datetime(d["created"], format="ISO8601")
    q.to_parquet(DATA_DIR / "questions.parquet", index=False)
    a.to_parquet(DATA_DIR / "answers.parquet", index=False)
    print(f"questions: {len(q):,}  answers: {len(a):,}")

    users_xml = RAW_DIR / "Users.xml"
    if users_xml.exists():
        owners = set(q.owner_user_id)
        u = pd.DataFrame([{"user_id": int(r["Id"]), "user_created": r["CreationDate"]}
                          for r in iter_rows(users_xml) if int(r["Id"]) in owners])
        u["user_created"] = pd.to_datetime(u["user_created"], format="ISO8601")
        u.to_parquet(DATA_DIR / "users.parquet", index=False)
        print(f"users: {len(u):,}")

    links_xml = RAW_DIR / "PostLinks.xml"
    if links_xml.exists():
        lk = pd.DataFrame([{"post_id": int(r["PostId"]), "related_post_id": int(r["RelatedPostId"]),
                            "link_type": int(r["LinkTypeId"])} for r in iter_rows(links_xml)
                           if r.get("LinkTypeId") == "3" and int(r["PostId"]) in kept])
        lk.to_parquet(DATA_DIR / "duplicate_links.parquet", index=False)
        print(f"duplicate links: {len(lk):,}")


if __name__ == "__main__":
    main()
