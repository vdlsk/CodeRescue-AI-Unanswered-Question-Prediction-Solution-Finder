"""Load the inspections CSV into DuckDB and build cleaned tables.  Usage:  python src/build_db.py"""
import json

import duckdb

from config import CSV_NAME, DB_PATH, NEW_CODE_START, OUT_DIR, RAW_DIR, ROOT


def main():
    csv = RAW_DIR / CSV_NAME
    if not csv.exists():
        raise SystemExit("Missing CSV. Run: python src/download.py")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute((ROOT / "sql" / "01_build.sql").read_text().format(csv=str(csv), new_code_start=NEW_CODE_START))
    one = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    stats = {
        "raw_rows": one("SELECT count(*) FROM raw_inspections"),
        "inspections": one("SELECT count(*) FROM inspections"),
        "inspections_with_outcome": one("SELECT count(*) FROM inspections WHERE has_outcome"),
        "establishments": one("SELECT count(*) FROM establishments"),
        "violation_citations": one("SELECT count(*) FROM violations"),
        "date_range": [str(x) for x in con.execute(
            "SELECT min(inspection_date), max(inspection_date) FROM inspections").fetchone()],
        "overall_fail_rate": one("SELECT avg(failed) FROM inspections WHERE has_outcome"),
        "result_counts": dict(con.execute("SELECT result, count(*) FROM inspections GROUP BY 1 ORDER BY 2 DESC").fetchall()),
    }
    con.execute("DROP TABLE raw_inspections")
    (OUT_DIR / "build_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
