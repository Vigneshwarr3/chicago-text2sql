#!/usr/bin/env python
"""
score_benchmark.py
==================
Deterministic scoring pass for the benchmark outputs in results/agent{1,2}/.
No LLM is used here. For every row of every results CSV it re-executes the
ground-truth SQL and the agent SQL against Postgres (read-only session,
per-query timeout) and adds these columns in place:

  ground_truth_result  JSON of the first MAX_ROWS rows of the ground-truth query
  agent_result         JSON of the first MAX_ROWS rows of the agent query
  gt_nrows/agent_nrows number of rows returned (up to FULL_ROWS; -1 on error)
  valid_sql            1 if the agent produced SQL that executed without error
                       (rows with no SQL, e.g. relevance rejections, count as 0)
  ex_match             1 if the two result sets are identical as value multisets:
                       row order, column names, case/whitespace and 2-decimal
                       rounding are ignored; column order and extra/missing
                       columns are not. This is a fast path, not a reported
                       metric: it spares the judge the obvious cases.

The final EX and the Synthesis Quality score (SQ) are added by judge_results.py,
which reads these columns.

Each distinct SQL string is executed once and reused across all files, so the
cost is ~(unique agent SQL + 96) queries rather than 2 x rows x files.

Usage
-----
    python score_benchmark.py                      # score every results/*/*.csv
    python score_benchmark.py --timeout 10         # per-query timeout, seconds
"""

import argparse
import csv
import glob
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import psycopg2
from dotenv import load_dotenv

load_dotenv()
csv.field_size_limit(sys.maxsize)

BASE_DIR = Path(__file__).resolve().parent
FULL_ROWS = 1000   # rows compared in memory; the CSV only stores a MAX_ROWS-row preview
MAX_ROWS = 20      # rows saved in the CSV, which is also all the judge ever sees


# -- result comparison helpers -------------------------------------------------------------

def _parse_result(json_str: str) -> Optional[list]:
    """Return parsed list-of-dicts, or None for empty / error results."""
    if not json_str or not json_str.strip():
        return None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    if isinstance(data[0], dict) and "error" in data[0]:
        return None
    return data


def _normalize(v):
    """Normalize a scalar value for column-agnostic comparison:
    numbers (and numeric strings) are rounded to 2 decimals, other strings
    are trimmed and lower-cased, booleans become 0/1."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, str):
        try:
            return round(float(v), 2)
        except ValueError:
            pass
        return v.strip().lower()
    return v


def _to_value_multiset(data: list) -> list:
    """
    Convert list-of-dicts to a sorted list of normalized value tuples.
    Rows are sorted (row order ignored); column names are ignored but the order of
    values within a row is kept, so swapped or extra columns do not match.
    """
    rows = []
    for row in data:
        if isinstance(row, dict):
            rows.append(tuple(_normalize(v) for v in row.values()))
        else:
            rows.append((_normalize(row),))
    return sorted(rows, key=lambda t: [str(x) for x in t])


NEW_COLUMNS = ["ground_truth_result", "agent_result", "gt_nrows", "agent_nrows", "valid_sql", "ex_match"]


def run_query(cur, conn, sql: str) -> str:
    """Execute `sql`; return JSON rows (first MAX_ROWS) or {"error": ...}."""
    if not sql or not sql.strip():
        return json.dumps({"error": "empty query"})
    try:
        cur.execute(sql)
        rows = cur.fetchmany(FULL_ROWS)
        cols = [d[0] for d in cur.description] if cur.description else []
        return json.dumps([dict(zip(cols, r)) for r in rows], default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc).strip().splitlines()[0]})
    finally:
        conn.rollback()


def preview(result_json: str) -> str:
    """First MAX_ROWS rows of a result, for storing in the CSV."""
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return result_json
    return json.dumps(data[:MAX_ROWS]) if isinstance(data, list) else result_json


def nrows(result_json: str) -> int:
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return -1
    return len(data) if isinstance(data, list) else -1


def is_error(result_json: str) -> bool:
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return True
    return isinstance(data, dict) and "error" in data


def ex_match(gt_json: str, ag_json: str) -> int:
    """Equality of the two result sets as multisets of normalized value tuples.
    Two valid, empty result sets are equal. A result of more than FULL_ROWS rows
    is compared on its first FULL_ROWS rows only (no reference query returns more
    than 77 rows, so this only bounds memory for runaway agent queries)."""
    if is_error(gt_json) or is_error(ag_json):
        return 0
    gt, ag = json.loads(gt_json), json.loads(ag_json)
    if not gt and not ag:
        return 1
    gt, ag = _parse_result(gt_json), _parse_result(ag_json)
    if gt is None or ag is None:
        return 0
    return int(_to_value_multiset(gt) == _to_value_multiset(ag))


def main():
    p = argparse.ArgumentParser(description="Score benchmark results (no LLM)")
    p.add_argument("--results-dir", type=Path, default=BASE_DIR / "results")
    p.add_argument("--dataset", default="data/evaluation_dataset.csv")
    p.add_argument("--db-name", default=os.getenv("DB_NAME", "chicago_crime_new"))
    p.add_argument("--timeout", type=float, default=120.0, help="Per-query timeout in seconds (default 120)")
    args = p.parse_args()

    files = sorted(glob.glob(str(args.results_dir / "agent*" / "*.csv")))
    if not files:
        sys.exit(f"No result CSVs under {args.results_dir}")

    conn = psycopg2.connect(host=os.getenv("DB_HOST", "127.0.0.1"), port=int(os.getenv("DB_PORT", "5440")),
                            dbname=args.db_name, user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASSWORD"), connect_timeout=10)
    # Agent SQL is untrusted, so the session is read-only (the database role is
    # read-only as well), and every query is rolled back after it runs.
    conn.set_session(readonly=True, autocommit=False)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = %s", (int(args.timeout * 1000),))
    conn.commit()

    # The same SQL string often appears in many of the 40 files (temperature 0),
    # so each distinct query is executed once and its result reused.
    cache: dict[str, str] = {}

    def result_for(sql: str) -> str:
        if sql not in cache:
            cache[sql] = run_query(cur, conn, sql)
            if len(cache) % 100 == 0:
                print(f"  … {len(cache)} distinct queries executed")
        return cache[sql]

    with open(BASE_DIR / args.dataset, newline="", encoding="utf-8") as f:
        gt_by_id = {r["id"]: r["sql_query"] for r in csv.DictReader(f)}

    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = list(reader.fieldnames)
        for row in rows:
            # The reference SQL is taken from the dataset, not from the run file.
            gt_sql = gt_by_id.get(row["id"], row["ground_truth_sql"])
            gt = result_for(gt_sql)
            ag = result_for(row["agent_sql_query"])
            row["ground_truth_result"] = preview(gt)
            row["agent_result"] = preview(ag)
            row["gt_nrows"] = nrows(gt)
            row["agent_nrows"] = nrows(ag)
            row["valid_sql"] = int(bool(row["agent_sql_query"].strip()) and not is_error(ag))
            row["ex_match"] = ex_match(gt, ag)
        for col in NEW_COLUMNS:
            if col not in fields:
                fields.append(col)
        # atomic in-place write
        fd, tmp = tempfile.mkstemp(dir=Path(path).parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as out:
            w = csv.DictWriter(out, fieldnames=fields, quoting=csv.QUOTE_ALL, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
        n = len(rows)
        print(f"{Path(path).relative_to(args.results_dir)}: VSR={sum(int(r['valid_sql']) for r in rows)/n:.0%} "
              f"exact-match={sum(int(r['ex_match']) for r in rows)/n:.0%}")

    gt_err = [i for i, s in gt_by_id.items() if is_error(cache.get(s, "{}"))]
    if gt_err:
        print(f"\nWARNING: ground-truth SQL errored/timed out for ids {gt_err}; those rows can never match.")
    cur.close(); conn.close()


if __name__ == "__main__":
    main()
