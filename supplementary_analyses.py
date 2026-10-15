#!/usr/bin/env python
"""
supplementary_analyses.py
=========================
Three analyses computed from the scored results CSVs in results/agent{1,2}/
(after score_benchmark.py and judge_results.py). No model is called.

  1. Matched subset   EM and EX of both agents on the questions that agent_2's
                      relevance checker accepted in every run, so the two agents
                      are compared on the same questions.
  2. Repaired queries EM and EX of the agent_2 queries that failed validation on
                      the first attempt and were repaired by the retry loop.
  3. Full-result check
                      The judge only sees the first 20 rows / 700 characters of
                      each result. This re-executes the reference and agent SQL of
                      every executed response that is not an exact match, compares
                      the complete results, and writes one row per judged item to
                      results/judge_audit.csv for manual review. It also checks
                      whether the exact matches (EM) depend on ignoring row order
                      or on 2-decimal rounding. Needs the benchmark database.

    python supplementary_analyses.py            # all three
    python supplementary_analyses.py --no-db    # 1 and 2 only
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv

from judge_results import GRADES_PATH, ex_key, needs_ex_judge
from score_benchmark import _normalize

load_dotenv()
csv.field_size_limit(sys.maxsize)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
STAGES = ["stage1_simple", "stage2_rules", "stage3_one_shot", "stage4_few_shot"]
RUNS = range(1, 6)
FETCH_ROWS = 100_000
PREVIEW_ROWS = 20      # rows stored in the CSV (score_benchmark.MAX_ROWS)


def load(agent, stage, run):
    with open(RESULTS_DIR / agent / f"{stage}_run{run}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(rows, col):
    return sum(int(r[col]) for r in rows) / len(rows) * 100


# -- 1. matched subset ---------------------------------------------------------------------

def matched_subset():
    accepted = None
    for stage in STAGES:
        for k in RUNS:
            ids = {r["id"] for r in load("agent2", stage, k) if r["agent_sql_query"].strip()}
            accepted = ids if accepted is None else accepted & ids
    print(f"## 1. Both agents on the {len(accepted)} questions accepted by agent_2 in every run\n")
    print("| Stage | agent_1 EM | agent_1 EX | agent_2 EM | agent_2 EX |")
    print("|---|---|---|---|---|")
    pooled = defaultdict(list)
    for stage in STAGES:
        cells = []
        for agent in ("agent1", "agent2"):
            runs = [[r for r in load(agent, stage, k) if r["id"] in accepted] for k in RUNS]
            em = [pct(r, "ex_match") for r in runs]
            ex = [pct(r, "ex") for r in runs]
            cells += [f"{st.mean(em):.1f} ± {st.stdev(em):.1f}", f"{st.mean(ex):.1f} ± {st.stdev(ex):.1f}"]
            if stage != "stage1_simple":
                pooled[agent, "EM"].append(st.mean(em))
                pooled[agent, "EX"].append(st.mean(ex))
        print(f"| {stage} | " + " | ".join(cells) + " |")
    print("| stages 2-4 mean | " + " | ".join(
        f"{st.mean(pooled[a, m]):.1f}" for a in ("agent1", "agent2") for m in ("EM", "EX")) + " |")

    rejected = None
    for stage in STAGES[1:]:
        for k in RUNS:
            rows = [r for r in load("agent1", stage, k) if r["id"] not in accepted]
            rejected = (rejected or []) + [sum(int(r["ex"]) for r in rows)]
    print(f"\nagent_1, stages 2-4: mean {st.mean(rejected):.1f} of the {96 - len(accepted)} "
          f"rejected questions correct (EX)\n")


# -- 2. repaired queries -------------------------------------------------------------------

def repaired_queries():
    print("## 2. agent_2 queries repaired by the retry loop\n")
    print("| Stage | Repaired | EM | EX |")
    print("|---|---|---|---|")
    total = Counter()
    for stage in STAGES:
        c = Counter()
        for k in RUNS:
            for r in load("agent2", stage, k):
                if r["agent_sql_query"].strip() and "Validator: SQL syntax error" in r["error_log"]:
                    c["n"] += 1
                    c["em"] += int(r["ex_match"])
                    c["ex"] += int(r["ex"])
                    c["valid"] += int(r["valid_sql"])
        total += c
        print(f"| {stage} | {c['n']} | {c['em']} | {c['ex']} |")
    print(f"| all | {total['n']} | {total['em']} ({total['em'] / total['n']:.1%}) "
          f"| {total['ex']} ({total['ex'] / total['n']:.1%}) |")
    print(f"\nAll repaired queries executed: {total['valid'] == total['n']}\n")


# -- 3. full-result check ------------------------------------------------------------------

_MIDNIGHT = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]00:00:00$")


def norm(v, ndigits=2):
    """score_benchmark's normalization, applied to raw psycopg2 values; a timestamp
    at midnight is read as a date, so 2024-01-01 and 2024-01-01 00:00:00 agree."""
    if v is not None and not isinstance(v, (bool, int, float, str)):
        v = str(v)
    if isinstance(v, str):
        m = _MIDNIGHT.match(v.strip())
        if m:
            v = m.group(1)
    if ndigits != 2 and isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), ndigits)
    if ndigits != 2 and isinstance(v, str):
        try:
            return round(float(v), ndigits)
        except ValueError:
            pass
    return _normalize(v)


def execute(cur, conn, sql):
    try:
        cur.execute(sql)
        return cur.fetchmany(FETCH_ROWS)
    except Exception as exc:
        return {"error": str(exc).strip().splitlines()[0]}
    finally:
        conn.rollback()


def projection(gt, ag, ndigits=2, subset=None):
    """An injective map from reference columns (all, or those in `subset`) to agent
    columns under which the agent rows, restricted to the mapped columns, equal the
    reference rows restricted to those columns as a multiset; extra agent columns are
    ignored. Returns the column map or None."""
    if len(gt) != len(ag):
        return None
    if not gt:
        return []
    G = [tuple(norm(v, ndigits) for v in r) for r in gt]
    A = [tuple(norm(v, ndigits) for v in r) for r in ag]
    if subset is not None:
        G = [tuple(r[i] for i in subset) for r in G]
    gcols = [Counter(r[i] for r in G) for i in range(len(G[0]))]
    acols = [Counter(r[j] for r in A) for j in range(len(A[0]))]
    cand = [[j for j, ac in enumerate(acols) if ac == gc] for gc in gcols]
    target = Counter(G)

    def search(i, used):
        if i == len(cand):
            return [] if Counter(tuple(r[j] for j in used) for r in A) == target else None
        for j in cand[i]:
            if j not in used:
                rest = search(i + 1, used + [j])
                if rest is not None:
                    return [j] + rest
        return None

    return search(0, [])


def unmatched_columns(gt, ag):
    """Fewest reference columns that must be dropped for the rest to be contained in
    the agent result (0 = fully contained). Equal row counts are assumed."""
    k = len(gt[0]) if gt else 0
    for size in range(k, 0, -1):
        if any(projection(gt, ag, subset=s) is not None for s in combinations(range(k), size)):
            return k - size
    return k


def top_level_order_by(sql):
    """True if the outermost SELECT has an ORDER BY (depth-0 match)."""
    depth, s = 0, sql.lower()
    for i, ch in enumerate(s):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and s.startswith("order by", i):
            return True
    return False


def same_order(gt, ag, cols):
    return [tuple(norm(v) for v in r) for r in gt] == [tuple(norm(r[j]) for j in cols) for r in ag]


def connect():
    import psycopg2

    conn = psycopg2.connect(host=os.getenv("DB_HOST", "127.0.0.1"), port=int(os.getenv("DB_PORT", "5440")),
                            dbname=os.getenv("DB_NAME", "chicago_crime_new"), user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASSWORD"), connect_timeout=10)
    # Agent SQL is untrusted: read-only session, every query rolled back.
    conn.set_session(readonly=True, autocommit=False)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 120000")
    conn.commit()
    return conn, cur


def execute_all(sqls, workers):
    """Run every distinct query once, over `workers` parallel connections."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    local, conns, cache = threading.local(), [], {}

    def run(sql):
        if not hasattr(local, "cur"):
            local.conn, local.cur = connect()
            conns.append(local.conn)
        return sql, execute(local.cur, local.conn, sql)

    with ThreadPoolExecutor(workers) as pool:
        for n, (sql, res) in enumerate(pool.map(run, sorted(sqls)), 1):
            cache[sql] = res
            if n % 100 == 0:
                print(f"  {n}/{len(sqls)} executed", file=sys.stderr)
    for c in conns:
        c.close()
    return cache


def full_result_check(workers, cache_path=None):
    rows = []
    for path in sorted(glob.glob(str(RESULTS_DIR / "agent*" / "*.csv"))):
        agent, stem = Path(path).parent.name, Path(path).stem
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["valid_sql"] == "1":
                    rows.append({**r, "_file": f"{agent}/{stem}"})

    # A result of at most 20 rows is stored complete in the CSV; only longer
    # results have to be re-executed.
    sqls = ({r["ground_truth_sql"] for r in rows if int(r["gt_nrows"]) > PREVIEW_ROWS}
            | {r["agent_sql_query"] for r in rows if int(r["agent_nrows"]) > PREVIEW_ROWS})
    print(f"## 3. Full-result check ({len(sqls)} distinct queries to re-execute)\n", file=sys.stderr)
    if cache_path and Path(cache_path).exists():
        cache = pickle.loads(Path(cache_path).read_bytes())
    else:
        cache = execute_all(sqls, workers)
        if cache_path:
            Path(cache_path).write_bytes(pickle.dumps(cache))

    def result(r, side):
        sql, n, stored = ((r["ground_truth_sql"], r["gt_nrows"], r["ground_truth_result"]) if side == "gt"
                          else (r["agent_sql_query"], r["agent_nrows"], r["agent_result"]))
        if int(n) > PREVIEW_ROWS:
            return cache[sql]
        return [tuple(d.values()) for d in json.loads(stored)]

    grades = json.loads(GRADES_PATH.read_text())["ex"]

    # 3a. executed responses without an exact match: the judge's cases
    def outcome(r):
        gt, ag = result(r, "gt"), result(r, "agent")
        if isinstance(gt, dict) or isinstance(ag, dict):
            return "re-execution error", None
        if len(gt) != len(ag):
            return "row count differs", None
        cols = projection(gt, ag)
        if cols is None:
            return "values differ / needs review", unmatched_columns(gt, ag)
        return "reference contained in full result", 0

    items = {}
    for r in rows:
        if not needs_ex_judge(r):
            continue
        k = ex_key(r)
        o, unmatched = outcome(r)
        it = items.setdefault(k, {"r": r, "outcomes": Counter(), "files": [], "unmatched": unmatched})
        it["outcomes"][o] += 1
        it["files"].append(r["_file"])

    print("### 3a. Judged responses (executed, no exact match)\n")
    for unit in ("responses", "distinct items"):
        table = Counter()
        for k, it in items.items():
            verdict = "accepted" if grades[k]["equivalent"] else "rejected"
            for o, c in it["outcomes"].items():
                table[verdict, o] += c if unit == "responses" else 1
        print(f"| Judge ({unit}) | reference contained | row count differs | values differ / review | error |")
        print("|---|---|---|---|---|")
        for v in ("accepted", "rejected"):
            print(f"| {v} | " + " | ".join(str(table[v, o]) for o in (
                "reference contained in full result", "row count differs",
                "values differ / needs review", "re-execution error")) + " |")
        print()
    mixed = [k for k, it in items.items() if len(it["outcomes"]) > 1]
    print(f"Items whose responses have different outcomes: {len(mixed)}\n")

    print("Same row count but not contained: reference columns that do not match jointly\n")
    print("| Judge | items, 1 column | items, 2 columns | items, 3+ columns |")
    print("|---|---|---|---|")
    for v in (True, False):
        c = Counter(min(it["unmatched"], 3) for k, it in items.items()
                    if grades[k]["equivalent"] == v and "values differ / needs review" in it["outcomes"])
        print(f"| {'accepted' if v else 'rejected'} | {c[1]} | {c[2]} | {c[3]} |")
    print()

    out = RESULTS_DIR / "judge_audit.csv"
    fields = ["item", "id", "question", "judge_equivalent", "judge_reason", "check_outcome",
              "unmatched_ref_columns", "gt_rows", "agent_rows", "n_responses", "files",
              "ground_truth_sql", "agent_sql",
              "ground_truth_preview", "agent_preview", "manual_verdict", "manual_note"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for k, it in sorted(items.items(), key=lambda kv: int(kv[1]["r"]["id"])):
            r = it["r"]
            gt, ag = result(r, "gt"), result(r, "agent")
            w.writerow({
                "item": k, "id": r["id"], "question": r["question"],
                "judge_equivalent": int(grades[k]["equivalent"]), "judge_reason": grades[k]["reason"],
                "check_outcome": "; ".join(f"{o} x{c}" for o, c in it["outcomes"].items()),
                "unmatched_ref_columns": "" if it["unmatched"] is None else it["unmatched"],
                "gt_rows": len(gt) if isinstance(gt, list) else "error",
                "agent_rows": len(ag) if isinstance(ag, list) else "error",
                "n_responses": sum(it["outcomes"].values()), "files": " ".join(sorted(set(it["files"]))),
                "ground_truth_sql": r["ground_truth_sql"], "agent_sql": r["agent_sql_query"],
                "ground_truth_preview": r["ground_truth_result"], "agent_preview": r["agent_result"],
                "manual_verdict": "", "manual_note": ""})
    print(f"Audit sheet: {out.relative_to(BASE_DIR)} ({len(items)} items)\n")

    # 3b. exact matches: do they depend on ignoring row order or on rounding?
    em = [r for r in rows if r["ex_match"] == "1"]
    ordered = order_diff = rounding = 0
    order_ids, rounding_ids = Counter(), Counter()
    for r in em:
        gt, ag = result(r, "gt"), result(r, "agent")
        if isinstance(gt, dict) or isinstance(ag, dict):
            continue
        cols = projection(gt, ag)
        if cols is None:
            continue
        if projection(gt, ag, ndigits=6) is None:
            rounding += 1
            rounding_ids[r["id"]] += 1
        if top_level_order_by(r["ground_truth_sql"]) and len(gt) > 1:
            ordered += 1
            if not same_order(gt, ag, cols):
                order_diff += 1
                order_ids[r["id"]] += 1
    print("### 3b. Exact matches (EM)\n")
    print(f"EM responses: {len(em)}")
    print(f"Match only after 2-decimal rounding (differ at 6 decimals): {rounding} "
          f"(question ids: {dict(rounding_ids)})")
    print(f"Reference is an ordered multi-row result: {ordered}; returned in a different order: "
          f"{order_diff} (question ids: {dict(order_ids)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-db", action="store_true", help="skip the full-result check")
    ap.add_argument("--workers", type=int, default=4, help="parallel database connections (default 4)")
    ap.add_argument("--cache", help="pickle file for the re-executed results (reused if it exists)")
    args = ap.parse_args()
    matched_subset()
    repaired_queries()
    if not args.no_db:
        full_result_check(args.workers, args.cache)


if __name__ == "__main__":
    main()
