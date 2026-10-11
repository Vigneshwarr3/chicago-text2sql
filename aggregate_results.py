#!/usr/bin/env python
"""
aggregate_results.py
====================
Aggregates the scored results CSVs (after score_benchmark.py and
judge_results.py) into the tables reported in results.md.

For every (agent, stage) it computes each metric per run, then reports the
mean and standard deviation across the runs:

  VSR       valid_sql            % of questions with executable SQL
  EX        ex                   % of questions whose result matches the reference
  SQ        sq_score             mean Likert 1-5 over all rows

Also per complexity tier (Easy/Medium/Hard) and a failure breakdown.
Prints markdown tables.
"""

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)
BASE = Path(__file__).resolve().parent
STAGES = ["stage1_simple", "stage2_rules", "stage3_one_shot", "stage4_few_shot"]
AGENTS = ["agent1", "agent2"]
TIERS = ["Easy", "Medium", "Hard"]


RESULTS_DIR = BASE / "results"   # overridden by --results-dir


def load(agent, stage, run):
    p = RESULTS_DIR / agent / f"{stage}_run{run}.csv"
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


NAN = float("nan")


def metrics(rows):
    """Per-run metrics. A metric whose column is absent or not yet filled in
    (EX / SQ before judge_results.py has been applied) is NaN."""
    n = len(rows)
    if not n:
        return {"VSR": NAN, "EX": NAN, "SQ": NAN, "n": 0}

    def pct(col):
        if col not in rows[0] or any(r[col] == "" for r in rows):
            return NAN
        return sum(int(r[col]) for r in rows) / n * 100

    sq = [int(r["sq_score"]) for r in rows if r.get("sq_score", "") != ""]
    return {"VSR": pct("valid_sql"), "EX": pct("ex"), "SQ": (sum(sq) / len(sq)) if len(sq) == n else NAN, "n": n}


def mean_sd(vals):
    if any(v != v for v in vals):          # NaN: metric not available for this run
        return NAN, NAN
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    RESULTS_DIR = ap.parse_args().results_dir
    out = {"overall": {}, "tier": {}, "runs": {}, "failures": {}}
    for agent in AGENTS:
        for stage in STAGES:
            runs = [load(agent, stage, k) for k in range(1, 6)]
            per_run = [metrics(r) for r in runs]
            out["runs"][f"{agent}/{stage}"] = per_run
            out["overall"][f"{agent}/{stage}"] = {
                m: mean_sd([p[m] for p in per_run]) for m in ("VSR", "EX", "SQ")}
            for tier in TIERS:
                tr = [metrics([r for r in rows if r["complexity"] == tier]) for rows in runs]
                out["tier"][f"{agent}/{stage}/{tier}"] = {
                    m: mean_sd([p[m] for p in tr]) for m in ("VSR", "EX", "SQ")} | {"n": tr[0]["n"]}
            # Classify every wrong answer (EX = 0) by why it is wrong.
            fail = defaultdict(float)
            for rows in runs:
                for r in rows:
                    if r.get("ex") == "1":
                        continue
                    if not r["agent_sql_query"].strip():
                        fail["no SQL (relevance rejection / invalid request)"] += 1
                    elif r.get("valid_sql") == "0":
                        fail["SQL failed to execute"] += 1
                    else:
                        fail["executed but wrong result"] += 1
            out["failures"][f"{agent}/{stage}"] = {k: v / 5 for k, v in fail.items()}

    fmt = lambda ms, d=1: "n/a" if ms[0] != ms[0] else f"{ms[0]:.{d}f} ± {ms[1]:.{d}f}"
    print("| Agent | Stage | VSR % | EX % | SQ (1-5) |")
    print("|---|---|---|---|---|")
    for agent in AGENTS:
        for stage in STAGES:
            o = out["overall"][f"{agent}/{stage}"]
            print(f"| {agent} | {stage} | {fmt(o['VSR'])} | {fmt(o['EX'])} | {fmt(o['SQ'],2)} |")
    print("\nPer tier (VSR %, EX %, SQ):")
    for agent in AGENTS:
        for stage in STAGES:
            cells = []
            for tier in TIERS:
                t = out["tier"][f"{agent}/{stage}/{tier}"]
                cells.append(f"{tier}(n={t['n']}) VSR {t['VSR'][0]:.0f} EX {t['EX'][0]:.0f} SQ {t['SQ'][0]:.2f}".replace("nan", "n/a"))
            print(agent, stage, " | ".join(cells))
    print("\nMean failures per run (EX = 0):")
    for key, d in out["failures"].items():
        print(key, {k: round(v, 1) for k, v in d.items()})


if __name__ == "__main__":
    main()
