#!/usr/bin/env python
"""
judge_results.py
================
Adds the two LLM-judged metrics to the scored results CSVs (run
score_benchmark.py first):

  ex_judge   1 if the judge deems a result equivalent to ground truth on rows
             where strict/relational matching failed (EX fallback)
  ex_soft    ex_relational OR ex_judge   (the hybrid EX metric)
  sq_score   1-5 Synthesis Quality of the agent's plain-English answer

Judging is keyed by content hash, so identical (question, result, answer)
triples across the 40 files are judged once. Rubrics: judge_rubric.py.

Three modes
-----------
  export  Write judge_items.json: every item that still needs a grade, plus
          rule-based SQ grades that need no model (see rule_sq).
  api     Grade the missing items with the Anthropic API (JUDGE_MODEL) and
          merge them into grades.json. Needs ANTHROPIC_API_KEY in the
          environment or .env. This is how others reproduce the grading.
  apply   Write ex_judge / ex_soft / sq_score into every results CSV from
          grades.json (no network, no model).

The released grades.json was produced by Claude Sonnet 5.5 reading
judge_items.json directly against the same rubrics rather than via `api`;
`api` regenerates equivalent grades for anyone who wants to verify them.

    python judge_results.py --mode export
    python judge_results.py --mode api          # optional, needs API key
    python judge_results.py --mode apply
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from judge_rubric import EX_RUBRIC, JUDGE_MODEL, SQ_RUBRIC

load_dotenv()
csv.field_size_limit(sys.maxsize)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"   # overridden by --results-dir
ITEMS_PATH = BASE_DIR / "judge_items.json"
GRADES_PATH = BASE_DIR / "grades.json"

MAX_RESULT_CHARS = 700
MAX_ANSWER_CHARS = 700
SYNTH_ROWS = 10          # the synthesizer only ever saw the first 10 rows


def _h(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]


def _trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " …[truncated]"


def _synth_view(agent_result: str) -> str:
    """The rows the answer synthesizer was given (first SYNTH_ROWS)."""
    try:
        data = json.loads(agent_result)
        if isinstance(data, list):
            return json.dumps(data[:SYNTH_ROWS], default=str)
    except json.JSONDecodeError:
        pass
    return agent_result


def is_error(result_json: str) -> bool:
    try:
        d = json.loads(result_json)
    except json.JSONDecodeError:
        return True
    return isinstance(d, dict) and "error" in d


def rule_sq(row: dict) -> tuple[int, str] | None:
    """SQ grades that follow mechanically from the rubric, needing no model:
    an in-scope question that got a refusal/'no results' answer because no
    query ran, or because the query errored, scores 1."""
    ans = row["agent_answer"].strip()
    if not row["agent_sql_query"].strip():
        return 1, "no SQL produced for an in-scope question; answer is a refusal/error message"
    if is_error(row["agent_result"]) and (ans.startswith("No results found") or not ans):
        return 1, "query errored but answer reports 'no results' / is empty"
    if not ans:
        return 1, "empty answer"
    return None


def load_rows() -> list[tuple[str, dict]]:
    out = []
    for path in sorted(glob.glob(str(RESULTS_DIR / "agent*" / "*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "ex_relational" not in row:
                    sys.exit(f"{path} has not been scored: run score_benchmark.py first")
                out.append((path, row))
    return out


def ex_key(row: dict) -> str:
    return _h(row["question"], row["ground_truth_result"], row["agent_result"])


def sq_key(row: dict) -> str:
    return _h(row["question"], _synth_view(row["agent_result"]), row["agent_answer"])


def needs_ex_judge(row: dict) -> bool:
    return (row["ex_relational"] == "0" and row["valid_sql"] == "1"
            and not is_error(row["ground_truth_result"]))


def export():
    grades = json.loads(GRADES_PATH.read_text()) if GRADES_PATH.exists() else {"ex": {}, "sq": {}}
    ex_items, sq_items = {}, {}
    for _, row in load_rows():
        if needs_ex_judge(row):
            k = ex_key(row)
            if k not in grades["ex"]:
                ex_items.setdefault(k, {
                    "question": row["question"],
                    "ground_truth": _trim(row["ground_truth_result"], MAX_RESULT_CHARS),
                    "agent": _trim(row["agent_result"], MAX_RESULT_CHARS),
                })
        if rule_sq(row) is None:
            k = sq_key(row)
            if k not in grades["sq"]:
                sq_items.setdefault(k, {
                    "question": row["question"],
                    "result": _trim(_synth_view(row["agent_result"]), MAX_RESULT_CHARS),
                    "answer": _trim(row["agent_answer"], MAX_ANSWER_CHARS),
                })
    ITEMS_PATH.write_text(json.dumps({"ex": ex_items, "sq": sq_items}, indent=1))
    print(f"Wrote {ITEMS_PATH.name}: {len(ex_items)} EX items, {len(sq_items)} SQ items still ungraded")


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def api():
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY (environment or .env) to use --mode api")
    client = anthropic.Anthropic()
    items = json.loads(ITEMS_PATH.read_text())
    grades = json.loads(GRADES_PATH.read_text()) if GRADES_PATH.exists() else {"ex": {}, "sq": {}}

    def ask(system: str, user: str) -> dict:
        r = client.messages.create(
            model=JUDGE_MODEL, max_tokens=2000, system=system,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": user}])
        return _parse_json(next(b.text for b in r.content if b.type == "text"))

    for k, it in items["ex"].items():
        v = ask(EX_RUBRIC, f"Question: {it['question']}\n\nGround-truth result:\n{it['ground_truth']}\n\n"
                           f"Agent result:\n{it['agent']}")
        grades["ex"][k] = {"equivalent": bool(v.get("equivalent")), "reason": v.get("reason", "")}
    for k, it in items["sq"].items():
        v = ask(SQ_RUBRIC, f"Question: {it['question']}\n\nAgent query result:\n{it['result']}\n\n"
                           f"Agent answer:\n{it['answer']}")
        grades["sq"][k] = {"score": int(v.get("score", 1)), "reason": v.get("reason", "")}
        GRADES_PATH.write_text(json.dumps(grades, indent=1))   # checkpoint
    GRADES_PATH.write_text(json.dumps(grades, indent=1))
    print(f"Graded {len(items['ex'])} EX and {len(items['sq'])} SQ items into {GRADES_PATH.name}")


def apply():
    grades = json.loads(GRADES_PATH.read_text()) if GRADES_PATH.exists() else {"ex": {}, "sq": {}}
    missing = {"ex": 0, "sq": 0}
    by_file: dict[str, list[dict]] = {}
    for path, row in load_rows():
        by_file.setdefault(path, []).append(row)
    for path, rows in by_file.items():
        for row in rows:
            judge = 0
            if needs_ex_judge(row):
                g = grades["ex"].get(ex_key(row))
                if g is None:
                    missing["ex"] += 1
                else:
                    judge = int(g["equivalent"])
            row["ex_judge"] = judge
            row["ex_soft"] = int(row["ex_relational"] == "1" or judge == 1)
            rs = rule_sq(row)
            if rs is not None:
                row["sq_score"] = rs[0]
            else:
                g = grades["sq"].get(sq_key(row))
                if g is None:
                    missing["sq"] += 1
                    row["sq_score"] = ""
                else:
                    row["sq_score"] = g["score"]
        fields = list(rows[0].keys())
        fd, tmp = tempfile.mkstemp(dir=Path(path).parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as out:
            w = csv.DictWriter(out, fieldnames=fields, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    print(f"Applied grades to {len(by_file)} files. Missing grades: EX={missing['ex']}, SQ={missing['sq']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["export", "api", "apply"], required=True)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                    help="Directory holding agent1/ and agent2/ (default: results/)")
    args = ap.parse_args()
    RESULTS_DIR = args.results_dir
    {"export": export, "api": api, "apply": apply}[args.mode]()
