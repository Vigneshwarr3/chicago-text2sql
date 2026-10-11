#!/usr/bin/env python
"""
judge_results.py
================
Adds the two judged metrics to the scored results CSVs, using the Anthropic API
(run score_benchmark.py first):

  ex         1 if the agent's result carries the same information as the reference
             result: either score_benchmark.py's exact match (ex_match = 1) or,
             when that fails, the judge says the results are equivalent
  sq_score   1-5 Synthesis Quality of the agent's plain-English answer

Rows whose SQL did not run (valid_sql = 0) get ex = 0, and answers with no query
or an errored query get sq_score = 1, without a judge call.
One command does everything: it finds the items that still need a grade, sends
them to the judge (JUDGE_MODEL, or --model), caches the grades in results/grades.json and
writes ex / sq_score into every results CSV. Judging is keyed by content hash,
so identical (question, result, answer) triples across the 40 files are judged
once, and re-running only grades what is new. Needs ANTHROPIC_API_KEY in the
environment or .env. Rubrics: judge_rubric.py.

    python judge_results.py --limit 5     # try the judge on five items first
    python judge_results.py               # grade everything and write the CSVs
    python judge_results.py --dry-run     # only count what needs grading
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
import time
from pathlib import Path

from dotenv import load_dotenv

from judge_rubric import EX_RUBRIC, JUDGE_MODEL, SQ_RUBRIC

load_dotenv()
csv.field_size_limit(sys.maxsize)

BASE_DIR = Path(__file__).resolve().parent
GRADES_PATH = BASE_DIR / "results" / "grades.json"   # the grade cache, kept next to the results

# What the judge is shown. Results and answers longer than this are cut, so the
# judge sees at most the first 20 stored rows (score_benchmark.MAX_ROWS) and 700
# characters of them. Known limitation, kept so the published grades reproduce:
# for long results the judge can compare or verify only the visible part.
MAX_RESULT_CHARS = 700
MAX_ANSWER_CHARS = 700
# SQ is judged on the first 10 stored rows. (The answer synthesizer itself is
# given the whole result: see _answer_synthesizer in sql_agent_pipeline.py.)
SYNTH_ROWS = 10


# Short content hash used as the cache key in grades.json.
def _h(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]


def _trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " …[truncated]"


def _synth_view(agent_result: str) -> str:
    """The result rows shown to the judge for SQ (first SYNTH_ROWS)."""
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


def load_files(results_dir: Path) -> dict[str, list[dict]]:
    files = {}
    for path in sorted(glob.glob(str(results_dir / "agent*" / "*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and "ex_match" not in rows[0]:
            sys.exit(f"{path} has not been scored: run score_benchmark.py first")
        files[path] = rows
    if not files:
        sys.exit(f"No result CSVs under {results_dir}")
    return files


# Cache keys: an EX grade depends on (question, reference result, agent result);
# an SQ grade on (question, result rows shown, answer). Identical items in
# different files therefore share one grade.
def ex_key(row: dict) -> str:
    return _h(row["question"], row["ground_truth_result"], row["agent_result"])


def sq_key(row: dict) -> str:
    return _h(row["question"], _synth_view(row["agent_result"]), row["agent_answer"])


# The judge decides EX only for a query that ran but did not match exactly
# (and whose reference query itself succeeded).
def needs_ex_judge(row: dict) -> bool:
    return (row["ex_match"] == "0" and row["valid_sql"] == "1"
            and not is_error(row["ground_truth_result"]))


def collect_items(files: dict[str, list[dict]], grades: dict) -> tuple[dict, dict]:
    """Items that have no cached grade yet, keyed by content hash."""
    ex_items, sq_items = {}, {}
    for rows in files.values():
        for row in rows:
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
    return ex_items, sq_items


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def grade(todo: list[tuple[str, str, dict]], grades: dict, model: str) -> None:
    """Send each (kind, key, item) to the judge and store the grade."""
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY (environment or .env) to grade the items")
    client = anthropic.Anthropic(max_retries=6)   # the SDK backs off on 429 / 5xx

    def ask(system: str, user: str) -> dict:
        """One judge call; a reply that is not valid JSON is retried once."""
        for attempt in (1, 2):
            r = client.messages.create(
                model=model, max_tokens=4000, system=system,
                messages=[{"role": "user", "content": user}])
            text = next((b.text for b in r.content if b.type == "text"), "")
            try:
                v = _parse_json(text)
            except json.JSONDecodeError:
                v = {}
            if v:
                return v
            print(f"  unparsable judge reply (stop_reason={r.stop_reason}), attempt {attempt}/2")
        raise RuntimeError("judge returned no JSON twice; item left ungraded")

    def save():
        GRADES_PATH.write_text(json.dumps(grades, indent=1))

    t0 = time.time()
    for n, (kind, k, it) in enumerate(todo, 1):
        try:
            if kind == "ex":
                v = ask(EX_RUBRIC, f"Question: {it['question']}\n\nGround-truth result:\n{it['ground_truth']}\n\n"
                                   f"Agent result:\n{it['agent']}")
                grades["ex"][k] = {"equivalent": bool(v.get("equivalent")), "reason": v.get("reason", "")}
            else:
                v = ask(SQ_RUBRIC, f"Question: {it['question']}\n\nAgent query result:\n{it['result']}\n\n"
                                   f"Agent answer:\n{it['answer']}")
                grades["sq"][k] = {"score": min(5, max(1, int(v["score"]))), "reason": v.get("reason", "")}
        except Exception as exc:
            print(f"  [{kind} {k}] skipped: {exc}")
            continue
        if n % 25 == 0:
            save()                                           # checkpoint
            print(f"  {n}/{len(todo)} graded ({time.time() - t0:.0f}s)")
    save()
    print(f"Graded {len(todo)} items using {model}; grades in {GRADES_PATH.name}")


def apply(files: dict[str, list[dict]], grades: dict) -> None:
    """Write ex / sq_score into every results CSV from the cached grades."""
    missing = {"ex": 0, "sq": 0}
    for path, rows in files.items():
        for row in rows:
            judge = 0
            if needs_ex_judge(row):
                g = grades["ex"].get(ex_key(row))
                if g is None:
                    missing["ex"] += 1
                else:
                    judge = int(g["equivalent"])
            row["ex"] = int(row["ex_match"] == "1" or judge == 1)
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
        for legacy in ("ex_hard", "ex_relational", "ex_judge", "ex_soft"):
            if legacy in fields:
                fields.remove(legacy)
        fd, tmp = tempfile.mkstemp(dir=Path(path).parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as out:
            w = csv.DictWriter(out, fieldnames=fields, quoting=csv.QUOTE_ALL, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    print(f"Wrote ex / sq_score to {len(files)} files. Still ungraded: EX={missing['ex']}, SQ={missing['sq']}")


def main():
    ap = argparse.ArgumentParser(description="Grade EX and SQ with the Anthropic API and write them into the results CSVs")
    ap.add_argument("--results-dir", type=Path, default=BASE_DIR / "results",
                    help="Directory holding agent1/ and agent2/ (default: results/)")
    ap.add_argument("--model", default=JUDGE_MODEL, help=f"Judge model (default: {JUDGE_MODEL})")
    ap.add_argument("--limit", type=int, help="Grade only the first N missing items (to try the judge out)")
    ap.add_argument("--dry-run", action="store_true", help="Only report how many items need grading")
    args = ap.parse_args()

    files = load_files(args.results_dir)
    grades = json.loads(GRADES_PATH.read_text()) if GRADES_PATH.exists() else {"ex": {}, "sq": {}}
    ex_items, sq_items = collect_items(files, grades)
    print(f"{len(ex_items)} EX items and {len(sq_items)} SQ items need grading")
    if args.dry_run:
        return
    todo = [("ex", k, it) for k, it in ex_items.items()] + [("sq", k, it) for k, it in sq_items.items()]
    if args.limit:
        todo = todo[:args.limit]
    if todo:
        grade(todo, grades, args.model)
    apply(files, grades)


if __name__ == "__main__":
    main()
