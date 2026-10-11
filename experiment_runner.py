#!/usr/bin/env python
"""
experiment_runner.py
=====================
Runs agents x prompt stages x repeats over data/evaluation_dataset.csv through the
agents in sql_agent_pipeline.py (agent_1 = simple, agent_2 = relevance check +
validation/retry). Each (agent, stage, run) writes its own file:

    results/agent1/{stage}_run{k}.csv
    results/agent2/{stage}_run{k}.csv

Usage
-----
    # Everything: 2 agents x 4 stages x 5 runs = 40 files
    python experiment_runner.py

    # One agent, all stages, 5 runs
    python experiment_runner.py --agent agent_1

    # Cheap smoke test: one agent/stage/run on the 20-question dev subset
    python experiment_runner.py --agent agent_1 --stage stage1_simple --runs 1 --dev-subset

    # Resume an interrupted run (skips question ids already in each file)
    python experiment_runner.py --agent agent_2 --resume

    # Only runs 3-5 of one stage
    python experiment_runner.py --agent agent_2 --stage stage2_rules --run-start 3 --runs 3
"""

import argparse
import csv
import io
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase

from eval_config import DEV_SUBSET_IDS, RESERVED_FEWSHOT_IDS, STAGES
from prompt_stages import STAGE_PROMPTS
from sql_agent_pipeline import AGENTS, invoke_agent

load_dotenv()   # reads GROQ_API_KEY and DB_* settings from .env

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_USER = os.getenv("DB_USER", "ai_agent")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5440")
DB_NAME = os.getenv("DB_NAME", "chicago_crime_new")
MODEL_NAME = "openai/gpt-oss-120b"   # the model under test, served by Groq

DATASET = "data/evaluation_dataset.csv"   # the 100 questions and their reference SQL
# Tables shown to the LLM. The benchmark database holds only these four tables;
# the list is kept explicit so the schema the model sees never depends on what
# else is installed in the database.
INCLUDE_TABLES = ["crime_incidents", "district_names", "community_area_names", "ward_names"]
DEFAULT_WORKERS = 8   # questions answered in parallel within one run

# Columns of every results file. score_benchmark.py and judge_results.py later
# append their own columns to the same files.
OUTPUT_FIELDS = [
    "id", "complexity", "question",
    "ground_truth_sql", "agent_sql_query", "agent_answer", "error_log",
]


def _output_csv_for(agent: str, stage: str, run: int) -> Path:
    """results/agent1|agent2/{stage}_run{k}.csv — one file per (agent, stage, run)."""
    return RESULTS_DIR / agent.replace("_", "") / f"{stage}_run{run}.csv"


def load_completed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["id"]) for row in csv.DictReader(f) if row.get("id", "").isdigit()}


def read_evaluation_dataset(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _invoke_with_retry(app, question: str, max_attempts: int = 6) -> dict:
    """Retry on Groq 429s (token-per-minute rate limits) with jittered
    exponential backoff — these are infra noise, not a signal about prompt
    quality, and would otherwise unfairly penalize whichever run happens to
    hit the limit. With parallel workers, 429s are expected occasionally."""
    for attempt in range(1, max_attempts + 1):
        try:
            return invoke_agent(app, question)
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
            if not is_rate_limit or attempt == max_attempts:
                raise
            wait = min(60, 4 * 2 ** (attempt - 1)) + random.uniform(0, 3)
            print(f"         [rate limited, retrying in {wait:.0f}s — attempt {attempt}/{max_attempts}]")
            time.sleep(wait)


def run_once(app, rows: list[dict], all_rows: list[dict], output_csv: Path,
             args, use_patch_mode: bool) -> tuple[int, int]:
    """Run one (agent, stage, run) over `rows` and write `output_csv`.
    Returns (success, failed).

    Two write modes:
      normal     write the file from scratch (or append to it with --resume)
      patch      used with --ids / --dev-subset: re-run only some questions and
                 replace just those rows in an existing file, keeping the others
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = load_completed_ids(output_csv) if args.resume else set()

    existing_results: dict[int, dict] = {}
    if use_patch_mode and output_csv.exists():
        with open(output_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("id", "").isdigit():
                    existing_results[int(row["id"])] = row

    if use_patch_mode:
        out_f = io.StringIO()
    else:
        out_f = open(output_csv, "a" if args.resume else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_ALL)
    if use_patch_mode or not output_csv.exists() or not args.resume:
        writer.writeheader()

    total = len(rows)
    success = 0
    failed = 0
    write_lock = threading.Lock()

    # Answer one question; called from the thread pool below. A failure is
    # recorded in error_log instead of stopping the whole run.
    def work(i: int, row: dict) -> tuple[bool, dict]:
        qid = int(row["id"])
        question = row["question"]
        print(f"[{i}/{total}] id={qid} ({row['complexity']}) — {question[:70]}…")
        out = {"id": qid, "complexity": row["complexity"], "question": question,
               "ground_truth_sql": row["sql_query"]}
        try:
            result = _invoke_with_retry(app, question)
            agent_answer = result.get("answer") or ""
            out.update(agent_sql_query=result.get("sql_query") or "", agent_answer=agent_answer,
                       error_log=" | ".join(result.get("error_log") or []))
            print(f"         id={qid} → {'OK' if out['agent_sql_query'] else 'NO_SQL'} | {agent_answer[:60]!r}")
            ok = True
        except Exception as exc:
            print(f"         id={qid} → ERROR: {exc}")
            out.update(agent_sql_query="", agent_answer="", error_log=f"Script exception: {exc}")
            ok = False
        with write_lock:                      # threads share one output file
            writer.writerow(out)
            out_f.flush()
        return ok, out

    todo = [(i, r) for i, r in enumerate(rows, start=1)
            if not (args.resume and int(r["id"]) in completed_ids)]
    if len(todo) < total:
        print(f"Skipping {total - len(todo)} rows already in {output_csv.name}.")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, i, r) for i, r in todo]
            for fut in as_completed(futures):
                if fut.result()[0]:
                    success += 1
                else:
                    failed += 1
    finally:
        if not use_patch_mode:
            out_f.close()

    if not use_patch_mode:
        # Rows were written in completion order; restore dataset order.
        with open(output_csv, newline="", encoding="utf-8") as f:
            by_id = {int(r["id"]): r for r in csv.DictReader(f)}
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(by_id[int(r["id"])] for r in all_rows if int(r["id"]) in by_id)

    # Patch mode: merge the re-run rows into the rows already in the file.
    if use_patch_mode:
        out_f.seek(0)
        for new_row in csv.DictReader(out_f):
            if new_row.get("id", "").isdigit():
                existing_results[int(new_row["id"])] = new_row

        id_order = [int(r["id"]) for r in all_rows]
        full_rows = [existing_results[qid] for qid in id_order if qid in existing_results]

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(full_rows[0].keys()) if full_rows else OUTPUT_FIELDS
            w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, extrasaction="ignore")
            w.writeheader()
            w.writerows(full_rows)

    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Run agents x prompt stages x repeats over data/evaluation_dataset.csv")
    parser.add_argument("--agent", default="all", choices=[*AGENTS, "all"],
                        help="agent_1 (simple), agent_2 (validate+retry) or all (default)")
    parser.add_argument("--stage", default="all", choices=[*STAGES, "all"],
                        help="Which prompt stage to run (default: all four)")
    parser.add_argument("--runs", type=int, default=5, help="Repeats per (agent, stage) (default: 5)")
    parser.add_argument("--run-start", type=int, default=1,
                        help="First run number, e.g. --run-start 3 --runs 3 does runs 3-5")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Questions run in parallel within each run (default: 8). "
                             "Limited by Groq's 250K tokens/min; lower it if you see many 429s.")
    parser.add_argument("--start", type=int, default=1, help="First question id to run (inclusive)")
    parser.add_argument("--end", type=int, default=100, help="Last question id to run (inclusive)")
    parser.add_argument("--resume", action="store_true", help="Skip rows already in each output file")
    parser.add_argument("--ids", type=int, nargs="+", help="Re-run only these specific question ids (overrides --start/--end)")
    parser.add_argument("--exclude-ids", type=int, nargs="+", default=RESERVED_FEWSHOT_IDS,
                        help="Ids to skip (default: the ids reserved as prompt examples)")
    parser.add_argument("--dev-subset", action="store_true",
                        help="Shortcut for --ids <the fixed 20-question dev subset>")
    args = parser.parse_args()

    agents = list(AGENTS) if args.agent == "all" else [args.agent]
    stages = STAGES if args.stage == "all" else [args.stage]
    runs = range(args.run_start, args.run_start + args.runs)

    input_csv = BASE_DIR / DATASET
    db_uri = f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD or '')}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    print(f"Connecting to database {DB_NAME} at {DB_HOST}:{DB_PORT} …")
    db = SQLDatabase.from_uri(db_uri, include_tables=INCLUDE_TABLES)
    llm = ChatGroq(model=MODEL_NAME, temperature=0, api_key=GROQ_API_KEY)
    print(f"DB dialect: {db.dialect}")

    all_rows = read_evaluation_dataset(input_csv)

    # Pick the questions: the dev subset, explicit --ids, or an id range.
    if args.dev_subset:
        id_set = set(DEV_SUBSET_IDS)
        rows = [r for r in all_rows if int(r["id"]) in id_set]
    elif args.ids:
        id_set = set(args.ids)
        rows = [r for r in all_rows if int(r["id"]) in id_set]
    else:
        rows = [r for r in all_rows if args.start <= int(r["id"]) <= args.end]

    exclude_set = set(args.exclude_ids or [])
    rows = [r for r in rows if int(r["id"]) not in exclude_set]
    use_patch_mode = bool(args.ids or args.dev_subset)

    # One compiled graph per (agent, stage); the same graph is reused for all
    # repeats of that pair, each writing its own file.
    summary = []
    for agent in agents:
        for stage in stages:
            app = AGENTS[agent](STAGE_PROMPTS[stage], llm, db)
            for run in runs:
                output_csv = _output_csv_for(agent, stage, run)
                label = f"{agent}_{stage}_run{run}"
                print(f"\n{'='*60}\n{label}  →  {output_csv.relative_to(BASE_DIR)}\n{'='*60}")
                success, failed = run_once(app, rows, all_rows, output_csv, args, use_patch_mode)
                summary.append((label, len(rows), success, failed))

    print(f"\n{'='*60}")
    for label, total, success, failed in summary:
        print(f"{label:<40} Total={total}  Success={success}  Failed={failed}")
    print(f"Results saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
