#!/usr/bin/env python
"""
chat.py
=======
Chat with the text-to-SQL agent in the terminal. Each question is turned into a
SQL query, run against the Chicago crime database (the Docker container from
docker-compose.yml) and answered in plain English.

This uses exactly the agents, prompts, model and database settings of the
benchmark (sql_agent_pipeline.py, prompt_stages.py, experiment_runner.py).
Every question is answered on its own: the agent has no memory of earlier
questions in the session.

Usage
-----
    docker compose up -d          # start the database first
    python chat.py                # agent_2 with the stage 4 (few-shot) prompt
    python chat.py --agent agent_1 --stage stage2_rules
    python chat.py --verbose      # also print each step of the agent

Type a question at the prompt; type "exit" or press Ctrl+C to quit.
Needs GROQ_API_KEY in .env (see .env.example).
"""

import argparse
import sys
import time
from urllib.parse import quote_plus

from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

from eval_config import STAGES
from experiment_runner import (DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER,
                               GROQ_API_KEY, INCLUDE_TABLES, MODEL_NAME, _invoke_with_retry)
from prompt_stages import STAGE_PROMPTS
from sql_agent_pipeline import AGENTS

MAX_RESULT_CHARS = 1500   # how much of the raw query result to print


def connect():
    """Build the model and database objects the agents need, with clear errors
    for the two usual setup problems (no API key, database not running)."""
    if not GROQ_API_KEY:
        sys.exit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    db_uri = f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD or '')}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        db = SQLDatabase.from_uri(db_uri, include_tables=INCLUDE_TABLES)
    except Exception as exc:
        sys.exit(f"Cannot connect to the database at {DB_HOST}:{DB_PORT} ({exc}).\n"
                 "Start it with `docker compose up -d` and wait until `docker compose ps` shows healthy.")
    llm = ChatGroq(model=MODEL_NAME, temperature=0, api_key=GROQ_API_KEY)
    return llm, db


def show(result: dict, verbose: bool) -> None:
    """Print the answer, then the SQL and the raw rows it was based on."""
    print(f"\nAgent: {result.get('answer') or '(no answer)'}\n")
    sql = result.get("sql_query")
    if sql:
        print("SQL:")
        print("  " + sql.replace("\n", "\n  "))
    rows = result.get("results")
    if rows:
        # The query runner returns the rows as one formatted string (see
        # _query_runner in sql_agent_pipeline.py); print it, shortened if long.
        text = "\n".join(str(r) for r in rows)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + " ... [shortened]"
        print(f"\nRows:\n  {text}")
    if verbose:
        print("\nSteps:")
        for line in result.get("error_log") or []:
            print(f"  - {line}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Chat with the Chicago crime text-to-SQL agent")
    ap.add_argument("--agent", default="agent_2", choices=list(AGENTS),
                    help="agent_2 (relevance check, validation, retry; default) or agent_1 (simple)")
    ap.add_argument("--stage", default="stage4_few_shot", choices=STAGES,
                    help="Prompt stage used to write the SQL (default: stage4_few_shot)")
    ap.add_argument("--verbose", action="store_true", help="Also print each step the agent took")
    args = ap.parse_args()

    llm, db = connect()
    app = AGENTS[args.agent](STAGE_PROMPTS[args.stage], llm, db)
    print(f"Chicago crime text-to-SQL agent ({args.agent}, {args.stage}, {MODEL_NAME})")
    print('Ask a question about Chicago crime, e.g. "How many thefts were reported in 2024?"')
    print('Type "exit" to quit.\n')

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        start = time.time()
        try:
            # Same call the benchmark uses, including the retry on Groq rate limits.
            result = _invoke_with_retry(app, question)
        except KeyboardInterrupt:
            print("\n(cancelled)\n")
            continue
        except Exception as exc:
            print(f"\nError: {exc}\n")
            continue
        show(result, args.verbose)
        print(f"({time.time() - start:.1f}s)\n")


if __name__ == "__main__":
    main()
