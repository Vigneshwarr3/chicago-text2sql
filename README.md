# Chicago Crime Text-to-SQL Benchmark

A controlled comparison of two LangGraph text-to-SQL agents on 96 natural-language questions about the
Chicago crime database (8.6M incidents), using `openai/gpt-oss-120b` on Groq.

- **agent_1**: simple pipeline: fetch schema → generate SQL → run → summarize.
  No relevance check, no validation, no retry.
- **agent_2**: full pipeline: relevance check → fetch schema → generate SQL → `EXPLAIN` validation
  with up to 3 regeneration attempts → run → summarize.
- **4 prompt stages** for the SQL-generation step: bare instruction, + rules, + one example, + four examples.
- **5 runs** of each agent × stage = 40 runs, 3,840 scored answers.

Metrics: **VSR** (valid SQL rate), **EX** (execution accuracy: strict, relational, and hybrid with a judge
fallback) and **SQ** (synthesis quality, 1–5).

The database is distributed as a Docker image (`vigneshwarr/chicago-crime-db`): PostgreSQL 15 with the
full data already loaded, so it works on Windows, macOS and Linux and starts in seconds.

## What you need

| | |
|---|---|
| Docker | Docker Desktop (Windows / macOS; WSL 2 backend on Windows) or Docker Engine + Compose v2 (Linux). About 6 GB free disk for the image, and at least 4 GB of memory available to Docker. |
| Python | 3.10 or newer |
| Groq API key | for the LLM. Groq's free tier is far too small for the full benchmark; use a paid (Developer) tier. |

## Quick start

```bash
git clone https://github.com/Vigneshwarr3/chicago-text2sql.git && cd chicago-text2sql

# 1. configuration
cp .env.example .env              # Windows PowerShell: Copy-Item .env.example .env
#    then edit .env and set GROQ_API_KEY

# 2. start the database (the first run downloads the image, a few GB)
docker compose up -d
docker compose ps                 # wait until STATUS says "healthy"

# 3. python environment
python -m venv .venv
source .venv/bin/activate         # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Check the database

```bash
docker compose exec db psql -U ai_agent -d chicago_crime_new -c "SELECT count(*) FROM crime_incidents"
```

Expected: `8643493`. The lookup tables hold 23 districts (`district_names`), 77 community areas
(`community_area_names`) and 50 wards (`ward_names`).

### Run the benchmark

```bash
# smoke test: 1 agent, 1 stage, 1 run, 20 questions (a few minutes, cents)
python experiment_runner.py --agent agent_1 --stage stage1_simple --runs 1 --dev-subset

# the full benchmark: 2 agents x 4 stages x 5 runs
python experiment_runner.py
```

Output goes to `results/agent1/` and `results/agent2/`, one file per run (`{stage}_run{k}.csv`).
The `results/` folder in this repository holds the author's full run on the 8.6M-row database: 40 files,
96 questions each, **raw agent output (not yet scored)**. Running `score_benchmark.py` on it adds the
scoring columns in place.
Useful flags: `--agent`, `--stage`, `--runs`, `--run-start`, `--workers` (lower it if you see many
`rate limited` lines), `--resume` (continue an interrupted run), `--ids`, `--dev-subset`.
Four questions (ids 4, 14, 22, 63) are the worked examples in the stage 3/4 prompts and are never scored.

### Score the runs

```bash
python score_benchmark.py              # VSR, EX hard, EX relational (re-executes every query, read-only)
python judge_results.py --mode export  # list the answers / results that need a judge grade
python judge_results.py --mode api     # grade them with Claude (needs ANTHROPIC_API_KEY in .env)
python judge_results.py --mode apply   # write EX hybrid and SQ into the run files
python aggregate_results.py            # mean ± SD tables
```

`score_benchmark.py` needs only the database and gives the deterministic metrics (VSR, EX hard,
EX relational). The judged metrics (EX hybrid, SQ) need a judge: the rubrics are in `judge_rubric.py`
and `--mode api` applies them with the model named there. Grades are cached in `grades.json` keyed by a hash
of the content judged, so re-running a step only grades what is new. Until the judged metrics exist,
`aggregate_results.py` prints `n/a` for them.

## Reproducibility notes

- **The database is a fixed snapshot.** The image contains every incident from 2001-01-01 through
  2026-09-29 as exported on 2026-10-10 (8,643,493 rows). The upstream database is refreshed weekly, so a fresh
  download would not match. For publications, pin the image by digest:
  `vigneshwarr/chicago-crime-db@sha256:594f1abaf67787b4cb3c2e1c5b97df0eced2a433a8d1cce6764d83fee60d7103`
  (tags `latest` and `2026-09-29`; set `DB_IMAGE` in `.env` to use it with `docker compose`).
- **Large tables, no indexes.** A few questions (window functions over the whole table) take 25–30 s per query.
  `score_benchmark.py` uses a 120 s per-query timeout (`--timeout`).
- **LLM nondeterminism.** Temperature is 0, but Groq inference is not bit-exact. Expect repeated runs to agree
  within a few points, not row by row.
- **Read-only database role.** The benchmark connects as `ai_agent`, a role that can only `SELECT`
  (`default_transaction_read_only`), because agent_1 has no SQL safety check.
- **Schema shown to the model.** Only the four tables below are given to the LLM (`INCLUDE_TABLES` in
  `experiment_runner.py`).

## The data

`crime_incidents` (8,643,493 rows, 38 columns) plus the lookup tables `district_names`,
`community_area_names` and `ward_names`, which `DISTRICT`, `COMMUNITY_AREA` and `WARD` in the incident table
join to by id. Incidents are Chicago Police Department records published by the City of Chicago on its open
data portal ("Crimes – 2001 to Present"). Several columns (`CRIME_CATEGORY`, `CRIME_OFFENSE_GROUP`,
`CRIME_SEVERITY_LEVEL`, `LOCATION_RISK_CATEGORY`, the `*_FLAG` columns, `PROCESS_DATE`, …) are not part of the
portal's original schema; they were added by the ETL pipeline that produced this database. Check the
portal's terms of use for the source data.

`docker/Dockerfile` and `docker/build_database.sh` document how the image was built: a PostgreSQL cluster is
initialised and loaded at build time and only the finished data directory is copied into the final image. The
raw snapshot is not stored in this repository.

## Repository layout

| Path | Purpose |
|---|---|
| `docker-compose.yml` | runs the published database image |
| `docker/` | how the database image was built |
| `evaluation_dataset_v2.csv` | the 100 questions with reference SQL (Easy 1–30, Medium 31–70, Hard 71–100) |
| `prompt_stages.py` | the four prompt variants |
| `sql_agent_pipeline.py` | agent_1 and agent_2 (LangGraph) |
| `experiment_runner.py` | runs agents × stages × repeats |
| `score_benchmark.py` | deterministic scoring against the database |
| `judge_results.py`, `judge_rubric.py` | judged metrics: rubrics and grading |
| `aggregate_results.py` | summary tables |
| `eval_config.py` | reserved example ids, dev subset, stage names |

## Troubleshooting

- **Port already in use.** Another PostgreSQL is on the same port. Set `DB_PORT` in `.env` to a free port
  (default 5440) and run `docker compose up -d` again.
- **`rate limited` messages.** Groq's tokens-per-minute cap. The runner retries with backoff; lower `--workers`
  if it happens constantly.
- **Cannot connect to the database from Python.** Check `docker compose ps` shows `healthy`, and that
  `DB_HOST`, `DB_PORT`, `DB_USER` and `DB_PASSWORD` in `.env` match `.env.example`.
- **Slow queries or Docker running out of memory.** Give Docker at least 4 GB (Docker Desktop → Settings →
  Resources).
