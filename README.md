# Chicago Crime Text-to-SQL Benchmark

A controlled comparison of two LangGraph text-to-SQL agents on 96 natural-language questions about the
Chicago crime database (8.6M incidents), using `openai/gpt-oss-120b` on Groq.

- **agent_1**: simple pipeline: fetch schema → generate SQL → run → summarize.
  No relevance check, no validation, no retry.
- **agent_2**: full pipeline: relevance check → fetch schema → generate SQL → `EXPLAIN` validation
  with up to 3 regeneration attempts → run → summarize.
- **4 prompt stages** for the SQL-generation step: bare instruction, + rules, + one example, + four examples.
- **5 runs** of each agent × stage = 40 runs, 3,840 scored answers.

Three metrics:

| Metric | Definition |
|---|---|
| **VSR** | valid SQL rate: 1 if the agent produced SQL and it ran without error (no SQL counts as 0) |
| **EX** | execution accuracy: 1 if the agent's result carries the same information as the reference result |
| **SQ** | synthesis quality: 1–5 score of the plain-English answer |

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

### Chat with the agent

```bash
python chat.py                                    # agent_2 with the stage 4 (few-shot) prompt
python chat.py --agent agent_1 --stage stage2_rules
python chat.py --verbose                          # also show each step the agent took
```

Type a question such as `How many thefts were reported in 2024?`. The agent prints its answer, the SQL it
wrote and the rows the query returned. Each question is answered on its own (no conversation memory). This is
exactly the benchmarked agent, so it has the same limitations (see [Known limitations](#known-limitations)):
agent_2, for example, rejects questions that contain no listed action phrase, such as "Which year had the
highest number of homicide offenses?". Needs `GROQ_API_KEY` in `.env` and the database running.

### Run the benchmark

```bash
# smoke test: 1 agent, 1 stage, 1 run, 20 questions (a few minutes, cents)
python experiment_runner.py --agent agent_1 --stage stage1_simple --runs 1 --dev-subset

# the full benchmark: 2 agents x 4 stages x 5 runs
python experiment_runner.py
```

Output goes to `results/agent1/` and `results/agent2/`, one file per run (`{stage}_run{k}.csv`).
The `results/` folder in this repository holds the author's full run on the 8.6M-row database: 40 files,
96 questions each, already scored (columns `valid_sql`, `ex_match`, `ex`, `sq_score`). A new run overwrites
these files, so copy the folder first if you want to keep them.
Useful flags: `--agent`, `--stage`, `--runs`, `--run-start`, `--workers` (lower it if you see many
`rate limited` lines), `--resume` (continue an interrupted run), `--ids`, `--dev-subset`.
Four questions (ids 4, 14, 22, 63) are the worked examples in the stage 3/4 prompts and are never scored.

### Score the runs

```bash
python score_benchmark.py              # VSR and the exact-match pre-check (re-executes every query, read-only)
python judge_results.py                # Claude Haiku 5.5 grades EX and SQ via the API (needs ANTHROPIC_API_KEY in .env)
                                       # and writes them into the run files
python aggregate_results.py            # mean ± SD tables
```

- **VSR** needs only the database.
- **EX**: a result that equals the reference after normalization counts as correct without a judge call.
  Normalization ignores row order, column names, case/whitespace and numeric precision beyond 2 decimals;
  column order and extra or missing columns are *not* ignored. Every other result from a query that ran is
  judged for equivalence by the judge model (`JUDGE_MODEL` in `judge_rubric.py`). EX is 0 when the SQL did
  not run.
- **SQ**: the judge applies the 1–5 rubric in `judge_rubric.py`. Answers to questions where no query ran, or the
  query errored and the answer says "no results", score 1 by rule, without a judge call.
- `judge_results.py` is one step: it sends every item that needs a grade to the API, caches the grades in
  `grades.json` keyed by a hash of the content judged (identical items are graded once, and a re-run only grades
  what is new), and writes `ex` and `sq_score` into the run files. `--limit 5` tries the judge on five items
  first; `--dry-run` only counts what needs grading. Until it has run, `aggregate_results.py` prints `n/a` for EX and SQ.

### Reproduce the published numbers

```bash
python aggregate_results.py            # tables from the scored files in results/ (no database, no API key)
```

To re-derive the scores from the raw agent output instead, run `score_benchmark.py` (needs the database) and then
`judge_results.py`. `grades.json` holds every grade the judge gave for the published run (1,412 items, each with a
one-line reason), so for these results `judge_results.py` makes no API calls and needs no key. Grading a new run
of your own needs an `ANTHROPIC_API_KEY`, because its answers will differ.

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

## Known limitations

The code is kept exactly as it was benchmarked, so that the published numbers reproduce. These issues are
therefore present on purpose and are documented rather than fixed:

- **agent_2's relevance checker rejects some in-scope questions.** It is a keyword rule: a question needs one
  listed action phrase ("how many", "list", "show", "top", ...) and one data word. Questions worded "Which ... had
  ..." or "What is the ... rate ..." have no action phrase, so the same 8 of the 96 benchmark questions (ids 28,
  35, 37, 42, 44, 62, 95, 96) are rejected in every run. This costs agent_2 8.3 points of VSR and EX.
- **The judge sees only part of long results.** `judge_results.py` shows the judge at most the 20 rows stored in
  each results file and 700 characters of them. For long results it can compare or verify only the visible part,
  so some EX acceptances rest on a partial comparison, and some SQ scores are lowered because figures "cannot be
  verified". The share of results that match exactly without the judge (`ex_match`) is the conservative
  figure.
- **SQ is affected by a digit-copying error in some runs.** In many runs the answer synthesizer changed digits
  when restating a number from the result (the query returned 2,158,897; the answer said 2,158,806). This
  happened in all agent_1 runs, agent_2 stage 1 and agent_2 stage 2 runs 1–2, and not afterwards, with no
  change to the code. SQ in these files should therefore not be used to compare the agents or the prompt
  stages.
- **The judge's grades have not been audited by hand.** Every grade and its reason are in `grades.json` for
  anyone who wants to check them.

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
| `score_benchmark.py` | VSR and the exact-match pre-check, against the database |
| `judge_results.py`, `judge_rubric.py` | EX and SQ judging: rubrics and grading |
| `aggregate_results.py` | summary tables |
| `chat.py` | chat with the agent in the terminal |
| `results/` | the author's 40 scored runs |
| `grades.json` | every judge grade (EX and SQ) with its reason |
| `agent_diagrams.ipynb`, `figures/` | diagrams of the two agent graphs (PNG, SVG, PDF) |
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

## License

The code is released under the [MIT License](LICENSE). The crime data comes from the City of Chicago's open
data portal ("Crimes – 2001 to Present") and is subject to the portal's terms of use.
