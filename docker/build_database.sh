#!/bin/bash
# Runs inside the `loader` build stage.
set -euo pipefail
mkdir -p "$PGDATA" && chown postgres:postgres "$PGDATA" && chmod 700 "$PGDATA"

gosu postgres initdb -D "$PGDATA" --username=postgres --encoding=UTF8 \
    --auth-local=trust --auth-host=scram-sha-256

# Bulk-load settings, used for this build only (passed on the command line, not persisted).
gosu postgres pg_ctl -D "$PGDATA" -w start -o "-c listen_addresses='' \
    -c fsync=off -c synchronous_commit=off -c full_page_writes=off \
    -c maintenance_work_mem=512MB -c max_wal_size=2GB -c min_wal_size=80MB -c wal_recycle=off"

gosu postgres createdb chicago_crime_new
echo "Loading snapshot ..."
cat /snapshot/*.part-* | gunzip | gosu postgres psql -v ON_ERROR_STOP=1 -q -d chicago_crime_new -o /dev/null

gosu postgres psql -v ON_ERROR_STOP=1 -d chicago_crime_new <<'EOSQL'
-- Read-only role the benchmark connects as: agent_1 has no SQL safety check, so the
-- database itself must refuse anything but SELECT.
CREATE ROLE ai_agent LOGIN PASSWORD 'benchmark';
GRANT CONNECT ON DATABASE chicago_crime_new TO ai_agent;
GRANT USAGE ON SCHEMA public TO ai_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ai_agent;
ALTER ROLE ai_agent SET default_transaction_read_only = on;
-- Runtime settings (the defaults are too small for 8.6M-row aggregations on a laptop).
ALTER SYSTEM SET shared_buffers = '512MB';
ALTER SYSTEM SET work_mem = '32MB';
ALTER SYSTEM SET effective_cache_size = '2GB';
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
EOSQL

# set hint bits and planner statistics so the first benchmark queries are not slow
gosu postgres psql -v ON_ERROR_STOP=1 -d chicago_crime_new -c "VACUUM (ANALYZE)"
gosu postgres psql -d chicago_crime_new -tAc "select 'crime_incidents rows: ' || count(*) from crime_incidents"
gosu postgres pg_ctl -D "$PGDATA" -m fast -w stop

# network access only for the read-only role; admin (postgres) only via `docker exec`
cat > "$PGDATA/pg_hba.conf" <<'EOF2'
local all all trust
host chicago_crime_new ai_agent all scram-sha-256
EOF2
