---
created: 2026-06-16
project: fifa-wc-2026-predictions
status: draft
tags: [data, duckdb, schema, design]
---

# Data store: CSV → DuckDB migration

## Context

The predictor's data currently lives in CSV files under `data/`: an upstream
historical feed (`results.csv`, ~49.4k international matches), a curated
tournament file (`wc-2026-games.csv`, the 72 group-stage fixtures with results
filled in as games are played), plus `goalscorers.csv` and `shootouts.csv`.
Results are maintained by hand-editing CSVs.

The user wants to (1) stop hand-editing CSVs, (2) auto-fetch results from an
online source, and (3) render played + predicted games as an HTML site. These
are three independent subsystems. This spec covers **only the first: the data
store.** It is the foundation the other two code against.

Out of scope (separate specs): the auto-fetcher and the HTML site. This spec
provides the schema and write hooks they will use, but does not implement them.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Engine | **DuckDB** | Dominant workload is bulk-read-into-pandas → fit → write-back; near-free CSV migration via native readers; embedded, zero-server; fits the existing duckdb tooling. |
| Migration scope | **Full** | DB is the single source of truth: historical corpus + WC matches + predictions + goalscorers + shootouts. CSVs become import sources / git-tracked exports. |
| Prediction history | **Committed + latest** | `committed` = the locked pick snapshotted ~90 min before kickoff (immutable, the honestly-scored pre-game pick); `latest` = the live forecast each `/reforecast` overwrites. |
| Predicted vs actual | **Separate tables joined on `match_id`** | Predictions are never overwritten by results, enabling "where was the model right?" analysis. |
| `match_id` | **Deterministic slug** | Stable across DB rebuilds so the predictions↔matches join never breaks. |
| Git tracking | **Track `data/wc2026.duckdb` binary** | DB is the one source of truth; no CSV export step. Binary diffs accepted; prediction history lives in-DB via `committed` rows + `forecast_ts`. |

## Storage & source-of-truth

- `data/wc2026.duckdb` is the **single source of truth, tracked in git.** All
  writes (migration, model predictions, later the fetcher) go to the DB; there
  is no CSV export step.
- **Upstream seed CSVs** (`results.csv`, `goalscorers.csv`, `shootouts.csv`)
  remain in git as *import sources* only: they seed the DB initially and are
  re-imported when the martj42 feed updates. They are inputs, not the record.
- `wc-2026-games.csv` is a one-time seed for the WC fixtures; after migration
  the DB is authoritative for WC data and it is no longer hand-edited.
- Tradeoff (accepted): tracking the `.duckdb` binary means git diffs are not
  human-readable, so the "picks evolution = `git log predictions.csv`" audit
  trail is given up. Point-in-time prediction history instead lives *inside* the
  DB via the `committed` rows and `forecast_ts`.

## Schema

```sql
-- All matches: historical + WC fixtures/results. score NULL = unplayed.
matches(
  match_id     TEXT PRIMARY KEY,      -- deterministic slug, see below
  date         DATE,
  home_team    TEXT, away_team TEXT,
  home_score   INT,  away_score INT,  -- NULL until played
  tournament   TEXT,
  neutral      BOOLEAN,
  city         TEXT, country TEXT,
  stage        TEXT,                  -- 'group' | 'knockout' | NULL (non-WC)
  round        TEXT,                  -- 'MD1'..'MD3' | 'R32','R16','QF','SF','3P','F' | NULL
  group_label  TEXT,                  -- 'A'..'L' for group stage | NULL
  source       TEXT                   -- 'upstream' | 'wc2026'
)

-- Predictions, never overwritten by results. committed + latest per match.
predictions(
  match_id        TEXT REFERENCES matches(match_id),
  kind            TEXT,               -- 'committed' | 'latest'
  pred_home_goals INT, pred_away_goals INT,
  outcome         TEXT,               -- 'H' | 'D' | 'A'
  lam_h DOUBLE, lam_a DOUBLE,
  p_result DOUBLE, p_home_g DOUBLE, p_away_g DOUBLE, p_gd DOUBLE, ep DOUBLE,
  model_as_of     DATE,               -- the as_of that produced this pick
  forecast_ts     TIMESTAMP,          -- when this row was written
  PRIMARY KEY (match_id, kind)
)

-- Bonus, not in the model path.
goalscorers(
  match_id TEXT REFERENCES matches(match_id),
  team TEXT, scorer TEXT, minute INT, own_goal BOOLEAN, penalty BOOLEAN
)

-- Shootouts (FT scores read as draws; shootout decides the winner).
shootouts(
  match_id TEXT REFERENCES matches(match_id),
  winner TEXT, first_shooter TEXT
)
```

**`match_id`**: deterministic, derived from the natural key as a readable slug,
`{YYYYMMDD}-{home}-{away}` lowercased/slugified (two national teams do not play
twice on the same date). NOT an autoincrement: a rebuild must reproduce the same
IDs or the predictions join breaks.

**View `v_model_report`**: joins played `matches` to their `committed`
predictions and computes the office-pool points (the a=3 / b=1 / c=1 rule) plus
correctness flags (outcome / goal-diff / exact). Makes "where was the model
right?" a single `SELECT`.

**Stage / round / group labels**: `stage` and group matchday are derivable (all
72 current WC rows are group stage; matchday from date buckets). `group_label`
is left `NULL` for now and sourced later (by the fetcher). Knockout fixtures get
`stage='knockout'` as they are added later.

## Code layout

**New module `wc2026/db.py`** (the only module that touches DuckDB):
- `connect(path=DB_PATH, read_only=False)`
- `init_schema(con)` — idempotent `CREATE TABLE IF NOT EXISTS` for the four
  tables + `v_model_report`.
- `make_match_id(date, home, away) -> str` — the deterministic slug, used
  everywhere a match is referenced.
- `load_matches(con) -> DataFrame` — returns exactly the columns the pipeline
  expects today.
- `upsert_latest_predictions(con, df, model_as_of)` — writes `kind='latest'`.
- `commit_predictions(con, match_ids, force=False)` — snapshots `latest` →
  `committed`; refuses to overwrite an existing committed row unless `force`
  (the 90-min lock).
- `upsert_results(con, rows)` — the hook the fetcher will use.

**Change `wc2026/data.py`**: `load_results()` reads from the DB via
`db.load_matches()` instead of the CSV, keeping its return contract identical.
`build_training_frame`, `upcoming_fixtures`, the model, and the existing tests
are untouched, the data source is swapped underneath them.

**Change `scripts/run_schedule.py`**: after predicting, write picks as
`kind='latest'`. A `--commit` flag (or small `scripts/commit_picks.py`) performs
the `latest`→`committed` snapshot for matches kicking off soon.

**New `scripts/migrate_to_duckdb.py`** (idempotent, re-runnable): seeds the DB
from the existing CSVs (the same `results.csv` + `wc-2026-games.csv` union we
already build, plus `goalscorers.csv` and `shootouts.csv`), assigns `match_id`,
and sets `stage`/`round` for WC rows (`group_label` left `NULL`, sourced later).
The seed CSVs remain, so the migration is re-runnable.

## Testing

In-memory DuckDB (`:memory:`) fixtures. New tests:
- `make_match_id` stability (same inputs → same id across runs).
- `init_schema` idempotency.
- seed migration row counts match the source CSVs.
- **commit immutability**: a `latest` re-forecast must not clobber an existing
  `committed` pick.
- `v_model_report` points equal `score_prediction` for the same inputs.

The existing 25-test suite stays green because `load_results`'s contract is
unchanged.

## Risks / notes

- **DuckDB single-writer concurrency**: fine for a single-user batch workflow;
  the fetcher and site run sequentially, not concurrently.
- **Binary `.duckdb` in git**: accepted; diffs are not human-readable. Data is
  tiny so the file stays small; rely on the DB's `committed` rows + `forecast_ts`
  for prediction history rather than CSV diffs.
- **`group_label` deferred**: left `NULL` until the fetcher can source the group
  draw; `stage` and `round` are populated now.
- **`model_cache.json` is unaffected**: the fitted-model warm-start cache stays
  a JSON file; only match/prediction *data* moves into the DB.

## Build-order context

This is sub-project 1 of 3. Next: (2) auto-fetcher writing results via
`upsert_results`, (3) static HTML site reading `matches` + `v_model_report`.
Each gets its own spec → plan → implementation cycle.
