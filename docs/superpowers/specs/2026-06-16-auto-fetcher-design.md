---
created: 2026-06-16
project: fifa-wc-2026-predictions
status: draft
tags: [data, fetcher, providers, duckdb, design]
---

# Auto-fetcher: pull match results into the DuckDB store

## Context

The data store (sub-project 1) made `data/wc2026.duckdb` the single source of
truth and exposed write hooks (`db.upsert_results`, and the `matches` table with
deferred `group_label`). Results are still filled by hand-editing or re-seeding
from CSVs. This spec covers **sub-project 2: the auto-fetcher**, which pulls
match results from online sources and writes them to the DB so the predictor
stays current during the tournament without manual data entry.

Out of scope (sub-project 3, separate spec): the static HTML site that renders
`matches` + `v_model_report`.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Sources | **Tiered, behind one interface** | A live API for fresh WC scores; the martj42 CSV feed for the historical-corpus refresh. Both normalize to the same record type so the rest of the pipeline is source-agnostic. |
| Provider abstraction | **`ResultsProvider` Protocol** | The concrete live API is chosen later; a small pluggable interface lets it drop in without reworking orchestration or DB writes. |
| Build scope (this cycle) | **martj42 provider concrete; live provider stubbed** | martj42's feed contains only *played* matches, so it refreshes the corpus and backfills WC scores (lagged). The live provider (fresh scores + unplayed knockout fixtures) is defined as an interface + stub; orchestration and DB writes are built now so only the provider remains. |
| Write scope | **Scores + new fixtures** | Fill final scores AND insert newly-scheduled fixtures (knockout bracket with stage/round/group_label) so `run_schedule` predicts them automatically. Goalscorers/shootouts deferred (no model consumers today). |
| Trigger | **Manual command** | `uv run python -m scripts.fetch_results`, composable into `/reforecast`. The user controls when network calls happen; matches the existing manual flow. |
| Write policy | **Final-only, overwrite-if-changed, report** | Write only `FINISHED` matches; overwrite a present score only when it differs and report the change (no silent clobber); insert new fixtures; update fixture metadata if it shifted. |
| HTTP client | **httpx** | Single modern client for both providers (sync API is sufficient for a manual batch command). |

## Architecture

```
provider.fetch(window) -> list[MatchRecord]  ->  reconcile()   ->  db.upsert_matches()
        (CSV or API)         (normalized)        (policy+diff)       (scores + fixtures)
```

Three new units, each independently testable:

**`wc2026/providers.py`** (source adapters; the only modules touching the network)
- `MatchRecord` (frozen dataclass): normalized match in the DB's own naming
  convention. Fields: `date`, `home_team`, `away_team`, `home_score`,
  `away_score`, `status` (`'FINISHED' | 'SCHEDULED' | 'IN_PLAY' | ...`),
  `tournament`, `neutral`, `city`, `country`, `stage`, `round`, `group_label`.
  Score/venue/label fields are optional (a scheduled fixture has no score).
- `ResultsProvider` (typing.Protocol): `name: str`; `fetch() -> list[MatchRecord]`.
- `Martj42CsvProvider`: downloads the three raw GitHub CSVs
  (`results.csv`, `goalscorers.csv`, `shootouts.csv`) and yields
  `MatchRecord`s. Names already match the DB (the DB was seeded from this
  feed), so no name mapping. Played matches only (`status='FINISHED'`).
- `LiveApiProvider`: `fetch()` raises `NotImplementedError` with the documented
  contract (reads `WC_RESULTS_API_KEY` from the env, maps provider team names to
  the DB convention via a name-map, marks finished vs scheduled). Filled in when
  the concrete API is chosen.

**`wc2026/fetch.py`** (provider-agnostic orchestration; no network, no SQL beyond
calling `db`)
- `reconcile(con, records) -> ChangeReport`: applies the write policy and calls
  `db.upsert_matches`. `ChangeReport` (frozen dataclass) lists inserted
  fixtures, score changes (`match_id`, old -> new), and skipped non-final rows.

**`scripts/fetch_results.py`** (CLI)
- `--source {martj42,live}` (default `martj42`), `--corpus-refresh` (also
  refresh the full historical corpus, not just WC rows; default off),
  `--dry-run` (report without writing). Prints the `ChangeReport`. Designed to
  run before `run_schedule` in the `/reforecast` workflow.

## DB changes (`wc2026/db.py`)

Add one helper:

```python
def upsert_matches(con, rows) -> None
```

- Pure write, no policy: inserts rows absent by `match_id` (derived via
  `make_match_id`), carrying `date, home_team, away_team, tournament, neutral,
  city, country, stage, round, group_label, source` and any score; updates the
  carried columns for rows already present.
- Sets `source` appropriately (`'upstream'` for martj42, a provider tag for the
  live API). `group_label` (left NULL by the data-store migration) is populated
  here when a provider supplies it.

The write *policy* and the change report live in `fetch.reconcile`, not here:
`reconcile` reads current state, computes the diff (what to insert, which scores
actually changed, what to skip), then hands the approved rows to
`upsert_matches`. `db` stays policy-free, consistent with the data-store
module's existing split.

`upsert_results` stays as the minimal score-only hook (still valid; the fetcher
uses the richer `upsert_matches`).

## Write policy (in `reconcile`)

1. Drop records whose `status` is not `FINISHED` for *score* writes; a scheduled
   future fixture still inserts as an unplayed row (NULL score) so it appears in
   `upcoming_fixtures`.
2. For a match already scored: overwrite only if the incoming score differs;
   record the change in the report. Never silently clobber.
3. Insert new fixtures (by `match_id`); update fixture metadata if changed.
4. Scores are FT incl. extra time, excl. penalties; a shootout reads as a draw
   (existing project convention, unchanged).
5. On any network or parse error: fail loudly, write nothing (no partial state).

## Secrets & dependencies

- The live provider reads its key from `WC_RESULTS_API_KEY` (env), never
  hard-coded; placeholders only in code/docs. The martj42 path needs no key.
- New dependency: `httpx` (added via `uv add httpx`).

## Testing

In-memory DuckDB; no live network.
- `Martj42CsvProvider` parses sample CSV bytes into `MatchRecord`s (fixture
  bytes, not a live download).
- `reconcile` against `:memory:` with synthetic `MatchRecord`s: final-only
  filtering, overwrite-if-changed reporting, new-fixture insert, unplayed
  fixture insert, idempotent re-run (second run reports no changes).
- `upsert_matches` insert-vs-update paths.
- `LiveApiProvider.fetch()` raises `NotImplementedError` (until the API lands;
  then parsing gets tests against a captured sample response).
- The existing suite stays green (the fetcher only adds write paths).

## Risks / notes

- **martj42 lag**: the CSV feed updates hours-to-days after kickoff; it is the
  backfill/corpus path, not the live path. Fresh WC scores wait on the live
  provider.
- **Live provider name mapping**: provider team names will not match the DB
  slugs; a name-map at the provider boundary is required when the API is chosen,
  with a loud error on any unmapped team (never a silent miss).
- **No partial writes**: a failed fetch must leave the DB untouched.
- **Interaction with committed picks**: filling a WC result flips that match
  from fixture to training data and makes it scoreable in `v_model_report`. The
  fetcher does not touch `predictions`; `commit_picks` remains the only writer of
  committed picks. Workflow ordering (commit picks pre-kickoff, fetch results
  post-FT) is a usage convention, not enforced here.

## Build-order context

Sub-project 2 of 3. Depends on sub-project 1 (data store, merged). Next: (3) the
static HTML site reading `matches` + `v_model_report`. Each gets its own
spec -> plan -> implementation cycle.
