---
created: 2026-06-17
project: fifa-wc-2026-predictions
status: design-approved
tags: [html, site, scoreboard, design]
---

# HTML site (sub-project 3) — design

A single-page, self-contained HTML **personal scoreboard** that reads the DuckDB
store and answers "how is my model doing this World Cup?": predicted vs actual
for played matches, a headline accuracy summary, and the current picks for
upcoming fixtures.

## Purpose & scope

- **Audience:** just the user. Honest, low-polish, single viewer.
- **Centerpiece:** predicted-vs-actual for played matches, with points earned.
- **In scope:** model scorecard summary, results-so-far table, upcoming
  fixtures + current picks.
- **Out of scope (YAGNI):** group standings (no `group_label` data — column is
  100% NULL), calibration / accuracy-over-time, multi-page site, live
  auto-refresh, any JS or charting.

## Key decision: honest pre-game predictions

`v_model_report` is empty because it joins **committed** picks to results, and
there are 0 committed picks (the 90-min `commit_picks` lock has never run on
real data). Stored `latest` predictions cover only 4 of the 20 played matches.

**Resolution:** reconstruct an honest pre-game prediction for every played match
by refitting the model with `as_of` set to the eve of each match — the same
leak-free pattern the 2018/2022 backtests use. These are stored under a **new
prediction kind `pregame`**, distinct from `committed` (real 90-min lock) and
`latest` (current rolling forecast). `v_model_report` is left untouched.

## Components

### 1. `scripts/backfill_predictions.py` (new)

Reconstructs honest pre-game picks for played matches.

- Query the distinct dates of played WC matches (currently 6: 2026-06-11 …
  2026-06-16).
- For each date `d`, fit the model with `PreprocessConfig.as_of = d - 1 day`
  (the eve), **warm-started** from `model_cache.json` so only the first fit is
  cold (~95 s) and the rest are seconds. Re-save the cache as `run_schedule`
  does.
- For every played match on `d`, compute `best_prediction` and upsert the
  result as kind **`pregame`** (reusing the existing prediction
  upsert/commit helpers in `db.py`).
- **Idempotent:** re-running overwrites the `pregame` rows for those matches.
  Safe to run again after each new matchday.
- Pure local compute; no network.

### 2. `v_site_report` view in `wc2026/db.py` (new)

Generalize the existing `v_model_report` scoring SQL into a CTE keyed by
prediction kind, then expose a site-facing view that, per match, selects the
best available pre-game prediction — **`committed` if one exists, else
`pregame`** — joins it to actual results, and emits the same columns as
`v_model_report`: `match_id, date, home_team, away_team, actual_h, actual_a,
pred_h, pred_a, outcome_ok, side_goals, gd_ok, exact_ok, points`.

Scoring is unchanged from the project objective: `a`=3 correct outcome, `b`=1
per correct team goal count, `c`=1 correct signed goal difference, exact = 6.

`v_model_report` itself is **not modified**.

### 3. Upcoming picks

Come straight from the existing `latest` predictions joined to unplayed
fixtures — no refit. `run_schedule` already maintains these. Each row exposes
the predicted scoreline and confidence (`p_result`, i.e. P(outcome)).

### 4. `scripts/build_site.py` (new)

The manual build command: `uv run python -m scripts.build_site`.

- Opens the DB **read-only**.
- Reads `v_site_report` (played matches, scored), `latest`-joined-fixtures
  (upcoming), and aggregates the scorecard metrics from `v_site_report`:
  total points, matches scored, outcome-accuracy %, exact-score %, average
  points/match.
- Renders `templates/site.html.j2` via Jinja2 into a single self-contained
  `docs/index.html` with inline `<style>` (no external assets — double-clickable
  locally and GitHub-Pages-ready).
- Prints a one-line summary (matches scored, total points, output path).
- Does **not** fit or fetch; pure read + render.

### 5. `templates/site.html.j2` (new) — Layout B

- Header: title + "last built" timestamp.
- Row of 4 metric cards: points / outcome % / exact % / pts-per-match
  (dark-green accent).
- Two columns:
  - **Results so far** — date, teams, predicted score, actual score, points;
    correct outcomes visually highlighted.
  - **Upcoming** — date, teams, pick (predicted scoreline), confidence
    (P(outcome)).
- One CSS media query collapses to a single column under ~700 px.
- Hand-rolled CSS, no framework, no JS, no charts.

## Data flow

```
backfill_predictions  (refit per eve-date  ->  kind 'pregame')
        |
        v
v_site_report  (pregame/committed joined to results, scored)  +  latest (upcoming)
        |
        v
build_site.py  ->  Jinja2 (templates/site.html.j2)  ->  docs/index.html
```

## Dependencies

- `uv add jinja2`.

## Refresh cadence

After a matchday: run `backfill_predictions` (picks up the newly played
matches), then `build_site`. Both are manual, matching the project's
manual-command convention.

## Testing (`tests/`, mirrors src, no model fit)

- `v_site_report` scoring matches `v_model_report`'s logic on a seeded fixture
  (a couple of `pregame` rows + results → known points).
- `build_site` against a tiny temp DB produces an `index.html` containing the
  expected rows and metric values (assert on substrings, not pixel layout).
- `backfill_predictions`' date-grouping + `as_of`-selection logic covered by a
  thin unit test; the actual model fit is **not** run in tests (too slow),
  consistent with the existing suite.

## Conventions

- Run entry points as modules: `uv run python -m scripts.build_site`.
- `db.py` owns all DB access; the site scripts go through it.
- Output committed to `docs/index.html` (git-tracked, like the rest of the
  repo's artifacts).
