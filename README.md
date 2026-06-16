# World Cup 2026 scoreline predictor

Predicts an **exact scoreline** for every match of the 2026 World Cup,
optimised for an office-pool scoring rule. It is not trying to guess the
most likely score — it maximises *expected pool points*.

## Scoring rule

Points are additive per fixture:

| Component | Points | Condition |
|---|---|---|
| Correct outcome | `a` = 3 | home-win / draw / away-win matches |
| Each team's goal count | `b` = 1 each | predicted home goals = actual, and/or away goals = actual |
| Correct goal difference | `c` = 1 | signed (home − away) matches (implies correct outcome) |

An exact scoreline scores `a + 2b + c = 6`. All three weights live in
`ScoringConfig` and the model is ratio-agnostic — change them and re-run.

## The model

A **bivariate-Poisson goal model with the Dixon-Coles low-score
correction**, fit by weighted maximum likelihood (`scipy.optimize`):

```
mu_home = exp(intercept + attack[home] − defense[away] + home_att·[venue≠neutral])
mu_away = exp(intercept + attack[away] − defense[home] − home_def·[venue≠neutral])
```

- Per-team **attack** / **defense** strengths (sum-to-zero for identifiability).
- **Two-parameter home advantage** — a home scoring boost *and* an away
  suppression. The split keeps neutral-venue goal totals unbiased, which
  matters because ~all of WC-2026 is at neutral venues.
- Joint law is a bivariate Poisson (shared component `lambda3`) with the
  Dixon-Coles `tau` adjustment (`rho`) on the {0-0, 1-0, 0-1, 1-1} cells.
- Training matches are weighted by **exponential recency decay** (half-life
  240 d) × a **competition-tier weight** (friendlies down-weighted to 0.3).
- Window: last 4 years; venue comes from the `neutral` flag, never a
  team/country string match.

### Picking a scoreline

For each fixture we build the DC-corrected score matrix `P(i, j)` and pick
the `(i, j)` maximising the expected-points objective:

```
EP(i, j) = a·P(outcome) + b·P(home goals = i) + b·P(away goals = j) + c·P(diff = i−j)
```

evaluated over the whole grid. Picks come out conservative (lots of 1-0,
2-1, the occasional 2-0 / draw), as a goals model should.

## Data

Source: [`martj42/international_results`](https://github.com/martj42/international_results)
(CC0). Drop `results.csv` (and optionally `goalscorers.csv`, `shootouts.csv`)
into `data/`. Schema:
`date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`.

> Scores are full-time **incl. extra time, excl. penalties**. A penalty
> shootout shows as a draw; that's intended, don't "fix" it.

### Data store

The data lives in a single git-tracked DuckDB database, `data/wc2026.duckdb`,
which is the source of truth (matches, predictions, goalscorers, shootouts).
The CSVs above are **import sources**, not hand-edited records. Seed or rebuild
the database from them with:

```bash
uv run python -m scripts.migrate_to_duckdb   # CSVs -> data/wc2026.duckdb
```

`load_results()` reads matches from the DB; `NA`-score rows are the fixture
list. `run_schedule` writes each forecast as the `latest` prediction per match.
On matchday, lock in the honestly-scored pre-game picks (immutable thereafter):

```bash
uv run python -m scripts.commit_picks 2026-06-16   # snapshot latest -> committed
```

The `v_model_report` view joins committed picks to actual results for "where was
the model right?" analysis. `wc2026/db.py` owns all database access.

## Install & run

Uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                                   # create env from uv.lock
uv run python -m scripts.run_schedule     # fit + write predictions.csv
```

`predictions.csv` has one row per remaining fixture: `date, home, away,
venue, pick, lam_h, lam_a, result, P_result, P_home_g, P_away_g, P_gd, EP`.

### Reforecasting mid-tournament

1. Fill in the latest results in `data/wc2026.duckdb` (via `db.upsert_results`,
   or re-seed from refreshed CSVs with `scripts.migrate_to_duckdb`).
2. `uv run python -m scripts.run_schedule`

`as_of` defaults to **today**, so new results land inside the window and
dominate via the recency decay — the model sharpens as the tournament goes
on. The fit is **warm-started** from the cached previous fit
(`model_cache.json`), so re-runs take seconds, not ~95 s.

```bash
uv run python -m scripts.run_schedule --cached   # re-emit CSV, no refit (~0.5s)
rm model_cache.json                              # force a clean cold fit
```

When the knockout bracket is set, add those fixtures as `NA`-score rows
(`date, home_team, away_team, tournament=FIFA World Cup, neutral`) and they
are predicted automatically.

## Other scripts

| Script | Purpose |
|---|---|
| `scripts/demo.py` | Fit + print per-team strengths and sample predictions |
| `scripts/backtest.py [2018\|2022]` | Refit as-of a past WC eve and score all 64 matches |
| `scripts/diagnose_goals.py` | Goal-volume / venue calibration diagnostics |
| `scripts/compare_weights.py` | Backtest competition-tier weighting schemes |
| `scripts/sweep_halflife.py <days>` | Backtest a single decay half-life |

## What the backtests say (2018 + 2022, 128 held-out matches)

- The model is a well-calibrated **favourite-ranker** (~50% exact-outcome,
  reliable win probabilities) with unbiased goal volumes after the
  home-advantage fix.
- Under this scoring, the exact-score optimisation is **not** a points edge:
  a trivial "1-0 to the model's favourite" matches or beats the EP-optimal
  pick. Most points live in ranking the favourite, which the model does well.
- Friendlies are **essential** (cross-confederation bridges) — down-weight,
  don't drop. Aggressive competition-tier weighting and shorter half-lives
  were tested and did **not** help.

## Layout

```
wc2026/
  config.py    # all tunables (window, decay, tier weights, scoring a/b/c)
  data.py      # load matches from the DB, window filter, recency × tier weights
  db.py        # all DuckDB access: schema, match_id, read/write, scoring view
  model.py     # bivariate-Poisson / Dixon-Coles fit, score matrix, save/load
  predict.py   # expected-points objective + scoreline pick, point scoring
scripts/       # demo, run_schedule, commit_picks, migrate_to_duckdb, backtest, ...
data/          # wc2026.duckdb (source of truth) + seed CSVs
predictions.csv
```
