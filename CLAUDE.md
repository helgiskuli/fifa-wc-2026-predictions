# CLAUDE.md

Guidance for AI agents working in this repo. See `README.md` for the
user-facing overview; this file captures conventions and the decisions/
findings that are **not obvious from the code** and should not be redone.

## Environment

- **Use `uv` for everything** — `uv sync`, `uv add <pkg>`, `uv run python -m scripts.<x>`.
  Do **not** use `python -m venv`, raw `pip`, or `pyenv`.
- Python package is `wc2026/`; runnable entry points are `scripts/*` (run as
  modules: `uv run python -m scripts.run_schedule`).

## Architecture (one line each)

- `wc2026/config.py` — all tunables as frozen dataclasses (`PreprocessConfig`,
  `ModelConfig`, `ScoringConfig`). Change knobs here, not in code.
- `wc2026/data.py` — load matches from the DuckDB store (via `db.load_matches`),
  window filter, per-match weight (recency decay × competition-tier), tier
  classifier, fixture extraction.
- `wc2026/db.py` — owns all DuckDB access: `connect`, `init_schema`,
  deterministic `make_match_id`, `load_matches`, prediction upsert/commit
  helpers, `upsert_results`, and the `v_model_report` / `v_site_report`
  scoring views (shared SQL via `_scored_report_select`).
- `wc2026/model.py` — bivariate-Poisson / Dixon-Coles weighted-MLE `fit()`,
  `FittedModel` (rates, score matrix, `save`/`load`, warm-start).
- `wc2026/predict.py` — expected-points objective, `best_prediction`, and
  `score_prediction` (used by backtests; mirrors the objective exactly).

## Data store

- Data lives in `data/wc2026.duckdb` (git-tracked binary, the single source of
  truth), seeded by `scripts/migrate_to_duckdb.py` from the CSVs.
- `results.csv` is the upstream historical seed; `wc-2026-games.csv` was the
  one-time WC seed; `goalscorers.csv` / `shootouts.csv` are bonus seeds. All are
  import sources, **not** hand-edited records.
- `wc2026/db.py` owns all DB access; `load_results()` reads matches from the DB.
- `match_id` is `YYYYMMDD-home-away` (slug). Known limitation: it cannot
  represent two genuinely distinct same-day same-teams matches; the migration
  collapses the lone corpus case (1974 Tahiti vs New Caledonia) into one row.
  Accepted (zero-weight friendly).
- Predictions are stored as `committed` (locked ~90 min pre-kickoff via
  `scripts.commit_picks`, immutable without `--force`), `latest` (overwritten
  each `run_schedule`), and `pregame` (honest eve-of-match picks for played
  games, written by `scripts.backfill_predictions`; see the HTML scoreboard
  section). `v_model_report` joins committed picks to actual results for model
  evaluation; `v_site_report` (committed-or-pregame) backs the scoreboard.

## Auto-fetcher

- `wc2026/providers.py` is the only network-touching module: a `ResultsProvider`
  protocol + normalized `MatchRecord`; `Martj42CsvProvider` (concrete, the
  GitHub CSV feed) and `LiveApiProvider` (stub; reads `WC_RESULTS_API_KEY`,
  concrete API TBD).
- `wc2026/fetch.py` `reconcile()` holds the write policy (final-only,
  overwrite-if-changed, change report); `db` stays policy-free.
  `db.upsert_matches` inserts new matches and updates **only** scores on
  conflict (preserving seeded WC labels/source).
- `scripts/fetch_results.py`: manual command. Default pulls this WC's results
  via martj42; `--corpus-refresh` does the full historical feed; `--dry-run`
  reports without writing (opens the DB read-only).

## HTML scoreboard (sub-project 3)

- `scripts/backfill_predictions.py` reconstructs honest pre-game picks for
  played WC matches: it refits the model as-of the eve of each played date
  (the backtest's leak-free pattern) and writes one EP-optimal pick per match
  under prediction kind **`pregame`**. It warm-starts but **does not** write
  `model_cache.json` (that is run_schedule's as-of-today cache) and never
  touches `latest`/`committed`. Re-run it after each new matchday.
- `db.v_site_report` scores the best available pre-game pick per match
  (**`committed` if one exists, else `pregame`**) against results, reusing the
  `v_model_report` scoring SQL via `_scored_report_select`. `v_model_report`
  itself (committed-only) is unchanged.
- `scripts/build_site.py` is pure read + render: it reads `v_site_report`
  (played) + `latest` (upcoming) and renders `templates/site.html.j2` (Jinja2)
  into a single self-contained `docs/index.html`. Refresh cadence: run
  `backfill_predictions` then `build_site` after a matchday.

## Decided approach — do NOT re-litigate

From the project kickoff (`wc-2026-predictor-kickoff.md`):
bivariate Poisson + Dixon-Coles low-score correction + exponential time
decay; **plain MLE** (no Bayesian/hierarchical — overkill for picking one
score/game); **no xG** (fit raw goals); Python. Venue from the `neutral`
flag, never a team/country string match.

## Validated findings — don't repeat these experiments

Everything below was settled by the 2018 + 2022 backtests (`scripts/backtest.py`).
Re-running an experiment that's already been answered wastes a ~95s fit each.

- **Friendlies are essential — down-weight (0.3), don't drop.** Setting
  `weight_friendly=0` cratered 2022 outcome accuracy 48%→34% (−45 pts/128
  matches). They are the main cross-confederation bridges.
- **Aggressive competition-tier weighting doesn't help.** The proposed
  1/.5/.3/.3 tiering was a wash-to-worse vs flat competitive tiers; qual/NL
  is 51% of the corpus and cutting it loses signal. Defaults keep major /
  qual_nl / other = 1.0, friendly = 0.3.
- **Half-life: keep 240 d. Do NOT shorten.** Sweep showed <240 d clearly
  worse (effective sample is only ~10 decay-weighted matches/team); 240–730 d
  is a noisy plateau. ~75% of effective weight already comes from 2025–26.
- **Two-parameter home advantage is load-bearing.** A single home term
  over-attributed goals to home scoring and under-predicted neutral-venue
  goal totals by ~0.22/match (hits ~all of WC-2026). Split into `home_att`
  (home boost) + `home_def` (away suppression). Don't collapse it.
- **The model's value is favourite-ranking, not exact scores.** Under this
  scoring a naive "1-0 to the favourite" matches/beats the EP-optimal pick.
  Don't oversell the score optimiser; the user knows and chose EP-optimal.

## Gotchas

- **`PreprocessConfig.as_of` defaults to `date.today()`** and is both the
  decay anchor *and* the training cutoff (`date <= as_of`). Backtests must
  override it to the eve of the target tournament, or they leak/exclude data.
- **Warm-start + cache:** `run_schedule` warm-starts `fit()` from
  `model_cache.json` (keyed by team name, robust to pool changes) and
  re-saves. `--cached` skips the fit entirely. Delete the cache to force cold.
- **Scores are FT incl. extra time, excl. penalties.** Shootouts read as
  draws — intended.
- **Fit cost:** cold ~95 s (≈440 params, numerical gradient); warm-started
  re-runs are seconds. If you change the model structure, the warm-start
  globals transform (inverse sigmoid/tanh in `fit`) must stay in sync.
- After changing the model/weights/window, **re-run the relevant backtest**
  before trusting output — that's the project's validation habit.

## Working with the slow fits (process notes)

- A cold fit is ~95 s. **Always warm-start / `--cached`; never cold-fit in a
  loop.** Backtest sweeps should parallelise across processes (see how
  `sweep_halflife.py` is launched) rather than run serially.
- When a fit runs in the background, **rely on the harness's
  background-completion notification** to resume — do not spawn
  `while pgrep …; do :; done` busy-wait loops (they peg a core, steal CPU
  from the fit, and pile up redundant wait-tasks).

## Tests

`uv run python -m pytest -q` — fast (~0.4 s, no model fit). Covers the
scoring rule, the EP objective, the score matrix, the param transforms, and
model save/load. Run it after touching `predict.py` / `model.py`. If you
change the warm-start transforms, the round-trip tests in `test_model.py`
must stay green.

## Reforecasting command

`/reforecast` (`.claude/commands/reforecast.md`) wraps the tournament
workflow: warm re-run of `run_schedule`, then `scripts/diff_predictions.py`
to report only the picks that moved vs the committed sheet.

## Scoring objective (the thing being optimised)

`a`=3 correct outcome, `b`=1 per correct team goal count, `c`=1 correct
signed goal difference; exact = 6. Pick maximises
`a·P(outcome) + b·P(home=i) + b·P(away=j) + c·P(diff=i−j)` over the grid.
