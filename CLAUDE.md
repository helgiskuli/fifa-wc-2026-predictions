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
- `wc2026/data.py` — load `results.csv`, window filter, per-match weight
  (recency decay × competition-tier), tier classifier, fixture extraction.
- `wc2026/model.py` — bivariate-Poisson / Dixon-Coles weighted-MLE `fit()`,
  `FittedModel` (rates, score matrix, `save`/`load`, warm-start).
- `wc2026/predict.py` — expected-points objective, `best_prediction`, and
  `score_prediction` (used by backtests; mirrors the objective exactly).

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

## Scoring objective (the thing being optimised)

`a`=3 correct outcome, `b`=1 per correct team goal count, `c`=1 correct
signed goal difference; exact = 6. Pick maximises
`a·P(outcome) + b·P(home=i) + b·P(away=j) + c·P(diff=i−j)` over the grid.
