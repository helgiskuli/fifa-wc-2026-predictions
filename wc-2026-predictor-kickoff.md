# Project: World Cup 2026 scoreline predictor (office pool)

## Objective
Predict an **exact scoreline** for every match in the 2026 World Cup, optimized for our office pool's scoring rule:
- `a` points for a correct **result** (home win / draw / away win)
- `b` **bonus** points for the exact goal count of each team

The pick per fixture is the scoreline that maximizes `a·P(outcome) + b·P(exact score)` — NOT the most likely exact score, and NOT the most likely outcome. Both `a` and `b` are config params (I'll set the real values; build it ratio-agnostic).

## Approach (decided — don't re-litigate)
- **Model:** bivariate Poisson goal model with the **Dixon-Coles low-score correction** (the τ adjustment for 0-0/1-0/0-1/1-1) and **exponential time decay** on match recency.
- **Fit:** plain MLE via `scipy.optimize` (Bayesian/hierarchical is overkill — we're picking one score per game, not simulating a bracket with uncertainty propagation).
- **No xG.** Not reliably available historically; fit on raw goals.
- **Language:** Python.

## Data
Source: `martj42/international_results` (GitHub, CC0). Pull `results.csv`.
Schema: `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`.

Preprocessing:
- Filter to recent competitive history — start with the **last ~3-4 years**, make the window a param.
- **Down-weight friendlies** via the `tournament` column (squads rotate; friendlies are noisy). Make the weight a param.
- Apply **exponential time decay** so recent matches dominate (half-life a param; start ~6-9 months).
- Use the **`neutral` flag for venue**, not a string match between team and country (team names are normalized to *current* identity; country names are *historical*, so they legitimately mismatch).

⚠️ The GitHub snapshot ends at **2024**. The most predictive window (2025-26 qualifiers, Nations League, recent friendlies) is the part doing the real work — check for an updated repo pull or top up the last ~18 months before trusting output.

## Model details
- Per-team **attack** and **defense** strength params, plus a **home advantage** term that only applies when `neutral == False`.
- 2026 is mostly neutral venues, but US/Mexico/Canada get a real home edge — model the venue, not a blanket home dummy.
- Dixon-Coles τ correction on the four low-score cells; decay weights in the log-likelihood.
- Optional: seed/fallback to **Elo** (eloratings.net) for teams with sparse data or no recent head-to-heads.

## Output pipeline
For each fixture:
1. Compute `λ_home, λ_away` from fitted strengths + venue.
2. Build the score matrix `P(i,j)` over the bivariate Poisson (with DC correction), `max_goals` ~6.
3. Run the expected-points optimizer. It collapses to **three candidates** — the modal score within each of H/D/A — pick the best by `a·P(outcome) + b·P(exact)`.

Expected behavior sanity check: picks should look **conservative** — lots of 1-0, 2-1, 1-1, occasional 2-0. If the sheet is full of 3-1s and 3-2s, something's wrong.

## Gotchas to respect
- Scores in the data are **full-time incl. extra time, excl. penalties** — slightly inflates goals on ET knockout matches. Negligible here, but don't "fix" it by accident.
- Don't predict the mean scoreline. Don't predict the modal exact score either. Optimize the weighted objective.

## Build order
1. Loader + preprocessing (filter, friendly down-weight, decay weights) against the exact CSV schema above.
2. Bivariate Poisson + Dixon-Coles MLE fit; return per-team attack/defense + home term.
3. `score_matrix(lam_home, lam_away)` → DC-corrected `P(i,j)`.
4. `best_prediction(P, a, b)` → expected-points-maximizing scoreline.
5. Fixture runner: take the 2026 schedule, emit one predicted score per match as a table/CSV.
6. (Optional) backtest on a held-out recent tournament (e.g. 2022) to sanity-check calibration before committing the sheet.

Start with steps 1-4 and show me per-team strengths + a couple of sample fixture predictions before wiring the full schedule.
