---
description: Refit the model on the latest results and report which picks changed
argument-hint: (optional notes, e.g. "after round 2")
allowed-tools: Bash(uv run python -m scripts.run_schedule), Bash(uv run python -m scripts.diff_predictions), Bash(git status:*), Bash(git diff:*)
---

Reforecast the World Cup sheet with the latest results. Assume the new scores
are already in the DuckDB store (`data/wc2026.duckdb`, via `db.upsert_results`
or a re-seed from refreshed CSVs).

Do this:

0. (Optional) Run `uv run python -m scripts.fetch_results` to pull the latest
   finished results into the DB before refitting. It reports any score changes;
   run with `--dry-run` first to preview. Skip if results are already current.
1. Run `uv run python -m scripts.run_schedule`. It warm-starts from the
   cached fit, so this should take seconds, not ~95s. Note the printed
   `Forecasting as-of <date>` line and confirm the fit converged
   (`success=True`).
2. Run `uv run python -m scripts.diff_predictions` to see which fixtures'
   picks moved versus the last committed `predictions.csv`.
3. Summarize concisely: the as-of date, how many of the N picks changed,
   and call out any *notable* swings (a flipped result H/D/A, or a big
   favourite's margin moving) — not every 1-0 ↔ 2-0 nudge.
4. Sanity-check the pick distribution is still conservative (mostly
   1-0 / 0-1 / 2-1 / draws; no flood of 3-x). Flag it if it isn't.
5. Remind the user to commit `predictions.csv` if it changed, and offer to
   do it. Do **not** commit without their go-ahead.

If the run reports skipped fixtures (teams not in the fitted pool), surface
them — that usually means a knockout fixture was added with a placeholder
team name that needs the real name.

$ARGUMENTS
