---
created: 2026-06-17
project: fifa-wc-2026-predictions
status: brainstorm-in-progress
tags: [html, site, handoff]
---

# HTML site (sub-project 3) — brainstorm handoff

Temporary handoff note so the brainstorm can resume on another machine (visual
work needs a browser; this was started on a headless VM). Delete this file once
the real design spec (`2026-06-17-html-site-design.md`) is written.

## Where we are

- Sub-project 1 (DuckDB data store) and 2 (auto-fetcher) are **merged to main**.
- Starting the brainstorm for **sub-project 3: a static HTML site** reading
  `matches` + `v_model_report` (predicted vs actual, "where was the model
  right?", upcoming picks).
- No design questions answered yet. The brainstorm (purpose/scope/layout) still
  needs to happen, ideally with the visual companion on the laptop.

## To resume on the laptop

1. `git pull` (everything below is pushed to origin/main).
2. Start Claude Code in the repo and say: *"continue the HTML site brainstorm,
   see docs/superpowers/specs/2026-06-17-html-site-BRAINSTORM-HANDOFF.md"*.
3. Accept the visual companion offer for layout mockups.

## Data findings that shape the design (already explored)

- **`v_model_report` is currently empty (0 rows).** It joins **committed** picks
  to actual results, and there are **0 committed predictions** (the 90-min
  `commit_picks` lock has never been run on real data). So "predicted vs actual"
  has no committed source yet. The site must decide what to show:
  committed (honest pre-game, none yet) vs `latest` (current forecast) vs a
  pre-kickoff snapshot. **This is the key open design question.**
- Predictions table: **56 `latest`** rows (includes stale rows for matches that
  have since been played), **0 `committed`**.
- **20 WC matches played, 52 fixtures remain** (as of 2026-06-17).
- `matches` columns include `stage` / `round` / `group_label` (currently NULL,
  deferred) / `source`. `v_model_report` columns: match_id, date, home_team,
  away_team, actual_h, actual_a, pred_h, pred_a, outcome_ok, side_goals, gd_ok,
  exact_ok, points.
- **No web dependencies** yet (no jinja/flask/fastapi). Static generation is
  open: hand-rolled templates, Jinja2 (add dep), or a generator.

## Open brainstorm questions (queued, ask one at a time)

1. **Output form & hosting:** single self-contained HTML file vs multi-page
   static site; served from `docs/` (GitHub Pages) vs opened as a local file.
2. **Pages/content:** results-so-far (predicted vs actual + points), upcoming
   fixtures with picks, a model scorecard/accuracy summary, group standings?
3. **Predicted-vs-actual source** (the gap above): start committing picks going
   forward, show `latest` as a proxy, or both.
4. **Styling:** hand-rolled CSS vs a small framework; any charts.
5. **Build trigger:** a manual `scripts/build_site.py` (consistent with the
   project's manual-command pattern) and regen cadence.

## Recommendation seeds (not decided)

- A single-command static generator `scripts/build_site.py` reading the DB and
  emitting to `docs/` (so GitHub Pages can serve it), Jinja2 templates, plain
  CSS. Matches the project's manual-command, DB-as-source-of-truth conventions.
