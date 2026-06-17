---
created: 2026-06-17
project: fifa-wc-2026-predictions
status: design-approved
tags: [ci, github-actions, github-pages, hosting, deploy]
---

# Scheduled GitHub Pages deploy — design

Host the HTML scoreboard publicly and keep it fresh automatically: a scheduled
GitHub Actions workflow rebuilds the site from the latest results and deploys it
to GitHub Pages, with an on-demand manual trigger.

## Decisions (settled in brainstorm)

- **Public site, public repo.** The repo is flipped to public so native GitHub
  Pages is free and Actions minutes are unlimited. The repo holds only model
  code and CC0 football data (martj42); nothing sensitive.
- **Stateless runs.** CI commits nothing. Each run rebuilds from the committed
  DuckDB (historical seed) plus a fresh martj42 fetch, and deploys the rendered
  HTML as a Pages artifact. The 8.3 MB `data/wc2026.duckdb` stays the
  manually-maintained source of truth via local `/reforecast`; git history stays
  clean (no daily binary commits).
- **Cadence.** Daily at 07:00 UTC (catches the prior day's late kickoffs), plus
  `workflow_dispatch` for manual runs. Tunable later.

## One-time setup

1. `gh repo edit --visibility public --accept-visibility-change-consequences`.
2. Set the Pages build source to GitHub Actions:
   `gh api -X POST repos/{owner}/{repo}/pages -f build_type=workflow`
   (idempotent-ish: if Pages already exists, use `PUT` to update `build_type`).

## Component: `.github/workflows/scoreboard.yml`

- **Triggers:** `schedule` (`cron: "0 7 * * *"`) and `workflow_dispatch`.
- **Permissions:** `contents: read`, `pages: write`, `id-token: write`.
- **Concurrency:** group `pages`, `cancel-in-progress: false` (don't interrupt an
  in-flight deploy).
- **`build` job** (ubuntu-latest):
  1. `actions/checkout`.
  2. `astral-sh/setup-uv` (with its built-in cache).
  3. `actions/cache` for `model_cache.json` (key with a stable prefix +
     `github.run_id`, `restore-keys` on the prefix) so fits warm-start across
     runs. First-ever run is a ~95 s cold fit; subsequent runs are seconds.
  4. `uv sync`.
  5. `uv run python -m scripts.fetch_results` — pull latest finished results
     (martj42, public CSV, no secret).
  6. `uv run python -m scripts.run_schedule` — refresh `latest` picks for
     upcoming fixtures.
  7. `uv run python -m scripts.backfill_predictions` — honest `pregame` picks for
     newly played matches.
  8. `uv run python -m scripts.build_site` — render `docs/index.html`.
  9. `actions/upload-pages-artifact` with `path: docs`.
- **`deploy` job:** `needs: build`, `environment: github-pages`,
  `actions/deploy-pages`.

## Failure behavior

Any failing step fails the run; the last successful deploy stays live. No
partial deploys (artifact upload + deploy are the final steps).

## Out of scope (YAGNI)

- Committing results/DB back from CI (explicitly rejected — bloats history).
- Multiple/region-specific cron schedules; private hosting / auth; the
  not-yet-implemented live results provider.

## Verification

After setup: trigger the workflow manually (`gh workflow run`), confirm both jobs
go green, and load the published Pages URL to see the scoreboard.
