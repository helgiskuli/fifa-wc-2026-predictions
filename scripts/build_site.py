"""Render the personal scoreboard to a single self-contained docs/index.html.

Pure read + render: reads v_site_report (played, scored) and the latest picks
for upcoming fixtures, then renders templates/site.html.j2 with inline CSS.
Run scripts.backfill_predictions first so played matches have pregame picks.

    uv run python -m scripts.build_site
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from wc2026 import db

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT / "templates"
OUT_PATH = ROOT / "docs" / "index.html"


def scorecard(con) -> dict:
    """Headline metrics aggregated from v_site_report."""
    n, pts, outcome_acc, exact_rate = con.execute("""
        SELECT count(*),
               coalesce(sum(points), 0),
               coalesce(avg(outcome_ok), 0),
               coalesce(avg(exact_ok), 0)
        FROM v_site_report
    """).fetchone()
    return {
        "matches": int(n),
        "points": int(pts),
        "outcome_pct": round(100 * outcome_acc, 1),
        "exact_pct": round(100 * exact_rate, 1),
        "pts_per_match": round(pts / n, 2) if n else 0.0,
    }


def results_rows(con) -> list[dict]:
    """Played, scored matches (newest scoring view), with a display date."""
    return con.execute("""
        SELECT strftime(date, '%b %d') AS day, home_team, away_team,
               actual_h, actual_a, pred_h, pred_a, outcome_ok, exact_ok, points
        FROM v_site_report
        ORDER BY date, home_team
    """).df().to_dict("records")


def upcoming_rows(con) -> list[dict]:
    """Unplayed WC-2026 fixtures with their current latest pick + confidence."""
    rows = con.execute("""
        SELECT strftime(m.date, '%b %d') AS day, m.home_team, m.away_team,
               p.pred_home_goals AS pred_h, p.pred_away_goals AS pred_a,
               p.p_result
        FROM matches m
        JOIN predictions p ON p.match_id = m.match_id AND p.kind = 'latest'
        WHERE m.home_score IS NULL AND m.tournament = 'FIFA World Cup'
          AND year(m.date) = 2026
        ORDER BY m.date, m.home_team
    """).df().to_dict("records")
    for r in rows:
        r["conf_pct"] = int(round(100 * (r["p_result"] or 0)))
    return rows


def render(card: dict, results: list[dict], upcoming: list[dict],
           built_at: str) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    return env.get_template("site.html.j2").render(
        card=card, results=results, upcoming=upcoming, built_at=built_at)


def main() -> None:
    con = db.connect(db.DB_PATH, read_only=True)
    try:
        card = scorecard(con)
        results = results_rows(con)
        upcoming = upcoming_rows(con)
    finally:
        con.close()

    built_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    html = render(card, results, upcoming, built_at)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html)
    print(f"wrote {OUT_PATH}  ({card['matches']} scored, {card['points']} pts, "
          f"{len(upcoming)} upcoming)")


if __name__ == "__main__":
    main()
