import pytest

from wc2026 import db
from scripts import build_site


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.init_schema(c)
    # one played match with a pregame pick (1-0 vs actual 3-1 -> 3 pts: outcome only)
    c.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, home_score, "
        "away_score, tournament, neutral, source) VALUES "
        "('m1', DATE '2026-06-11', 'France', 'Senegal', 3, 1, 'FIFA World Cup', TRUE, 'wc2026')"
    )
    c.execute(
        "INSERT INTO predictions (match_id, kind, pred_home_goals, pred_away_goals, "
        "outcome, p_result) VALUES ('m1', 'pregame', 1, 0, 'H', 0.5)"
    )
    # one upcoming fixture with a latest pick
    c.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, home_score, "
        "away_score, tournament, neutral, source) VALUES "
        "('m2', DATE '2026-06-20', 'England', 'Croatia', NULL, NULL, 'FIFA World Cup', TRUE, 'wc2026')"
    )
    c.execute(
        "INSERT INTO predictions (match_id, kind, pred_home_goals, pred_away_goals, "
        "outcome, p_result) VALUES ('m2', 'latest', 1, 0, 'H', 0.58)"
    )
    yield c
    c.close()


def test_scorecard_aggregates_v_site_report(con):
    card = build_site.scorecard(con)
    assert card["matches"] == 1
    assert card["points"] == 3
    assert card["outcome_pct"] == 100.0
    assert card["exact_pct"] == 0.0
    assert card["pts_per_match"] == 3.0


def test_results_and_upcoming_rows(con):
    results = build_site.results_rows(con)
    assert len(results) == 1 and results[0]["home_team"] == "France"
    assert results[0]["actual_h"] == 3 and results[0]["pred_h"] == 1
    upcoming = build_site.upcoming_rows(con)
    assert len(upcoming) == 1 and upcoming[0]["away_team"] == "Croatia"


def test_render_produces_self_contained_html(con):
    html = build_site.render(
        build_site.scorecard(con), build_site.results_rows(con),
        build_site.upcoming_rows(con), "2026-06-17 10:00",
    )
    assert "<html" in html.lower() and "<style" in html.lower()
    assert "France" in html and "Senegal" in html
    assert "England" in html and "Croatia" in html
    assert "2026-06-17 10:00" in html
