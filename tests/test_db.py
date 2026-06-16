import pandas as pd
import pytest

from wc2026 import db


def test_make_match_id_is_deterministic_slug():
    a = db.make_match_id("2026-06-16", "France", "Senegal")
    b = db.make_match_id(pd.Timestamp("2026-06-16"), "France", "Senegal")
    assert a == b == "20260616-france-senegal"


def test_make_match_id_slugifies_spaces_and_case():
    mid = db.make_match_id("2026-06-24", "Bosnia and Herzegovina", "Qatar")
    assert mid == "20260624-bosnia-and-herzegovina-qatar"


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.init_schema(c)
    yield c
    c.close()


def test_init_schema_creates_tables(con):
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"matches", "predictions", "goalscorers", "shootouts"} <= names


def test_init_schema_is_idempotent(con):
    db.init_schema(con)  # second call must not raise
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "matches" in names


from wc2026 import ScoringConfig, score_prediction


def _seed_match(con, mid, h, a, hs, as_):
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, "
        "home_score, away_score, source) VALUES (?, DATE '2026-06-16', ?, ?, ?, ?, 'wc2026')",
        [mid, h, a, hs, as_],
    )


def _seed_committed(con, mid, ph, pa):
    con.execute(
        "INSERT INTO predictions (match_id, kind, pred_home_goals, "
        "pred_away_goals, outcome) VALUES (?, 'committed', ?, ?, 'H')",
        [mid, ph, pa],
    )


def test_report_view_points_match_score_prediction(con):
    cfg = ScoringConfig()
    cases = [("m1", 2, 1, 2, 1), ("m2", 1, 0, 0, 2), ("m3", 1, 1, 2, 0)]
    for mid, ph, pa, ah, ay in cases:
        _seed_match(con, mid, "H", "A", ah, ay)
        _seed_committed(con, mid, ph, pa)
    rows = con.execute(
        "SELECT match_id, points FROM v_model_report ORDER BY match_id"
    ).fetchall()
    got = {mid: pts for mid, pts in rows}
    for mid, ph, pa, ah, ay in cases:
        assert got[mid] == score_prediction(ph, pa, ah, ay, cfg)


def test_load_matches_returns_loader_contract(con):
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, home_score, "
        "away_score, tournament, neutral, city, country, source) VALUES "
        "('m1', DATE '2022-06-01', 'Spain', 'Italy', 2, 1, 'Friendly', TRUE, 'X', 'Y', 'upstream'),"
        "('m2', DATE '2026-06-16', 'France', 'Senegal', NULL, NULL, 'FIFA World Cup', TRUE, 'Z', 'W', 'wc2026')"
    )
    df = db.load_matches(con)
    assert list(df.columns) == [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral",
    ]
    assert str(df["date"].dtype).startswith("datetime64")
    assert df["neutral"].dtype == bool
    # unplayed match has NaN score (mirrors the CSV na_values behaviour)
    assert df.set_index("home_team").loc["France", "home_score"] != \
        df.set_index("home_team").loc["France", "home_score"]  # NaN != NaN
