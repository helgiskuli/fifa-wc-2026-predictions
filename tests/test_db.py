import pandas as pd
import pytest

from scripts import commit_picks
from scripts import migrate_to_duckdb as mig
from scripts import run_schedule as rs
from wc2026 import ScoringConfig, db, score_prediction
from wc2026 import data as wcdata


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


def _pred_df(mid, ph, pa, outcome="H"):
    return pd.DataFrame([{
        "match_id": mid, "pred_home_goals": ph, "pred_away_goals": pa,
        "outcome": outcome, "lam_h": 1.5, "lam_a": 0.8, "p_result": 0.5,
        "p_home_g": 0.3, "p_away_g": 0.4, "p_gd": 0.2, "ep": 2.5,
    }])


def test_upsert_latest_overwrites_in_place(con):
    db.upsert_latest_predictions(con, _pred_df("m1", 1, 0), "2026-06-10")
    db.upsert_latest_predictions(con, _pred_df("m1", 2, 1), "2026-06-15")
    rows = con.execute(
        "SELECT pred_home_goals, pred_away_goals FROM predictions "
        "WHERE match_id='m1' AND kind='latest'"
    ).fetchall()
    assert rows == [(2, 1)]  # single latest row, overwritten


def test_commit_snapshots_latest(con):
    db.upsert_latest_predictions(con, _pred_df("m1", 2, 1), "2026-06-15")
    n = db.commit_predictions(con, ["m1"], now="2026-06-16 12:00:00")
    assert n == 1
    row = con.execute(
        "SELECT pred_home_goals, pred_away_goals FROM predictions "
        "WHERE match_id='m1' AND kind='committed'"
    ).fetchone()
    assert row == (2, 1)


def test_commit_is_immutable_without_force(con):
    db.upsert_latest_predictions(con, _pred_df("m1", 2, 1), "2026-06-15")
    db.commit_predictions(con, ["m1"], now="2026-06-16 12:00:00")
    # a later re-forecast changes latest, then a second commit attempt
    db.upsert_latest_predictions(con, _pred_df("m1", 0, 0), "2026-06-16")
    n = db.commit_predictions(con, ["m1"], now="2026-06-16 13:00:00")
    assert n == 0  # refused
    row = con.execute(
        "SELECT pred_home_goals, pred_away_goals FROM predictions "
        "WHERE match_id='m1' AND kind='committed'"
    ).fetchone()
    assert row == (2, 1)  # original committed pick preserved


def test_commit_force_overwrites(con):
    db.upsert_latest_predictions(con, _pred_df("m1", 2, 1), "2026-06-15")
    db.commit_predictions(con, ["m1"], now="2026-06-16 12:00:00")
    db.upsert_latest_predictions(con, _pred_df("m1", 0, 0), "2026-06-16")
    n = db.commit_predictions(con, ["m1"], force=True, now="2026-06-16 13:00:00")
    assert n == 1
    row = con.execute(
        "SELECT pred_home_goals, pred_away_goals FROM predictions "
        "WHERE match_id='m1' AND kind='committed'"
    ).fetchone()
    assert row == (0, 0)


def test_upsert_results_fills_scores(con):
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, source) "
        "VALUES ('m1', DATE '2026-06-16', 'France', 'Senegal', 'wc2026')"
    )
    db.upsert_results(con, [{"match_id": "m1", "home_score": 2, "away_score": 1}])
    row = con.execute(
        "SELECT home_score, away_score FROM matches WHERE match_id='m1'"
    ).fetchone()
    assert row == (2, 1)


def test_assign_match_ids():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-16", "2026-06-16"]),
        "home_team": ["France", "Iraq"], "away_team": ["Senegal", "Norway"],
    })
    out = db.assign_match_ids(df)
    assert out["match_id"].tolist() == [
        "20260616-france-senegal", "20260616-iraq-norway"]


def test_derive_group_matchday_counts_appearances():
    # 2 groups (ABCD, EFGH style) over 3 rounds; matchday = each team's Nth game
    rounds = [
        ("2026-06-11", "A", "B"), ("2026-06-11", "C", "D"),
        ("2026-06-15", "A", "C"), ("2026-06-15", "B", "D"),
        ("2026-06-19", "A", "D"), ("2026-06-19", "B", "C"),
    ]
    df = pd.DataFrame(
        [{"date": pd.Timestamp(d), "home_team": h, "away_team": a}
         for d, h, a in rounds]
    )
    md = db.derive_group_matchday(df)
    assert md.tolist() == [1, 1, 2, 2, 3, 3]


def test_build_matches_labels_wc_rows(tmp_path):
    results = tmp_path / "results.csv"
    wc = tmp_path / "wc.csv"
    hdr = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    results.write_text(
        hdr + "2024-01-01,Spain,Italy,2,1,Friendly,X,Y,FALSE\n"
    )
    wc.write_text(
        hdr
        + "2026-06-11,France,Senegal,1,0,FIFA World Cup,A,US,TRUE\n"
        + "2026-06-15,France,Iraq,NA,NA,FIFA World Cup,B,US,TRUE\n"
    )
    df = mig.build_matches(results, wc)
    by_id = df.set_index("match_id")
    assert by_id.loc["20240101-spain-italy", "source"] == "upstream"
    assert by_id.loc["20240101-spain-italy", "stage"] is None or \
        pd.isna(by_id.loc["20240101-spain-italy", "stage"])
    assert by_id.loc["20260611-france-senegal", "stage"] == "group"
    assert by_id.loc["20260611-france-senegal", "round"] == "MD1"
    assert by_id.loc["20260615-france-iraq", "round"] == "MD2"
    assert pd.isna(by_id.loc["20260611-france-senegal", "group_label"])


def test_load_results_reads_from_db(tmp_path, monkeypatch):
    dbfile = tmp_path / "t.duckdb"
    c = db.connect(dbfile)
    db.init_schema(c)
    c.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, home_score, "
        "away_score, tournament, neutral, city, country, source) VALUES "
        "('m1', DATE '2022-06-01', 'Spain', 'Italy', 2, 1, 'Friendly', TRUE, 'X', 'Y', 'upstream')"
    )
    c.close()
    monkeypatch.setattr(db, "DB_PATH", dbfile)  # load_results reads db.DB_PATH at call time
    df = wcdata.load_results()
    assert {"home_team", "away_team", "home_score", "neutral"} <= set(df.columns)
    assert df.iloc[0]["home_team"] == "Spain"


def test_predictions_frame_shape():
    rows = [{
        "date": "2026-06-16", "home": "France", "away": "Senegal",
        "pick": "1-0", "lam_h": 1.5, "lam_a": 1.1, "result": "H",
        "P_result": 0.46, "P_home_g": 0.33, "P_away_g": 0.34, "P_gd": 0.23,
        "EP": 2.29,
    }]
    df = rs.predictions_frame(rows)
    assert df.loc[0, "match_id"] == "20260616-france-senegal"
    assert df.loc[0, "pred_home_goals"] == 1
    assert df.loc[0, "pred_away_goals"] == 0
    assert df.loc[0, "outcome"] == "H"
    assert set(["lam_h", "lam_a", "p_result", "p_home_g", "p_away_g",
                "p_gd", "ep"]).issubset(df.columns)


def test_match_ids_for_date_filters_unplayed(con):
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, source) VALUES "
        "('20260616-france-senegal', DATE '2026-06-16', 'France', 'Senegal', 'wc2026'),"
        "('20260617-england-croatia', DATE '2026-06-17', 'England', 'Croatia', 'wc2026')"
    )
    ids = commit_picks.match_ids_for_date(con, "2026-06-16")
    assert ids == ["20260616-france-senegal"]
