from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = _DATA_DIR / "wc2026.duckdb"


def _slug(s: str) -> str:
    s = str(s).strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def make_match_id(match_date, home: str, away: str) -> str:
    """Deterministic, rebuild-stable id: YYYYMMDD-home-away (slugified).

    NOT an autoincrement: a DB rebuild must reproduce the same ids or the
    predictions <-> matches join breaks."""
    d = pd.Timestamp(match_date).strftime("%Y%m%d")
    return f"{d}-{_slug(home)}-{_slug(away)}"


def connect(path=DB_PATH, read_only: bool = False):
    return duckdb.connect(str(path), read_only=read_only)


_TABLES = [
    """CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        date DATE,
        home_team TEXT, away_team TEXT,
        home_score INTEGER, away_score INTEGER,
        tournament TEXT,
        neutral BOOLEAN,
        city TEXT, country TEXT,
        stage TEXT, round TEXT, group_label TEXT,
        source TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS predictions (
        match_id TEXT,
        kind TEXT,
        pred_home_goals INTEGER, pred_away_goals INTEGER,
        outcome TEXT,
        lam_h DOUBLE, lam_a DOUBLE,
        p_result DOUBLE, p_home_g DOUBLE, p_away_g DOUBLE, p_gd DOUBLE, ep DOUBLE,
        model_as_of DATE,
        forecast_ts TIMESTAMP,
        PRIMARY KEY (match_id, kind)
    )""",
    """CREATE TABLE IF NOT EXISTS goalscorers (
        match_id TEXT,
        team TEXT, scorer TEXT, minute INTEGER, own_goal BOOLEAN, penalty BOOLEAN
    )""",
    """CREATE TABLE IF NOT EXISTS shootouts (
        match_id TEXT,
        winner TEXT, first_shooter TEXT
    )""",
]


def init_schema(con) -> None:
    for stmt in _TABLES:
        con.execute(stmt)
    _create_report_view(con)


def _create_report_view(con) -> None:
    # Office-pool scoring: a=3 correct outcome, b=1 per correct team goal
    # count, c=1 correct signed goal difference. Mirrors predict.score_prediction.
    con.execute("""
        CREATE OR REPLACE VIEW v_model_report AS
        SELECT
            m.match_id, m.date, m.home_team, m.away_team,
            m.home_score AS actual_h, m.away_score AS actual_a,
            p.pred_home_goals AS pred_h, p.pred_away_goals AS pred_a,
            (sign(p.pred_home_goals - p.pred_away_goals)
               = sign(m.home_score - m.away_score))::INT      AS outcome_ok,
            (p.pred_home_goals = m.home_score)::INT
               + (p.pred_away_goals = m.away_score)::INT       AS side_goals,
            ((p.pred_home_goals - p.pred_away_goals)
               = (m.home_score - m.away_score))::INT           AS gd_ok,
            (p.pred_home_goals = m.home_score
               AND p.pred_away_goals = m.away_score)::INT      AS exact_ok,
            ( 3 * (sign(p.pred_home_goals - p.pred_away_goals)
                     = sign(m.home_score - m.away_score))::INT
            + 1 * (p.pred_home_goals = m.home_score)::INT
            + 1 * (p.pred_away_goals = m.away_score)::INT
            + 1 * ((p.pred_home_goals - p.pred_away_goals)
                     = (m.home_score - m.away_score))::INT )    AS points
        FROM matches m
        JOIN predictions p
          ON p.match_id = m.match_id AND p.kind = 'committed'
        WHERE m.home_score IS NOT NULL AND m.away_score IS NOT NULL
    """)


_MATCH_COLS = ["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "city", "country", "neutral"]


def load_matches(con) -> pd.DataFrame:
    """All matches (played + unplayed), in the exact column shape that
    data.load_results historically returned from the CSV."""
    df = con.execute(
        f"SELECT {', '.join(_MATCH_COLS)} FROM matches ORDER BY date"
    ).df()
    df["date"] = pd.to_datetime(df["date"])
    # scores as float64 with NaN for unplayed, mirroring read_csv(na_values=["NA"])
    df["home_score"] = df["home_score"].astype("float64")
    df["away_score"] = df["away_score"].astype("float64")
    df["neutral"] = df["neutral"].astype(bool)
    for c in ["home_team", "away_team", "tournament", "city", "country"]:
        df[c] = df[c].astype("string")
    return df


_PRED_COLS = ["match_id", "kind", "pred_home_goals", "pred_away_goals",
              "outcome", "lam_h", "lam_a", "p_result", "p_home_g",
              "p_away_g", "p_gd", "ep", "model_as_of", "forecast_ts"]


def upsert_latest_predictions(con, df: pd.DataFrame, model_as_of, now=None) -> None:
    """Write/replace the single kind='latest' row per match."""
    rows = df.copy()
    rows["kind"] = "latest"
    rows["model_as_of"] = pd.Timestamp(model_as_of)
    rows["forecast_ts"] = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow()
    rows = rows[_PRED_COLS]
    con.register("_preds", rows)
    con.execute(
        f"INSERT OR REPLACE INTO predictions ({', '.join(_PRED_COLS)}) "
        f"SELECT {', '.join(_PRED_COLS)} FROM _preds"
    )
    con.unregister("_preds")


def commit_predictions(con, match_ids, force: bool = False, now=None) -> int:
    """Snapshot current 'latest' rows to 'committed' for the given matches.
    Refuses to overwrite an existing committed row unless force=True.
    Returns the number of committed rows written."""
    ids = list(dict.fromkeys(match_ids))
    if not ids:
        return 0
    ts = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow()
    con.register("_ids", pd.DataFrame({"match_id": ids}))
    guard = "" if force else (
        "AND p.match_id NOT IN (SELECT match_id FROM predictions WHERE kind='committed')"
    )
    written = con.execute(
        f"""INSERT OR REPLACE INTO predictions
            (match_id, kind, pred_home_goals, pred_away_goals, outcome,
             lam_h, lam_a, p_result, p_home_g, p_away_g, p_gd, ep,
             model_as_of, forecast_ts)
            SELECT p.match_id, 'committed', p.pred_home_goals, p.pred_away_goals,
             p.outcome, p.lam_h, p.lam_a, p.p_result, p.p_home_g, p.p_away_g,
             p.p_gd, p.ep, p.model_as_of, ?
            FROM predictions p
            WHERE p.kind='latest'
              AND p.match_id IN (SELECT match_id FROM _ids) {guard}
            RETURNING 1""",
        [ts],
    ).fetchall()
    con.unregister("_ids")
    return len(written)


def upsert_results(con, rows) -> None:
    """Fill scores for existing matches (the fetcher's write hook). `rows` is a
    list of dicts or a DataFrame with match_id, home_score, away_score."""
    df = pd.DataFrame(rows)
    if df.empty:
        return
    con.register("_res", df)
    con.execute(
        "UPDATE matches AS m SET home_score = r.home_score, "
        "away_score = r.away_score FROM _res r WHERE m.match_id = r.match_id"
    )
    con.unregister("_res")


def assign_match_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_id"] = [
        make_match_id(d, h, a)
        for d, h, a in zip(df["date"], df["home_team"], df["away_team"])
    ]
    return df


def derive_group_matchday(group_df: pd.DataFrame) -> pd.Series:
    """Matchday (1..3) per group game, via each team's Nth appearance in date
    order. Both teams in a round-robin game share the same appearance count.
    Returns a Series aligned to group_df's index."""
    ordered = group_df.sort_values("date")
    counts: dict[str, int] = {}
    md: dict = {}
    for idx, r in ordered.iterrows():
        n = max(counts.get(r["home_team"], 0), counts.get(r["away_team"], 0)) + 1
        counts[r["home_team"]] = n
        counts[r["away_team"]] = n
        md[idx] = n
    return pd.Series(md).reindex(group_df.index)
