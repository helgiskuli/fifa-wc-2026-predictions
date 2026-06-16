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
