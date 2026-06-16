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
    # Placeholder until Task 4 fills in the scoring view.
    pass
