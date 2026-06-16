"""Step 1: loader + preprocessing.

Reads the martj42/international_results `results.csv` (live each call, so
manual top-ups during the tournament flow straight through), filters to a
recent competitive window, and attaches a per-match weight combining
friendly down-weighting with exponential time decay.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import db
from .config import PreprocessConfig

# Schema per the kickoff:
# date, home_team, away_team, home_score, away_score, tournament, city,
# country, neutral
RESULTS_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_RESULTS_PATH = _DATA_DIR / "results.csv"
# Curated WC-2026 fixtures/results, kept separate from the upstream historical
# feed (results.csv) so the feed can be refreshed without clobbering tournament
# data. Unioned in by load_results; played rows train, NA rows are fixtures.
DEFAULT_WC_GAMES_PATH = _DATA_DIR / "wc-2026-games.csv"

# Major finals = the World Cup plus the continental championships. These are
# the highest-quality, full-strength-squad games and get the top weight.
MAJOR_FINALS = frozenset({
    "FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations",
    "AFC Asian Cup", "Gold Cup", "CONCACAF Championship",
    "Confederations Cup", "OFC Nations Cup",
})


def tournament_tier(name: str) -> str:
    """Classify a tournament into a weight tier: 'major', 'qual_nl',
    'friendly', or 'other'."""
    if name in MAJOR_FINALS:
        return "major"
    if name == "Friendly":
        return "friendly"
    if "qualification" in name or "Nations League" in name:
        return "qual_nl"
    return "other"


def _read_results_csv(path: Path | str) -> pd.DataFrame:
    """Read one results-schema CSV, typed."""
    df = pd.read_csv(
        path,
        dtype={"home_team": "string", "away_team": "string",
               "tournament": "string", "city": "string", "country": "string"},
        # `neutral` is TRUE/FALSE text; scores are NA for unplayed fixtures.
        na_values=["NA"],
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    # neutral -> bool. Use the flag for venue, NOT a team/country string match
    # (team names are current identity, country names historical).
    df["neutral"] = df["neutral"].astype("string").str.upper().eq("TRUE")
    return df


def load_results(path: Path | str | None = None,
                 wc_games_path: Path | str | None = None) -> pd.DataFrame:
    """Load all matches (played + unplayed) from the DuckDB store, in the same
    column shape the CSV loader used to return. The `path`/`wc_games_path`
    arguments are retained for backwards compatibility and ignored (the data
    now lives in the database; see scripts.migrate_to_duckdb)."""
    con = db.connect(db.DB_PATH, read_only=True)
    try:
        return db.load_matches(con)
    finally:
        con.close()


def _decay_weight(age_days: np.ndarray, half_life_days: float) -> np.ndarray:
    """0.5 ** (age / half_life): recent matches dominate."""
    return np.power(0.5, age_days / half_life_days)


def build_training_frame(
    df: pd.DataFrame, cfg: PreprocessConfig
) -> pd.DataFrame:
    """Filter to the competitive window and attach match weights.

    Returns played matches only (real scores), with columns:
    home_team, away_team, home_score, away_score, neutral, weight.
    """
    as_of = pd.Timestamp(cfg.as_of)
    cutoff = as_of - pd.DateOffset(years=int(cfg.window_years),
                                   days=int((cfg.window_years % 1) * 365.25))

    played = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    played = played[(played["date"] >= cutoff) & (played["date"] <= as_of)]
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)

    # Weight = time decay * per-competition-tier weight.
    age_days = (as_of - played["date"]).dt.days.to_numpy(dtype=float)
    weight = _decay_weight(age_days, cfg.half_life_days)
    tier_weight = {
        "major": cfg.weight_major, "qual_nl": cfg.weight_qual_nl,
        "friendly": cfg.weight_friendly, "other": cfg.weight_other,
    }
    tiers = played["tournament"].map(tournament_tier)
    played["tier"] = tiers.to_numpy()
    weight = weight * tiers.map(tier_weight).to_numpy(dtype=float)
    played["weight"] = weight

    # Drop any tier zero-weighted out entirely.
    played = played[played["weight"] > 0]

    # Drop teams with too few matches in the window.
    played = _filter_sparse_teams(played, cfg.min_matches)

    return played.reset_index(drop=True)[
        ["date", "home_team", "away_team", "home_score", "away_score",
         "neutral", "tournament", "weight"]
    ]


def _filter_sparse_teams(played: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """Iteratively drop teams appearing in < min_matches rows (dropping one
    team can push another below threshold)."""
    while True:
        counts = pd.concat([played["home_team"], played["away_team"]]).value_counts()
        keep = set(counts[counts >= min_matches].index)
        mask = played["home_team"].isin(keep) & played["away_team"].isin(keep)
        if mask.all():
            return played
        played = played[mask]


def upcoming_fixtures(df: pd.DataFrame, tournament: str = "FIFA World Cup",
                      season_year: int = 2026) -> pd.DataFrame:
    """The unplayed (NA-score) fixtures for the target tournament — the
    matches we ultimately need to predict."""
    fx = df[df["home_score"].isna() & (df["tournament"] == tournament)
            & (df["date"].dt.year == season_year)].copy()
    return fx.reset_index(drop=True)[
        ["date", "home_team", "away_team", "neutral", "city", "country"]
    ]
