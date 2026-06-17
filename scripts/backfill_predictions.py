"""Reconstruct honest pre-game picks for played WC-2026 matches.

For each distinct date with a played WC match, refit the model as-of the eve
of that date (the backtest's leak-free pattern) and store one EP-optimal pick
per match under prediction kind 'pregame'. Warm-started for speed; the first
fit is cold (~95 s), the rest are seconds. Does NOT touch model_cache.json
(that is run_schedule's as-of-today cache) and does NOT touch 'latest' or
'committed' rows.

    uv run python -m scripts.backfill_predictions
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from wc2026 import (FittedModel, ModelConfig, PreprocessConfig, ScoringConfig,
                    best_prediction, fit, load_results)
from wc2026 import db

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "model_cache.json"


def played_wc_matches(df: pd.DataFrame, tournament: str = "FIFA World Cup",
                      season_year: int = 2026) -> pd.DataFrame:
    """Played (real-score) matches of the target tournament/season."""
    m = df[(df["tournament"] == tournament)
           & (df["date"].dt.year == season_year)
           & df["home_score"].notna() & df["away_score"].notna()].copy()
    return m.reset_index(drop=True)[["date", "home_team", "away_team", "neutral"]]


def pregame_prediction_rows(model: FittedModel, matches: pd.DataFrame,
                            scoring: ScoringConfig) -> tuple[pd.DataFrame, list[str]]:
    """EP-optimal pick per match as a predictions-table payload, plus the list
    of skipped fixtures whose teams are not in the fitted pool."""
    rows, skipped = [], []
    for _, r in matches.iterrows():
        home, away, neutral = r.home_team, r.away_team, bool(r.neutral)
        if home not in model.attack or away not in model.attack:
            skipped.append(f"{home} vs {away}")
            continue
        mu_h, mu_a = model.rates(home, away, neutral)
        P = model.score_matrix(mu_h, mu_a)
        pred = best_prediction(P, scoring)
        rows.append({
            "match_id": db.make_match_id(r.date, home, away),
            "pred_home_goals": pred.home_goals, "pred_away_goals": pred.away_goals,
            "outcome": pred.outcome,
            "lam_h": mu_h, "lam_a": mu_a,
            "p_result": pred.p_outcome, "p_home_g": pred.p_home_goals,
            "p_away_g": pred.p_away_goals, "p_gd": pred.p_goaldiff,
            "ep": pred.exp_points,
        })
    return pd.DataFrame(rows), skipped


def main() -> None:
    mcfg, scoring = ModelConfig(), ScoringConfig()
    df = load_results()
    played = played_wc_matches(df)
    if played.empty:
        print("no played WC matches to backfill")
        return

    dates = sorted({d.date() for d in played["date"]})
    # First fit warm-starts from the production cache if present; thereafter we
    # roll the previous eve's fit forward as the warm start (kept in memory).
    warm = FittedModel.load(CACHE_PATH) if CACHE_PATH.exists() else None

    con = db.connect(db.DB_PATH)
    try:
        for d in dates:
            as_of = d - timedelta(days=1)
            pre = PreprocessConfig(as_of=as_of)
            model = fit(df, pre, mcfg, warm_start=warm)
            warm = model
            day = played[played["date"].dt.date == d]
            rows, skipped = pregame_prediction_rows(model, day, scoring)
            if not rows.empty:
                db.upsert_predictions(con, rows, "pregame", as_of)
            msg = f"{d}: wrote {len(rows)} pregame picks (as_of {as_of})"
            if skipped:
                msg += f"; skipped {len(skipped)}: " + ", ".join(skipped)
            print(msg)
    finally:
        con.close()


if __name__ == "__main__":
    main()
