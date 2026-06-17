import pandas as pd

from wc2026 import ModelConfig, ScoringConfig
from wc2026.model import FittedModel
from scripts import backfill_predictions as bf


def _toy_model():
    return FittedModel(
        teams=["A", "B"],
        attack={"A": 0.3, "B": -0.3}, defense={"A": 0.2, "B": -0.2},
        intercept=0.0, home_att=0.15, home_def=0.1,
        lambda3=0.1, rho=-0.05, cfg=ModelConfig(),
        neg_loglik=0.0, n_matches=10,
    )


def test_played_wc_matches_filters_to_played_wc_2026():
    df = pd.DataFrame([
        {"date": pd.Timestamp("2026-06-11"), "home_team": "A", "away_team": "B",
         "home_score": 1.0, "away_score": 0.0, "tournament": "FIFA World Cup",
         "neutral": True},
        {"date": pd.Timestamp("2026-06-20"), "home_team": "A", "away_team": "B",
         "home_score": float("nan"), "away_score": float("nan"),
         "tournament": "FIFA World Cup", "neutral": True},   # unplayed -> excluded
        {"date": pd.Timestamp("2024-06-01"), "home_team": "A", "away_team": "B",
         "home_score": 2.0, "away_score": 2.0, "tournament": "Friendly",
         "neutral": False},                                   # not WC -> excluded
    ])
    out = bf.played_wc_matches(df)
    assert len(out) == 1
    assert out.iloc[0]["home_team"] == "A"
    assert list(out.columns) == ["date", "home_team", "away_team", "neutral"]


def test_pregame_prediction_rows_maps_and_skips_unknown_teams():
    model = _toy_model()
    matches = pd.DataFrame([
        {"date": pd.Timestamp("2026-06-11"), "home_team": "A", "away_team": "B",
         "neutral": True},
        {"date": pd.Timestamp("2026-06-11"), "home_team": "A", "away_team": "Z",
         "neutral": True},  # Z not in model -> skipped
    ])
    rows, skipped = bf.pregame_prediction_rows(model, matches, ScoringConfig())
    assert skipped == ["A vs Z"]
    assert len(rows) == 1
    assert rows.iloc[0]["match_id"] == "20260611-a-b"
    assert {"pred_home_goals", "pred_away_goals", "outcome", "lam_h", "lam_a",
            "p_result", "p_home_g", "p_away_g", "p_gd", "ep"} <= set(rows.columns)
