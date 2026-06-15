"""Step 5: fixture runner.

Fits the model (warm-started from the cached fit when present, for fast
re-runs) and emits one predicted scoreline per remaining WC-2026 fixture,
as a printed table and a CSV (predictions.csv). The fit is cached to
model_cache.json.

    uv run python -m scripts.run_schedule            # refit (warm) + predict
    uv run python -m scripts.run_schedule --cached   # reuse cache, no refit
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from wc2026 import (FittedModel, ModelConfig, PreprocessConfig, ScoringConfig,
                    best_prediction, fit, load_results, upcoming_fixtures)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "predictions.csv"
CACHE_PATH = ROOT / "model_cache.json"

pd.set_option("display.width", 120)
pd.set_option("display.max_rows", 200)


def main(use_cached: bool = False) -> None:
    pre, mcfg, scoring = PreprocessConfig(), ModelConfig(), ScoringConfig()

    df = load_results()
    print(f"Forecasting as-of {pre.as_of} (training window: last "
          f"{pre.window_years:g}y, half-life {pre.half_life_days:g}d)")

    if use_cached and CACHE_PATH.exists():
        # Re-predict from the saved fit without refitting (instant).
        model = FittedModel.load(CACHE_PATH)
        print(f"Loaded cached model ({len(model.teams)} teams) -- no refit.")
    else:
        # Warm-start from the cache if we have one; refit; re-cache.
        warm = FittedModel.load(CACHE_PATH) if CACHE_PATH.exists() else None
        model = fit(df, pre, mcfg, warm_start=warm)
        model.save(CACHE_PATH)
        print(f"Cached fitted model -> {CACHE_PATH.name}")

    fx = upcoming_fixtures(df)

    rows = []
    skipped = []
    for _, r in fx.iterrows():
        home, away, neutral = r.home_team, r.away_team, bool(r.neutral)
        if home not in model.attack or away not in model.attack:
            skipped.append(f"{home} vs {away}")
            continue
        mu_h, mu_a = model.rates(home, away, neutral)
        P = model.score_matrix(mu_h, mu_a)
        pred = best_prediction(P, scoring)
        rows.append({
            "date": r.date.date().isoformat(),
            "home": home, "away": away,
            "venue": "home" if not neutral else "neutral",
            "pick": pred.score,
            "lam_h": round(mu_h, 2), "lam_a": round(mu_a, 2),
            "result": pred.outcome,
            "P_result": round(pred.p_outcome, 3),
            "P_home_g": round(pred.p_home_goals, 3),
            "P_away_g": round(pred.p_away_goals, 3),
            "P_gd": round(pred.p_goaldiff, 3),
            "EP": round(pred.exp_points, 3),
        })

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\n=== WC-2026 predictions  (a={scoring.a}, b={scoring.b}; "
          f"{len(out)} fixtures) ===")
    print(out.to_string(index=False))
    if skipped:
        print(f"\nskipped {len(skipped)} (team not in fitted pool): "
              + ", ".join(skipped))

    print(f"\nwrote {OUT_PATH}")
    print("\nPick distribution:")
    print(out["pick"].value_counts().to_string())


if __name__ == "__main__":
    main(use_cached="--cached" in sys.argv[1:])
