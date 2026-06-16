"""One-off: is the low draw-pick rate a fitting bias or a point-prediction artifact?

Compares, over the WC-2026 fixtures:
  * how often a DRAW is the EP-optimal pick (what the user sees),
  * the model's mean implied P(draw) per game (the calibrated belief),
  * how often a draw is the *modal* exact score and the *most likely outcome*.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wc2026 import (FittedModel, ModelConfig, PreprocessConfig, ScoringConfig,
                    best_prediction, fit, load_results, upcoming_fixtures)
from wc2026.predict import _outcome_probs

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "model_cache.json"


def main() -> None:
    pre, mcfg, scoring = PreprocessConfig(), ModelConfig(), ScoringConfig()
    df = load_results()
    warm = FittedModel.load(CACHE) if CACHE.exists() else None
    model = fit(df, pre, mcfg, warm_start=warm)
    model.save(CACHE)

    fx = upcoming_fixtures(df)
    p_draw_each = []
    ep_draw = mode_draw = outcome_draw = 0
    n = 0
    for _, r in fx.iterrows():
        home, away, neutral = r.home_team, r.away_team, bool(r.neutral)
        if home not in model.attack or away not in model.attack:
            continue
        n += 1
        P = model.fixture_matrix(home, away, neutral)
        pH, pD, pA = _outcome_probs(P)
        p_draw_each.append(pD)

        pred = best_prediction(P, scoring)
        if pred.outcome == "D":
            ep_draw += 1
        i, j = divmod(int(P.argmax()), P.shape[0])
        if i == j:
            mode_draw += 1
        if pD >= pH and pD >= pA:
            outcome_draw += 1

    pe = np.array(p_draw_each)
    print(f"\nfixtures analysed: {n}")
    print(f"EP-optimal pick is a draw:        {ep_draw:2d}  ({ep_draw/n:5.1%})")
    print(f"modal exact score is a draw:      {mode_draw:2d}  ({mode_draw/n:5.1%})")
    print(f"draw is most-likely OUTCOME:      {outcome_draw:2d}  ({outcome_draw/n:5.1%})")
    print(f"\nmodel mean implied P(draw)/game:  {pe.mean():5.1%}  "
          f"(=> expected ~{pe.sum():.1f} draws over {n} games)")
    print(f"P(draw) range over fixtures:      {pe.min():5.1%} .. {pe.max():5.1%}")
    print(f"games where P(draw) >= 30%:       {(pe>=0.30).sum()}")
    print(f"games where P(draw) >= 33%:       {(pe>=0.33).sum()}")
    print("\nhistorical reference (actual draw rate): WC2022 23.4%, "
          "WC2018 ~14% (group-only higher)")


if __name__ == "__main__":
    main()
