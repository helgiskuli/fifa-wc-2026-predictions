"""Calibration check for the winner-take-all decision.

For a held-out WC edition, refit as-of the eve of kickoff and ask three
questions the EP backtest does not answer directly:

  1. Aggregate: do predicted H/D/A frequencies match actual? (Is the model
     draw-shy or favourite-overconfident in aggregate?)
  2. Favourite reliability: when the model says the most-likely outcome has
     probability p, does it happen at rate p? (Overconfidence shows as
     actual < predicted in the high-confidence bins -- the exact thing that
     makes upsets underpriced by the whole field.)
  3. Brier score of the 3-way outcome forecast vs a chalk baseline.

    uv run python -m scripts.calibration [2018|2022]
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from wc2026 import (ModelConfig, PreprocessConfig, ScoringConfig,
                    fit, load_results)
from wc2026.predict import _outcome_probs

WC = {2018: (date(2018, 6, 14), date(2018, 7, 15)),
      2022: (date(2022, 11, 20), date(2022, 12, 18))}


def main(year: int = 2022) -> None:
    kickoff, final = WC[year]
    pre = PreprocessConfig(as_of=kickoff - timedelta(days=1))
    df = load_results()
    model = fit(df, pre, ModelConfig(), verbose=False)

    wc = df[(df.tournament == "FIFA World Cup")
            & (df.date >= pd.Timestamp(kickoff))
            & (df.date <= pd.Timestamp(final))
            & df.home_score.notna()].copy()

    rows = []
    for _, r in wc.iterrows():
        P = model.fixture_matrix(r.home_team, r.away_team, bool(r.neutral))
        pH, pD, pA = _outcome_probs(P)
        ah, aa = int(r.home_score), int(r.away_score)
        act = "H" if ah > aa else ("D" if ah == aa else "A")
        probs = {"H": pH, "D": pD, "A": pA}
        fav = max(probs, key=probs.get)
        rows.append({"pH": pH, "pD": pD, "pA": pA, "act": act,
                     "p_fav": probs[fav], "fav_hit": int(act == fav),
                     "p_act": probs[act]})
    o = pd.DataFrame(rows)
    n = len(o)

    print(f"\n================  {year} WC calibration  (n={n})  ================")

    print("\n[1] Aggregate predicted vs actual outcome frequency")
    for k, lab in [("H", "home win"), ("D", "draw    "), ("A", "away win")]:
        pred = o[f"p{k}"].mean()
        actual = (o.act == k).mean()
        print(f"    {lab}: predicted {pred:5.1%}   actual {actual:5.1%}   "
              f"diff {pred-actual:+5.1%}")

    print("\n[2] Favourite reliability (model's most-likely outcome)")
    bins = [0.33, 0.45, 0.55, 0.65, 0.80, 1.01]
    o["_b"] = pd.cut(o.p_fav, bins, right=False)
    rel = o.groupby("_b", observed=True).agg(
        n=("fav_hit", "size"), pred=("p_fav", "mean"), actual=("fav_hit", "mean"))
    for b, row in rel.iterrows():
        flag = "  <- overconfident" if row.actual < row.pred - 0.05 else ""
        print(f"    p_fav {str(b):13s} n={int(row.n):2d}  "
              f"predicted={row.pred:.2f}  actual={row.actual:.2f}{flag}")
    print(f"    overall: favourite predicted {o.p_fav.mean():.1%}  "
          f"actual hit {o.fav_hit.mean():.1%}")

    print("\n[3] Outcome Brier score (lower = better; 3-way multiclass)")
    y = pd.get_dummies(o.act)[["H", "D", "A"]].to_numpy(dtype=float)
    p = o[["pH", "pD", "pA"]].to_numpy()
    brier_model = float(((p - y) ** 2).sum(axis=1).mean())
    # chalk baseline: everyone's prior = base rates of this tournament
    base = y.mean(axis=0)
    brier_base = float(((base[None, :] - y) ** 2).sum(axis=1).mean())
    print(f"    model     {brier_model:.3f}")
    print(f"    base-rate {brier_base:.3f}  (predict tournament avg every game)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2022)
