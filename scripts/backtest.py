"""Step 6: backtest on the 2022 World Cup (held out).

Refits the model as-of the day before kickoff (no leakage), predicts all 64
tournament matches, and scores them under the finalised office-pool rule.
Reports points earned, hit rates, calibration (realised vs expected points,
outcome reliability, goal-total bias) and naive baselines for comparison.

    uv run python -m scripts.backtest [2018|2022]   (default 2022)
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from wc2026 import (ModelConfig, PreprocessConfig, ScoringConfig,
                    best_prediction, fit, load_results, modal_prediction,
                    score_prediction)

pd.set_option("display.width", 130)
pd.set_option("display.max_rows", 100)

# (kickoff, final) per World Cup edition; we refit as-of the eve of kickoff.
WC_EDITIONS = {
    2018: (date(2018, 6, 14), date(2018, 7, 15)),
    2022: (date(2022, 11, 20), date(2022, 12, 18)),
}


def main(year: int = 2022) -> None:
    scoring = ScoringConfig()
    kickoff, final = WC_EDITIONS[year]
    # Production preprocessing, but as-of the eve of the chosen WC.
    pre = PreprocessConfig(as_of=kickoff - timedelta(days=1))
    mcfg = ModelConfig()

    df = load_results()
    model = fit(df, pre, mcfg)

    wc = df[(df.tournament == "FIFA World Cup")
            & (df.date >= pd.Timestamp(kickoff))
            & (df.date <= pd.Timestamp(final))
            & df.home_score.notna()].copy()

    rows, ep_sum = [], 0.0
    for _, r in wc.iterrows():
        home, away, neutral = r.home_team, r.away_team, bool(r.neutral)
        ah, aa = int(r.home_score), int(r.away_score)
        P = model.fixture_matrix(home, away, neutral)
        pred = best_prediction(P, scoring)
        mi, mj = modal_prediction(P)
        # naive: 1-0 to the higher-lambda side
        mu_h, mu_a = model.rates(home, away, neutral)
        ni, nj = (1, 0) if mu_h >= mu_a else (0, 1)

        pts = score_prediction(pred.home_goals, pred.away_goals, ah, aa, scoring)
        pts_modal = score_prediction(mi, mj, ah, aa, scoring)
        pts_naive = score_prediction(ni, nj, ah, aa, scoring)
        ep_sum += pred.exp_points

        rows.append({
            "home": home, "away": away,
            "actual": f"{ah}-{aa}", "pick": pred.score, "modal": f"{mi}-{mj}",
            "pts": pts, "pts_modal": pts_modal, "pts_naive": pts_naive,
            "exact": int(pred.home_goals == ah and pred.away_goals == aa),
            "outcome_ok": int(np.sign(pred.home_goals - pred.away_goals)
                              == np.sign(ah - aa)),
            "gd_ok": int((pred.home_goals - pred.away_goals) == (ah - aa)),
            "side_goals": int(pred.home_goals == ah) + int(pred.away_goals == aa),
            "p_home_win": _p_home_win(P),
            "home_won": int(ah > aa),
            "lam_sum": mu_h + mu_a, "goals": ah + aa,
        })

    out = pd.DataFrame(rows)
    _report(out, ep_sum, scoring, year)


def _p_home_win(P: np.ndarray) -> float:
    n = P.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return float(P[ii > jj].sum())


def _report(out: pd.DataFrame, ep_sum: float, scoring: ScoringConfig,
            year: int) -> None:
    n = len(out)
    print(f"\n=== {year} WC backtest: {n} matches "
          f"(a={scoring.a}, b={scoring.b}, c={scoring.c}) ===\n")
    print(out[["home", "away", "actual", "pick", "pts"]].to_string(index=False))

    tot = out["pts"].sum()
    print("\n--- Points (EP-optimal pick) ---")
    print(f"  total            {tot:.0f} / {6*n} max   ({tot/n:.3f} per match)")
    print(f"  model expected   {ep_sum:.1f}  ({ep_sum/n:.3f} per match)  "
          f"<- realised should land near this if calibrated")
    print("\n--- Hit rates ---")
    print(f"  exact scoreline  {out.exact.mean():6.1%}  ({out.exact.sum()}/{n})")
    print(f"  outcome (H/D/A)  {out.outcome_ok.mean():6.1%}  ({out.outcome_ok.sum()}/{n})")
    print(f"  goal difference  {out.gd_ok.mean():6.1%}  ({out.gd_ok.sum()}/{n})")
    print(f"  side goal counts {out.side_goals.sum()}/{2*n} sides "
          f"({out.side_goals.mean():.2f} per match)")

    print("\n--- Baselines (total points) ---")
    print(f"  EP-optimal pick      {tot:.0f}   ({tot/n:.3f}/match)")
    print(f"  modal exact score    {out.pts_modal.sum():.0f}   "
          f"({out.pts_modal.mean():.3f}/match)")
    print(f"  naive 1-0 favourite  {out.pts_naive.sum():.0f}   "
          f"({out.pts_naive.mean():.3f}/match)")

    print("\n--- Calibration ---")
    # outcome reliability over predicted P(home win) bins
    print("  P(home win) reliability:")
    bins = [0, .2, .4, .6, .8, 1.01]
    out["_bin"] = pd.cut(out.p_home_win, bins, right=False)
    rel = out.groupby("_bin", observed=True).agg(
        n=("home_won", "size"), pred=("p_home_win", "mean"),
        actual=("home_won", "mean"))
    for b, row in rel.iterrows():
        print(f"    {str(b):14s} n={int(row.n):2d}  pred={row.pred:.2f}  "
              f"actual={row.actual:.2f}")
    # goal-total bias
    print(f"  mean goals/match  predicted(lam)={out.lam_sum.mean():.2f}  "
          f"actual={out.goals.mean():.2f}")


if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
    main(yr)
