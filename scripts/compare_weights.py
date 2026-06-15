"""Validate competition-tier weighting schemes on the 2018 + 2022 WCs.

Fits the model under several (major, qual_nl, friendly, other) weight
schemes, refit as-of each WC eve, and reports held-out office-pool points
(EP-optimal pick and the naive "1-0 to favourite" baseline) plus outcome
accuracy. Lets the backtest pick the weighting rather than assuming it.

    uv run python -m scripts.compare_weights
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from wc2026 import (ModelConfig, PreprocessConfig, ScoringConfig,
                    best_prediction, fit, load_results, score_prediction)
from scripts.backtest import WC_EDITIONS

# (major, qual_nl, friendly, other)
SCHEMES = {
    "flat-competitive (prev)": (1.0, 1.0, 0.3, 1.0),
    "tiered 1/.5/.3/.3 (proposed)": (1.0, 0.5, 0.3, 0.3),
    "steep 1/.4/.2/.2": (1.0, 0.4, 0.2, 0.2),
}


def eval_year(df, pre, year, scoring) -> dict:
    kickoff, final = WC_EDITIONS[year]
    model = fit(df, pre, ModelConfig(), verbose=False)
    wc = df[(df.tournament == "FIFA World Cup")
            & (df.date >= pd.Timestamp(kickoff))
            & (df.date <= pd.Timestamp(final))
            & df.home_score.notna()].copy()

    ep_pts = naive_pts = outcome_ok = 0.0
    for _, r in wc.iterrows():
        ah, aa = int(r.home_score), int(r.away_score)
        P = model.fixture_matrix(r.home_team, r.away_team, bool(r.neutral))
        pred = best_prediction(P, scoring)
        mu_h, mu_a = model.rates(r.home_team, r.away_team, bool(r.neutral))
        ni, nj = (1, 0) if mu_h >= mu_a else (0, 1)
        ep_pts += score_prediction(pred.home_goals, pred.away_goals, ah, aa, scoring)
        naive_pts += score_prediction(ni, nj, ah, aa, scoring)
        outcome_ok += (np.sign(pred.home_goals - pred.away_goals) == np.sign(ah - aa))
    return {"ep": ep_pts, "naive": naive_pts, "outcome_acc": outcome_ok / len(wc),
            "n": len(wc)}


def main() -> None:
    df = load_results()
    scoring = ScoringConfig()
    rows = []
    for name, (wm, wq, wf, wo) in SCHEMES.items():
        per_year = {}
        for year in (2018, 2022):
            kickoff, _ = WC_EDITIONS[year]
            pre = PreprocessConfig(as_of=kickoff - timedelta(days=1),
                                   weight_major=wm, weight_qual_nl=wq,
                                   weight_friendly=wf, weight_other=wo)
            per_year[year] = eval_year(df, pre, year, scoring)
            print(f"  [{name}] {year}: EP={per_year[year]['ep']:.0f} "
                  f"naive={per_year[year]['naive']:.0f} "
                  f"acc={per_year[year]['outcome_acc']:.1%}")
        rows.append({
            "scheme": name,
            "EP_2018": per_year[2018]["ep"], "EP_2022": per_year[2022]["ep"],
            "EP_total": per_year[2018]["ep"] + per_year[2022]["ep"],
            "naive_total": per_year[2018]["naive"] + per_year[2022]["naive"],
            "acc_2018": per_year[2018]["outcome_acc"],
            "acc_2022": per_year[2022]["outcome_acc"],
        })

    out = pd.DataFrame(rows)
    print("\n=== Tier-weighting comparison (128 held-out matches) ===")
    print(out.to_string(index=False,
          formatters={"acc_2018": "{:.1%}".format, "acc_2022": "{:.1%}".format}))


if __name__ == "__main__":
    main()
