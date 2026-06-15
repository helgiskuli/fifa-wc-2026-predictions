"""Diagnose the goal-volume bias seen in the 2022 backtest.

Question 1: does the fit reproduce the mean goals of its OWN training set?
            (a Poisson-style MLE with an intercept should ~moment-match)
Question 2: do major tournaments score more than the training mix (the
            qualifiers/friendlies that dominate the window)? -> shift, not bug.
Question 3: is the bias concentrated at neutral venues (no home_adv)?

    uv run python -m scripts.diagnose_goals
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from wc2026 import ModelConfig, PreprocessConfig, fit, load_results
from wc2026.data import build_training_frame

pd.set_option("display.width", 120)


def main() -> None:
    df = load_results()
    pre = PreprocessConfig(as_of=date(2022, 11, 19))
    model = fit(df, pre, ModelConfig())
    train = build_training_frame(df, pre)

    mu_h = np.empty(len(train))
    mu_a = np.empty(len(train))
    for k, r in enumerate(train.itertuples(index=False)):
        mu_h[k], mu_a[k] = model.rates(r.home_team, r.away_team, bool(r.neutral))
    w = train["weight"].to_numpy()
    yh = train["home_score"].to_numpy()
    ya = train["away_score"].to_numpy()
    train = train.assign(pred_total=mu_h + mu_a, actual_total=yh + ya,
                         pred_h=mu_h, pred_a=mu_a)

    print("\n=== Q1: does the fit moment-match its own training data? ===")
    print(f"  weighted actual total goals/match : {np.average(yh+ya, weights=w):.3f}")
    print(f"  weighted PRED   total goals/match : {np.average(mu_h+mu_a, weights=w):.3f}")
    print(f"  unweighted actual total           : {(yh+ya).mean():.3f}")
    print(f"  unweighted pred   total           : {(mu_h+mu_a).mean():.3f}")

    print("\n=== Q2: mean goals by competition category (training window) ===")
    cat = np.where(train.tournament == "Friendly", "Friendly",
          np.where(train.tournament.str.contains("qualification"), "Qualifier",
          np.where(train.tournament.isin(
              ["FIFA World Cup", "UEFA Euro", "Copa América",
               "African Cup of Nations", "AFC Asian Cup", "Gold Cup"]),
              "MajorFinals",
          np.where(train.tournament.str.contains("Nations League"),
                   "NationsLeague", "Other"))))
    train = train.assign(cat=cat)
    by_cat = train.groupby("cat").agg(
        n=("actual_total", "size"),
        actual=("actual_total", "mean"),
        pred=("pred_total", "mean"),
        weight_share=("weight", "sum")).sort_values("actual", ascending=False)
    by_cat["weight_share"] = (by_cat["weight_share"]
                              / by_cat["weight_share"].sum()).round(3)
    print(by_cat.round(3).to_string())

    print("\n=== Q3: bias at neutral vs home venues (training) ===")
    by_neu = train.groupby("neutral").agg(
        n=("actual_total", "size"),
        actual=("actual_total", "mean"),
        pred=("pred_total", "mean")).round(3)
    print(by_neu.to_string())

    print(f"\n  fitted home_att = {model.home_att:.3f} "
          f"(x{np.exp(model.home_att):.3f} home scoring), home_def "
          f"= {model.home_def:.3f} (x{np.exp(-model.home_def):.3f} away "
          f"scoring), intercept = {model.intercept:.3f} "
          f"(=> {np.exp(model.intercept):.3f} goals/side at neutral, "
          f"average matchup)")


if __name__ == "__main__":
    main()
