"""Steps 1-4 demo: fit the model, show per-team strengths, and predict a
few sample fixtures. Run with:  uv run python -m scripts.demo
"""
from __future__ import annotations

import pandas as pd

from wc2026 import (ModelConfig, PreprocessConfig, ScoringConfig,
                    best_prediction, fit, load_results, upcoming_fixtures)

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 12)


def main() -> None:
    pre = PreprocessConfig()
    mcfg = ModelConfig()
    scoring = ScoringConfig()

    df = load_results()
    model = fit(df, pre, mcfg)

    # ---- per-team strengths -------------------------------------------
    table = model.strength_table()
    print("\n=== Top 20 by overall strength (attack + defense) ===")
    print(table.head(20).to_string(index=False,
          formatters={c: "{:+.3f}".format for c in ["attack", "defense", "overall"]}))
    print("\n=== Bottom 5 (of WC-relevant pool) ===")
    print(table.tail(5).to_string(index=False,
          formatters={c: "{:+.3f}".format for c in ["attack", "defense", "overall"]}))
    print(f"\nhome_att = {model.home_att:+.3f}  home_def = {model.home_def:+.3f}"
          f"  (total home edge {model.home_adv:+.3f}; applies only when "
          f"neutral == False)")
    print(f"lambda3 (biv-Poisson shared) = {model.lambda3:.3f}   "
          f"rho (Dixon-Coles) = {model.rho:+.3f}")

    # ---- sample fixture predictions -----------------------------------
    samples = [
        ("Spain", "Germany", True),
        ("United States", "England", False),   # US at home: real home edge
        ("Argentina", "Brazil", True),
        ("France", "Norway", True),
    ]
    print(f"\n=== Sample fixture predictions (a={scoring.a}, b={scoring.b}) ===")
    _print_predictions(model, scoring, samples)

    # ---- a couple of real upcoming WC fixtures ------------------------
    fx = upcoming_fixtures(df)
    print(f"\n=== Next 5 real WC-2026 fixtures from the schedule "
          f"({len(fx)} unplayed) ===")
    real = []
    for _, r in fx.head(5).iterrows():
        if r.home_team in model.attack and r.away_team in model.attack:
            real.append((r.home_team, r.away_team, bool(r.neutral)))
    _print_predictions(model, scoring, real)


def _print_predictions(model, scoring, fixtures) -> None:
    rows = []
    for home, away, neutral in fixtures:
        if home not in model.attack or away not in model.attack:
            rows.append({"fixture": f"{home} vs {away}", "pick": "n/a (sparse)",
                         "lam_h": None, "lam_a": None, "P(out)": None,
                         "P(exact)": None, "EP": None})
            continue
        mu_h, mu_a = model.rates(home, away, neutral)
        P = model.score_matrix(mu_h, mu_a)
        pred = best_prediction(P, scoring)
        rows.append({
            "fixture": f"{home} vs {away}" + (" (N)" if neutral else " (H)"),
            "pick": pred.score, "lam_h": round(mu_h, 2), "lam_a": round(mu_a, 2),
            "P(out)": round(pred.p_outcome, 3), "P(exact)": round(pred.p_exact, 3),
            "EP": round(pred.exp_points, 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
