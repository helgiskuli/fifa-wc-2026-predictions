"""Backtest a single time-decay half-life on the 2018 + 2022 WCs.

Designed to be launched once per half-life value (in parallel from the
shell) so the whole sweep finishes in roughly one fit's wall-time.

    uv run python -m scripts.sweep_halflife <half_life_days>
"""
from __future__ import annotations

import sys
from datetime import timedelta

from wc2026 import PreprocessConfig, ScoringConfig, load_results
from scripts.backtest import WC_EDITIONS
from scripts.compare_weights import eval_year


def main(half_life: float) -> None:
    df = load_results()
    scoring = ScoringConfig()
    for year in (2018, 2022):
        kickoff, _ = WC_EDITIONS[year]
        pre = PreprocessConfig(as_of=kickoff - timedelta(days=1),
                               half_life_days=half_life)
        m = eval_year(df, pre, year, scoring)
        print(f"HL={half_life:.0f} year={year} EP={m['ep']:.0f} "
              f"naive={m['naive']:.0f} acc={m['outcome_acc']:.3f}",
              flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]))
