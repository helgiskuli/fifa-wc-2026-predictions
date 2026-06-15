"""Step 4: expected-points-maximising scoreline pick.

Office-pool scoring rule (confirmed with the user) — additive components:
  * a points for a correct result (home win / draw / away win)
  * b points for EACH correctly predicted team goal count (+b home, +b away)
  * c points for the correct signed goal difference (only counts when the
    outcome is correct, which a matching signed difference guarantees).

An exact scoreline therefore scores a + 2b + c (= 6 under the defaults).

The expected points of a predicted scoreline (i, j):

    EP(i, j) = a * P(outcome of (i,j))
             + b * P(home goals == i)          # row marginal of P
             + b * P(away goals == j)          # col marginal of P
             + c * P(home - away == i - j)     # signed-difference mass

All four terms are computable from the score matrix, so we evaluate EP over
the whole grid and take the argmax. This optimises the weighted objective;
it returns neither the mean nor the modal score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ScoringConfig


@dataclass
class Prediction:
    home_goals: int
    away_goals: int
    outcome: str            # "H", "D", "A"
    p_outcome: float        # P(home win / draw / away win)
    p_home_goals: float     # P(home goals == home_goals)  (marginal)
    p_away_goals: float     # P(away goals == away_goals)  (marginal)
    p_goaldiff: float       # P(home - away == home_goals - away_goals)
    p_exact: float          # P(this exact score)  -- reported, not optimised
    exp_points: float       # full weighted objective at this pick

    @property
    def score(self) -> str:
        return f"{self.home_goals}-{self.away_goals}"


def _outcome_probs(P: np.ndarray) -> tuple[float, float, float]:
    n = P.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (float(P[ii > jj].sum()),   # H
            float(P[ii == jj].sum()),  # D
            float(P[ii < jj].sum()))   # A


def _diff_prob_grid(P: np.ndarray) -> np.ndarray:
    """pdiff[i,j] = P(actual home - away == i - j), broadcast over the grid."""
    n = P.shape[0]
    ii, jj = np.indices((n, n))
    diff = ii - jj
    pdiff = np.zeros_like(P)
    for d in range(-(n - 1), n):
        mask = diff == d
        pdiff[mask] = P[mask].sum()
    return pdiff


def expected_points_grid(P: np.ndarray, cfg: ScoringConfig) -> np.ndarray:
    """EP(i, j) for every cell of the score grid."""
    n = P.shape[0]
    p_home = P.sum(axis=1)   # marginal P(home goals == i)
    p_away = P.sum(axis=0)   # marginal P(away goals == j)
    pH, pD, pA = _outcome_probs(P)

    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    result_p = np.where(ii > jj, pH, np.where(ii == jj, pD, pA))
    pdiff = _diff_prob_grid(P)

    return (cfg.a * result_p
            + cfg.b * p_home[:, None]
            + cfg.b * p_away[None, :]
            + cfg.c * pdiff)


def score_prediction(pred_h: int, pred_a: int, act_h: int, act_a: int,
                     cfg: ScoringConfig) -> float:
    """Office-pool points scored by a predicted scoreline against the actual
    result. Mirrors the EP objective exactly so backtest totals are
    comparable to the model's expected points."""
    pts = 0.0
    pred_sign = (pred_h > pred_a) - (pred_h < pred_a)
    act_sign = (act_h > act_a) - (act_h < act_a)
    if pred_sign == act_sign:                       # correct outcome
        pts += cfg.a
    if pred_h == act_h:                             # home goal count
        pts += cfg.b
    if pred_a == act_a:                             # away goal count
        pts += cfg.b
    if (pred_h - pred_a) == (act_h - act_a):        # signed goal difference
        pts += cfg.c
    return pts


def modal_prediction(P: np.ndarray) -> tuple[int, int]:
    """Most likely exact scoreline (a naive baseline, NOT the EP pick)."""
    n = P.shape[0]
    i, j = divmod(int(P.argmax()), n)
    return i, j


def best_prediction(P: np.ndarray, cfg: ScoringConfig) -> Prediction:
    """Pick the scoreline maximising the full weighted objective."""
    n = P.shape[0]
    EP = expected_points_grid(P, cfg)
    i, j = divmod(int(EP.argmax()), n)

    pH, pD, pA = _outcome_probs(P)
    outcome = "H" if i > j else ("D" if i == j else "A")
    p_outcome = pH if i > j else (pD if i == j else pA)
    pdiff = _diff_prob_grid(P)

    return Prediction(
        home_goals=i, away_goals=j, outcome=outcome,
        p_outcome=float(p_outcome),
        p_home_goals=float(P[i, :].sum()),
        p_away_goals=float(P[:, j].sum()),
        p_goaldiff=float(pdiff[i, j]),
        p_exact=float(P[i, j]),
        exp_points=float(EP[i, j]),
    )
