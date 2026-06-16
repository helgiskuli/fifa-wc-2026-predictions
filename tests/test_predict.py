"""Expected-points objective and scoreline pick (predict.py)."""
import numpy as np

from wc2026 import ScoringConfig, best_prediction, score_matrix
from wc2026.predict import expected_points_grid, _outcome_probs

CFG = ScoringConfig()


def _P():
    # A realistic-ish DC-corrected matrix for a mild home favourite.
    return score_matrix(1.4, 1.0, lambda3=0.1, rho=-0.05, max_goals=8)


def test_ep_grid_matches_manual_formula():
    P = _P()
    EP = expected_points_grid(P, CFG)
    p_home = P.sum(axis=1)
    p_away = P.sum(axis=0)
    pH, pD, pA = _outcome_probs(P)
    n = P.shape[0]
    for i, j in [(0, 0), (1, 0), (2, 1), (0, 2), (3, 3)]:
        outcome_p = pH if i > j else (pD if i == j else pA)
        diff_mass = P[np.add.outer(np.arange(n), -np.arange(n)) == (i - j)].sum()
        expected = (CFG.a * outcome_p + CFG.b * p_home[i]
                    + CFG.b * p_away[j] + CFG.c * diff_mass)
        assert EP[i, j] == np.float64(expected) or abs(EP[i, j] - expected) < 1e-12


def test_best_prediction_is_grid_argmax():
    P = _P()
    EP = expected_points_grid(P, CFG)
    pred = best_prediction(P, CFG)
    i, j = np.unravel_index(int(EP.argmax()), EP.shape)
    assert (pred.home_goals, pred.away_goals) == (int(i), int(j))
    assert abs(pred.exp_points - EP[i, j]) < 1e-12


def test_picks_are_conservative():
    # A goals model should not pick high scores for a mild favourite.
    pred = best_prediction(_P(), CFG)
    assert pred.home_goals + pred.away_goals <= 3


def test_reported_probabilities_consistent():
    P = _P()
    pred = best_prediction(P, CFG)
    assert abs(pred.p_home_goals - P[pred.home_goals, :].sum()) < 1e-12
    assert abs(pred.p_away_goals - P[:, pred.away_goals].sum()) < 1e-12
