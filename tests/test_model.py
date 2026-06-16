"""Score matrix, parameter transforms, and model caching (model.py)."""
import numpy as np
import pytest

from wc2026 import ModelConfig, FittedModel, score_matrix
from wc2026.model import (_lambda3_to_raw, _raw_to_lambda3, _raw_to_rho,
                          _rho_to_raw, _LAMBDA3_MAX)


def test_score_matrix_is_a_distribution():
    P = score_matrix(1.6, 0.9, lambda3=0.12, rho=-0.07, max_goals=8)
    assert P.shape == (9, 9)
    assert (P >= 0).all()
    assert abs(P.sum() - 1.0) < 1e-12


def test_score_matrix_reduces_to_independent_poisson_marginals():
    # With lambda3=0 and rho=0 the marginals should match Poisson means.
    P = score_matrix(1.5, 1.0, lambda3=0.0, rho=0.0, max_goals=12)
    mean_home = (P.sum(axis=1) * np.arange(P.shape[0])).sum()
    mean_away = (P.sum(axis=0) * np.arange(P.shape[1])).sum()
    assert mean_home == pytest.approx(1.5, abs=1e-3)
    assert mean_away == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("lambda3", [1e-4, 0.05, 0.2, 0.44])
def test_lambda3_transform_roundtrip(lambda3):
    assert _raw_to_lambda3(_lambda3_to_raw(lambda3)) == pytest.approx(lambda3, rel=1e-6)


@pytest.mark.parametrize("rho", [-0.9, -0.1, 0.0, 0.3, 0.8])
def test_rho_transform_roundtrip(rho):
    assert _raw_to_rho(_rho_to_raw(rho)) == pytest.approx(rho, abs=1e-6)


def test_lambda3_transform_stays_in_bounds():
    # (0, _LAMBDA3_MAX]: the upper cap is reached only at float saturation.
    for raw in [-50, -1, 0, 1, 50]:
        assert 0.0 < _raw_to_lambda3(raw) <= _LAMBDA3_MAX


def test_save_load_roundtrip(tmp_path):
    m = FittedModel(
        teams=["A", "B", "C"],
        attack={"A": 0.3, "B": -0.1, "C": -0.2},
        defense={"A": 0.2, "B": 0.0, "C": -0.2},
        intercept=0.05, home_att=0.15, home_def=0.18,
        lambda3=0.12, rho=-0.08, cfg=ModelConfig(),
        neg_loglik=123.4, n_matches=2000,
    )
    p = tmp_path / "m.json"
    m.save(p)
    back = FittedModel.load(p)
    assert back.teams == m.teams
    assert back.attack == m.attack and back.defense == m.defense
    assert back.home_att == m.home_att and back.home_def == m.home_def
    assert back.lambda3 == m.lambda3 and back.rho == m.rho
    assert back.cfg == m.cfg
    # rates must be identical after a round-trip (warm-start relies on this)
    assert back.rates("A", "B", neutral=False) == m.rates("A", "B", neutral=False)
