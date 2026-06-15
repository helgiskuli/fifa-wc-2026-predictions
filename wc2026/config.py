"""Configuration for the WC-2026 scoreline predictor.

Everything the kickoff calls a "param" lives here so the model stays
ratio-agnostic and easy to tune without touching the fitting code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class PreprocessConfig:
    """Controls the training-data window and match weighting."""

    # As-of date for time decay AND the training-window cutoff (only matches
    # with date <= as_of are used). Defaults to *today* so that re-running
    # mid-tournament automatically pulls in newly-added results and decays
    # them relative to now -- no constant to bump between rounds. Backtests
    # override this to the eve of a past tournament.
    as_of: date = field(default_factory=date.today)

    # Only fit on matches within this many years of `as_of`. The kickoff
    # asks for ~3-4 years; the most predictive part (2025-26 qualifiers,
    # Nations League, live WC) all falls inside a 4y window.
    window_years: float = 4.0

    # Per-competition-tier weights (multiplied with the time-decay weight).
    # Higher-tier games are more predictive of tournament football. Friendlies
    # are kept (not dropped): they are the main cross-confederation bridges --
    # dropping them cratered the 2022 backtest (48%->34% outcome acc.).
    # Tiers: major finals (WC + continental), qualifiers + Nations League,
    # friendlies, and other (minor regional cups). Ratios are validated by
    # the 2018/2022 backtests, NOT assumed: down-weighting qualifiers/NL
    # (the proposed 1/.5/.3/.3 tiering) was a wash-to-worse vs leaving the
    # competitive tiers flat (300 vs 302 EP / 128 matches), because qual/NL
    # is 51% of the corpus and cutting it loses signal. So only friendlies
    # are down-weighted by default; the other knobs are here to tune.
    weight_major: float = 1.0       # FIFA WC, Euro, Copa, AFCON, Asian Cup, Gold Cup
    weight_qual_nl: float = 1.0     # *_qualification, UEFA/CONCACAF Nations League
    weight_friendly: float = 0.3    # Friendly
    weight_other: float = 1.0       # minor regional cups (Gulf Cup, COSAFA, ...)

    # Exponential time-decay half-life in days. ~8 months by default.
    half_life_days: float = 240.0

    # Drop teams with fewer than this many matches in the window (their
    # attack/defense would be wild and can destabilise the fit). WC teams
    # all clear this comfortably.
    min_matches: int = 5


@dataclass(frozen=True)
class ModelConfig:
    """Controls the bivariate-Poisson / Dixon-Coles fit."""

    # Score grid for the prediction matrix: goals 0..max_goals inclusive.
    max_goals: int = 8

    # Tiny ridge penalty on attack/defense strengths for numerical
    # stability. The kickoff wants plain MLE, so this defaults to ~0;
    # raise it only if the optimiser produces extreme outliers.
    ridge: float = 1e-3

    # Optimiser settings.
    max_iter: int = 1200


@dataclass(frozen=True)
class ScoringConfig:
    """Office-pool scoring rule (additive components; an exact scoreline
    scores a + 2b + c = 6 under the defaults). Build everything
    ratio-agnostic."""

    a: float = 3.0   # points for a correct result (H / D / A)
    b: float = 1.0   # points per correct team goal count (max 2b for both)
    c: float = 1.0   # points for the correct signed goal difference
