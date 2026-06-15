"""WC-2026 office-pool scoreline predictor.

Bivariate-Poisson / Dixon-Coles goal model fit by weighted MLE, with an
expected-points-maximising scoreline pick for the office-pool scoring rule.
"""
from .config import ModelConfig, PreprocessConfig, ScoringConfig
from .data import build_training_frame, load_results, upcoming_fixtures
from .model import FittedModel, fit, score_matrix
from .predict import (Prediction, best_prediction, modal_prediction,
                      score_prediction)

__all__ = [
    "ModelConfig", "PreprocessConfig", "ScoringConfig",
    "load_results", "build_training_frame", "upcoming_fixtures",
    "fit", "score_matrix", "FittedModel",
    "best_prediction", "Prediction", "modal_prediction", "score_prediction",
]
