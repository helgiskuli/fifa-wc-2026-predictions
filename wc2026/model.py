"""Steps 2-3: bivariate-Poisson goal model with Dixon-Coles low-score
correction, fit by weighted MLE, plus the DC-corrected score matrix.

Parametrisation (standard Dixon-Coles, with a bivariate-Poisson shared
component on top):

    mu_home = exp(intercept + attack[home] - defense[away] + home_adv*[!neutral])
    mu_away = exp(intercept + attack[away] - defense[home])

mu_home / mu_away are the *marginal* expected goals. The joint law of
(home goals, away goals) is a bivariate Poisson with shared component
lambda3 (positive goal correlation), multiplied by the Dixon-Coles tau
adjustment on the four low-score cells {0-0, 1-0, 0-1, 1-1}.

Identifiability: attack and defense are constrained sum-to-zero, with the
overall level carried by `intercept`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .config import ModelConfig, PreprocessConfig
from .data import build_training_frame

_EPS = 1e-10


@dataclass
class FittedModel:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    intercept: float
    # Two-parameter home advantage (only applies when neutral == False):
    #   home_att boosts the home team's scoring,
    #   home_def suppresses the away team's scoring.
    # Splitting the effect lets the model capture "home teams win more"
    # via the home/away split without inflating total goals, which keeps
    # neutral-venue predictions (almost all of WC-2026) unbiased.
    home_att: float
    home_def: float
    lambda3: float          # bivariate-Poisson shared component
    rho: float              # Dixon-Coles low-score correction
    cfg: ModelConfig
    neg_loglik: float
    n_matches: int

    @property
    def home_adv(self) -> float:
        """Total home edge on the outcome (sum of the two home effects)."""
        return self.home_att + self.home_def

    # -- rates -------------------------------------------------------------
    def rates(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        """Marginal expected goals (mu_home, mu_away) for a fixture."""
        ah, dh = self.attack[home], self.defense[home]
        aa, da = self.attack[away], self.defense[away]
        h_att = self.home_att if not neutral else 0.0
        h_def = self.home_def if not neutral else 0.0
        mu_home = np.exp(self.intercept + ah - da + h_att)
        mu_away = np.exp(self.intercept + aa - dh - h_def)
        return float(mu_home), float(mu_away)

    # -- score matrix (step 3) --------------------------------------------
    def score_matrix(self, mu_home: float, mu_away: float) -> np.ndarray:
        """DC-corrected bivariate-Poisson P(i, j) over goals 0..max_goals.

        Rows = home goals, cols = away goals. Normalised to sum to 1 over
        the truncated grid."""
        return score_matrix(mu_home, mu_away, self.lambda3, self.rho,
                            self.cfg.max_goals)

    def fixture_matrix(self, home: str, away: str, neutral: bool) -> np.ndarray:
        mu_home, mu_away = self.rates(home, away, neutral)
        return self.score_matrix(mu_home, mu_away)

    def strength_table(self) -> pd.DataFrame:
        """Per-team attack/defense plus a single overall rating.

        A strong team has high attack (scores more) AND high defense
        (concedes less, since mu_conceded ~ exp(... - defense)), so the
        overall rating is attack + defense."""
        rows = [
            {"team": t, "attack": self.attack[t], "defense": self.defense[t],
             "overall": self.attack[t] + self.defense[t]}
            for t in self.teams
        ]
        return (pd.DataFrame(rows)
                .sort_values("overall", ascending=False)
                .reset_index(drop=True))

    # -- caching (JSON: portable, inspectable, version-stable) ------------
    def save(self, path: Path | str) -> None:
        """Persist the fitted model so it can be reloaded without refitting
        (or reused as a warm-start)."""
        data = {
            "schema": 1, "teams": self.teams,
            "attack": self.attack, "defense": self.defense,
            "intercept": self.intercept, "home_att": self.home_att,
            "home_def": self.home_def, "lambda3": self.lambda3, "rho": self.rho,
            "cfg": asdict(self.cfg),
            "neg_loglik": self.neg_loglik, "n_matches": self.n_matches,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "FittedModel":
        data = json.loads(Path(path).read_text())
        return cls(
            teams=list(data["teams"]),
            attack=dict(data["attack"]), defense=dict(data["defense"]),
            intercept=data["intercept"], home_att=data["home_att"],
            home_def=data["home_def"], lambda3=data["lambda3"], rho=data["rho"],
            cfg=ModelConfig(**data["cfg"]),
            neg_loglik=data["neg_loglik"], n_matches=data["n_matches"],
        )


# --------------------------------------------------------------------------
# Dixon-Coles tau + bivariate-Poisson pmf
# --------------------------------------------------------------------------
def _dc_tau(x, y, mu1, mu2, rho):
    """Dixon-Coles low-score adjustment, vectorised over matches."""
    tau = np.ones_like(mu1, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - mu1[m00] * mu2[m00] * rho
    tau[m01] = 1.0 + mu1[m01] * rho
    tau[m10] = 1.0 + mu2[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


def _bivpois_logpmf(x, y, mu1, mu2, lambda3):
    """log P(X=x, Y=y) for a bivariate Poisson with marginals mu1, mu2 and
    shared component lambda3. X = Y1 + Y3, Y = Y2 + Y3, with
    lambda1 = mu1 - lambda3, lambda2 = mu2 - lambda3."""
    lam1 = np.maximum(mu1 - lambda3, _EPS)
    lam2 = np.maximum(mu2 - lambda3, _EPS)
    base = (-(lam1 + lam2 + lambda3)
            + x * np.log(lam1) + y * np.log(lam2)
            - gammaln(x + 1) - gammaln(y + 1))
    # Sum_{k=0}^{min(x,y)} C(x,k) C(y,k) k! (lambda3/(lam1*lam2))^k
    if lambda3 <= _EPS:
        return base  # reduces to independent Poisson
    ratio = lambda3 / (lam1 * lam2)
    kmax = int(min(x.max(), y.max()))
    s = np.ones_like(mu1, dtype=float)  # k = 0 term
    for k in range(1, kmax + 1):
        mask = (x >= k) & (y >= k)
        if not mask.any():
            continue
        log_coeff = (gammaln(x[mask] + 1) - gammaln(k + 1) - gammaln(x[mask] - k + 1)
                     + gammaln(y[mask] + 1) - gammaln(k + 1) - gammaln(y[mask] - k + 1)
                     + gammaln(k + 1))
        s[mask] += np.exp(log_coeff + k * np.log(ratio[mask]))
    return base + np.log(s)


def score_matrix(mu_home: float, mu_away: float, lambda3: float, rho: float,
                 max_goals: int) -> np.ndarray:
    """DC-corrected bivariate-Poisson probability matrix over the grid."""
    n = max_goals + 1
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = ii.ravel().astype(float)
    y = jj.ravel().astype(float)
    mu1 = np.full(x.shape, mu_home, dtype=float)
    mu2 = np.full(x.shape, mu_away, dtype=float)
    logp = _bivpois_logpmf(x, y, mu1, mu2, lambda3)
    p = np.exp(logp) * np.maximum(_dc_tau(x, y, mu1, mu2, rho), _EPS)
    P = p.reshape(n, n)
    return P / P.sum()


# --------------------------------------------------------------------------
# Fit (step 2)
# --------------------------------------------------------------------------
def _pack_indices(teams: list[str]):
    return {t: i for i, t in enumerate(teams)}


def fit(df: pd.DataFrame, pre: PreprocessConfig, mcfg: ModelConfig,
        verbose: bool = True,
        warm_start: "FittedModel | None" = None) -> FittedModel:
    """Weighted-MLE fit of the bivariate-Poisson / Dixon-Coles model.

    If `warm_start` is given (e.g. the previous matchday's fit), the
    optimiser is seeded from its parameters instead of cold defaults. Since
    a new matchday only nudges strengths slightly, this converges in a
    fraction of the iterations. Teams absent from the warm-start (or absent
    now) are handled gracefully -- the layout is rebuilt by team name."""
    train = build_training_frame(df, pre)
    teams = sorted(set(train["home_team"]) | set(train["away_team"]))
    idx = _pack_indices(teams)
    n_teams = len(teams)

    h = train["home_team"].map(idx).to_numpy()
    a = train["away_team"].map(idx).to_numpy()
    x = train["home_score"].to_numpy(dtype=float)
    y = train["away_score"].to_numpy(dtype=float)
    not_neutral = (~train["neutral"].to_numpy()).astype(float)
    w = train["weight"].to_numpy(dtype=float)

    # Parameter vector layout:
    #   attack[0..n-2], defense[0..n-2]  (last team set by sum-to-zero)
    #   intercept, home_att, home_def, raw_lambda3, raw_rho
    n_free = n_teams - 1
    p_intercept = 2 * n_free
    p_hatt = p_intercept + 1
    p_hdef = p_hatt + 1
    p_l3 = p_hdef + 1
    p_rho = p_l3 + 1
    n_params = p_rho + 1

    def unpack(p):
        att = np.empty(n_teams)
        dfn = np.empty(n_teams)
        att[:n_free] = p[:n_free]
        att[-1] = -att[:n_free].sum()           # sum-to-zero
        dfn[:n_free] = p[n_free:2 * n_free]
        dfn[-1] = -dfn[:n_free].sum()
        intercept = p[p_intercept]
        home_att = p[p_hatt]                     # home scoring boost
        home_def = p[p_hdef]                     # away scoring suppression
        # lambda3 in [0, 0.45] via scaled sigmoid (keeps lam1,lam2 > 0 for
        # realistic international scoring rates); rho in (-1, 1) via tanh.
        lambda3 = 0.45 / (1.0 + np.exp(-p[p_l3]))
        rho = np.tanh(p[p_rho])
        return att, dfn, intercept, home_att, home_def, lambda3, rho

    def neg_loglik(p):
        att, dfn, intercept, home_att, home_def, lambda3, rho = unpack(p)
        mu_home = np.exp(intercept + att[h] - dfn[a] + home_att * not_neutral)
        mu_away = np.exp(intercept + att[a] - dfn[h] - home_def * not_neutral)
        logbp = _bivpois_logpmf(x, y, mu_home, mu_away, lambda3)
        tau = np.maximum(_dc_tau(x, y, mu_home, mu_away, rho), _EPS)
        ll = w * (logbp + np.log(tau))
        nll = -ll.sum()
        # Tiny ridge on strengths for numerical stability (≈0 by default).
        nll += mcfg.ridge * (np.dot(att, att) + np.dot(dfn, dfn))
        if not np.isfinite(nll):
            return 1e12
        return nll

    p0 = np.zeros(n_params)
    p0[p_intercept] = np.log(max(x.mean(), 0.1))  # sensible scoring level
    p0[p_hatt] = 0.20   # home scoring boost
    p0[p_hdef] = 0.10   # away scoring suppression
    p0[p_l3] = -2.0   # small lambda3
    p0[p_rho] = -0.1  # mild negative rho (typical for football)

    if warm_start is not None:
        # Seed per-team strengths by name (unknown teams stay at 0), and the
        # global params via the inverse of their constraining transforms.
        for i, t in enumerate(teams[:n_free]):
            p0[i] = warm_start.attack.get(t, 0.0)
            p0[n_free + i] = warm_start.defense.get(t, 0.0)
        p0[p_intercept] = warm_start.intercept
        p0[p_hatt] = warm_start.home_att
        p0[p_hdef] = warm_start.home_def
        s = min(max(warm_start.lambda3 / 0.45, 1e-6), 1 - 1e-6)
        p0[p_l3] = np.log(s / (1.0 - s))               # inverse scaled-sigmoid
        p0[p_rho] = np.arctanh(min(max(warm_start.rho, -0.999), 0.999))  # inverse tanh

    if verbose:
        start = "warm" if warm_start is not None else "cold"
        print(f"Fitting {n_teams} teams on {len(train)} weighted matches "
              f"({n_params} params, {start}-start)...")
    res = minimize(neg_loglik, p0, method="L-BFGS-B",
                   options={"maxiter": mcfg.max_iter, "maxfun": 1_000_000,
                            "ftol": 1e-9, "gtol": 1e-6})

    att, dfn, intercept, home_att, home_def, lambda3, rho = unpack(res.x)
    if verbose:
        print(f"  done: success={res.success}, nll={res.fun:.1f}, "
              f"home_att={home_att:.3f}, home_def={home_def:.3f}, "
              f"lambda3={lambda3:.3f}, rho={rho:.3f}")
        print(f"  optimiser: {res.message} (nit={res.nit}, nfev={res.nfev})")

    return FittedModel(
        teams=teams,
        attack={t: float(att[i]) for t, i in idx.items()},
        defense={t: float(dfn[i]) for t, i in idx.items()},
        intercept=float(intercept),
        home_att=float(home_att),
        home_def=float(home_def),
        lambda3=float(lambda3),
        rho=float(rho),
        cfg=mcfg,
        neg_loglik=float(res.fun),
        n_matches=len(train),
    )
