"""Winner-take-all pool simulator: how contrarian should the sheet be?

Loads the cached fitted model (no refit), then Monte-Carlos a medium casual
pool. Each trial samples BOTH a full tournament outcome (from the model's
per-fixture score matrices) AND a fresh set of casual opponents' picks, scores
everyone under the office-pool rule, and decides the winner with a fair random
tiebreak. My entry is the EP-optimal sheet with K deliberate deviations layered
on (draws first, then mid-confidence underdogs). Sweeping K shows which
deviation budget maximises P(I finish first).

    uv run python -m scripts.pool_sim [pool_size] [n_sims]

ASSUMPTIONS (the load-bearing ones -- edit CASUAL_* to change the field model):
  * truth is sampled from the fitted model. Calibration showed the model
    OVER-rates favourites, so real upsets are MORE frequent than this -> the
    contrarian benefit reported here is conservative (a lower bound).
  * casual opponents pick the stronger side to win ~83% of the time, an upset
    ~12%, a draw ~5%, with round casual scorelines. Independent across
    opponents (the herd emerges from the shared favourite bias).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from wc2026 import (FittedModel, ScoringConfig, best_prediction, load_results,
                    upcoming_fixtures)
from wc2026.predict import _outcome_probs

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "model_cache.json"
SEED = 20260616

# --- casual-field model (the assumption that drives the result) -----------
CASUAL_P_DRAW = 0.05      # P(a casual entrant picks a draw at all)
CASUAL_P_UPSET = 0.12     # P(picks the weaker side to win)
# favourite-side scoreline mix (round numbers casuals like)
FAV_SCORES = {(1, 0): 0.34, (2, 0): 0.22, (2, 1): 0.26, (3, 1): 0.10, (3, 0): 0.08}
DRAW_SCORES = {(1, 1): 0.60, (0, 0): 0.25, (2, 2): 0.15}

# sharp opponents converge on the model's EP pick, with mild perturbation.
SHARP_P_EXACT = 0.65      # P(plays the EP pick verbatim) -> heavy clustering

# DGP="real": the backtest showed the model OVER-rates favourites (2022:
# favourite predicted 57% vs 48% actual). Move this fraction of favourite-win
# mass to the underdog-win block (draws were ~calibrated, so leave them).
FAV_SHRINK = 0.15

# --- deviation candidate thresholds ---------------------------------------
DRAW_FLOOR = 0.27         # treat as a draw-pick candidate if P(draw) >= this
MIDCONF_LO, MIDCONF_HI = 0.45, 0.60   # favourite-prob band for underdog picks


def score_vec(ph: int, pa: int, ah: np.ndarray, aa: np.ndarray,
              cfg: ScoringConfig) -> np.ndarray:
    """Office-pool points for a fixed pick (ph,pa) vs arrays of actuals."""
    psign = (ph > pa) - (ph < pa)
    asign = np.sign(ah - aa)
    pts = cfg.a * (psign == asign)
    pts = pts + cfg.b * (ah == ph) + cfg.b * (aa == pa)
    pts = pts + cfg.c * ((ph - pa) == (ah - aa))
    return pts.astype(float)


def sample_from_mix(rng, mix: dict, size: int):
    keys = list(mix.keys())
    p = np.array(list(mix.values()), dtype=float)
    p /= p.sum()
    idx = rng.choice(len(keys), size=size, p=p)
    arr = np.array(keys)
    return arr[idx, 0], arr[idx, 1]


def recalibrate(P: np.ndarray, fav_home: bool) -> np.ndarray:
    """DGP='real': deflate the favourite-win block by FAV_SHRINK, move that
    mass to the underdog-win block (draws untouched). Encodes the backtest's
    favourite-overconfidence finding so 'truth' is more upset-prone than the
    model believes -- the regime where contrarian picks can pay."""
    n = P.shape[0]
    ii, jj = np.indices((n, n))
    home_blk, away_blk = ii > jj, ii < jj
    fav_blk = home_blk if fav_home else away_blk
    und_blk = away_blk if fav_home else home_blk
    mfav, mund = P[fav_blk].sum(), P[und_blk].sum()
    if mund <= 0:
        return P
    moved = FAV_SHRINK * mfav
    Q = P.copy()
    Q[fav_blk] *= (mfav - moved) / mfav
    Q[und_blk] *= (mund + moved) / mund
    return Q / Q.sum()


def main(pool_size: int = 20, n_sims: int = 20000,
         dgp: str = "model", field: str = "casual") -> None:
    cfg = ScoringConfig()
    rng = np.random.default_rng(SEED)
    if not CACHE.exists():
        sys.exit("no model_cache.json -- run scripts.run_schedule first")
    model = FittedModel.load(CACHE)
    df = load_results()
    fx = upcoming_fixtures(df)

    fixtures = []
    for _, r in fx.iterrows():
        h, a, neu = r.home_team, r.away_team, bool(r.neutral)
        if h not in model.attack or a not in model.attack:
            continue
        P = model.fixture_matrix(h, a, neu)
        n = P.shape[0]
        pH, pD, pA = _outcome_probs(P)
        mu_h, mu_a = model.rates(h, a, neu)
        fav_home = mu_h >= mu_a
        fav_prob = pH if fav_home else pA
        pred = best_prediction(P, cfg)

        # contrarian picks for this fixture
        diag = np.array([P[d, d] for d in range(n)])
        dd = int(diag.argmax())                  # modal draw scoreline (d,d)
        ii, jj = np.indices((n, n))
        und_mask = (ii > jj) if not fav_home else (ii < jj)  # underdog-win cells
        und_cell = np.where(und_mask, P, -1).argmax()
        ui, uj = divmod(int(und_cell), n)

        P_truth = recalibrate(P, fav_home) if dgp == "real" else P
        fixtures.append({
            "h": h, "a": a, "P": P, "P_truth": P_truth, "pD": pD,
            "fav_home": fav_home, "fav_prob": fav_prob,
            "ep": (pred.home_goals, pred.away_goals),
            "draw_pick": (dd, dd), "und_pick": (ui, uj),
        })

    F = len(fixtures)
    O = pool_size - 1

    # --- sample one tournament outcome per trial, per fixture -------------
    truth_h = np.empty((F, n_sims), dtype=int)
    truth_a = np.empty((F, n_sims), dtype=int)
    for k, fxd in enumerate(fixtures):
        Pt = fxd["P_truth"]
        n = Pt.shape[0]
        cell = rng.choice(n * n, size=n_sims, p=Pt.ravel())
        truth_h[k], truth_a[k] = np.divmod(cell, n)

    # --- opponents: (n_sims, O) picks per fixture, scored vs truth --------
    opp_tot = np.zeros((n_sims, O), dtype=float)
    for k, fxd in enumerate(fixtures):
        P = fxd["P"]
        n = P.shape[0]
        fav_home = fxd["fav_home"]
        m = n_sims * O
        if field == "casual":
            u = rng.random(m)
            oh = np.empty(m, dtype=int)
            oa = np.empty(m, dtype=int)
            is_draw = u < CASUAL_P_DRAW
            is_upset = (u >= CASUAL_P_DRAW) & (u < CASUAL_P_DRAW + CASUAL_P_UPSET)
            is_fav = ~(is_draw | is_upset)
            fh, fa = sample_from_mix(rng, FAV_SCORES, m)   # favourite-side score
            if fav_home:
                oh[is_fav], oa[is_fav] = fh[is_fav], fa[is_fav]
                oh[is_upset], oa[is_upset] = 0, 1          # away upset 0-1
            else:
                oh[is_fav], oa[is_fav] = fa[is_fav], fh[is_fav]
                oh[is_upset], oa[is_upset] = 1, 0          # home upset 1-0
            dh, da = sample_from_mix(rng, DRAW_SCORES, m)
            oh[is_draw], oa[is_draw] = dh[is_draw], da[is_draw]
        else:  # sharp: cluster on the EP pick, perturb one side by +-1
            eh, ea = fxd["ep"]
            oh = np.full(m, eh, dtype=int)
            oa = np.full(m, ea, dtype=int)
            pert = rng.random(m) >= SHARP_P_EXACT
            side = rng.random(m) < 0.5
            delta = rng.choice([-1, 1], size=m)
            oh[pert & side] = np.maximum(0, oh[pert & side] + delta[pert & side])
            oa[pert & ~side] = np.maximum(0, oa[pert & ~side] + delta[pert & ~side])

        oh = oh.reshape(n_sims, O)
        oa = oa.reshape(n_sims, O)
        th = truth_h[k][:, None]
        ta = truth_a[k][:, None]
        psign = np.sign(oh - oa)
        asign = np.sign(th - ta)
        opp_tot += (cfg.a * (psign == asign) + cfg.b * (oh == th)
                    + cfg.b * (oa == ta) + cfg.c * ((oh - oa) == (th - ta)))

    opp_max = opp_tot.max(axis=1)

    # --- my base (EP) score per trial, plus per-fixture deltas ------------
    base = np.zeros(n_sims, dtype=float)
    for k, fxd in enumerate(fixtures):
        ph, pa = fxd["ep"]
        base += score_vec(ph, pa, truth_h[k], truth_a[k], cfg)

    # deviation priority: draw candidates (by P(draw) desc), then mid-conf
    # underdog candidates (most coin-flip first)
    draw_cands = sorted(
        [k for k, f in enumerate(fixtures) if f["pD"] >= DRAW_FLOOR],
        key=lambda k: -fixtures[k]["pD"])
    und_cands = sorted(
        [k for k, f in enumerate(fixtures)
         if MIDCONF_LO <= f["fav_prob"] <= MIDCONF_HI and k not in draw_cands],
        key=lambda k: abs(fixtures[k]["fav_prob"] - 0.5))
    priority = [("D", k) for k in draw_cands] + [("U", k) for k in und_cands]

    def win_prob(my_tot: np.ndarray) -> tuple[float, float]:
        """P(strictly first) and P(win under fair random tiebreak)."""
        top = np.maximum(my_tot, opp_max)
        i_am_top = my_tot >= top
        strict = float(np.mean(my_tot > opp_max))
        n_opp_at_top = (opp_tot == top[:, None]).sum(axis=1)
        winners = n_opp_at_top + i_am_top
        share = np.where(i_am_top, 1.0 / winners, 0.0)
        return strict, float(np.mean(share))

    print(f"\n=== winner-take-all pool sim: {pool_size} entrants "
          f"({O} {field} opponents), {F} fixtures, {n_sims} trials ===")
    print(f"DGP={dgp} (truth source), field={field}")
    print(f"deviation pool: {len(draw_cands)} draw-likely + {len(und_cands)} "
          f"mid-confidence underdog games available\n")

    fair_chance = 1.0 / pool_size
    print(f"{'K':>3} {'added':<26} {'my pts (mean)':>13} "
          f"{'P(strict 1st)':>14} {'P(win,tiebrk)':>14} {'vs fair':>9}")
    cur = base.copy()
    results = []
    for K in range(0, len(priority) + 1):
        if K > 0:
            typ, k = priority[K - 1]
            ph, pa = fixtures[k][("draw_pick" if typ == "D" else "und_pick")]
            cur = cur - score_vec(*fixtures[k]["ep"], truth_h[k], truth_a[k], cfg)
            cur = cur + score_vec(ph, pa, truth_h[k], truth_a[k], cfg)
            added = f'{typ} {fixtures[k]["h"][:11]}-{fixtures[k]["a"][:11]}'
        else:
            added = "(pure EP-optimal)"
        strict, fair = win_prob(cur)
        results.append((K, fair))
        print(f"{K:>3} {added:<26} {cur.mean():>13.2f} "
              f"{strict:>13.1%} {fair:>13.1%} {fair/fair_chance:>8.2f}x")

    bestK, bestp = max(results, key=lambda t: t[1])
    print(f"\noptimum: K={bestK} deviations -> P(win) {bestp:.1%} "
          f"({bestp/fair_chance:.2f}x a fair 1/{pool_size} share)")


if __name__ == "__main__":
    ps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    dgp = sys.argv[3] if len(sys.argv) > 3 else "model"
    field = sys.argv[4] if len(sys.argv) > 4 else "casual"
    main(ps, ns, dgp, field)
