"""Two-reagent river: a commons game where harm comes from a combination of actions (C1).

Claude audit, October 2026. Protocol:
notes/claude_audit_20261005/studies/C1_compositional_harm/protocol.md

Game
----
6 agents; agents 0-2 discharge reagent X, agents 3-5 discharge reagent Y.
Requests p_i in [0, 1]. Each step, with X = sum of executed X discharges and
Y = sum of executed Y discharges:

  compositional game ("comp"):  D = kappa * X * Y + lam * (X + Y)
  additive control   ("add"):   D = lam_add * (X + Y)
  Q' = clip(Q - D + r Q (1 - Q/100) + eps, 0, 100),  eps ~ N(0, sigma^2)
  a step is unsafe if Q' < 30.

Clipping never changes whether Q' < 30, so the one-step risk of an action is
P(Q - D + g(Q) + eps < 30) = Phi((30 - Q + D - g(Q)) / sigma).

Reviewers choose one shared cut from SCALES with a 5% chance constraint
(estimated with Monte Carlo weather draws through their own prediction), as
in R1, except `quota`, which caps each agent at S/6 and does not use the
shared cut (see `quota_executed`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from fishery_sim.calibrated_oversight import SCALES, wilson

N_AGENTS = 6
X_IDX = (0, 1, 2)
Y_IDX = (3, 4, 5)
IS_X = np.array([True, True, True, False, False, False])
Q_INIT = 80.0
Q_UNSAFE = 30.0
Q_MAX = 100.0
P_MAX = 1.0
TOL = 0.05
Z95 = 1.6448536269514722  # one-sided 95% normal quantile, used only for the lam_add matching rule
REVIEWERS = ("joint", "quota", "local_optimistic", "local_bounded")
GAMES = ("comp", "add")

# Request range sets considered in the calibration (Amendment 1 picks one).
# "fishery": the Fishery ranges (claude_oversight_common.fishery_setup), thresholds U(35, 70).
# "wide" / "wide_low": a wider gap between low and high requests, thresholds U(35, 75).
RANGE_SETS = {
    "fishery": dict(normal=((0.05, 0.20), (0.20, 0.40)), stress=((0.45, 0.65), (0.75, 0.95)), thresholds=(35.0, 70.0)),
    "wide": dict(normal=((0.00, 0.10), (0.35, 0.65)), stress=((0.15, 0.35), (0.75, 0.95)), thresholds=(35.0, 75.0)),
    "wide_low": dict(normal=((0.00, 0.10), (0.25, 0.50)), stress=((0.10, 0.30), (0.70, 0.95)), thresholds=(35.0, 75.0)),
}

# Frozen by Amendment 1 (calibration) in protocol 21. lam = LAM_RATIO * kappa (fixed before calibration).
LAM_RATIO = 0.05
FROZEN = None  # Params(game, kappa, lam, lam_add, r, sigma), set after calibration
FROZEN_RANGES = None


@dataclass(frozen=True)
class Params:
    game: str
    kappa: float
    lam: float
    lam_add: float
    r: float
    sigma: float

    def as_dict(self):
        return dict(game=self.game, kappa=self.kappa, lam=self.lam, lam_add=self.lam_add, r=self.r, sigma=self.sigma)


# ------------------------------------------------------------------ dynamics
def regrowth(r, q):
    return r * q * (1.0 - q / Q_MAX)


def damage(P: Params, x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if P.game == "comp":
        return P.kappa * x * y + P.lam * (x + y)
    if P.game == "add":
        return P.lam_add * (x + y)
    raise ValueError(P.game)


def totals(p):
    p = np.asarray(p, float)
    return float(p[list(X_IDX)].sum()), float(p[list(Y_IDX)].sum())


def damage_of(P: Params, p):
    return float(damage(P, *totals(p)))


def next_q(P: Params, q, p, eps):
    return float(np.clip(q - damage_of(P, p) + regrowth(P.r, q) + eps, 0.0, Q_MAX))


def exact_risk(P: Params, q, p):
    """Exact one-step risk P(Q' < 30) under Gaussian weather."""
    margin = q - damage_of(P, p) + regrowth(P.r, q) - Q_UNSAFE  # nominal Q' - 30
    if P.sigma <= 0:
        return float(margin < 0)
    return 0.5 * math.erfc(margin / (P.sigma * math.sqrt(2.0)))


# ------------------------------------------------------------------ typical mix
def total_for_damage(P: Params, d):
    """Largest total T (split equally between X and Y) with damage(T/2, T/2) <= d; 0 if d <= 0; capped at 6."""
    if d <= 0:
        return 0.0
    if P.game == "add":
        t = d / P.lam_add
    elif P.kappa == 0:
        t = d / P.lam if P.lam > 0 else float("inf")
    else:  # kappa T^2 / 4 + lam T - d = 0
        a, b = P.kappa / 4.0, P.lam
        t = (-b + math.sqrt(b * b + 4 * a * d)) / (2 * a)
    return float(min(t, N_AGENTS * P_MAX))


def exact_headroom(r, sigma, q):
    """Damage that keeps one-step risk at exactly 5% (analytic normal quantile)."""
    return q + regrowth(r, q) - Q_UNSAFE - Z95 * sigma


def matching_lam_add(kappa, lam, r, sigma, q=Q_INIT):
    """lam_add so that the safe total at Q=80 (typical equal mix) equals the compositional game's."""
    h = exact_headroom(r, sigma, q)
    s = total_for_damage(Params("comp", kappa, lam, 1.0, r, sigma), h)
    return h / s


def make_params(game, kappa, r, sigma, lam=None):
    lam = LAM_RATIO * kappa if lam is None else lam
    return Params(game, float(kappa), float(lam), float(matching_lam_add(kappa, lam, r, sigma)), float(r), float(sigma))


# ------------------------------------------------------------------ reviewers
def reviewer_draws(P: Params, seed, draws=400):
    return np.random.default_rng(seed).normal(0.0, 1.0, size=draws) * P.sigma


def predicted_damage(P: Params, believed, scale, reviewer):
    """The damage a shared-cut reviewer checks (the worst of its checks)."""
    b = np.asarray(believed, float) * scale
    if reviewer == "joint":
        return damage_of(P, b)
    other_max = scale * P_MAX * len(Y_IDX)  # the other type at its maximum, after the shared cut
    checks = []
    for i in range(N_AGENTS):
        own = b[i]
        if reviewer == "local_optimistic":
            x, y = (own, 0.0) if IS_X[i] else (0.0, own)
        elif reviewer == "local_bounded":
            x, y = (own, other_max) if IS_X[i] else (other_max, own)
        else:
            raise ValueError(reviewer)
        checks.append(float(damage(P, x, y)))
    return max(checks)


def estimated_risk(P: Params, q, d, eps):
    return float(np.mean(q - d + regrowth(P.r, q) + eps < Q_UNSAFE))


def choose_scale(P: Params, q, believed, reviewer, eps, tol=TOL):
    """Largest scale in SCALES with estimated risk <= tol. Returns (scale, est_risk, candidates)."""
    risk = 1.0
    for count, s in enumerate(SCALES, start=1):
        risk = estimated_risk(P, q, predicted_damage(P, believed, s, reviewer), eps)
        if risk <= tol:
            return s, risk, count
    return 0.0, risk, len(SCALES)


def empirical_headroom(P: Params, q, eps, tol=TOL):
    """Largest damage d with estimated risk mean(q - d + g + eps < 30) <= tol, from the same draws."""
    e = np.sort(np.asarray(eps, float))
    m = int(math.floor(tol * len(e) + 1e-9))
    if m >= len(e):
        return float("inf")
    return float(q + regrowth(P.r, q) - Q_UNSAFE + e[m])


def quota_cap(P: Params, q, eps, tol=TOL):
    """Per-agent quota S/6, S = total safe at the typical (equal) mix at the current Q (chance constraint, same draws)."""
    return total_for_damage(P, empirical_headroom(P, q, eps, tol)) / N_AGENTS


def review(P: Params, q, believed, true_requests, reviewer, eps):
    """Apply a reviewer. Returns dict(executed, scale, approved, est_risk)."""
    req = np.asarray(true_requests, float)
    if reviewer == "none":
        return dict(executed=req.copy(), scale=1.0, approved=1, est_risk=None)
    if reviewer == "quota":
        cap = quota_cap(P, q, eps)
        b = np.asarray(believed, float)
        cut = b > cap  # the quota is applied to what the reviewer believes each agent requests
        executed = np.where(cut, np.minimum(req, cap), req)
        tot = req.sum()
        return dict(executed=executed, scale=float(executed.sum() / tot) if tot > 0 else 1.0,
                    approved=int(not cut.any()), est_risk=None, cap=cap)
    s, est, cand = choose_scale(P, q, believed, reviewer, eps)
    return dict(executed=req * s, scale=s, approved=int(s == 1.0), est_risk=est, candidates=cand)


# ------------------------------------------------------------------ labels
def reference(P: Params, q, executed, seed, draws=2000, tol=TOL):
    """Monte Carlo reference label with Wilson 95% interval at the 5% threshold (R1 convention)."""
    eps = np.random.default_rng(seed).normal(0.0, 1.0, size=draws) * P.sigma
    nominal = q - damage_of(P, executed) + regrowth(P.r, q)
    failures = int((nominal + eps < Q_UNSAFE).sum())
    lo, hi = wilson(failures, draws)
    label = "safe" if hi <= tol else "risky" if lo > tol else "unresolved"
    return dict(risk=failures / draws, risk_lower=lo, risk_upper=hi, label=label)


# ------------------------------------------------------------------ audits
def sensitivity(P: Params, believed):
    """dD/dp_i * p_i at the believed vector: how much agent i's report moves predicted damage."""
    b = np.asarray(believed, float)
    x, y = totals(b)
    if P.game == "comp":
        grad = np.where(IS_X, P.kappa * y + P.lam, P.kappa * x + P.lam)
    else:
        grad = np.full(N_AGENTS, P.lam_add)
    return grad * b


def targeted_agent(P: Params, believed):
    return int(np.argmax(sensitivity(P, believed)))  # ties: lowest index


def memory_belief(reports, records, audited=None, true_requests=None):
    """S3 Part D memory: unaudited agent with records -> min(1, r / (1 - mean d_hat)); audited -> truth."""
    b = np.asarray(reports, float).copy()
    for i, ds in records.items():
        if ds:
            b[i] = min(P_MAX, b[i] / (1.0 - float(np.mean(ds))))
    if audited is not None:
        b[audited] = true_requests[audited]
    return b


# ------------------------------------------------------------------ population
def make_population(context, pop_base, ranges=None):
    R = RANGE_SETS[ranges or FROZEN_RANGES]
    rng = np.random.default_rng(pop_base + context)
    stressed = [int(rng.integers(0, 3)), int(3 + rng.integers(0, 3))]  # one X and one Y stress agent
    kind = ["stress" if i in stressed else "normal" for i in range(N_AGENTS)]
    low = np.array([rng.uniform(*R[k][0]) for k in kind])
    high = np.array([rng.uniform(*R[k][1]) for k in kind])
    thresholds = rng.uniform(*R["thresholds"], N_AGENTS)
    return dict(low=low, high=high, thresholds=thresholds, stressed=stressed)


def policy_requests(pol, q):
    return np.where(q < pol["thresholds"], pol["low"], pol["high"])


def weather_shocks(context, weather_base, horizon):
    """Standard normal shocks per step (multiplied by sigma in use), shared by all arms of a context."""
    return np.random.default_rng(weather_base + context).normal(0.0, 1.0, size=horizon)


def frozen_params(game):
    if FROZEN is None:
        raise RuntimeError("C1 parameters not frozen yet (Amendment 1 missing)")
    return replace(FROZEN, game=game)


# ------------------------------------------------------------------ frozen (Amendment 1, calibration, 2026-10-05)
# Chosen by the calibration rule in run_c1_compositional_harm.calibrate on the 8 calibration contexts
# (results/runs/claude_c1_calibration_v1/calibration.json). lam = 0.05 kappa; lam_add from the matching rule.
FROZEN_RANGES = "wide_low"
FROZEN = make_params("comp", 32.0, 1.0, 3.0)
