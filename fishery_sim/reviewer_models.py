"""Reviewer model error for experiment R2, Part B (Claude audit, October 2026).

Protocol: notes/claude_audit_20261005/studies/R2_robustness_and_reviewer_model/protocol.md

The environment always runs on its TRUE configuration. A reviewer predicts with
its OWN copy (``dataclasses.replace``), which may be wrong by a fixed factor,
learned from observed transitions, or (``allee``) right in its parameters but
wrong in form because the environment has critical depensation.

Reference labels (safe / risky) are always computed with the true model.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .calibrated_oversight import fishery_next, fishery_reference, fishery_residual
from .config import FisheryConfig
from .fishery_oversight import FisherySnapshot
from .harvest import HarvestCommonsConfig

ALLEE_A = 20.0
MIN_LEARN_STEPS = 5
HARVEST_CONDITIONS = ("exact", "noise_low", "noise_high", "regen_low", "regen_high", "learned")
FISHERY_CONDITIONS = ("exact", "regen_low", "regen_high", "K_low", "K_high", "learned", "allee")


def game_of(cfg) -> str:
    if isinstance(cfg, HarvestCommonsConfig):
        return "harvest"
    if isinstance(cfg, FisheryConfig):
        return "fishery"
    raise TypeError(f"Unknown configuration type {type(cfg).__name__}")


def _cap_regen(value: float) -> float:
    """Regrowth is capped at 1 (the environments require regen_rate <= 1)."""
    return min(1.0, float(value))


def reviewer_config(true_cfg, condition: str):
    """Return the reviewer's own copy of the configuration under ``condition``.

    The true configuration object is never modified. For ``learned`` this is the
    optimistic starting model used before 5 steps have been seen.
    """
    game = game_of(true_cfg)
    if game == "harvest":
        if condition not in HARVEST_CONDITIONS:
            raise ValueError(f"{condition} is not a Harvest condition")
        changes = dict(
            exact={},
            noise_low=dict(weather_noise_std=true_cfg.weather_noise_std * 0.5),
            noise_high=dict(weather_noise_std=true_cfg.weather_noise_std * 2.0),
            regen_low=dict(regen_rate=true_cfg.regen_rate * 0.75),
            regen_high=dict(regen_rate=_cap_regen(true_cfg.regen_rate * 1.25)),
            learned=dict(regen_rate=_cap_regen(true_cfg.regen_rate * 1.25),
                         weather_noise_std=true_cfg.weather_noise_std * 0.5),
        )[condition]
    else:
        if condition not in FISHERY_CONDITIONS:
            raise ValueError(f"{condition} is not a Fishery condition")
        changes = dict(
            exact={},
            allee={},  # the reviewer keeps the (correctly parameterised) logistic model
            regen_low=dict(regen_rate=true_cfg.regen_rate * 0.75),
            regen_high=dict(regen_rate=_cap_regen(true_cfg.regen_rate * 1.25)),
            K_low=dict(stock_max=true_cfg.stock_max * 0.75),
            K_high=dict(stock_max=true_cfg.stock_max * 1.25),
            learned=dict(regen_rate=_cap_regen(true_cfg.regen_rate * 1.25),
                         stock_max=true_cfg.stock_max * 1.25),
        )[condition]
    return replace(true_cfg, **changes)


def env_allee(condition: str) -> float:
    """Critical-depensation level A used by the TRUE Fishery environment."""
    return ALLEE_A if condition == "allee" else 0.0


# ----------------------------------------------------------------- estimators
def fit_logistic_rK(residual, next_stock):
    """Least squares on the logistic form S' - R = r R - (r/K) R^2 (no intercept).

    Returns (r, K) or None when the data do not identify a logistic curve
    (fewer than two distinct positive R, or r <= 0, or no negative curvature).
    """
    R = np.asarray(residual, float)
    S = np.asarray(next_stock, float)
    m = np.isfinite(R) & np.isfinite(S) & (R > 0)
    R, S = R[m], S[m]
    if len(R) < 2 or np.ptp(R) <= 1e-9 * max(1.0, float(R.max())):
        return None
    X = np.column_stack([R, R * R])
    (a, b), *_ = np.linalg.lstsq(X, S - R, rcond=None)
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b >= 0:
        return None
    return float(a), float(-a / b)


def fit_logistic_r(remaining, next_health, patch_max):
    """Least squares for r in H' - R = r R (1 - R/P) + noise, with P known.

    Returns (r, sigma), sigma being the residual standard deviation (ddof=1),
    or None if the data carry no information about r.
    """
    R = np.asarray(remaining, float).ravel()
    H = np.asarray(next_health, float).ravel()
    m = np.isfinite(R) & np.isfinite(H)
    R, H = R[m], H[m]
    g = R * (1.0 - R / patch_max)
    y = H - R
    gg = float(np.dot(g, g))
    if len(R) < 2 or gg <= 0:
        return None
    r = float(np.dot(g, y)) / gg
    resid = y - r * g
    sigma = float(np.sqrt(np.dot(resid, resid) / (len(R) - 1)))
    if not (np.isfinite(r) and np.isfinite(sigma)):
        return None
    return r, sigma


# ----------------------------------------------------------------- Harvest pieces
def harvest_remaining(cfg: HarvestCommonsConfig, health, executed):
    """Patch health after harvest and neighbour spill-over, before regrowth and weather.

    Uses only parts of the model that are not misspecified in any R2 condition.
    """
    health = np.asarray(health, float)
    taken = np.minimum(np.asarray(executed, float) * cfg.max_harvest_per_agent, health)
    excess = np.maximum(0.0, taken - cfg.sustainable_harvest_frac * cfg.max_harvest_per_agent)
    return np.maximum(0.0, health - taken - cfg.neighbor_externality * (np.roll(excess, 1) + np.roll(excess, -1)))


# ----------------------------------------------------------------- Fishery pieces
def fishery_step(cfg: FisheryConfig, state: FisherySnapshot, requests, allee: float = 0.0):
    """One step of the TRUE Fishery, replicating FisheryEnv.step (governance off).

    With ``allee == 0`` this equals ``fishery_oversight.transition`` exactly:
    logistic growth floored at 0, clipped to [0, K], harvest scaled down when the
    total exceeds the stock, collapse after ``collapse_patience`` consecutive
    steps below the threshold (stock then set to 0, and nothing more happens).
    With ``allee == A > 0`` the growth is r R (1 - R/K)(R/A - 1), which is
    negative below A, so it is NOT floored at 0.
    Returns (next state, payoffs, realized total harvest).
    """
    requests = np.asarray(requests, dtype=float)
    if (requests.shape != (cfg.n_agents,) or not np.isfinite(requests).all()
            or np.any((requests < 0) | (requests > 1))):
        raise ValueError("Expected normalized requests")
    if cfg.monitoring_prob or cfg.quota_fraction or cfg.obs_noise_std:
        raise ValueError("Matched Fishery disables legacy governance and observation noise")
    if not 0 <= cfg.regen_rate <= 1:
        raise ValueError("Monotone accounting bounds require regen_rate <= 1")
    if allee < 0:
        raise ValueError("Allee level must be nonnegative")
    if state.collapsed:
        return FisherySnapshot(state.stock, state.below_count, True), np.zeros(cfg.n_agents), 0.0
    max_h = float(cfg.max_harvest_per_agent)
    stock = float(state.stock)
    K = float(cfg.stock_max)
    harvests = np.clip(requests * cfg.max_harvest_per_agent, 0.0, max_h).copy()
    total = harvests.sum()
    if total > stock and total > 0:
        harvests *= stock / total
        total = harvests.sum()
    payoffs = np.maximum(0.0, harvests - np.zeros(cfg.n_agents))
    remaining = max(0.0, stock - total)
    growth = float(cfg.regen_rate) * remaining * (1.0 - remaining / K)
    if allee > 0:
        new = float(np.clip(remaining + growth * (remaining / allee - 1.0), 0.0, K))
    else:
        new = float(np.clip(remaining + max(0.0, growth), 0.0, K))
    below = state.below_count + 1 if new < float(cfg.collapse_threshold) else 0
    collapsed = below >= int(cfg.collapse_patience)
    if collapsed:
        new = 0.0
    return FisherySnapshot(new, below, collapsed), payoffs, float(total)


def fishery_true_next(cfg, stock, total, allee: float = 0.0):
    """True one-step projection used for labels (no collapse bookkeeping, as in R1)."""
    if allee <= 0:
        return fishery_next(cfg, stock, total)
    r = fishery_residual(stock, total)
    return min(cfg.stock_max, max(0.0, r + cfg.regen_rate * r * (1 - r / cfg.stock_max) * (r / allee - 1)))


def fishery_true_reference(cfg, stock, executed, target, allee: float = 0.0):
    """Label of an executed joint action under the TRUE model (equals R1's with allee=0)."""
    if allee <= 0:
        return fishery_reference(cfg, stock, executed, target)
    total = float(np.asarray(executed, float).sum() * cfg.max_harvest_per_agent)
    if target == "one_step":
        ok = fishery_true_next(cfg, stock, total, allee) >= cfg.collapse_threshold
    elif target == "msy":
        ok = fishery_residual(stock, total) >= cfg.stock_max / 2
    else:
        raise ValueError(f"Unknown target {target}")
    return dict(risk=0.0 if ok else 1.0, label="safe" if ok else "risky")


# ----------------------------------------------------------------- reviewer model state
class ReviewerModel:
    """The reviewer's belief about the game. Fixed except under ``learned``.

    ``learned`` re-estimates after every observed step, once at least
    MIN_LEARN_STEPS steps have been seen, from ALL transitions seen so far:
    - Fishery: r and K by least squares on the logistic form (transitions that
      end in the environment's collapse reset are not logistic and are skipped);
    - Harvest: r by least squares on the logistic form (patch_max known) and the
      weather noise as the residual standard deviation.
    If a fit is not identified the previous model is kept. r is capped at 1.
    """

    def __init__(self, true_cfg, condition: str):
        self.true_cfg, self.condition, self.game = true_cfg, condition, game_of(true_cfg)
        self.cfg = reviewer_config(true_cfg, condition)
        self.steps = 0
        self.R, self.S = [], []

    def config(self):
        return self.cfg

    def params(self):
        if self.game == "harvest":
            return dict(regen_rate=self.cfg.regen_rate, weather_noise_std=self.cfg.weather_noise_std)
        return dict(regen_rate=self.cfg.regen_rate, stock_max=self.cfg.stock_max)

    def observe_harvest(self, health, executed, next_health):
        if self.condition != "learned":
            return
        self.R.extend(harvest_remaining(self.cfg, health, executed).tolist())
        self.S.extend(np.asarray(next_health, float).tolist())
        self.steps += 1
        if self.steps >= MIN_LEARN_STEPS:
            fit = fit_logistic_r(self.R, self.S, self.true_cfg.patch_max)
            if fit is not None:
                r, sigma = fit
                self.cfg = replace(self.cfg, regen_rate=float(np.clip(r, 0.0, 1.0)), weather_noise_std=sigma)

    def observe_fishery(self, stock, executed, next_state: FisherySnapshot):
        if self.condition != "learned":
            return
        self.steps += 1
        if not next_state.collapsed:
            total = float(np.asarray(executed, float).sum() * self.true_cfg.max_harvest_per_agent)
            self.R.append(fishery_residual(stock, total))
            self.S.append(float(next_state.stock))
        if self.steps >= MIN_LEARN_STEPS:
            fit = fit_logistic_rK(self.R, self.S)
            if fit is not None:
                r, K = fit
                self.cfg = replace(self.cfg, regen_rate=_cap_regen(r), stock_max=K)
