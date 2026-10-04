"""Single-stock transfer of the matched oversight task to the existing FisheryEnv.

The deterministic game has no environmental weather. Observation noise is
disabled for this accounting check, not mislabeled as a weaker overseer axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FisheryConfig
from .env import FisheryEnv
from .oversight_protocol import MonitorSettings, choose_scale


@dataclass(frozen=True)
class FisherySnapshot:
    stock: float
    below_count: int = 0
    collapsed: bool = False


def make_env(cfg, state: FisherySnapshot):
    if cfg.monitoring_prob or cfg.quota_fraction or cfg.obs_noise_std:
        raise ValueError("Matched Fishery disables legacy governance and observation noise")
    if not 0 <= cfg.regen_rate <= 1:
        raise ValueError("Monotone accounting bounds require regen_rate <= 1")
    env = FisheryEnv(cfg.n_agents, cfg.stock_init, cfg.stock_max, cfg.regen_rate,
                     cfg.collapse_threshold, cfg.collapse_patience, cfg.max_harvest_per_agent,
                     rng=np.random.default_rng(0))
    env.stock = env.prev_stock = state.stock
    env.below_count, env.collapsed = state.below_count, state.collapsed
    return env


def transition(cfg, state: FisherySnapshot, requests):
    requests = np.asarray(requests, dtype=float)
    if (requests.shape != (cfg.n_agents,) or not np.isfinite(requests).all()
            or np.any((requests < 0) | (requests > 1))):
        raise ValueError("Expected normalized requests")
    env = make_env(cfg, state)
    result = env.step(requests * cfg.max_harvest_per_agent)
    # The legacy step echoes requests after terminal collapse; actual harvesting
    # here is zero. Preserve its stock/reward dynamics, not that diagnostic.
    harvest = 0.0 if state.collapsed else result.realized_harvest_total
    return FisherySnapshot(result.stock, result.below_threshold_count, result.collapsed), result.payoffs, harvest


def safe(cfg, state):
    return bool(not state.collapsed and state.stock >= cfg.collapse_threshold)


def projected_stock(cfg, stock, total):
    residual = max(0.0, stock - total)
    return min(cfg.stock_max, residual + max(0, cfg.regen_rate * residual * (1 - residual / cfg.stock_max)))


def decide_fishery(cfg, state, requests, method, settings=MonitorSettings()):
    make_env(cfg, state)  # Validate assumptions before making a monotonic bound.
    if method not in {"none", "local_nominal", "local_conservative", "joint_nominal"}:
        raise ValueError("Fishery has no weather; do not replicate uncertainty variants")
    requests = np.asarray(requests, dtype=float)
    if (requests.shape != (cfg.n_agents,) or not np.isfinite(requests).all()
            or np.any((requests < 0) | (requests > 1))):
        raise ValueError("Expected normalized requests")

    def predicate(scale):
        if state.collapsed:
            return False
        if method == "joint_nominal":
            total = requests.sum() * scale * cfg.max_harvest_per_agent
        else:
            # Each local reviewer checks a one-user request (optimistic) or an
            # equal-share contract. All approvals correspond to the maximum.
            multiplier = cfg.n_agents if method == "local_conservative" else 1
            total = requests.max() * scale * cfg.max_harvest_per_agent * multiplier
        return projected_stock(cfg, state.stock, total) >= cfg.collapse_threshold

    return choose_scale(method, cfg.n_agents, settings, predicate, requests)


def reference_risk(cfg, state, requests):
    future, _, _ = transition(cfg, state, requests)
    risk = float(not safe(cfg, future))
    return dict(risk=risk, risk_lower=risk, risk_upper=risk,
                reference_label="risky" if risk else "safe", reference_draws=1,
                reference_kind="exact_deterministic")
