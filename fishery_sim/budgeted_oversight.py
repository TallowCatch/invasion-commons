"""One-step resource-limited reviewers with a common target and action menu."""
from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
from scipy.stats import norm

from .config import FisheryConfig
from .fishery_oversight import FisherySnapshot, make_env, projected_stock
from .harvest import HarvestCommonsConfig
from .oversight_protocol import (
    Decision, MonitorSettings, choose_scale, harvest_nominal_next, harvest_safe,
    local_patch_report, validate_harvest,
)

MODES = ("local_optimistic", "local_bounded", "joint")
DIAGNOSTIC_MODE = "local_coupled"


def inspection_order(n: int, identity: str) -> np.ndarray:
    if n < 1:
        raise ValueError("Population must be nonempty")
    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).permutation(n)


def mask_requests(requests, count: int, identity: str) -> np.ndarray:
    values = np.asarray(requests, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1)) or not 0 <= count <= len(values)):
        raise ValueError("Invalid requests or inspection count")
    visible = np.full(len(values), np.nan)
    indices = inspection_order(len(values), identity)[:count]
    visible[indices] = values[indices]
    return visible


def _check_visible(visible, n: int):
    values = np.asarray(visible, dtype=float)
    if (values.shape != (n,) or np.isinf(values).any()
            or np.any((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1))):
        raise ValueError("Expected visible request or NaN for each agent")
    return values, int(np.isfinite(values).sum())


def _decision(mode, n, visible, settings, predicate):
    if mode not in (*MODES, DIAGNOSTIC_MODE):
        raise ValueError(f"Unknown reviewer mode: {mode}")
    observed, k = _check_visible(visible, n)
    # choose_scale controls the identical repair menu and candidate limit.
    # The mode-specific prediction sees observed entries plus explicit bounds.
    alias = "joint_uncertain" if mode == "joint" else "local_conservative_uncertain"
    decision = choose_scale(alias, n, settings, predicate,
                            np.where(np.isfinite(observed), observed, 1.0))
    if mode == DIAGNOSTIC_MODE:
        # One patch/contribution report per candidate; Harvest also sends
        # the inspected request to each of its two neighbouring reporters.
        transmissions = n * decision.candidate_evaluations
    elif mode.startswith("local"):
        transmissions = n * decision.candidate_evaluations
    else:
        transmissions = k if decision.candidate_evaluations else 0
    return replace(decision, method=mode, request_inspections=k,
                   transmitted_scalars=transmissions)


def decide_budgeted_harvest(cfg: HarvestCommonsConfig, health, visible_requests,
                            mode: str, settings=MonitorSettings()) -> Decision:
    validate_harvest(cfg)
    health = np.asarray(health, dtype=float)
    if health.shape != (cfg.n_agents,) or not np.isfinite(health).all() or np.any(health < 0):
        raise ValueError("Invalid public patch health")
    visible, _ = _check_visible(visible_requests, cfg.n_agents)
    upper = np.where(np.isfinite(visible), visible, 1.0)
    margin = cfg.weather_noise_std * norm.isf(settings.risk_tolerance / cfg.n_agents)

    def predicate(scale):
        if mode == "joint":
            predicted = harvest_nominal_next(cfg, health, upper * scale)
        elif mode == DIAGNOSTIC_MODE:
            taken = np.minimum(upper * scale * cfg.max_harvest_per_agent, health)
            excess = np.maximum(0, taken - cfg.sustainable_harvest_frac * cfg.max_harvest_per_agent)
            predicted = np.empty(cfg.n_agents, dtype=float)
            for i in range(cfg.n_agents):
                residual = max(0.0, health[i] - taken[i] - cfg.neighbor_externality *
                               (excess[(i - 1) % cfg.n_agents] + excess[(i + 1) % cfg.n_agents]))
                predicted[i] = residual + max(0.0, cfg.regen_rate * residual *
                                               (1 - residual / cfg.patch_max))
        elif mode in {"local_optimistic", "local_bounded"}:
            predicted = np.array([local_patch_report(cfg, float(h), float(a), scale,
                         conservative=mode == "local_bounded") for h, a in zip(health, upper)])
        else:
            raise ValueError(f"Unknown reviewer mode: {mode}")
        return harvest_safe(cfg, np.clip(predicted - margin, 0, cfg.patch_max))

    decision = _decision(mode, cfg.n_agents, visible, settings, predicate)
    if mode == DIAGNOSTIC_MODE:
        # A logical communication count, not a measured network cost.
        decision = replace(decision, transmitted_scalars=decision.transmitted_scalars +
                           2 * int(np.isfinite(visible).sum()) * decision.candidate_evaluations)
    return decision


def decide_budgeted_fishery(cfg: FisheryConfig, state: FisherySnapshot,
                            visible_requests, mode: str,
                            settings=MonitorSettings()) -> Decision:
    make_env(cfg, state)
    visible, _ = _check_visible(visible_requests, cfg.n_agents)
    upper = np.where(np.isfinite(visible), visible, 1.0)

    def predicate(scale):
        if state.collapsed:
            return False
        if mode == "joint":
            total = upper.sum() * scale * cfg.max_harvest_per_agent
        elif mode == DIAGNOSTIC_MODE:
            contributions = [float(value) * scale * cfg.max_harvest_per_agent for value in upper]
            total = sum(contributions)
        elif mode == "local_bounded":
            total = cfg.n_agents * upper.max() * scale * cfg.max_harvest_per_agent
        elif mode == "local_optimistic":
            total = upper.max() * scale * cfg.max_harvest_per_agent
        else:
            raise ValueError(f"Unknown reviewer mode: {mode}")
        return projected_stock(cfg, state.stock, total) >= cfg.collapse_threshold

    return _decision(mode, cfg.n_agents, visible, settings, predicate)
