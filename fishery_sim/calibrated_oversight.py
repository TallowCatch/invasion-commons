"""Chance-constrained reviewers and reference labels for experiments R1 and S1.

Claude audit, 5 October 2026. Protocols:
notes/claude_audit_20261005/07_PROTOCOL_R1_REPAIRED_REVIEWER.md and
notes/claude_audit_20261005/08_PROTOCOL_S1_REPORTING_AND_AUDITS.md.

Every reviewer chooses the largest uniform scale in a fixed menu whose
*estimated* one-step risk is at most ``tolerance``. For Harvest the estimate
uses Monte Carlo weather draws applied to the reviewer's own prediction model
(a sample-based chance constraint). Fishery is deterministic, so its risk is
0 or 1. No fixed safety buffer is used.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from fishery_sim.oversight_protocol import harvest_nominal_next, local_patch_report
from fishery_sim.harvest import HarvestCommonsConfig

SCALES = (1.0, 0.75, 0.5, 0.25, 0.0)
REVIEWERS = ("joint", "local_bounded", "local_optimistic")


def stable_seed(base: int, *parts) -> int:
    content = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=float)
    return int(base) + int(hashlib.sha256(content.encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------- Harvest
def harvest_unsafe_mask(cfg: HarvestCommonsConfig, futures: np.ndarray) -> np.ndarray:
    """futures: (..., n) next patch health after weather and clipping."""
    mean_ok = futures.mean(axis=-1) >= cfg.global_min_mean_patch_health
    frac_ok = (futures < cfg.local_patch_failure_threshold).mean(axis=-1) < cfg.failure_fraction_threshold
    return ~(mean_ok & frac_ok)


def harvest_prediction(cfg, health, believed, scale, reviewer):
    health = np.asarray(health, float)
    believed = np.asarray(believed, float)
    if reviewer == "joint":
        return harvest_nominal_next(cfg, health, believed * scale)
    if reviewer in ("local_bounded", "local_optimistic"):
        return np.array([local_patch_report(cfg, float(h), float(a), float(scale),
                                            conservative=reviewer == "local_bounded")
                         for h, a in zip(health, believed)])
    raise ValueError(f"Unknown reviewer {reviewer}")


def harvest_choose_scale(cfg, health, believed, reviewer, seed, draws=400, tolerance=0.05):
    """Return (scale, estimated_risk_at_chosen_scale, candidates_evaluated)."""
    weather = np.random.default_rng(seed).normal(0.0, cfg.weather_noise_std, size=(draws, cfg.n_agents))
    for count, scale in enumerate(SCALES, start=1):
        nominal = harvest_prediction(cfg, health, believed, scale, reviewer)
        futures = np.clip(nominal[None, :] + weather, 0.0, cfg.patch_max)
        risk = float(harvest_unsafe_mask(cfg, futures).mean())
        if risk <= tolerance:
            return scale, risk, count
    return 0.0, risk, len(SCALES)


def wilson(failures: int, trials: int, z: float = 1.959963984540054):
    p = failures / trials
    den = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / den
    radius = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return max(0.0, centre - radius), min(1.0, centre + radius)


def harvest_reference(cfg, health, executed, seed, draws=2000, tolerance=0.05):
    """Reference one-step risk of an executed (true) joint action."""
    nominal = harvest_nominal_next(cfg, np.asarray(health, float), np.asarray(executed, float))
    weather = np.random.default_rng(seed).normal(0.0, cfg.weather_noise_std, size=(draws, cfg.n_agents))
    futures = np.clip(nominal[None, :] + weather, 0.0, cfg.patch_max)
    failures = int(harvest_unsafe_mask(cfg, futures).sum())
    lo, hi = wilson(failures, draws)
    label = "safe" if hi <= tolerance else "risky" if lo > tolerance else "unresolved"
    return dict(risk=failures / draws, risk_lower=lo, risk_upper=hi, label=label)


# ----------------------------------------------------------------- Fishery
def fishery_residual(stock, total):
    return max(0.0, float(stock) - float(total))


def fishery_next(cfg, stock, total):
    r = fishery_residual(stock, total)
    return min(cfg.stock_max, r + max(0.0, cfg.regen_rate * r * (1 - r / cfg.stock_max)))


def fishery_target_ok(cfg, stock, total, target):
    if target == "one_step":
        return fishery_next(cfg, stock, total) >= cfg.collapse_threshold
    if target == "msy":
        return fishery_residual(stock, total) >= cfg.stock_max / 2
    raise ValueError(f"Unknown target {target}")


def fishery_predicted_total(cfg, believed, scale, reviewer):
    b = np.asarray(believed, float) * scale * cfg.max_harvest_per_agent
    if reviewer == "joint":
        return float(b.sum())
    if reviewer == "local_bounded":
        return float(cfg.n_agents * b.max())
    if reviewer == "local_optimistic":
        return float(b.max())
    raise ValueError(f"Unknown reviewer {reviewer}")


def fishery_choose_scale(cfg, stock, believed, reviewer, target, collapsed=False):
    if collapsed:
        return 0.0, len(SCALES)
    for count, scale in enumerate(SCALES, start=1):
        if fishery_target_ok(cfg, stock, fishery_predicted_total(cfg, believed, scale, reviewer), target):
            return scale, count
    return 0.0, len(SCALES)


def fishery_reference(cfg, stock, executed, target):
    total = float(np.asarray(executed, float).sum() * cfg.max_harvest_per_agent)
    ok = fishery_target_ok(cfg, stock, total, target)
    return dict(risk=0.0 if ok else 1.0, label="safe" if ok else "risky")
