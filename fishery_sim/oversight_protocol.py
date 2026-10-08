"""Matched one-step oversight. No old simulation or governance defaults change.

Local Harvest reviewers see only their own patch/request and emit predicted
patch-health reports. A public aggregator applies the SAME global target as
the joint reviewer. This is pooled local evidence, not autonomous local control.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm

from .harvest import HarvestCommonsConfig, harvest_global_safe


METHODS = (
    "none", "cutoff_uniform", "local_nominal", "local_uncertain",
    "local_conservative", "local_conservative_uncertain", "joint_nominal", "joint_uncertain",
)


@dataclass(frozen=True)
class MonitorSettings:
    scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    risk_tolerance: float = 0.05
    candidate_budget: int = 5

    def __post_init__(self):
        if (not self.scales or self.scales[0] != 1 or self.scales[-1] != 0
                or any(not 0 <= x <= 1 for x in self.scales)
                or any(a <= b for a, b in zip(self.scales, self.scales[1:]))):
            raise ValueError("Scales must decrease strictly from 1 to 0")
        if not 0 < self.risk_tolerance < 1 or self.candidate_budget < 0:
            raise ValueError("Invalid risk tolerance or budget")


@dataclass(frozen=True)
class Decision:
    method: str
    scale: float
    verdict: str
    status: str
    candidate_evaluations: int
    component_evaluations: int
    request_inspections: int
    transmitted_scalars: int
    predicted_safe: bool | None

    def record(self):
        return asdict(self)


def check_vectors(health, requests, n):
    h, a = np.asarray(health, dtype=float), np.asarray(requests, dtype=float)
    if (h.shape != (n,) or a.shape != (n,) or not np.isfinite(h).all()
            or not np.isfinite(a).all() or np.any(h < 0) or np.any((a < 0) | (a > 1))):
        raise ValueError("Expected finite nonnegative health and unit-interval request vectors")
    return h, a


def validate_harvest(cfg):
    if (cfg.n_agents < 3 or not 0 <= cfg.regen_rate <= 1 or cfg.patch_max <= 0
            or cfg.max_harvest_per_agent <= 0 or cfg.weather_noise_std < 0
            or cfg.neighbor_externality < 0):
        raise ValueError("Bounds require >=3 agents, monotone regrowth and nonnegative externality/noise")


def harvest_nominal_next(cfg: HarvestCommonsConfig, health, requests):
    """Exact deterministic part of the existing Harvest step, before weather."""
    validate_harvest(cfg)
    health, requests = check_vectors(health, requests, cfg.n_agents)
    taken = np.minimum(requests * cfg.max_harvest_per_agent, health)
    excess = np.maximum(0, taken - cfg.sustainable_harvest_frac * cfg.max_harvest_per_agent)
    residual = np.maximum(0, health - taken - cfg.neighbor_externality *
                          (np.roll(excess, 1) + np.roll(excess, -1)))
    return residual + np.maximum(0, cfg.regen_rate * residual * (1 - residual / cfg.patch_max))


def local_patch_report(cfg, own_health: float, own_request: float, scale: float,
                       conservative: bool) -> float:
    """No neighbour state or neighbour request enters this interface.

    Every executed request is at most scale * max_harvest. Thus the conservative
    unknown-neighbour bound is valid under the common uniform-scale authority.
    The nominal alternative explicitly assumes zero neighbour excess.
    """
    taken = min(own_request * scale * cfg.max_harvest_per_agent, own_health)
    excess_bound = max(0, (scale - cfg.sustainable_harvest_frac) * cfg.max_harvest_per_agent)
    spillover = 2 * cfg.neighbor_externality * excess_bound if conservative else 0.0
    residual = max(0.0, own_health - taken - spillover)
    return residual + max(0, cfg.regen_rate * residual * (1 - residual / cfg.patch_max))


def harvest_prediction(cfg, health, requests, method, scale, risk_tolerance):
    validate_harvest(cfg)
    health, requests = check_vectors(health, requests, cfg.n_agents)
    if method.startswith("local_"):
        predicted = np.array([local_patch_report(cfg, float(h), float(a), scale,
                              "conservative" in method) for h, a in zip(health, requests)])
    elif method.startswith("joint_"):
        predicted = harvest_nominal_next(cfg, health, requests * scale)
    else:
        raise ValueError(f"No prediction model for {method}")
    if "uncertain" in method:
        # Union bound: with probability >= 1-delta every disturbance exceeds -q.
        # Only conservative/joint predictions also account for neighbour damage.
        predicted -= cfg.weather_noise_std * norm.isf(risk_tolerance / cfg.n_agents)
    return np.clip(predicted, 0, cfg.patch_max)


def harvest_safe(cfg, health):
    return harvest_global_safe(
        health, min_mean_patch_health=cfg.global_min_mean_patch_health,
        local_patch_failure_threshold=cfg.local_patch_failure_threshold,
        failure_fraction_threshold=cfg.failure_fraction_threshold,
    )


def choose_scale(method: str, n: int, settings: MonitorSettings,
                 predict_safe: Callable[[float], bool], requests,
                 cutoff: float = 0.4) -> Decision:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    if method == "none":
        return Decision(method, 1.0, "approve", "unmonitored", 0, 0, 0, 0, None)
    if method == "cutoff_uniform":
        # Same finite intervention menu; this is not the legacy componentwise clip.
        scale = next(s for s in settings.scales if np.max(requests) * s <= cutoff + 1e-12)
        return Decision(method, scale, "approve" if scale == 1 else "reject",
                        "fixed_rule", 0, 0, n, n, None)
    tested = 0
    initial_safe = None
    for scale in settings.scales:
        if tested == settings.candidate_budget:
            break
        safe = bool(predict_safe(scale))
        tested += 1
        if initial_safe is None:
            initial_safe = safe
        if safe:
            return Decision(method, scale, "approve" if initial_safe else "reject", "feasible",
                            tested, n * tested, n, n * tested if method.startswith("local") else n, True)
    zero_tested = tested == len(settings.scales)
    # Failure to certify is not reported as a safe intervention. A zero-action
    # fallback is a declared executor rule, including on budget exhaustion.
    return Decision(method, 0.0, "abstain" if initial_safe is None else "reject",
                    "infeasible" if zero_tested else "budget_exhausted", tested, n * tested,
                    n if tested else 0, n * tested if method.startswith("local") else (n if tested else 0),
                    False if zero_tested else None)


def decide_harvest(cfg, health, requests, method, settings=MonitorSettings()):
    validate_harvest(cfg)
    health, requests = check_vectors(health, requests, cfg.n_agents)
    return choose_scale(method, cfg.n_agents, settings, lambda s: harvest_safe(
        cfg, harvest_prediction(cfg, health, requests, method, s, settings.risk_tolerance)),
        requests, cfg.sustainable_harvest_frac + cfg.local_safety_margin)


def wilson_interval(failures: int, trials: int, confidence: float = .95):
    if trials < 1 or not 0 <= failures <= trials:
        raise ValueError("Invalid binomial counts")
    z = float(norm.ppf((1 + confidence) / 2))
    p, den = failures / trials, 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / den
    radius = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return max(0.0, centre - radius), min(1.0, centre + radius)


def harvest_reference_risk(cfg, health, requests, seed: int, draws: int = 128,
                           risk_tolerance: float = .05):
    """Independent one-step reference, not a claimed perfect ground-truth oracle."""
    if draws < 1:
        raise ValueError("draws must be positive")
    nominal = harvest_nominal_next(cfg, health, requests)
    if cfg.weather_noise_std == 0:
        risk = float(not harvest_safe(cfg, np.clip(nominal, 0, cfg.patch_max)))
        return dict(risk=risk, risk_lower=risk, risk_upper=risk, reference_label="risky" if risk else "safe",
                    reference_draws=1, reference_kind="exact_deterministic")
    weather = np.random.default_rng(seed).normal(0, cfg.weather_noise_std, size=(draws, cfg.n_agents))
    future = np.clip(nominal + weather, 0, cfg.patch_max)
    safe = (future.mean(axis=1) >= cfg.global_min_mean_patch_health) & (
        (future < cfg.local_patch_failure_threshold).mean(axis=1) < cfg.failure_fraction_threshold)
    failures = int((~safe).sum())
    lo, hi = wilson_interval(failures, draws)
    label = "safe" if hi <= risk_tolerance else "risky" if lo > risk_tolerance else "unresolved"
    return dict(risk=failures / draws, risk_lower=lo, risk_upper=hi, reference_label=label,
                reference_draws=draws, reference_kind="one_step_mc_wilson95")


def score_verdict(verdict: str, label: str):
    if verdict not in {"approve", "reject", "abstain"} or label not in {"safe", "risky", "unresolved"}:
        raise ValueError("Unknown verdict or reference label")
    return dict(risky_resolved=int(label == "risky"), safe_resolved=int(label == "safe"),
                unresolved=int(label == "unresolved"),
                harmful_accepted=int(verdict == "approve" and label == "risky"),
                safe_rejected=int(verdict == "reject" and label == "safe"),
                approved=int(verdict == "approve"), abstained=int(verdict == "abstain"))
