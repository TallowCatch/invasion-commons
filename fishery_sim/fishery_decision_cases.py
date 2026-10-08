"""Prospective structural challenges, not samples claimed to be reachable.

The grid is fixed before evaluation. Generation never simulates, labels, filters
by outcomes, or invokes reviewers. Case IDs identify design cells within a
configuration; callers must retain that configuration when comparing results.
"""
from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from numbers import Integral, Real

from .config import FisheryConfig
from .fishery_oversight import FisherySnapshot


_STOCK_GRID = (5.0, 8.0, 10.0, 12.0, 20.0, 35.0, 70.0)
_MEAN_REQUEST_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def _validate_config(cfg: FisheryConfig) -> None:
    """Validate physical inputs and the deterministic matched-task contract."""
    if not isinstance(cfg, FisheryConfig):
        raise ValueError("cfg must be a FisheryConfig")
    for name, minimum in (
        ("n_agents", 1), ("horizon", 1), ("collapse_patience", 1), ("seed", 0),
    ):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in (
        "stock_init", "stock_max", "regen_rate", "collapse_threshold",
        "max_harvest_per_agent", "obs_noise_std", "monitoring_prob", "quota_fraction",
    ):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    if cfg.stock_max < _STOCK_GRID[-1]:
        raise ValueError("stock_max must be >= 70 for the frozen stock grid")
    if not 0 <= cfg.stock_init <= cfg.stock_max:
        raise ValueError("stock_init must lie in [0, stock_max]")
    if not 0 <= cfg.collapse_threshold <= cfg.stock_max:
        raise ValueError("collapse_threshold must lie in [0, stock_max]")
    if not 0 <= cfg.regen_rate <= 1:
        raise ValueError("regen_rate must lie in [0, 1] for monotone accounting")
    if cfg.max_harvest_per_agent <= 0:
        raise ValueError("max_harvest_per_agent must be positive")
    if cfg.obs_noise_std or cfg.monitoring_prob or cfg.quota_fraction:
        raise ValueError("obs_noise_std, monitoring_prob and quota_fraction must be zero")
    try:
        finite_demand = isfinite(int(cfg.n_agents) * cfg.max_harvest_per_agent)
    except OverflowError:
        finite_demand = False
    if not finite_demand:
        raise ValueError("n_agents * max_harvest_per_agent must be finite")


def fishery_challenge_cases(cfg: FisheryConfig) -> list[dict]:
    """Return the deduplicated 7-stock x 6-mean x 3-allocation design.

    Proposals are normalized requests, not realized harvest. Each allocation
    requests the same total: n_agents * mean * max_harvest_per_agent, even when
    stock is insufficient. Uniform requests all equal the grid mean; concentrated
    requests fill successive slots to one, then a fractional slot, then zeros.
    The reverse allocation reverses those slots, without changing demand.

    Deduplication uses the exact snapshot and ordered proposal vector. The first
    shape supplies the stable ID; allocation_aliases retains all design shapes
    represented by that record. This gives 98 cases for n_agents > 1, or 42 for
    one agent. Every snapshot explicitly resets collapse history, including
    below-threshold stocks: these are structural challenges, not trajectories.
    """
    _validate_config(cfg)
    n_agents = int(cfg.n_agents)
    by_record: dict[tuple[FisherySnapshot, tuple[float, ...]], dict] = {}
    for stock in _STOCK_GRID:
        state = FisherySnapshot(stock=stock, below_count=0, collapsed=False)
        for mean in _MEAN_REQUEST_GRID:
            total = n_agents * mean
            concentrated = [min(1.0, max(0.0, total - i)) for i in range(n_agents)]
            allocations = (
                ("uniform", [mean] * n_agents),
                ("concentrated", concentrated),
                ("reverse_concentrated", concentrated[::-1]),
            )
            for allocation, proposals in allocations:
                key = (state, tuple(proposals))
                if key in by_record:
                    by_record[key]["design"]["allocation_aliases"].append(allocation)
                    continue
                by_record[key] = {
                    "case_id": (
                        f"fishery_challenge_v1-n{n_agents}-s{stock:g}-m{mean:g}-{allocation}"
                    ),
                    "state": asdict(state),
                    "proposals": proposals,
                    "design": {
                        "family": "fishery_structural_challenge_v1",
                        "state_origin": "structural_not_claimed_reachable",
                        "stock": stock,
                        "mean_normalized_request": mean,
                        "total_normalized_request": total,
                        "total_requested_harvest": float(total * cfg.max_harvest_per_agent),
                        "allocation": allocation,
                        "allocation_aliases": [allocation],
                    },
                }
    return list(by_record.values())
