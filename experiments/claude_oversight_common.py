"""Shared population and environment setup for experiments R1 and S1 (Claude audit, Oct 2026).

Population generators and game settings are copied from the 23 September
confirmation (`experiments/run_matched_oversight.py`), so R1/S1 use the same
policy classes and dynamics with fresh seeds.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from fishery_sim.config import FisheryConfig
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_scenario
from fishery_sim.harvest_evolution import adversarial_harvest_strategy, cooperative_harvest_strategy


def harvest_setup(context: int, pop_base: int, weather_base: int, horizon: int = 80, n_stress: int = 2):
    cfg = make_harvest_cfg_for_scenario("forest_co_management", horizon=horizon,
                                        seed=weather_base + 1000 * context,
                                        communication_enabled=False, side_payments_enabled=False)
    cfg.regen_rate *= .85  # slow renewal, as in the confirmation
    rng = np.random.default_rng(pop_base + context)
    stressed = sorted(rng.permutation(cfg.n_agents)[:n_stress].tolist())
    specs = [(adversarial_harvest_strategy if i in stressed else cooperative_harvest_strategy)(
        f"heldout_{i}", cfg.patch_max, rng) for i in range(cfg.n_agents)]
    return cfg, specs, stressed


def fishery_setup(context: int, pop_base: int, horizon: int = 80, n_stress: int = 4):
    cfg = FisheryConfig(n_agents=6, horizon=horizon, stock_init=70, stock_max=100,
                        regen_rate=.7, obs_noise_std=0, max_harvest_per_agent=6)
    rng = np.random.default_rng(pop_base + context)
    ranges = {"useful": ((.05, .20), (.20, .40)), "stress": ((.45, .65), (.75, .95))}
    stressed = sorted(rng.permutation(cfg.n_agents)[:n_stress].tolist())
    low = np.array([rng.uniform(*ranges["stress" if i in stressed else "useful"][0]) for i in range(cfg.n_agents)])
    high = np.array([rng.uniform(*ranges["stress" if i in stressed else "useful"][1]) for i in range(cfg.n_agents)])
    thresholds = rng.uniform(25, 65, cfg.n_agents)
    return cfg, dict(low=low, high=high, thresholds=thresholds), stressed


def fishery_requests(policy, stock):
    return np.where(stock < policy["thresholds"], policy["low"], policy["high"])


def write_jsonl_gz(path: Path, rows):
    data = "".join(json.dumps(row, default=float) + "\n" for row in rows).encode()
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(data)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
