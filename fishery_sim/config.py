from dataclasses import dataclass
from typing import Dict, Any
import yaml


@dataclass
class FisheryConfig:
    # Environment
    n_agents: int = 8
    horizon: int = 200

    stock_init: float = 100.0
    stock_max: float = 200.0

    regen_rate: float = 1.5          # growth strength
    collapse_threshold: float = 10.0  # below this is "danger"
    collapse_patience: int = 5        # K consecutive rounds below threshold => collapse

    # Observability / noise
    obs_noise_std: float = 20.0        # 0 means fully observed
    max_harvest_per_agent: float = 10.0

    # Governance knobs (disabled by default)
    monitoring_prob: float = 0.0       # chance each agent is audited each step
    quota_fraction: float = 0.0        # per-agent quota = quota_fraction * stock
    base_fine_rate: float = 1.0        # fine per unit of quota violation
    fine_growth: float = 0.5           # repeated violations increase fines
    governance_variant: str = "static"
    adaptive_quota_min_scale: float = 0.4
    adaptive_quota_sensitivity: float = 0.8
    temporary_closure_trigger: float = 15.0
    temporary_closure_quota_fraction: float = 0.01

    # Randomness
    seed: int = 0


def load_config(path: str) -> FisheryConfig:
    with open(path, "r", encoding="utf-8") as f:
        d: Dict[str, Any] = yaml.safe_load(f) or {}
    return FisheryConfig(**d)
