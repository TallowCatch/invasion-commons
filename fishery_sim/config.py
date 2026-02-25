from dataclasses import dataclass
from typing import Optional, Dict, Any
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

    # Randomness
    seed: int = 0


def load_config(path: str) -> FisheryConfig:
    with open(path, "r") as f:
        d: Dict[str, Any] = yaml.safe_load(f)
    return FisheryConfig(**d)