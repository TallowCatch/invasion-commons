from __future__ import annotations

from dataclasses import asdict

from .harvest import HarvestCommonsConfig


HARVEST_TIER_PRESETS: dict[str, dict[str, float]] = {
    "easy_h1": {
        "patch_init": 15.5,
        "regen_rate": 0.82,
        "weather_noise_std": 0.20,
        "neighbor_externality": 0.08,
    },
    "medium_h1": {
        "patch_init": 14.0,
        "regen_rate": 0.70,
        "weather_noise_std": 0.30,
        "neighbor_externality": 0.12,
    },
    "hard_h1": {
        "patch_init": 12.5,
        "regen_rate": 0.56,
        "weather_noise_std": 0.42,
        "neighbor_externality": 0.18,
    },
}


HARVEST_PARTNER_MIX_PRESETS: dict[str, dict[str, float]] = {
    "cooperative_heavy": {"cooperative_weight": 0.6, "balanced_weight": 0.3, "adversarial_weight": 0.1},
    "balanced": {"cooperative_weight": 0.34, "balanced_weight": 0.33, "adversarial_weight": 0.33},
    "adversarial_heavy": {"cooperative_weight": 0.1, "balanced_weight": 0.3, "adversarial_weight": 0.6},
}


def get_harvest_tier_preset(name: str) -> dict[str, float]:
    if name not in HARVEST_TIER_PRESETS:
        raise ValueError(f"Unknown Harvest Commons tier '{name}'.")
    return dict(HARVEST_TIER_PRESETS[name])


def get_harvest_partner_mix_preset(name: str) -> dict[str, float]:
    if name not in HARVEST_PARTNER_MIX_PRESETS:
        raise ValueError(f"Unknown Harvest Commons partner mix '{name}'.")
    return dict(HARVEST_PARTNER_MIX_PRESETS[name])


def make_harvest_cfg_for_tier(name: str, **overrides: float | int | bool) -> HarvestCommonsConfig:
    cfg_dict = asdict(HarvestCommonsConfig())
    cfg_dict.update(get_harvest_tier_preset(name))
    cfg_dict.update(overrides)
    return HarvestCommonsConfig(**cfg_dict)


def get_harvest_regime_pack(tier_name: str) -> list[dict[str, dict[str, float]]]:
    base = get_harvest_tier_preset(tier_name)
    patch_init = float(base["patch_init"])
    regen_rate = float(base["regen_rate"])
    weather_noise_std = float(base["weather_noise_std"])
    neighbor_externality = float(base["neighbor_externality"])
    return [
        {
            "name": "noisy_weather",
            "overrides": {
                "weather_noise_std": round(weather_noise_std * 1.35, 6),
            },
        },
        {
            "name": "strong_externality",
            "overrides": {
                "neighbor_externality": round(neighbor_externality * 1.30, 6),
            },
        },
        {
            "name": "low_init",
            "overrides": {
                "patch_init": round(max(0.0, patch_init - 1.5), 6),
            },
        },
        {
            "name": "slow_regen",
            "overrides": {
                "regen_rate": round(regen_rate * 0.85, 6),
            },
        },
    ]
