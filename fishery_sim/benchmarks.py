from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml


# Fixed held-out benchmark pack to stress generalization under harsh conditions.
DEFAULT_BENCHMARK_PACKS: dict[str, list[dict[str, Any]]] = {
    "easy_v1": [
        {
            "name": "abundant_clear_signal",
            "overrides": {"regen_rate": 2.35, "obs_noise_std": 4.0},
        },
        {
            "name": "abundant_mild_noise",
            "overrides": {"regen_rate": 2.10, "obs_noise_std": 7.0},
        },
        {
            "name": "steady_balanced",
            "overrides": {"regen_rate": 1.95, "obs_noise_std": 8.0},
        },
        {
            "name": "moderate_starting_stock",
            "overrides": {"stock_init": 180.0, "regen_rate": 1.90, "obs_noise_std": 9.0},
        },
    ],
    "medium_v1": [
        {
            "name": "moderate_regen_low_noise",
            "overrides": {"regen_rate": 1.70, "obs_noise_std": 10.0},
        },
        {
            "name": "moderate_regen_moderate_noise",
            "overrides": {"regen_rate": 1.55, "obs_noise_std": 14.0},
        },
        {
            "name": "scarcity_start_moderate",
            "overrides": {"stock_init": 130.0, "regen_rate": 1.45, "obs_noise_std": 16.0},
        },
        {
            "name": "fragile_threshold_moderate",
            "overrides": {
                "regen_rate": 1.40,
                "obs_noise_std": 15.0,
                "collapse_threshold": 12.0,
                "collapse_patience": 4,
            },
        },
    ],
    "hard_v1": [
        {
            "name": "low_regen_high_noise",
            "overrides": {"regen_rate": 1.10, "obs_noise_std": 28.0},
        },
        {
            "name": "very_low_regen_very_noisy",
            "overrides": {"regen_rate": 0.95, "obs_noise_std": 35.0},
        },
        {
            "name": "fragile_threshold",
            "overrides": {
                "regen_rate": 1.05,
                "obs_noise_std": 25.0,
                "collapse_threshold": 14.0,
                "collapse_patience": 3,
            },
        },
        {
            "name": "scarcity_shock_like",
            "overrides": {"stock_init": 80.0, "regen_rate": 1.00, "obs_noise_std": 30.0},
        },
    ],
    "heldout_v1": [
        {
            "name": "easy_high_regen_low_noise",
            "overrides": {"regen_rate": 2.20, "obs_noise_std": 5.0},
        },
        {
            "name": "moderate_shift",
            "overrides": {"regen_rate": 1.55, "obs_noise_std": 12.0},
        },
        {
            "name": "scarcity_start",
            "overrides": {"stock_init": 120.0, "regen_rate": 1.45, "obs_noise_std": 14.0},
        },
        {
            "name": "harsh_low_regen_high_noise",
            "overrides": {"regen_rate": 1.10, "obs_noise_std": 25.0},
        },
    ],
    "mixed_v1": [
        {
            "name": "moderate_regen_moderate_noise",
            "overrides": {"regen_rate": 1.30, "obs_noise_std": 14.0},
        },
        {
            "name": "low_regen_moderate_noise",
            "overrides": {"regen_rate": 1.20, "obs_noise_std": 18.0},
        },
        {
            "name": "fragile_threshold_light",
            "overrides": {
                "regen_rate": 1.25,
                "obs_noise_std": 16.0,
                "collapse_threshold": 12.0,
                "collapse_patience": 4,
            },
        },
        {
            "name": "scarcity_start_moderate",
            "overrides": {"stock_init": 120.0, "regen_rate": 1.22, "obs_noise_std": 15.0},
        },
    ],
    "harsh_v1": [
        {
            "name": "low_regen_high_noise",
            "overrides": {"regen_rate": 1.10, "obs_noise_std": 28.0},
        },
        {
            "name": "very_low_regen_very_noisy",
            "overrides": {"regen_rate": 0.95, "obs_noise_std": 35.0},
        },
        {
            "name": "fragile_threshold",
            "overrides": {
                "regen_rate": 1.05,
                "obs_noise_std": 25.0,
                "collapse_threshold": 14.0,
                "collapse_patience": 3,
            },
        },
        {
            "name": "scarcity_shock_like",
            "overrides": {"stock_init": 80.0, "regen_rate": 1.00, "obs_noise_std": 30.0},
        },
    ],
    "harsh_v2": [
        {
            "name": "ultra_scarce",
            "overrides": {"stock_init": 70.0, "regen_rate": 0.9, "obs_noise_std": 32.0},
        },
        {
            "name": "signal_breakdown",
            "overrides": {"regen_rate": 1.15, "obs_noise_std": 40.0},
        },
        {
            "name": "fast_cliff",
            "overrides": {
                "regen_rate": 1.0,
                "obs_noise_std": 27.0,
                "collapse_threshold": 16.0,
                "collapse_patience": 2,
            },
        },
    ],
}


def get_benchmark_pack(name: str) -> list[dict[str, Any]]:
    if name not in DEFAULT_BENCHMARK_PACKS:
        raise ValueError(f"Unknown benchmark pack '{name}'.")
    return deepcopy(DEFAULT_BENCHMARK_PACKS[name])


def load_benchmark_pack_file(path: str, pack_name: str | None = None) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if isinstance(data, list):
        pack = data
    elif isinstance(data, dict):
        if pack_name:
            if pack_name not in data:
                raise ValueError(f"Pack name '{pack_name}' not found in {path}.")
            pack = data[pack_name]
        else:
            # If no pack name provided and exactly one mapping key exists, use it.
            if "regimes" in data and isinstance(data["regimes"], list):
                pack = data["regimes"]
            elif len(data) == 1:
                only_value = next(iter(data.values()))
                pack = only_value
            else:
                raise ValueError(
                    f"Ambiguous benchmark pack file {path}; pass --benchmark-pack-file-name."
                )
    else:
        raise ValueError(f"Unsupported benchmark pack format in {path}.")

    if not isinstance(pack, list):
        raise ValueError("Benchmark pack must resolve to a list of regimes.")

    validated: list[dict[str, Any]] = []
    for i, regime in enumerate(pack):
        if not isinstance(regime, dict):
            raise ValueError(f"Regime at index {i} is not a mapping.")
        name = str(regime.get("name", f"regime_{i}"))
        overrides = regime.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"'overrides' for regime {name} must be a mapping.")
        validated.append({"name": name, "overrides": dict(overrides)})

    return validated
