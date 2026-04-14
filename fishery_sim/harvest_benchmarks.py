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


HARVEST_GOVERNANCE_FRICTION_REGIMES: dict[str, dict[str, float | int]] = {
    "ideal": {
        "detection_recall": 1.0,
        "enforcement_delay_rounds": 0,
        "max_target_share": 1.0,
        "governance_budget_cost": 0.0,
    },
    "constrained": {
        "detection_recall": 0.7,
        "enforcement_delay_rounds": 1,
        "max_target_share": 0.5,
        "governance_budget_cost": 0.02,
    },
}


HARVEST_SCENARIO_PRESETS: dict[str, dict[str, object]] = {
    "regulated_fishery": {
        "label": "Regulated fishery",
        "institutional_archetype": "Centralized quota-setting, monitoring, and enforcement",
        "tier": "medium_h1",
        "partner_mix": "balanced",
        "cfg_overrides": {
            "neighbor_externality": 0.10,
            "communication_enabled": False,
            "side_payments_enabled": False,
        },
        "default_friction_regime": "constrained",
        "preferred_governance_comparison": ["top_down_only", "hybrid"],
        "citations": [
            "Hilborn, Orensanz, and Parma (2005), Institutions, incentives and the future of fisheries.",
            "Gutierrez, Hilborn, and Defeo (2011), Leadership, social capital and incentives promote successful fisheries.",
        ],
        "notes": "Regulator-facing renewable commons with meaningful monitoring and enforcement capacity.",
    },
    "community_irrigation": {
        "label": "Community irrigation",
        "institutional_archetype": "Locally monitored and self-governed irrigation management",
        "tier": "medium_h1",
        "partner_mix": "balanced",
        "cfg_overrides": {
            "regen_rate": 0.64,
            "neighbor_externality": 0.15,
            "communication_enabled": True,
            "side_payments_enabled": False,
        },
        "default_friction_regime": "constrained",
        "preferred_governance_comparison": ["bottom_up_only", "hybrid"],
        "citations": [
            "Ostrom and Gardner (1993), Coping with asymmetries in the commons.",
            "Ostrom, Lam, and Lee (1994), The performance of self-governing irrigation systems in Nepal.",
            "Villamayor-Tomas (2020), Robust irrigation system institutions: A global comparison.",
        ],
        "notes": "User participation, local monitoring, and rights to organize are central to performance.",
    },
    "forest_co_management": {
        "label": "Forest co-management",
        "institutional_archetype": "Polycentric stewardship with local management and higher-level intervention",
        "tier": "hard_h1",
        "partner_mix": "adversarial_heavy",
        "cfg_overrides": {
            "neighbor_externality": 0.22,
            "communication_enabled": True,
            "side_payments_enabled": True,
        },
        "default_friction_regime": "constrained",
        "preferred_governance_comparison": ["top_down_only", "hybrid"],
        "citations": [
            "Nagendra and Ostrom (2012), Polycentric governance of multifunctional forested landscapes.",
        ],
        "notes": "Mixed governance setting where neither pure centralization nor pure decentralization is sufficient.",
    },
}


def get_harvest_tier_preset(name: str) -> dict[str, float]:
    if name not in HARVEST_TIER_PRESETS:
        raise ValueError(f"Unknown Harvest Commons tier '{name}'.")
    return dict(HARVEST_TIER_PRESETS[name])


def get_harvest_partner_mix_preset(name: str) -> dict[str, float]:
    if name not in HARVEST_PARTNER_MIX_PRESETS:
        raise ValueError(f"Unknown Harvest Commons partner mix '{name}'.")
    return dict(HARVEST_PARTNER_MIX_PRESETS[name])


def get_harvest_governance_friction_regime(name: str) -> dict[str, float | int]:
    if name not in HARVEST_GOVERNANCE_FRICTION_REGIMES:
        raise ValueError(f"Unknown Harvest governance friction regime '{name}'.")
    return dict(HARVEST_GOVERNANCE_FRICTION_REGIMES[name])


def get_harvest_scenario_preset(name: str) -> dict[str, object]:
    if name not in HARVEST_SCENARIO_PRESETS:
        raise ValueError(f"Unknown Harvest Commons scenario '{name}'.")
    preset = dict(HARVEST_SCENARIO_PRESETS[name])
    preset["cfg_overrides"] = dict(preset.get("cfg_overrides", {}))
    preset["preferred_governance_comparison"] = list(preset.get("preferred_governance_comparison", []))
    preset["citations"] = list(preset.get("citations", []))
    return preset


def make_harvest_cfg_for_tier(name: str, **overrides: float | int | bool) -> HarvestCommonsConfig:
    cfg_dict = asdict(HarvestCommonsConfig())
    cfg_dict.update(get_harvest_tier_preset(name))
    cfg_dict.update(overrides)
    return HarvestCommonsConfig(**cfg_dict)


def make_harvest_cfg_for_scenario(
    scenario_name: str,
    *,
    n_agents: int = 6,
    seed: int = 0,
    **overrides: float | int | bool,
) -> HarvestCommonsConfig:
    preset = get_harvest_scenario_preset(scenario_name)
    cfg_overrides = dict(preset["cfg_overrides"])
    cfg_overrides.update({"n_agents": n_agents, "seed": seed})
    cfg_overrides.update(overrides)
    return make_harvest_cfg_for_tier(str(preset["tier"]), **cfg_overrides)


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
