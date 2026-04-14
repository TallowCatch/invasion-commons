from __future__ import annotations

import itertools
from copy import deepcopy


_BASE_GOVERNMENT_PARAMS = {
    "government_trigger": 16.0,
    "strict_cap_frac": 0.18,
    "relaxed_cap_frac": 0.35,
    "soft_trigger": 18.0,
    "deterioration_threshold": 0.35,
    "activation_warmup": 3,
    "aggressive_request_threshold": 0.75,
    "aggressive_agent_fraction_trigger": 0.34,
    "local_neighborhood_trigger": 0.67,
}


_STAGE_CONFIGS: dict[str, dict] = {
    "stage_a_llm_reliability": {
        "explicit_cells": [
            {
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "condition": "top_down_only",
                "injector_mode": "llm_json",
                "pressure": "0.3",
            },
            {
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "condition": "hybrid",
                "injector_mode": "llm_json",
                "pressure": "0.3",
            },
            {
                "tier": "medium_h1",
                "partner_mix": "adversarial_heavy",
                "condition": "hybrid",
                "injector_mode": "llm_json",
                "pressure": "0.3",
            },
        ],
        "n_runs": "1",
        "generations": "4",
        "population_size": "6",
        "seeds_per_generation": "8",
        "test_seeds_per_generation": "8",
        "replacement_fraction": "0.2",
        "run_name": "harvest_invasion_llm_stageA_reliability",
        "experiment_tag": "harvest_invasion_llm_stageA_reliability",
        "summary_prefix": "results/runs/showcase/curated/harvest_invasion_llm_stageA_reliability",
        "plot_mode": "legacy",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "stage_b_llm": {
        "tiers": ["medium_h1", "hard_h1"],
        "partner_mixes": ["balanced", "adversarial_heavy"],
        "conditions": ["top_down_only", "hybrid"],
        "injector_modes": ["mutation", "llm_json"],
        "pressures": ["0.3", "0.5"],
        "n_runs": "3",
        "generations": "12",
        "population_size": "6",
        "seeds_per_generation": "24",
        "test_seeds_per_generation": "24",
        "replacement_fraction": "0.2",
        "run_name": "harvest_invasion_llm_stageB_local",
        "experiment_tag": "harvest_invasion_llm_stageB_local",
        "summary_prefix": "results/runs/showcase/curated/harvest_invasion_llm_stageB_local",
        "plot_mode": "legacy",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "stage_b_llm_narrow": {
        "tiers": ["medium_h1", "hard_h1"],
        "partner_mixes": ["balanced", "adversarial_heavy"],
        "conditions": ["top_down_only", "hybrid"],
        "injector_modes": ["mutation", "llm_json"],
        "pressures": ["0.3"],
        "n_runs": "3",
        "generations": "12",
        "population_size": "6",
        "seeds_per_generation": "24",
        "test_seeds_per_generation": "24",
        "replacement_fraction": "0.2",
        "run_name": "harvest_invasion_llm_stageB_narrow",
        "experiment_tag": "harvest_invasion_llm_stageB_narrow",
        "summary_prefix": "results/runs/showcase/curated/harvest_invasion_llm_stageB_narrow",
        "plot_mode": "legacy",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "stage_c_llm": {
        "tiers": ["medium_h1", "hard_h1"],
        "partner_mixes": ["balanced", "adversarial_heavy"],
        "conditions": ["top_down_only", "hybrid"],
        "injector_modes": ["llm_json"],
        "pressures": ["0.3"],
        "n_runs": "5",
        "generations": "15",
        "population_size": "6",
        "seeds_per_generation": "32",
        "test_seeds_per_generation": "32",
        "replacement_fraction": "0.2",
        "run_name": "harvest_invasion_llm_stageC_local",
        "experiment_tag": "harvest_invasion_llm_stageC_local",
        "summary_prefix": "results/runs/showcase/curated/harvest_invasion_llm_stageC_local",
        "plot_mode": "legacy",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "stage_c_llm_narrow": {
        "tiers": ["medium_h1", "hard_h1"],
        "partner_mixes": ["balanced", "adversarial_heavy"],
        "conditions": ["top_down_only", "hybrid"],
        "injector_modes": ["llm_json"],
        "pressures": ["0.3"],
        "n_runs": "5",
        "generations": "15",
        "population_size": "6",
        "seeds_per_generation": "32",
        "test_seeds_per_generation": "32",
        "replacement_fraction": "0.2",
        "run_name": "harvest_invasion_llm_stageC_narrow",
        "experiment_tag": "harvest_invasion_llm_stageC_narrow",
        "summary_prefix": "results/runs/showcase/curated/harvest_invasion_llm_stageC_narrow",
        "plot_mode": "legacy",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "architecture_stageB": {
        "tiers": ["medium_h1", "hard_h1"],
        "partner_mixes": ["balanced", "adversarial_heavy"],
        "conditions": ["none", "bottom_up_only", "top_down_only", "hybrid"],
        "injector_modes": ["mutation", "search_mutation"],
        "pressures": ["0.3"],
        "n_runs": "3",
        "generations": "12",
        "population_size": "6",
        "seeds_per_generation": "24",
        "test_seeds_per_generation": "24",
        "replacement_fraction": "0.2",
        "run_name": "harvest_architecture_stageB",
        "experiment_tag": "harvest_architecture_stageB",
        "summary_prefix": "results/runs/showcase/curated/harvest_architecture_stageB",
        "plot_mode": "institutional",
        "ci_plot_filter": "--tiers medium_h1,hard_h1 --partner-mixes balanced,adversarial_heavy --injector-modes mutation,search_mutation --pressures 0.3",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "architecture_stageC": {
        "tiers": ["medium_h1", "hard_h1"],
        "partner_mixes": ["balanced", "adversarial_heavy"],
        "conditions": ["bottom_up_only", "top_down_only", "hybrid"],
        "injector_modes": ["search_mutation"],
        "pressures": ["0.3", "0.5"],
        "n_runs": "5",
        "generations": "15",
        "population_size": "6",
        "seeds_per_generation": "32",
        "test_seeds_per_generation": "32",
        "replacement_fraction": "0.2",
        "run_name": "harvest_architecture_stageC",
        "experiment_tag": "harvest_architecture_stageC",
        "summary_prefix": "results/runs/showcase/curated/harvest_architecture_stageC",
        "plot_mode": "institutional",
        "ci_plot_filter": "--tiers medium_h1,hard_h1 --partner-mixes balanced,adversarial_heavy --injector-modes search_mutation --pressures 0.3,0.5",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "capability_ladder_stageB": {
        "tiers": ["medium_h1", "hard_h1"],
        "partner_mixes": ["balanced", "adversarial_heavy"],
        "conditions": ["bottom_up_only", "top_down_only", "hybrid"],
        "injector_modes": ["random", "mutation", "adversarial_heuristic", "search_mutation"],
        "pressures": ["0.3", "0.5"],
        "n_runs": "3",
        "generations": "12",
        "population_size": "6",
        "seeds_per_generation": "24",
        "test_seeds_per_generation": "24",
        "replacement_fraction": "0.2",
        "run_name": "harvest_capability_ladder_stageB",
        "experiment_tag": "harvest_capability_ladder_stageB",
        "summary_prefix": "results/runs/showcase/curated/harvest_capability_ladder_stageB",
        "plot_mode": "institutional",
        "ci_plot_filter": "--tiers medium_h1,hard_h1 --partner-mixes balanced,adversarial_heavy --injector-modes random,mutation,adversarial_heuristic,search_mutation --pressures 0.3,0.5",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "institutional_friction_moduleA": {
        "scenario_presets": ["regulated_fishery", "community_irrigation", "forest_co_management"],
        "governance_friction_regimes": ["ideal", "constrained"],
        "conditions": ["bottom_up_only", "top_down_only", "hybrid"],
        "injector_modes": ["search_mutation"],
        "pressures": ["0.3", "0.5"],
        "n_runs": "5",
        "generations": "15",
        "population_size": "6",
        "seeds_per_generation": "32",
        "test_seeds_per_generation": "32",
        "replacement_fraction": "0.2",
        "run_name": "harvest_institutional_friction_moduleA",
        "experiment_tag": "harvest_institutional_friction_moduleA",
        "summary_prefix": "results/runs/showcase/curated/harvest_institutional_friction_moduleA",
        "plot_mode": "institutional",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "institutional_friction_regulated_fishery": {
        "scenario_presets": ["regulated_fishery"],
        "governance_friction_regimes": ["ideal", "constrained"],
        "conditions": ["bottom_up_only", "top_down_only", "hybrid"],
        "injector_modes": ["search_mutation"],
        "pressures": ["0.3", "0.5"],
        "n_runs": "5",
        "generations": "15",
        "population_size": "6",
        "seeds_per_generation": "32",
        "test_seeds_per_generation": "32",
        "replacement_fraction": "0.2",
        "run_name": "harvest_institutional_friction_regulated_fishery",
        "experiment_tag": "harvest_institutional_friction_regulated_fishery",
        "summary_prefix": "results/runs/showcase/curated/harvest_institutional_friction_regulated_fishery",
        "plot_mode": "institutional",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "institutional_friction_community_irrigation": {
        "scenario_presets": ["community_irrigation"],
        "governance_friction_regimes": ["ideal", "constrained"],
        "conditions": ["bottom_up_only", "top_down_only", "hybrid"],
        "injector_modes": ["search_mutation"],
        "pressures": ["0.3", "0.5"],
        "n_runs": "5",
        "generations": "15",
        "population_size": "6",
        "seeds_per_generation": "32",
        "test_seeds_per_generation": "32",
        "replacement_fraction": "0.2",
        "run_name": "harvest_institutional_friction_community_irrigation",
        "experiment_tag": "harvest_institutional_friction_community_irrigation",
        "summary_prefix": "results/runs/showcase/curated/harvest_institutional_friction_community_irrigation",
        "plot_mode": "institutional",
        **_BASE_GOVERNMENT_PARAMS,
    },
    "institutional_friction_forest_co_management": {
        "scenario_presets": ["forest_co_management"],
        "governance_friction_regimes": ["ideal", "constrained"],
        "conditions": ["bottom_up_only", "top_down_only", "hybrid"],
        "injector_modes": ["search_mutation"],
        "pressures": ["0.3", "0.5"],
        "n_runs": "5",
        "generations": "15",
        "population_size": "6",
        "seeds_per_generation": "32",
        "test_seeds_per_generation": "32",
        "replacement_fraction": "0.2",
        "run_name": "harvest_institutional_friction_forest_co_management",
        "experiment_tag": "harvest_institutional_friction_forest_co_management",
        "summary_prefix": "results/runs/showcase/curated/harvest_institutional_friction_forest_co_management",
        "plot_mode": "institutional",
        **_BASE_GOVERNMENT_PARAMS,
    },
}


def stage_names() -> list[str]:
    return sorted(_STAGE_CONFIGS)


def stage_config(stage: str) -> dict:
    if stage not in _STAGE_CONFIGS:
        raise ValueError(f"Unknown stage: {stage}")
    return deepcopy(_STAGE_CONFIGS[stage])


def stage_cells(cfg: dict) -> list[dict[str, str]]:
    if "explicit_cells" in cfg:
        return [dict(cell) for cell in cfg["explicit_cells"]]
    scenario_presets = cfg.get("scenario_presets")
    governance_friction_regimes = cfg.get("governance_friction_regimes", ["ideal"])
    if scenario_presets:
        cells: list[dict[str, str]] = []
        for scenario_preset, governance_friction_regime, condition, injector_mode, pressure in itertools.product(
            scenario_presets,
            governance_friction_regimes,
            cfg["conditions"],
            cfg["injector_modes"],
            cfg["pressures"],
        ):
            cells.append(
                {
                    "scenario_preset": scenario_preset,
                    "governance_friction_regime": governance_friction_regime,
                    "tier": "",
                    "partner_mix": "",
                    "condition": condition,
                    "injector_mode": injector_mode,
                    "pressure": pressure,
                }
            )
        return cells
    cells: list[dict[str, str]] = []
    for tier, partner_mix, condition, injector_mode, pressure in itertools.product(
        cfg["tiers"],
        cfg["partner_mixes"],
        cfg["conditions"],
        cfg["injector_modes"],
        cfg["pressures"],
    ):
        cells.append(
            {
                "tier": tier,
                "partner_mix": partner_mix,
                "condition": condition,
                "injector_mode": injector_mode,
                "pressure": pressure,
            }
        )
    return cells


def shard_slug(tier: str, partner_mix: str, condition: str, injector_mode: str, pressure: str) -> str:
    return f"{tier}__{partner_mix}__{condition}__{injector_mode}__p{pressure}".replace(".", "p")


def github_matrix_payload(stage: str) -> dict:
    cfg = stage_config(stage)
    include = []
    for cell in stage_cells(cfg):
        include.append(
            {
                **cell,
                "slug": shard_slug(
                    cell.get("scenario_preset") or cell["tier"],
                    cell.get("governance_friction_regime") or cell["partner_mix"],
                    cell["condition"],
                    cell["injector_mode"],
                    cell["pressure"],
                ),
                "n_runs": int(cfg["n_runs"]),
                "generations": int(cfg["generations"]),
                "population_size": int(cfg["population_size"]),
                "seeds_per_generation": int(cfg["seeds_per_generation"]),
                "test_seeds_per_generation": int(cfg["test_seeds_per_generation"]),
                "replacement_fraction": float(cfg["replacement_fraction"]),
            }
        )
    return {
        "matrix": {"include": include},
        "run_name": cfg["run_name"],
        "summary_prefix": cfg["summary_prefix"],
        "plot_mode": cfg.get("plot_mode", "legacy"),
        "ci_plot_filter": cfg.get("ci_plot_filter", ""),
    }
