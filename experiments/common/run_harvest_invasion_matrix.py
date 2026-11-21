import argparse
import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from fishery_sim.harvest_benchmarks import (
    get_harvest_actor_capability_preset,
    get_harvest_governance_friction_regime,
    get_harvest_overseer_capability_preset,
    get_harvest_regime_pack,
    get_harvest_scenario_preset,
    harvest_capability_gap,
    make_harvest_cfg_for_scenario,
    make_harvest_cfg_for_tier,
)
from fishery_sim.harvest_evolution import (
    HARVEST_LLM_PARSE_ERROR_TYPES,
    _llm_integrity_base_df,
    make_harvest_strategy_injector,
    run_harvest_invasion,
)
from fishery_sim.llm_adapter import (
    FileReplayPolicyLLMClient,
    OllamaPolicyLLMClient,
    OpenAIResponsesPolicyLLMClient,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Harvest Commons invasion matrix sweeps.")
    parser.add_argument("--tiers", default="easy_h1,medium_h1,hard_h1")
    parser.add_argument("--partner-mixes", default="cooperative_heavy,balanced,adversarial_heavy")
    parser.add_argument("--scenario-presets", default="")
    parser.add_argument("--conditions", default="none,top_down_only,bottom_up_only,hybrid")
    parser.add_argument("--injector-modes", default="random,mutation,adversarial_heuristic")
    parser.add_argument("--adversarial-pressures", default="0.3")
    parser.add_argument("--governance-friction-regimes", default="ideal")
    parser.add_argument("--actor-capability-levels", default="")
    parser.add_argument("--overseer-capability-levels", default="")
    parser.add_argument("--search-candidates", type=int, default=0)
    parser.add_argument("--search-eval-horizon", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--seeds-per-generation", type=int, default=24)
    parser.add_argument("--test-seeds-per-generation", type=int, default=24)
    parser.add_argument("--replacement-fraction", type=float, default=0.2)
    parser.add_argument("--rng-seed-start", type=int, default=0)
    parser.add_argument("--cfg-seed-start", type=int, default=0)
    parser.add_argument("--run-seed-stride", type=int, default=1000)
    parser.add_argument("--local-safety-margin", type=float, default=0.05)
    parser.add_argument("--global-min-mean-patch-health", type=float, default=10.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--output-prefix", default="results/runs/harvest_invasion/harvest_invasion_matrix")
    parser.add_argument("--experiment-tag", default="harvest_invasion")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--llm-policy-replay-file", default=None)
    parser.add_argument("--llm-provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:3b-instruct")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout-s", type=float, default=120.0)
    parser.add_argument("--llm-temperature", type=float, default=0.2)

    parser.add_argument("--government-trigger", type=float, default=16.0)
    parser.add_argument("--strict-cap-frac", type=float, default=0.18)
    parser.add_argument("--relaxed-cap-frac", type=float, default=0.35)
    parser.add_argument("--soft-trigger", type=float, default=18.0)
    parser.add_argument("--deterioration-threshold", type=float, default=0.35)
    parser.add_argument("--activation-warmup", type=int, default=3)
    parser.add_argument("--aggressive-request-threshold", type=float, default=0.75)
    parser.add_argument("--aggressive-agent-fraction-trigger", type=float, default=0.34)
    parser.add_argument("--local-neighborhood-trigger", type=float, default=0.67)
    return parser.parse_args()


def _parse_csv_arg(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _regime_failure_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("test_") and c.endswith("_garden_failure_rate") and c != "test_garden_failure_rate"]


def _regime_name_from_col(col: str) -> str:
    name = col.removeprefix("test_").removesuffix("_garden_failure_rate")
    return name or "overall_test"


def _first_generation_at_or_above(generation_df: pd.DataFrame, column: str, threshold: float, default_generation: float) -> float:
    hits = generation_df[generation_df[column] >= threshold]
    if hits.empty:
        return float(default_generation)
    return float(hits["generation"].iloc[0])


def _summary_from_generation_df(condition: str, injector_mode: str, run_id: int, generation_df: pd.DataFrame) -> dict:
    n_generations = float(generation_df["generation"].max() + 1)
    regime_cols = _regime_failure_columns(generation_df)
    row = {
        "condition": condition,
        "injector_mode_requested": injector_mode,
        "run_id": run_id,
        "train_garden_failure_mean": float(generation_df["train_garden_failure_rate"].mean()),
        "test_garden_failure_mean": float(generation_df["test_garden_failure_rate"].mean()),
        "train_garden_failure_last": float(generation_df["train_garden_failure_rate"].iloc[-1]),
        "test_garden_failure_last": float(generation_df["test_garden_failure_rate"].iloc[-1]),
        "test_mean_patch_health_mean": float(generation_df["test_mean_patch_health"].mean()),
        "test_mean_welfare_mean": float(generation_df["test_mean_welfare"].mean()),
        "test_mean_credit_transferred_mean": float(generation_df["test_mean_credit_transferred"].mean()),
        "test_mean_aggressive_request_fraction_mean": float(generation_df["test_mean_aggressive_request_fraction"].mean()),
        "test_mean_max_local_aggression_mean": float(generation_df["test_mean_max_local_aggression"].mean()),
        "test_mean_neighborhood_overharvest_mean": float(generation_df["test_mean_neighborhood_overharvest"].mean()),
        "test_mean_capped_action_fraction_mean": float(generation_df["test_mean_capped_action_fraction"].mean()),
        "test_mean_targeted_agent_fraction_mean": float(generation_df["test_mean_targeted_agent_fraction"].mean()),
        "test_missed_target_rate_mean": float(generation_df["test_missed_target_rate"].mean()) if "test_missed_target_rate" in generation_df.columns else 0.0,
        "test_targeted_share_mean": float(generation_df["test_targeted_share"].mean()) if "test_targeted_share" in generation_df.columns else 0.0,
        "test_delayed_intervention_count_mean": float(generation_df["test_delayed_intervention_count"].mean()) if "test_delayed_intervention_count" in generation_df.columns else 0.0,
        "test_governance_budget_spent_mean": float(generation_df["test_governance_budget_spent"].mean()) if "test_governance_budget_spent" in generation_df.columns else 0.0,
        "test_mean_prevented_harvest_mean": float(generation_df["test_mean_prevented_harvest"].mean()),
        "test_mean_patch_variance_mean": float(generation_df["test_mean_patch_variance"].mean()),
        "test_mean_requested_harvest_mean": float(generation_df["test_mean_requested_harvest"].mean()),
        "test_mean_realized_harvest_mean": float(generation_df["test_mean_realized_harvest"].mean()),
        "test_local_safe_action_fraction_mean": float(generation_df["test_local_safe_action_fraction"].mean()) if "test_local_safe_action_fraction" in generation_df.columns else 0.0,
        "test_all_local_safe_step_fraction_mean": float(generation_df["test_all_local_safe_step_fraction"].mean()) if "test_all_local_safe_step_fraction" in generation_df.columns else 0.0,
        "test_global_unsafe_rate_mean": float(generation_df["test_global_unsafe_rate"].mean()) if "test_global_unsafe_rate" in generation_df.columns else 0.0,
        "test_local_pass_global_fail_rate_mean": float(generation_df["test_local_pass_global_fail_rate"].mean()) if "test_local_pass_global_fail_rate" in generation_df.columns else 0.0,
        "test_time_to_first_global_unsafe_mean": float(generation_df["test_time_to_first_global_unsafe"].mean()) if "test_time_to_first_global_unsafe" in generation_df.columns else n_generations,
        "time_to_garden_failure": _first_generation_at_or_above(
            generation_df=generation_df,
            column="test_garden_failure_rate",
            threshold=1.0 - 1e-12,
            default_generation=n_generations,
        ),
        "first_generation_test_failure_ge_0_8": _first_generation_at_or_above(
            generation_df=generation_df,
            column="test_garden_failure_rate",
            threshold=0.8,
            default_generation=n_generations,
        ),
        "test_regime_count": float(generation_df["test_regime_count"].iloc[0]) if "test_regime_count" in generation_df.columns else 1.0,
        "population_diversity_mean": float(generation_df["population_diversity"].mean()),
    }
    if regime_cols:
        per_regime_survival = []
        for col in regime_cols:
            val = float((1.0 - generation_df[col]).mean())
            row[f"per_regime_health_survival_over_generations__{_regime_name_from_col(col)}"] = val
            per_regime_survival.append(val)
        row["per_regime_health_survival_over_generations_mean"] = float(np.mean(per_regime_survival))
    else:
        row["per_regime_health_survival_over_generations_mean"] = float((1.0 - generation_df["test_garden_failure_rate"]).mean())
    return row


def _llm_integrity_summary(strategy_df: pd.DataFrame) -> dict[str, float | int]:
    integrity_df = _llm_integrity_base_df(strategy_df)
    if integrity_df.empty:
        row: dict[str, float | int] = {
            "llm_json_fraction": 0.0,
            "llm_fallback_fraction": 0.0,
            "direct_json_fraction": 0.0,
            "repaired_json_fraction": 0.0,
            "effective_llm_fraction": 0.0,
            "unrepaired_fallback_fraction": 0.0,
        }
        for error_type in HARVEST_LLM_PARSE_ERROR_TYPES:
            row[f"llm_parse_error_count__{error_type}"] = 0
        return row

    llm_parse_status = integrity_df["llm_parse_status"] if "llm_parse_status" in integrity_df.columns else pd.Series("", index=integrity_df.index)
    llm_parse_error_type = (
        integrity_df["llm_parse_error_type"] if "llm_parse_error_type" in integrity_df.columns else pd.Series("", index=integrity_df.index)
    )
    row = {
        "llm_json_fraction": float((integrity_df["origin"] == "llm_json").mean()),
        "llm_fallback_fraction": float((integrity_df["origin"] == "llm_fallback_mutation").mean()),
        "direct_json_fraction": float((llm_parse_status == "direct_json").mean()),
        "repaired_json_fraction": float((llm_parse_status == "repaired_json").mean()),
        "effective_llm_fraction": float(((llm_parse_status == "direct_json") | (llm_parse_status == "repaired_json")).mean()),
        "unrepaired_fallback_fraction": float((llm_parse_status == "fallback_mutation").mean()),
    }
    for error_type in HARVEST_LLM_PARSE_ERROR_TYPES:
        row[f"llm_parse_error_count__{error_type}"] = int((llm_parse_error_type == error_type).sum())
    return row


def _run_job(job: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tier = str(job["tier"])
    partner_mix = str(job["partner_mix"])
    condition = str(job["condition"])
    injector_mode = str(job["injector_mode"])
    pressure = float(job["pressure"])
    run_id = int(job["run_id"])
    scenario_preset = str(job.get("scenario_preset", ""))
    governance_friction_regime = str(job.get("governance_friction_regime", "ideal"))
    actor_capability_level = str(job.get("actor_capability_level", ""))
    overseer_capability_level = str(job.get("overseer_capability_level", ""))
    actor_capability_rank = job.get("actor_capability_rank", "")
    overseer_capability_rank = job.get("overseer_capability_rank", "")
    capability_gap = job.get("capability_gap", "")
    search_candidates = int(job.get("search_candidates", 0) or 0)
    search_eval_horizon = int(job.get("search_eval_horizon", 0) or 0)
    args = job["args"]
    llm_client = None
    if args["llm_policy_replay_file"]:
        llm_client = FileReplayPolicyLLMClient(path=str(args["llm_policy_replay_file"]))
    elif injector_mode == "llm_json":
        if args["llm_provider"] == "ollama":
            llm_client = OllamaPolicyLLMClient(
                model=str(args["llm_model"]),
                base_url=args["llm_base_url"],
                timeout_s=float(args["llm_timeout_s"]),
                temperature=float(args["llm_temperature"]),
            )
        elif args["llm_provider"] == "openai":
            api_key = os.environ.get(str(args["llm_api_key_env"]))
            if not api_key:
                raise ValueError(
                    f"{args['llm_api_key_env']} is required for OpenAI injection. "
                    "Set the env var or switch to --llm-provider ollama / replay file."
                )
            llm_client = OpenAIResponsesPolicyLLMClient(
                model=str(args["llm_model"]),
                api_key=api_key,
                base_url=args["llm_base_url"],
                timeout_s=float(args["llm_timeout_s"]),
                temperature=float(args["llm_temperature"]),
            )
        else:
            raise ValueError(f"Unsupported llm provider: {args['llm_provider']}")

    if scenario_preset:
        cfg = make_harvest_cfg_for_scenario(
            scenario_preset,
            n_agents=int(args["population_size"]),
            seed=int(args["cfg_seed_start"]) + run_id * int(args["run_seed_stride"]),
        )
    else:
        cfg = make_harvest_cfg_for_tier(
            tier,
            n_agents=int(args["population_size"]),
            seed=int(args["cfg_seed_start"]) + run_id * int(args["run_seed_stride"]),
        )
    cfg.local_safety_margin = float(args["local_safety_margin"])
    cfg.global_min_mean_patch_health = float(args["global_min_mean_patch_health"])
    generation_df, strategy_df, agent_history_df = run_harvest_invasion(
        base_cfg=copy.deepcopy(cfg),
        condition=condition,
        generations=int(args["generations"]),
        population_size=int(args["population_size"]),
        seeds_per_generation=int(args["seeds_per_generation"]),
        test_seeds_per_generation=int(args["test_seeds_per_generation"]),
        replacement_fraction=float(args["replacement_fraction"]),
        adversarial_pressure=pressure,
        rng_seed=int(args["rng_seed_start"]) + run_id * int(args["run_seed_stride"]),
        partner_mix_preset=partner_mix,
        injector=make_harvest_strategy_injector(
            injector_mode,
            llm_client=llm_client,
            search_candidates=search_candidates,
            search_eval_horizon=search_eval_horizon,
        ),
        test_regimes=get_harvest_regime_pack(tier),
        government_params=dict(args["government_params"]),
        progress_callback=None,
    )
    generation_df = generation_df.assign(
        tier=tier,
        partner_mix=partner_mix,
        condition=condition,
        injector_mode_requested=injector_mode,
        adversarial_pressure=pressure,
        run_id=run_id,
        experiment_tag=str(args["experiment_tag"]),
        llm_provider=str(args["llm_provider"] if injector_mode == "llm_json" else "none"),
        llm_model=str(args["llm_model"] if injector_mode == "llm_json" else ""),
        scenario_preset=scenario_preset,
        governance_friction_regime=governance_friction_regime,
        actor_capability_level=actor_capability_level,
        overseer_capability_level=overseer_capability_level,
        actor_capability_rank=actor_capability_rank,
        overseer_capability_rank=overseer_capability_rank,
        capability_gap=capability_gap,
        search_candidates=search_candidates,
        search_eval_horizon=search_eval_horizon,
        local_safety_margin=float(args["local_safety_margin"]),
        global_min_mean_patch_health=float(args["global_min_mean_patch_health"]),
    )
    strategy_df = strategy_df.assign(
        tier=tier,
        partner_mix=partner_mix,
        condition=condition,
        injector_mode_requested=injector_mode,
        adversarial_pressure=pressure,
        run_id=run_id,
        experiment_tag=str(args["experiment_tag"]),
        llm_provider=str(args["llm_provider"] if injector_mode == "llm_json" else "none"),
        llm_model=str(args["llm_model"] if injector_mode == "llm_json" else ""),
        scenario_preset=scenario_preset,
        governance_friction_regime=governance_friction_regime,
        actor_capability_level=actor_capability_level,
        overseer_capability_level=overseer_capability_level,
        actor_capability_rank=actor_capability_rank,
        overseer_capability_rank=overseer_capability_rank,
        capability_gap=capability_gap,
        search_candidates=search_candidates,
        search_eval_horizon=search_eval_horizon,
        local_safety_margin=float(args["local_safety_margin"]),
        global_min_mean_patch_health=float(args["global_min_mean_patch_health"]),
    )
    agent_history_df = agent_history_df.assign(
        tier=tier,
        partner_mix=partner_mix,
        condition=condition,
        injector_mode_requested=injector_mode,
        adversarial_pressure=pressure,
        run_id=run_id,
        experiment_tag=str(args["experiment_tag"]),
        llm_provider=str(args["llm_provider"] if injector_mode == "llm_json" else "none"),
        llm_model=str(args["llm_model"] if injector_mode == "llm_json" else ""),
        scenario_preset=scenario_preset,
        governance_friction_regime=governance_friction_regime,
        actor_capability_level=actor_capability_level,
        overseer_capability_level=overseer_capability_level,
        actor_capability_rank=actor_capability_rank,
        overseer_capability_rank=overseer_capability_rank,
        capability_gap=capability_gap,
        search_candidates=search_candidates,
        search_eval_horizon=search_eval_horizon,
        local_safety_margin=float(args["local_safety_margin"]),
        global_min_mean_patch_health=float(args["global_min_mean_patch_health"]),
    )
    summary_row = _summary_from_generation_df(condition, injector_mode, run_id, generation_df)
    summary_row.update(
        {
            "tier": tier,
            "partner_mix": partner_mix,
            "adversarial_pressure": pressure,
            "generations": int(args["generations"]),
            "population_size": int(args["population_size"]),
            "seeds_per_generation": int(args["seeds_per_generation"]),
            "test_seeds_per_generation": int(args["test_seeds_per_generation"]),
            "replacement_fraction": float(args["replacement_fraction"]),
            "experiment_tag": str(args["experiment_tag"]),
            "llm_provider": str(args["llm_provider"] if injector_mode == "llm_json" else "none"),
            "llm_model": str(args["llm_model"] if injector_mode == "llm_json" else ""),
            "scenario_preset": scenario_preset,
            "governance_friction_regime": governance_friction_regime,
            "actor_capability_level": actor_capability_level,
            "overseer_capability_level": overseer_capability_level,
            "actor_capability_rank": actor_capability_rank,
            "overseer_capability_rank": overseer_capability_rank,
            "capability_gap": capability_gap,
            "search_candidates": search_candidates,
            "search_eval_horizon": search_eval_horizon,
            "local_safety_margin": float(args["local_safety_margin"]),
            "global_min_mean_patch_health": float(args["global_min_mean_patch_health"]),
        }
    )
    summary_row.update(_llm_integrity_summary(strategy_df))
    return summary_row, generation_df, strategy_df, agent_history_df


def main() -> None:
    args = parse_args()
    tiers = _parse_csv_arg(args.tiers)
    partner_mixes = _parse_csv_arg(args.partner_mixes)
    scenario_presets = _parse_csv_arg(args.scenario_presets)
    conditions = _parse_csv_arg(args.conditions)
    injector_modes = _parse_csv_arg(args.injector_modes)
    pressures = [float(x) for x in _parse_csv_arg(args.adversarial_pressures)]
    governance_friction_regimes = _parse_csv_arg(args.governance_friction_regimes)
    actor_capability_levels = _parse_csv_arg(args.actor_capability_levels)
    overseer_capability_levels = _parse_csv_arg(args.overseer_capability_levels)

    government_params = {
        "trigger": args.government_trigger,
        "strict_cap_frac": args.strict_cap_frac,
        "relaxed_cap_frac": args.relaxed_cap_frac,
        "soft_trigger": args.soft_trigger,
        "deterioration_threshold": args.deterioration_threshold,
        "activation_warmup": args.activation_warmup,
        "aggressive_request_threshold": args.aggressive_request_threshold,
        "aggressive_agent_fraction_trigger": args.aggressive_agent_fraction_trigger,
        "local_neighborhood_trigger": args.local_neighborhood_trigger,
    }

    job_cells = []
    if scenario_presets:
        for scenario_name in scenario_presets:
            scenario = get_harvest_scenario_preset(scenario_name)
            regimes = governance_friction_regimes or [""]
            for regime in regimes:
                job_cells.append(
                    {
                        "scenario_preset": scenario_name,
                        "tier": str(scenario["tier"]),
                        "partner_mix": str(scenario["partner_mix"]),
                        "governance_friction_regime": regime,
                    }
                )
    else:
        for tier in tiers:
            for partner_mix in partner_mixes:
                for regime in governance_friction_regimes or [""]:
                    job_cells.append(
                        {
                            "scenario_preset": "",
                            "tier": tier,
                            "partner_mix": partner_mix,
                            "governance_friction_regime": regime,
                        }
                    )

    actor_grid = actor_capability_levels or [""]
    overseer_grid = overseer_capability_levels or [""]
    injector_grid = [""] if actor_capability_levels else injector_modes
    pressure_grid = [0.0] if actor_capability_levels else pressures
    total_jobs = len(job_cells) * len(conditions) * len(actor_grid) * len(overseer_grid) * len(injector_grid) * len(pressure_grid) * args.n_runs
    total_steps = total_jobs * args.generations
    progress_bar = None
    use_job_progress = args.max_workers > 1
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(total=total_jobs if use_job_progress else total_steps, desc="Harvest invasion matrix")
    elif not args.no_progress:
        denom = total_jobs if use_job_progress else total_steps
        unit = "jobs" if use_job_progress else "generation-steps"
        print(f"Progress: 0/{denom} {unit}")

    def _progress(done: int, total: int) -> None:
        del done, total
        if progress_bar is not None:
            progress_bar.update(1)
            return
        _progress.state += 1
        denom = total_jobs if use_job_progress else total_steps
        unit = "jobs" if use_job_progress else "generation-steps"
        if _progress.state == denom or _progress.state % max(1, denom // 20) == 0:
            print(f"Progress: {_progress.state}/{denom} {unit}")

    _progress.state = 0  # type: ignore[attr-defined]

    per_run_rows: list[dict] = []
    generation_histories: list[pd.DataFrame] = []
    strategy_histories: list[pd.DataFrame] = []
    agent_histories: list[pd.DataFrame] = []

    job_specs = []
    for cell in job_cells:
        for condition in conditions:
            for actor_level in actor_grid:
                actor_preset = get_harvest_actor_capability_preset(actor_level) if actor_level else None
                resolved_injectors = [str(actor_preset["injector_mode"])] if actor_preset else injector_grid
                resolved_pressures = [float(actor_preset["adversarial_pressure"])] if actor_preset else pressure_grid
                resolved_search_candidates = int(actor_preset["search_candidates"]) if actor_preset else int(args.search_candidates)
                resolved_search_eval_horizon = int(actor_preset["search_eval_horizon"]) if actor_preset else int(args.search_eval_horizon)
                actor_rank = int(actor_preset["rank"]) if actor_preset else ""
                for overseer_level in overseer_grid:
                    merged_government_params = dict(government_params)
                    regime_name = str(cell["governance_friction_regime"])
                    if overseer_level:
                        overseer_preset = get_harvest_overseer_capability_preset(overseer_level)
                        merged_government_params.update(
                            {
                                key: value
                                for key, value in overseer_preset.items()
                                if key
                                in {
                                    "detection_recall",
                                    "enforcement_delay_rounds",
                                    "max_target_share",
                                    "governance_budget_cost",
                                }
                            }
                        )
                        overseer_rank = int(overseer_preset["rank"])
                        gap = harvest_capability_gap(actor_level or "low_actor", overseer_level)
                    else:
                        regime_params = get_harvest_governance_friction_regime(regime_name or "ideal")
                        merged_government_params.update(regime_params)
                        overseer_rank = ""
                        gap = ""
                    for injector_mode in resolved_injectors:
                        for pressure in resolved_pressures:
                            for run_id in range(args.n_runs):
                                job_specs.append(
                                    {
                                        "scenario_preset": str(cell["scenario_preset"]),
                                        "governance_friction_regime": regime_name,
                                        "tier": str(cell["tier"]),
                                        "partner_mix": str(cell["partner_mix"]),
                                        "condition": condition,
                                        "injector_mode": injector_mode,
                                        "pressure": float(pressure),
                                        "run_id": run_id,
                                        "actor_capability_level": actor_level,
                                        "overseer_capability_level": overseer_level,
                                        "actor_capability_rank": actor_rank,
                                        "overseer_capability_rank": overseer_rank,
                                        "capability_gap": gap,
                                        "search_candidates": resolved_search_candidates,
                                        "search_eval_horizon": resolved_search_eval_horizon,
                                        "args": {
                                            "generations": args.generations,
                                            "population_size": args.population_size,
                                            "seeds_per_generation": args.seeds_per_generation,
                                            "test_seeds_per_generation": args.test_seeds_per_generation,
                                            "replacement_fraction": args.replacement_fraction,
                                            "rng_seed_start": args.rng_seed_start,
                                            "cfg_seed_start": args.cfg_seed_start,
                                            "run_seed_stride": args.run_seed_stride,
                                            "local_safety_margin": args.local_safety_margin,
                                            "global_min_mean_patch_health": args.global_min_mean_patch_health,
                                            "government_params": merged_government_params,
                                            "experiment_tag": args.experiment_tag,
                                            "llm_policy_replay_file": args.llm_policy_replay_file,
                                            "llm_provider": args.llm_provider,
                                            "llm_model": args.llm_model,
                                            "llm_base_url": args.llm_base_url,
                                            "llm_api_key_env": args.llm_api_key_env,
                                            "llm_timeout_s": args.llm_timeout_s,
                                            "llm_temperature": args.llm_temperature,
                                        },
                                    }
                                )

    try:
        if args.max_workers > 1:
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(_run_job, job) for job in job_specs]
                for future in as_completed(futures):
                    summary_row, generation_df, strategy_df, agent_history_df = future.result()
                    generation_histories.append(generation_df)
                    strategy_histories.append(strategy_df)
                    agent_histories.append(agent_history_df)
                    per_run_rows.append(summary_row)
                    if not args.no_progress:
                        _progress(0, 0)
        else:
            for job in job_specs:
                summary_row, generation_df, strategy_df, agent_history_df = _run_job(job)
                generation_histories.append(generation_df)
                strategy_histories.append(strategy_df)
                agent_histories.append(agent_history_df)
                per_run_rows.append(summary_row)
                if not args.no_progress:
                    for _ in range(args.generations):
                        _progress(0, 0)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    runs_df = pd.DataFrame(per_run_rows)
    history_df = pd.concat(generation_histories, ignore_index=True) if generation_histories else pd.DataFrame()
    strategy_df = pd.concat(strategy_histories, ignore_index=True) if strategy_histories else pd.DataFrame()
    agent_history_df = pd.concat(agent_histories, ignore_index=True) if agent_histories else pd.DataFrame()

    runs_csv = f"{args.output_prefix}_runs.csv"
    history_csv = f"{args.output_prefix}_generation_history.csv"
    strategy_csv = f"{args.output_prefix}_strategy_history.csv"
    agent_history_csv = f"{args.output_prefix}_agent_history.csv"
    runs_df.to_csv(runs_csv, index=False)
    history_df.to_csv(history_csv, index=False)
    strategy_df.to_csv(strategy_csv, index=False)
    agent_history_df.to_csv(agent_history_csv, index=False)
    print(f"Saved: {runs_csv}")
    print(f"Saved: {history_csv}")
    print(f"Saved: {strategy_csv}")
    print(f"Saved: {agent_history_csv}")


if __name__ == "__main__":
    main()
