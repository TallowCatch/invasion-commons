from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fishery_sim.harvest_benchmarks import (
    get_harvest_governance_friction_regime,
    make_harvest_cfg_for_scenario,
)
from fishery_sim.harvest_evolution import evaluate_harvest_population
from fishery_sim.harvest_llm_population import load_harvest_strategy_bank, sample_population_from_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Harvest LLM-population governance maps.")
    parser.add_argument("--bank-csv", required=True)
    parser.add_argument("--scenarios", default="regulated_fishery,community_irrigation")
    parser.add_argument("--conditions", default="none,top_down_only,hybrid")
    parser.add_argument("--governance-friction-regime", default="ideal", choices=["ideal", "constrained"])
    parser.add_argument("--exploitative-shares", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--n-populations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--evaluation-seeds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    scenarios = _parse_csv(args.scenarios)
    conditions = _parse_csv(args.conditions)
    exploitative_shares = [float(part) for part in _parse_csv(args.exploitative_shares)]
    rng = np.random.default_rng(args.seed)

    bank_df = load_harvest_strategy_bank(args.bank_csv)
    model_labels = sorted(bank_df["bank_model_label"].dropna().astype(str).unique().tolist())
    if not model_labels:
        raise ValueError("No model labels found in bank CSV")

    regime_params = get_harvest_governance_friction_regime(args.governance_friction_regime)
    summary_rows: list[dict] = []
    episode_rows: list[dict] = []
    attitude_rows: list[dict] = []

    for scenario in scenarios:
        for model_label in model_labels:
            for condition in conditions:
                for exploitative_share in exploitative_shares:
                    for population_id in range(int(args.n_populations)):
                        cfg_seed = int(rng.integers(0, 1_000_000_000))
                        base_cfg = make_harvest_cfg_for_scenario(
                            scenario,
                            n_agents=int(args.population_size),
                            seed=cfg_seed,
                        )
                        population, attitudes = sample_population_from_bank(
                            bank_df,
                            population_size=int(args.population_size),
                            exploitative_share=exploitative_share,
                            rng=rng,
                            model_label=model_label,
                        )
                        eval_seeds = list(range(cfg_seed, cfg_seed + int(args.evaluation_seeds)))
                        episode_df, score_df, agent_df = evaluate_harvest_population(
                            base_cfg=base_cfg,
                            condition=condition,
                            population=population,
                            seeds=eval_seeds,
                            government_params=regime_params,
                        )
                        summary_rows.append(
                            {
                                "scenario_preset": scenario,
                                "bank_model_label": model_label,
                                "condition": condition,
                                "governance_friction_regime": args.governance_friction_regime,
                                "exploitative_share": exploitative_share,
                                "population_id": population_id,
                                "mean_patch_health": float(episode_df["mean_patch_health"].mean()),
                                "mean_neighborhood_overharvest": float(episode_df["mean_neighborhood_overharvest"].mean()),
                                "mean_welfare": float(episode_df["mean_welfare"].mean()),
                                "garden_failure_rate": float(episode_df["garden_failure_event"].mean()),
                                "mean_governance_budget_spent": float(episode_df["governance_budget_spent"].mean()),
                                "mean_missed_target_rate": float(episode_df["missed_target_rate"].mean()),
                                "mean_targeted_share": float(episode_df["targeted_share"].mean()),
                                "mean_exploitative_action_share": float(agent_df["aggressive_request_fraction"].mean())
                                if "aggressive_request_fraction" in agent_df.columns and not agent_df.empty
                                else float("nan"),
                            }
                        )
                        if not score_df.empty:
                            score_tmp = score_df.copy()
                            score_tmp["scenario_preset"] = scenario
                            score_tmp["bank_model_label"] = model_label
                            score_tmp["condition"] = condition
                            score_tmp["governance_friction_regime"] = args.governance_friction_regime
                            score_tmp["exploitative_share"] = exploitative_share
                            score_tmp["population_id"] = population_id
                            episode_rows.extend(score_tmp.to_dict("records"))
                        for agent_index, attitude in enumerate(attitudes):
                            attitude_rows.append(
                                {
                                    "scenario_preset": scenario,
                                    "bank_model_label": model_label,
                                    "condition": condition,
                                    "governance_friction_regime": args.governance_friction_regime,
                                    "exploitative_share": exploitative_share,
                                    "population_id": population_id,
                                    "agent_index": agent_index,
                                    "bank_attitude": attitude,
                                }
                            )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        aggregate_df = (
            summary_df.groupby(
                ["scenario_preset", "bank_model_label", "condition", "governance_friction_regime", "exploitative_share"],
                as_index=False,
            )
            .agg(
                n_populations=("population_id", "count"),
                mean_patch_health=("mean_patch_health", "mean"),
                mean_neighborhood_overharvest=("mean_neighborhood_overharvest", "mean"),
                mean_welfare=("mean_welfare", "mean"),
                garden_failure_rate=("garden_failure_rate", "mean"),
                mean_governance_budget_spent=("mean_governance_budget_spent", "mean"),
                mean_missed_target_rate=("mean_missed_target_rate", "mean"),
                mean_targeted_share=("mean_targeted_share", "mean"),
                mean_exploitative_action_share=("mean_exploitative_action_share", "mean"),
            )
            .reset_index(drop=True)
        )
    else:
        aggregate_df = pd.DataFrame()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = str(output_prefix.with_name(output_prefix.name + "_samples.csv"))
    aggregate_csv = str(output_prefix.with_name(output_prefix.name + "_summary.csv"))
    episode_csv = str(output_prefix.with_name(output_prefix.name + "_strategy_scores.csv"))
    attitude_csv = str(output_prefix.with_name(output_prefix.name + "_attitudes.csv"))
    summary_df.to_csv(summary_csv, index=False)
    aggregate_df.to_csv(aggregate_csv, index=False)
    pd.DataFrame(episode_rows).to_csv(episode_csv, index=False)
    pd.DataFrame(attitude_rows).to_csv(attitude_csv, index=False)
    print(f"Saved: {summary_csv}")
    print(f"Saved: {aggregate_csv}")
    print(f"Saved: {episode_csv}")
    print(f"Saved: {attitude_csv}")


if __name__ == "__main__":
    main()
