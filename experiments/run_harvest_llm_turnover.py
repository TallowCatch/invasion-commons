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
from fishery_sim.harvest_llm_population import load_harvest_strategy_bank, sample_strategy_from_gene_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Harvest LLM-population turnover pilot.")
    parser.add_argument("--bank-csv", required=True)
    parser.add_argument("--scenarios", default="regulated_fishery,community_irrigation")
    parser.add_argument("--conditions", default="none,top_down_only,hybrid")
    parser.add_argument("--governance-friction-regime", default="ideal", choices=["ideal", "constrained"])
    parser.add_argument("--population-size", type=int, default=128)
    parser.add_argument("--survivor-count", type=int, default=16)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=30)
    parser.add_argument("--evaluation-seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _initial_gene_population(model_labels: list[str], population_size: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    genes: list[tuple[str, str]] = []
    attitudes = ["cooperative", "exploitative"]
    for _ in range(population_size):
        genes.append((str(rng.choice(model_labels)), str(rng.choice(attitudes))))
    return genes


def _positive_weights(values: np.ndarray) -> np.ndarray:
    shifted = values - values.min() + 1e-6
    total = shifted.sum()
    if total <= 0:
        return np.full_like(shifted, 1.0 / len(shifted))
    return shifted / total


def main() -> None:
    args = parse_args()
    scenarios = _parse_csv(args.scenarios)
    conditions = _parse_csv(args.conditions)
    rng = np.random.default_rng(args.seed)
    regime_params = get_harvest_governance_friction_regime(args.governance_friction_regime)
    bank_df = load_harvest_strategy_bank(args.bank_csv)
    model_labels = sorted(bank_df["bank_model_label"].dropna().astype(str).unique().tolist())
    if not model_labels:
        raise ValueError("No model labels found in bank CSV")

    trajectory_rows: list[dict] = []
    gene_rows: list[dict] = []

    for scenario in scenarios:
        for condition in conditions:
            for simulation_id in range(int(args.simulations)):
                gene_population = _initial_gene_population(model_labels, int(args.population_size), rng)
                for generation in range(int(args.generations)):
                    cfg_seed = int(rng.integers(0, 1_000_000_000))
                    base_cfg = make_harvest_cfg_for_scenario(
                        scenario,
                        n_agents=int(args.population_size),
                        seed=cfg_seed,
                    )
                    population = [
                        sample_strategy_from_gene_bank(
                            bank_df,
                            model_label=model_label,
                            attitude=attitude,
                            rng=rng,
                            strategy_id=f"sim{simulation_id}_g{generation}_a{i}",
                        )
                        for i, (model_label, attitude) in enumerate(gene_population)
                    ]
                    eval_seeds = list(range(cfg_seed, cfg_seed + int(args.evaluation_seeds)))
                    episode_df, score_df, _ = evaluate_harvest_population(
                        base_cfg=base_cfg,
                        condition=condition,
                        population=population,
                        seeds=eval_seeds,
                        government_params=regime_params,
                    )
                    cooperative_share = float(np.mean([1.0 if attitude == "cooperative" else 0.0 for _, attitude in gene_population]))
                    exploitative_share = 1.0 - cooperative_share
                    trajectory_rows.append(
                        {
                            "scenario_preset": scenario,
                            "condition": condition,
                            "governance_friction_regime": args.governance_friction_regime,
                            "simulation_id": simulation_id,
                            "generation": generation,
                            "cooperative_gene_share": cooperative_share,
                            "exploitative_gene_share": exploitative_share,
                            "mean_patch_health": float(episode_df["mean_patch_health"].mean()),
                            "mean_neighborhood_overharvest": float(episode_df["mean_neighborhood_overharvest"].mean()),
                            "mean_welfare": float(episode_df["mean_welfare"].mean()),
                            "garden_failure_rate": float(episode_df["garden_failure_event"].mean()),
                            "mean_governance_budget_spent": float(episode_df["governance_budget_spent"].mean()),
                        }
                    )
                    if not score_df.empty:
                        tmp_score_df = score_df.copy()
                        tmp_score_df["model_label"] = [model_label for model_label, _ in gene_population]
                        tmp_score_df["bank_attitude"] = [attitude for _, attitude in gene_population]
                        tmp_score_df["scenario_preset"] = scenario
                        tmp_score_df["condition"] = condition
                        tmp_score_df["governance_friction_regime"] = args.governance_friction_regime
                        tmp_score_df["simulation_id"] = simulation_id
                        tmp_score_df["generation"] = generation
                        gene_rows.extend(tmp_score_df.to_dict("records"))

                    if generation == int(args.generations) - 1:
                        continue

                    ranked = score_df.reset_index().rename(columns={"index": "population_index"}).sort_values("fitness", ascending=False).reset_index(drop=True)
                    survivor_count = min(int(args.survivor_count), len(ranked))
                    survivor_indices = ranked["population_index"].iloc[:survivor_count].tolist()
                    survivor_genes = [gene_population[int(idx)] for idx in survivor_indices]
                    survivor_fitness = ranked["fitness"].iloc[:survivor_count].to_numpy(dtype=float)
                    survivor_weights = _positive_weights(survivor_fitness)
                    next_population = list(survivor_genes)
                    while len(next_population) < int(args.population_size):
                        choice = int(rng.choice(len(survivor_genes), p=survivor_weights))
                        next_population.append(survivor_genes[choice])
                    rng.shuffle(next_population)
                    gene_population = next_population

    trajectory_df = pd.DataFrame(trajectory_rows)
    if not trajectory_df.empty:
        summary_df = (
            trajectory_df.groupby(
                ["scenario_preset", "condition", "governance_friction_regime", "generation"],
                as_index=False,
            )
            .agg(
                n_simulations=("simulation_id", "count"),
                cooperative_gene_share=("cooperative_gene_share", "mean"),
                exploitative_gene_share=("exploitative_gene_share", "mean"),
                mean_patch_health=("mean_patch_health", "mean"),
                mean_neighborhood_overharvest=("mean_neighborhood_overharvest", "mean"),
                mean_welfare=("mean_welfare", "mean"),
                garden_failure_rate=("garden_failure_rate", "mean"),
                mean_governance_budget_spent=("mean_governance_budget_spent", "mean"),
            )
            .reset_index(drop=True)
        )
    else:
        summary_df = pd.DataFrame()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trajectory_csv = str(output_prefix.with_name(output_prefix.name + "_trajectories.csv"))
    summary_csv = str(output_prefix.with_name(output_prefix.name + "_summary.csv"))
    genes_csv = str(output_prefix.with_name(output_prefix.name + "_gene_scores.csv"))
    trajectory_df.to_csv(trajectory_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    pd.DataFrame(gene_rows).to_csv(genes_csv, index=False)
    print(f"Saved: {trajectory_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {genes_csv}")


if __name__ == "__main__":
    main()
