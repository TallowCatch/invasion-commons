import argparse
import copy
import os

import pandas as pd

from fishery_sim.harvest import BaseHarvestAgent
from fishery_sim.harvest import CreditSharingHarvestAgent
from fishery_sim.harvest import GovernmentAgent
from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.harvest import ReciprocalHarvestAgent
from fishery_sim.harvest import SelfInterestedHarvestAgent
from fishery_sim.harvest import run_harvest_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study 2 harvest commons governance comparison.")
    parser.add_argument("--n-runs", type=int, default=12)
    parser.add_argument("--output-prefix", default="results/runs/harvest/harvest_commons_study")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--social-mixes", default="cooperative_heavy,mixed_pressure")
    parser.add_argument("--government-trigger", type=float, default=16.0)
    parser.add_argument("--strict-cap-frac", type=float, default=0.18)
    parser.add_argument("--relaxed-cap-frac", type=float, default=0.35)
    parser.add_argument("--soft-trigger", type=float, default=18.0)
    parser.add_argument("--deterioration-threshold", type=float, default=0.35)
    parser.add_argument("--activation-warmup", type=int, default=3)
    return parser.parse_args()


def _parse_csv_arg(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _agents_for_social_mix(social_mix: str, n_agents: int) -> list[BaseHarvestAgent]:
    if n_agents != 6:
        raise ValueError("Current harvest study presets assume n_agents=6.")
    if social_mix == "cooperative_heavy":
        return [
            ReciprocalHarvestAgent(),
            CreditSharingHarvestAgent(),
            ReciprocalHarvestAgent(),
            CreditSharingHarvestAgent(),
            ReciprocalHarvestAgent(),
            CreditSharingHarvestAgent(),
        ]
    if social_mix == "mixed_pressure":
        return [
            SelfInterestedHarvestAgent(),
            ReciprocalHarvestAgent(),
            CreditSharingHarvestAgent(),
            SelfInterestedHarvestAgent(),
            ReciprocalHarvestAgent(),
            CreditSharingHarvestAgent(),
        ]
    if social_mix == "adversarial_heavy":
        return [
            SelfInterestedHarvestAgent(),
            SelfInterestedHarvestAgent(),
            SelfInterestedHarvestAgent(),
            SelfInterestedHarvestAgent(),
            ReciprocalHarvestAgent(),
            CreditSharingHarvestAgent(),
        ]
    raise ValueError(f"Unknown social mix: {social_mix}")


def _condition_setup(
    condition: str,
    social_mix: str,
    n_agents: int,
    governor_params: dict[str, float | int],
) -> tuple[list[BaseHarvestAgent], GovernmentAgent | None, HarvestCommonsConfig]:
    agents = _agents_for_social_mix(social_mix, n_agents=n_agents)
    cfg = HarvestCommonsConfig(n_agents=n_agents)
    governor = GovernmentAgent(**governor_params)

    if condition == "none":
        cfg.communication_enabled = False
        cfg.side_payments_enabled = False
        return agents, None, cfg
    if condition == "top_down_only":
        cfg.communication_enabled = False
        cfg.side_payments_enabled = False
        return agents, governor, cfg
    if condition == "bottom_up_only":
        return agents, None, cfg
    if condition == "hybrid":
        return agents, governor, cfg
    raise ValueError(f"Unknown condition: {condition}")


def main() -> None:
    args = parse_args()
    conditions = ["none", "top_down_only", "bottom_up_only", "hybrid"]
    social_mixes = _parse_csv_arg(args.social_mixes)
    rows = []

    for social_mix in social_mixes:
        governor_params = {
            "trigger": args.government_trigger,
            "strict_cap_frac": args.strict_cap_frac,
            "relaxed_cap_frac": args.relaxed_cap_frac,
            "soft_trigger": args.soft_trigger,
            "deterioration_threshold": args.deterioration_threshold,
            "activation_warmup": args.activation_warmup,
        }
        for condition in conditions:
            for run_id in range(args.n_runs):
                agents, condition_governor, cfg = _condition_setup(
                    condition=condition,
                    social_mix=social_mix,
                    n_agents=args.n_agents,
                    governor_params=governor_params,
                )
                cfg = copy.deepcopy(cfg)
                cfg.seed = args.seed_start + run_id
                out = run_harvest_episode(cfg, agents, governor=condition_governor)
                rows.append(
                    {
                        "social_mix": social_mix,
                        "condition": condition,
                        "run_id": run_id,
                        "mean_patch_health": out["mean_patch_health"],
                        "final_patch_health": out["final_patch_health"],
                        "total_welfare": out["total_welfare"],
                        "mean_welfare": out["mean_welfare"],
                        "payoff_gini": out["payoff_gini"],
                        "total_credit_transferred": out["total_credit_transferred"],
                        "mean_credit_transferred": out["mean_credit_transferred"],
                        "mean_government_cap": out["mean_government_cap"],
                    }
                )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["social_mix", "condition"], sort=False)
        .agg(
            mean_patch_health_mean=("mean_patch_health", "mean"),
            final_patch_health_mean=("final_patch_health", "mean"),
            mean_welfare_mean=("mean_welfare", "mean"),
            payoff_gini_mean=("payoff_gini", "mean"),
            total_credit_transferred_mean=("total_credit_transferred", "mean"),
            mean_government_cap_mean=("mean_government_cap", "mean"),
            n_runs=("run_id", "count"),
        )
        .reset_index()
    )

    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    runs_csv = f"{args.output_prefix}_runs.csv"
    table_csv = f"{args.output_prefix}_table.csv"
    df.to_csv(runs_csv, index=False)
    summary.to_csv(table_csv, index=False)
    print(f"Saved: {runs_csv}")
    print(f"Saved: {table_csv}")


if __name__ == "__main__":
    main()
