import argparse
import copy
import os
from datetime import datetime

import pandas as pd

from fishery_sim.benchmarks import get_benchmark_pack
from fishery_sim.config import load_config
from fishery_sim.evolution import make_strategy_injector, run_evolutionary_invasion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a project showcase report with rationale + empirical summaries."
    )
    parser.add_argument("--config", default="experiments/configs/base.yaml")
    parser.add_argument("--ablation-table", default="results/runs/ablation/governance_ablation_table.csv")
    parser.add_argument("--invasion-generations", default="results/runs/invasion/invasion_generations.csv")
    parser.add_argument("--output", default="results/runs/showcase/showcase_report.md")
    parser.add_argument("--run-quick-ablation", action="store_true")
    parser.add_argument("--quick-generations", type=int, default=10)
    parser.add_argument("--quick-seeds", type=int, default=24)
    parser.add_argument("--quick-population", type=int, default=10)
    return parser.parse_args()


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _quick_ablation_table(cfg_path: str, generations: int, seeds: int, population: int) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    injector = make_strategy_injector("mutation")
    conditions = [
        ("none", 0.0, 0.0, 0.0, 0.0),
        ("monitoring", 0.9, 0.07, 0.0, 0.0),
        ("monitoring_sanctions", 0.9, 0.07, 2.0, 0.8),
    ]
    rows = []
    for name, monitoring_prob, quota_fraction, base_fine_rate, fine_growth in conditions:
        c = copy.deepcopy(cfg)
        c.monitoring_prob = monitoring_prob
        c.quota_fraction = quota_fraction
        c.base_fine_rate = base_fine_rate
        c.fine_growth = fine_growth
        generation_df, _ = run_evolutionary_invasion(
            base_cfg=c,
            generations=generations,
            population_size=population,
            seeds_per_generation=seeds,
            replacement_fraction=0.3,
            collapse_penalty=50.0,
            adversarial_pressure=0.7,
            rng_seed=0,
            train_overrides=None,
            test_overrides=None,
            test_regimes=get_benchmark_pack("harsh_v1"),
            injector=injector,
        )
        rows.append(
            {
                "condition": name,
                "test_collapse_mean": round(float(generation_df["test_collapse_rate"].mean()), 4),
                "test_mean_stock_mean": round(float(generation_df["test_mean_stock"].mean()), 3),
                "test_mean_welfare_mean": round(float(generation_df["test_mean_welfare"].mean()), 3),
                "test_generations_collapse_le_0_5": int((generation_df["test_collapse_rate"] <= 0.5).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("test_collapse_mean", ascending=True).reset_index(drop=True)


def _markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No data available._"
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def _invasion_summary(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {}
    return {
        "generations": int(len(df)),
        "train_collapse_first": round(float(df["train_collapse_rate"].iloc[0]), 4),
        "train_collapse_last": round(float(df["train_collapse_rate"].iloc[-1]), 4),
        "test_collapse_first": round(float(df["test_collapse_rate"].iloc[0]), 4),
        "test_collapse_last": round(float(df["test_collapse_rate"].iloc[-1]), 4),
        "test_mean_stock_mean": round(float(df["test_mean_stock"].mean()), 3),
    }


def main() -> None:
    args = parse_args()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ablation_df = _safe_read_csv(args.ablation_table)
    if args.run_quick_ablation or ablation_df is None:
        ablation_df = _quick_ablation_table(
            cfg_path=args.config,
            generations=args.quick_generations,
            seeds=args.quick_seeds,
            population=args.quick_population,
        )

    invasion_df = _safe_read_csv(args.invasion_generations)
    inv = _invasion_summary(invasion_df)

    lines = []
    lines.append("# Fishery Commons Showcase")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("## Problem We Are Tackling")
    lines.append("")
    lines.append(
        "We are not only testing whether cooperation can emerge in a commons. "
        "We are testing whether cooperative norms remain stable under continuous strategy invasion pressure, "
        "especially in held-out harsher regimes."
    )
    lines.append("")
    lines.append("## Why These Design Choices")
    lines.append("")
    lines.append("- Minimal global-stock environment: keeps causal interpretation clean.")
    lines.append("- Strategy-level agents (not opaque token-by-token behavior): supports policy inspection.")
    lines.append("- Evolutionary turnover with replacement: pressure-tests norm stability against invaders.")
    lines.append("- Train/test regime split: avoids over-claiming from in-distribution behavior.")
    lines.append("- Governance knobs (monitoring, quota enforcement, sanctions): tests institutional defenses.")
    lines.append("")
    lines.append("## Current Invasion Snapshot")
    lines.append("")
    if inv:
        lines.append(f"- Generations analysed: {inv['generations']}")
        lines.append(
            f"- Train collapse rate: {inv['train_collapse_first']} -> {inv['train_collapse_last']}"
        )
        lines.append(
            f"- Test collapse rate: {inv['test_collapse_first']} -> {inv['test_collapse_last']}"
        )
        lines.append(f"- Mean held-out stock across generations: {inv['test_mean_stock_mean']}")
    else:
        lines.append("- No invasion generation CSV found yet.")
    lines.append("")
    lines.append("## Governance Ablation (Publishable Table)")
    lines.append("")
    lines.append(_markdown_table(ablation_df))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The ablation compares identical invasion settings while varying only governance. "
        "This isolates whether monitoring and sanctions increase resistance to collapse under adversarial pressure."
    )
    lines.append("")
    lines.append("## Recommended Next Experiments")
    lines.append("")
    lines.append("- Swap mutation injector for `llm_json` injector using replay or live model outputs.")
    lines.append("- Increase seeds and generations to tighten confidence intervals.")
    lines.append("- Run held-out stress regimes with lower regeneration and higher observation noise.")
    lines.append("")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved showcase report: {args.output}")


if __name__ == "__main__":
    main()
