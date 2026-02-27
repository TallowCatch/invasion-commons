import argparse
import copy
import math
import os

import numpy as np
import pandas as pd

from fishery_sim.benchmarks import get_benchmark_pack, load_benchmark_pack_file
from fishery_sim.config import load_config
from fishery_sim.evolution import make_strategy_injector, run_evolutionary_invasion
from fishery_sim.llm_adapter import FileReplayPolicyLLMClient, OpenAIResponsesPolicyLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Governance ablation with confidence intervals and fixed held-out benchmark packs."
    )
    parser.add_argument("--config", default="experiments/configs/base.yaml")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--seeds-per-generation", type=int, default=64)
    parser.add_argument("--test-seeds-per-generation", type=int, default=None)
    parser.add_argument("--replacement-fraction", type=float, default=0.3)
    parser.add_argument("--adversarial-pressure", type=float, default=0.7)
    parser.add_argument("--collapse-penalty", type=float, default=50.0)

    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--rng-seed-start", type=int, default=0)
    parser.add_argument("--cfg-seed-start", type=int, default=0)
    parser.add_argument("--run-seed-stride", type=int, default=10_000)

    parser.add_argument("--injector-mode", choices=["mutation", "llm_json"], default="mutation")
    parser.add_argument("--llm-policy-replay-file", default=None)
    parser.add_argument("--llm-provider", choices=["openai"], default="openai")
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout-s", type=float, default=45.0)
    parser.add_argument("--llm-temperature", type=float, default=0.8)

    parser.add_argument("--train-regen-rate", type=float, default=None)
    parser.add_argument("--train-obs-noise-std", type=float, default=None)
    parser.add_argument("--test-regen-rate", type=float, default=None)
    parser.add_argument("--test-obs-noise-std", type=float, default=None)
    parser.add_argument("--benchmark-pack", default="harsh_v1")
    parser.add_argument("--benchmark-pack-file", default=None)
    parser.add_argument("--benchmark-pack-file-name", default=None)

    parser.add_argument("--monitoring-prob", type=float, default=0.9)
    parser.add_argument("--quota-fraction", type=float, default=0.07)
    parser.add_argument("--sanction-base-fine-rate", type=float, default=2.0)
    parser.add_argument("--sanction-fine-growth", type=float, default=0.8)
    parser.add_argument("--output-prefix", default="results/runs/ablation/governance_ablation")
    return parser.parse_args()


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean()) if arr.size else 0.0
    if arr.size <= 1:
        return mean, mean, mean
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    delta = 1.96 * se
    return mean, mean - delta, mean + delta


def _build_llm_client(args: argparse.Namespace):
    if args.llm_policy_replay_file:
        return FileReplayPolicyLLMClient(path=args.llm_policy_replay_file)
    if args.injector_mode != "llm_json":
        return None
    api_key = os.environ.get(args.llm_api_key_env)
    if not api_key:
        raise ValueError(
            f"{args.llm_api_key_env} is required for live LLM injection. "
            "Set the env var or pass --llm-policy-replay-file."
        )
    if args.llm_provider != "openai":
        raise ValueError(f"Unsupported llm provider: {args.llm_provider}")
    return OpenAIResponsesPolicyLLMClient(
        model=args.llm_model,
        api_key=api_key,
        base_url=args.llm_base_url,
        timeout_s=args.llm_timeout_s,
        temperature=args.llm_temperature,
    )


def _summary_from_generation_df(condition: str, run_id: int, generation_df: pd.DataFrame) -> dict:
    return {
        "condition": condition,
        "run_id": run_id,
        "train_collapse_mean": float(generation_df["train_collapse_rate"].mean()),
        "test_collapse_mean": float(generation_df["test_collapse_rate"].mean()),
        "train_collapse_last": float(generation_df["train_collapse_rate"].iloc[-1]),
        "test_collapse_last": float(generation_df["test_collapse_rate"].iloc[-1]),
        "test_mean_stock_mean": float(generation_df["test_mean_stock"].mean()),
        "test_mean_welfare_mean": float(generation_df["test_mean_welfare"].mean()),
        "test_generations_collapse_le_0_5": float((generation_df["test_collapse_rate"] <= 0.5).sum()),
        "test_regime_count": float(generation_df["test_regime_count"].iloc[0])
        if "test_regime_count" in generation_df.columns
        else 1.0,
    }


def _aggregate_with_ci(per_run_df: pd.DataFrame) -> pd.DataFrame:
    metric_keys = [
        "train_collapse_mean",
        "test_collapse_mean",
        "train_collapse_last",
        "test_collapse_last",
        "test_mean_stock_mean",
        "test_mean_welfare_mean",
        "test_generations_collapse_le_0_5",
    ]
    rows = []
    for condition, cdf in per_run_df.groupby("condition", sort=False):
        row = {"condition": condition, "n_runs": int(len(cdf))}
        for key in metric_keys:
            mean, lo, hi = _ci95(cdf[key].tolist())
            row[f"{key}_mean"] = round(mean, 4)
            row[f"{key}_ci95_low"] = round(lo, 4)
            row[f"{key}_ci95_high"] = round(hi, 4)
        row["test_regime_count"] = int(cdf["test_regime_count"].iloc[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("test_collapse_mean_mean", ascending=True).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)

    llm_client = _build_llm_client(args)
    injector = make_strategy_injector(injector_mode=args.injector_mode, llm_client=llm_client)

    train_overrides = {
        "regen_rate": args.train_regen_rate,
        "obs_noise_std": args.train_obs_noise_std,
    }
    test_overrides = {
        "regen_rate": args.test_regen_rate,
        "obs_noise_std": args.test_obs_noise_std,
    }
    test_regimes = None
    if args.benchmark_pack_file:
        test_regimes = load_benchmark_pack_file(
            path=args.benchmark_pack_file,
            pack_name=args.benchmark_pack_file_name,
        )
    elif args.benchmark_pack:
        test_regimes = get_benchmark_pack(args.benchmark_pack)

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    conditions = {
        "none": {
            "monitoring_prob": 0.0,
            "quota_fraction": 0.0,
            "base_fine_rate": 0.0,
            "fine_growth": 0.0,
        },
        "monitoring": {
            "monitoring_prob": args.monitoring_prob,
            "quota_fraction": args.quota_fraction,
            "base_fine_rate": 0.0,
            "fine_growth": 0.0,
        },
        "monitoring_sanctions": {
            "monitoring_prob": args.monitoring_prob,
            "quota_fraction": args.quota_fraction,
            "base_fine_rate": args.sanction_base_fine_rate,
            "fine_growth": args.sanction_fine_growth,
        },
    }

    per_run_rows = []
    for name, knobs in conditions.items():
        for run_id in range(args.n_runs):
            cfg = copy.deepcopy(base_cfg)
            cfg.monitoring_prob = knobs["monitoring_prob"]
            cfg.quota_fraction = knobs["quota_fraction"]
            cfg.base_fine_rate = knobs["base_fine_rate"]
            cfg.fine_growth = knobs["fine_growth"]
            cfg.seed = args.cfg_seed_start + run_id * args.run_seed_stride

            run_rng_seed = args.rng_seed_start + run_id * args.run_seed_stride
            generation_df, strategy_df = run_evolutionary_invasion(
                base_cfg=cfg,
                generations=args.generations,
                population_size=args.population_size,
                seeds_per_generation=args.seeds_per_generation,
                test_seeds_per_generation=args.test_seeds_per_generation,
                replacement_fraction=args.replacement_fraction,
                collapse_penalty=args.collapse_penalty,
                adversarial_pressure=args.adversarial_pressure,
                rng_seed=run_rng_seed,
                train_overrides=train_overrides,
                test_overrides=test_overrides,
                test_regimes=test_regimes,
                injector=injector,
            )

            generation_path = f"{args.output_prefix}_{name}_run{run_id:02d}_generations.csv"
            strategy_path = f"{args.output_prefix}_{name}_run{run_id:02d}_strategies.csv"
            generation_df.to_csv(generation_path, index=False)
            strategy_df.to_csv(strategy_path, index=False)
            per_run_rows.append(_summary_from_generation_df(name, run_id, generation_df))

    per_run_df = pd.DataFrame(per_run_rows)
    table_df = _aggregate_with_ci(per_run_df)

    runs_csv = f"{args.output_prefix}_runs.csv"
    table_csv = f"{args.output_prefix}_table.csv"
    table_md = f"{args.output_prefix}_table.md"
    per_run_df.to_csv(runs_csv, index=False)
    table_df.to_csv(table_csv, index=False)
    with open(table_md, "w", encoding="utf-8") as f:
        f.write(_to_markdown_table(table_df))
        f.write("\n")

    print(f"Saved: {runs_csv}")
    print(f"Saved: {table_csv}")
    print(f"Saved: {table_md}")
    if test_regimes:
        print("Benchmark regimes:", ", ".join([r["name"] for r in test_regimes]))
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    main()

