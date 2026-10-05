import argparse
import math
import os

import numpy as np
import pandas as pd

from fishery_sim.config import load_config
from fishery_sim.fishery_rl import PPOTrainConfig
from fishery_sim.fishery_rl import apply_rl_condition
from fishery_sim.fishery_rl import evaluate_self_play_policy
from fishery_sim.fishery_rl import load_benchmark_pack_from_args
from fishery_sim.fishery_rl import resolve_torch_device
from fishery_sim.fishery_rl import save_rl_checkpoint
from fishery_sim.fishery_rl import train_self_play_policy

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-seed Fishery RL baseline sweep.")
    parser.add_argument("--config", default="experiments/configs/base.yaml")
    parser.add_argument("--conditions", default="none,monitoring_sanctions")
    parser.add_argument("--benchmark-pack", default="medium_v1")
    parser.add_argument("--benchmark-pack-file", default=None)
    parser.add_argument("--benchmark-pack-file-name", default=None)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--run-seed-start", type=int, default=0)
    parser.add_argument("--run-seed-stride", type=int, default=1_000)
    parser.add_argument("--train-regen-rate", type=float, default=2.0)
    parser.add_argument("--train-obs-noise-std", type=float, default=6.0)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--rollout-steps", type=int, default=1_024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--action-bins", type=int, default=11)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--train-eval-episodes", type=int, default=16)
    parser.add_argument("--n-eval-episodes", type=int, default=32)
    parser.add_argument("--collapse-penalty", type=float, default=500.0)
    parser.add_argument("--curriculum-regen-jitter", type=float, default=0.6)
    parser.add_argument("--curriculum-obs-noise-jitter", type=float, default=10.0)
    parser.add_argument("--curriculum-stock-init-jitter", type=float, default=20.0)
    parser.add_argument("--curriculum-eval-episodes", type=int, default=8)
    parser.add_argument("--stock-health-reward-weight", type=float, default=0.0)
    parser.add_argument("--low-stock-penalty-weight", type=float, default=0.0)
    parser.add_argument("--low-stock-penalty-fraction", type=float, default=0.35)
    parser.add_argument("--reward-mode", choices=["payoff", "net_utility"], default="payoff")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-prefix", default="results/runs/rl_fishery/fishery_rl_baseline")
    parser.add_argument("--experiment-tag", default="fishery_rl_baseline")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _parse_csv_arg(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean()) if arr.size else 0.0
    if arr.size <= 1:
        return mean, mean, mean
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    delta = 1.96 * se
    return mean, mean - delta, mean + delta


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def _build_condition_delta_table(table_df: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "benchmark_pack", "test_collapse_mean_mean", "test_mean_stock_mean_mean", "test_mean_welfare_mean_mean"}
    if table_df.empty or not required.issubset(table_df.columns):
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for benchmark_pack, gdf in table_df.groupby("benchmark_pack", dropna=False, sort=True):
        by_condition = {str(row["condition"]): row for _, row in gdf.iterrows()}
        if "none" not in by_condition or "monitoring_sanctions" not in by_condition:
            continue
        none_row = by_condition["none"]
        ms_row = by_condition["monitoring_sanctions"]
        rows.append(
            {
                "benchmark_pack": benchmark_pack,
                "delta_test_collapse_mean": round(
                    float(none_row["test_collapse_mean_mean"]) - float(ms_row["test_collapse_mean_mean"]),
                    4,
                ),
                "delta_test_mean_stock_mean": round(
                    float(ms_row["test_mean_stock_mean_mean"]) - float(none_row["test_mean_stock_mean_mean"]),
                    4,
                ),
                "delta_test_mean_welfare_mean": round(
                    float(ms_row["test_mean_welfare_mean_mean"]) - float(none_row["test_mean_welfare_mean_mean"]),
                    4,
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_runs(per_run_df: pd.DataFrame) -> pd.DataFrame:
    metric_keys = [
        "train_collapse_mean",
        "train_mean_welfare_mean",
        "test_collapse_mean",
        "test_mean_stock_mean",
        "test_mean_welfare_mean",
        "per_regime_survival_over_generations_mean",
    ]
    metric_keys.extend(sorted(c for c in per_run_df.columns if c.startswith("per_regime_survival_over_generations__")))

    rows: list[dict[str, object]] = []
    for (condition, benchmark_pack), gdf in per_run_df.groupby(["condition", "benchmark_pack"], dropna=False, sort=True):
        row: dict[str, object] = {
            "condition": condition,
            "benchmark_pack": benchmark_pack,
            "n_runs": int(len(gdf)),
        }
        for key in metric_keys:
            if key not in gdf.columns:
                continue
            mean, lo, hi = _ci95(gdf[key].tolist())
            row[f"{key}_mean"] = round(mean, 4)
            row[f"{key}_ci95_low"] = round(lo, 4)
            row[f"{key}_ci95_high"] = round(hi, 4)
        rows.append(row)

    table_df = pd.DataFrame(rows)
    if not table_df.empty and "test_collapse_mean_mean" in table_df.columns:
        table_df = table_df.sort_values("test_collapse_mean_mean", ascending=True).reset_index(drop=True)
    return table_df


def main() -> None:
    args = parse_args()
    conditions = _parse_csv_arg(args.conditions)
    if not conditions:
        raise ValueError("At least one condition is required.")

    device = resolve_torch_device(args.device)
    base_cfg = load_config(args.config)
    benchmark_pack = load_benchmark_pack_from_args(
        benchmark_pack=args.benchmark_pack,
        benchmark_pack_file=args.benchmark_pack_file,
        benchmark_pack_file_name=args.benchmark_pack_file_name,
    )
    train_cfg = PPOTrainConfig(
        total_timesteps=int(args.total_timesteps),
        rollout_steps=int(args.rollout_steps),
        update_epochs=int(args.update_epochs),
        minibatch_size=int(args.minibatch_size),
        learning_rate=float(args.learning_rate),
        hidden_size=int(args.hidden_size),
        action_bins=int(args.action_bins),
        eval_every=int(args.eval_every),
        train_eval_episodes=int(args.train_eval_episodes),
        collapse_penalty=float(args.collapse_penalty),
        curriculum_regen_jitter=float(args.curriculum_regen_jitter),
        curriculum_obs_noise_jitter=float(args.curriculum_obs_noise_jitter),
        curriculum_stock_init_jitter=float(args.curriculum_stock_init_jitter),
        curriculum_eval_episodes=int(args.curriculum_eval_episodes),
        stock_health_reward_weight=float(args.stock_health_reward_weight),
        low_stock_penalty_weight=float(args.low_stock_penalty_weight),
        low_stock_penalty_fraction=float(args.low_stock_penalty_fraction),
        reward_mode=str(args.reward_mode),
    )

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_jobs = len(conditions) * int(args.n_runs)
    progress_bar = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(total=total_jobs, desc="Fishery RL baseline runs")
    elif not args.no_progress:
        print(f"Progress: 0/{total_jobs} runs")

    per_run_rows: list[dict[str, object]] = []
    for condition in conditions:
        for run_id in range(int(args.n_runs)):
            run_seed = int(args.run_seed_start) + run_id * int(args.run_seed_stride)
            cfg = apply_rl_condition(base_cfg, condition)
            if args.train_regen_rate is not None:
                cfg.regen_rate = float(args.train_regen_rate)
            if args.train_obs_noise_std is not None:
                cfg.obs_noise_std = float(args.train_obs_noise_std)

            run_prefix = f"{args.output_prefix}_{condition}_run{run_id:02d}"
            policy, action_bins, history_df = train_self_play_policy(
                cfg=cfg,
                train_cfg=train_cfg,
                run_seed=run_seed,
                device=device,
                progress_callback=None,
            )

            checkpoint_path = f"{run_prefix}_checkpoint.pt"
            history_path = f"{run_prefix}_train_history.csv"
            save_rl_checkpoint(
                checkpoint_path,
                policy=policy,
                action_bins=action_bins,
                cfg=cfg,
                train_cfg=train_cfg,
                metadata={
                    "condition": condition,
                    "run_seed": run_seed,
                    "device": device,
                    "experiment_tag": args.experiment_tag,
                    "run_id": run_id,
                },
            )
            history_df.to_csv(history_path, index=False)

            train_episode_df, train_summary = evaluate_self_play_policy(
                cfg=cfg,
                policy=policy,
                action_bins=action_bins,
                n_eval_episodes=int(args.n_eval_episodes),
                seed=run_seed,
                benchmark_pack=None,
                deterministic=True,
                prefix="train",
                device=device,
            )
            train_episode_df["split"] = "train"

            test_episode_df, test_summary = evaluate_self_play_policy(
                cfg=cfg,
                policy=policy,
                action_bins=action_bins,
                n_eval_episodes=int(args.n_eval_episodes),
                seed=run_seed + 1_000_000,
                benchmark_pack=benchmark_pack,
                deterministic=True,
                prefix="test",
                device=device,
            )
            test_episode_df["split"] = "test"

            episodes_df = pd.concat([train_episode_df, test_episode_df], ignore_index=True)
            episodes_path = f"{run_prefix}_episodes.csv"
            summary_path = f"{run_prefix}_summary.csv"
            episodes_df.to_csv(episodes_path, index=False)

            summary_row: dict[str, object] = {
                "condition": condition,
                "run_id": run_id,
                "run_seed": run_seed,
                "experiment_tag": args.experiment_tag,
                "benchmark_pack": args.benchmark_pack or "",
                "n_eval_episodes": int(args.n_eval_episodes),
                "train_regen_rate": float(args.train_regen_rate) if args.train_regen_rate is not None else np.nan,
                "train_obs_noise_std": float(args.train_obs_noise_std) if args.train_obs_noise_std is not None else np.nan,
                "total_timesteps": int(args.total_timesteps),
                "rollout_steps": int(args.rollout_steps),
                "update_epochs": int(args.update_epochs),
                "minibatch_size": int(args.minibatch_size),
                "learning_rate": float(args.learning_rate),
                "hidden_size": int(args.hidden_size),
                "action_bins": int(args.action_bins),
                "collapse_penalty": float(args.collapse_penalty),
                "curriculum_regen_jitter": float(args.curriculum_regen_jitter),
                "curriculum_obs_noise_jitter": float(args.curriculum_obs_noise_jitter),
                "curriculum_stock_init_jitter": float(args.curriculum_stock_init_jitter),
                "curriculum_eval_episodes": int(args.curriculum_eval_episodes),
                "stock_health_reward_weight": float(args.stock_health_reward_weight),
                "low_stock_penalty_weight": float(args.low_stock_penalty_weight),
                "low_stock_penalty_fraction": float(args.low_stock_penalty_fraction),
                "reward_mode": str(args.reward_mode),
            }
            summary_row.update(train_summary)
            summary_row.update(test_summary)
            summary_row["test_regime_count"] = int(test_summary.get("test_regime_count", 1))
            pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
            per_run_rows.append(summary_row)

            if progress_bar is not None:
                progress_bar.update(1)
            elif not args.no_progress:
                done = len(per_run_rows)
                print(f"Progress: {done}/{total_jobs} runs")

    if progress_bar is not None:
        progress_bar.close()

    per_run_df = pd.DataFrame(per_run_rows)
    table_df = _aggregate_runs(per_run_df)
    runs_path = f"{args.output_prefix}_runs.csv"
    table_path = f"{args.output_prefix}_table.csv"
    delta_path = f"{args.output_prefix}_comparison.csv"
    md_path = f"{args.output_prefix}_table.md"
    per_run_df.to_csv(runs_path, index=False)
    table_df.to_csv(table_path, index=False)
    delta_df = _build_condition_delta_table(table_df)
    if not delta_df.empty:
        delta_df.to_csv(delta_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown_table(table_df))
        if not delta_df.empty:
            f.write("\n\n")
            f.write("## Condition Delta\n\n")
            f.write(_markdown_table(delta_df))
        f.write("\n")

    print(f"Saved: {runs_path}")
    print(f"Saved: {table_path}")
    if not delta_df.empty:
        print(f"Saved: {delta_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
