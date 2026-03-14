import argparse
import os

import pandas as pd

from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_rl import evaluate_self_play_policy
from fishery_sim.fishery_rl import load_benchmark_pack_from_args
from fishery_sim.fishery_rl import load_rl_checkpoint
from fishery_sim.fishery_rl import resolve_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Fishery Commons PPO checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-pack", default=None)
    parser.add_argument("--benchmark-pack-file", default=None)
    parser.add_argument("--benchmark-pack-file-name", default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-prefix", default="results/runs/rl_fishery/fishery_rl_eval")
    return parser.parse_args()


def _cfg_from_checkpoint(meta: dict) -> FisheryConfig:
    cfg_dict = dict(meta.get("fishery_config", {}))
    if not cfg_dict:
        raise ValueError("Checkpoint does not contain fishery_config metadata.")
    return FisheryConfig(**cfg_dict)


def main() -> None:
    args = parse_args()
    device = resolve_torch_device(args.device)
    policy, action_bins, meta = load_rl_checkpoint(args.checkpoint, device=device)
    cfg = _cfg_from_checkpoint(meta)
    checkpoint_meta = dict(meta.get("metadata", {}))
    benchmark_pack = load_benchmark_pack_from_args(
        benchmark_pack=args.benchmark_pack,
        benchmark_pack_file=args.benchmark_pack_file,
        benchmark_pack_file_name=args.benchmark_pack_file_name,
    )

    train_episode_df, train_summary = evaluate_self_play_policy(
        cfg=cfg,
        policy=policy,
        action_bins=action_bins,
        n_eval_episodes=int(args.n_eval_episodes),
        seed=int(args.seed),
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
        seed=int(args.seed + 1_000_000),
        benchmark_pack=benchmark_pack,
        deterministic=True,
        prefix="test",
        device=device,
    )
    test_episode_df["split"] = "test"

    episode_df = pd.concat([train_episode_df, test_episode_df], ignore_index=True)
    summary = {
        "condition": checkpoint_meta.get("condition", ""),
        "run_seed": int(checkpoint_meta.get("run_seed", 0)),
        "experiment_tag": checkpoint_meta.get("experiment_tag", ""),
        "checkpoint_path": args.checkpoint,
        "benchmark_pack": args.benchmark_pack or "",
        "n_eval_episodes": int(args.n_eval_episodes),
    }
    summary.update(train_summary)
    summary.update(test_summary)
    summary["test_regime_count"] = int(test_summary.get("test_regime_count", 1))
    summary_df = pd.DataFrame([summary])

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    episodes_path = f"{args.output_prefix}_episodes.csv"
    summary_path = f"{args.output_prefix}_summary.csv"
    episode_df.to_csv(episodes_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {episodes_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
