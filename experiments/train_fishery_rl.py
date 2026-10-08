import argparse
import json
import os
import subprocess
from datetime import datetime
from datetime import timezone

from fishery_sim.config import load_config
from fishery_sim.fishery_rl import PPOTrainConfig
from fishery_sim.fishery_rl import apply_rl_condition
from fishery_sim.fishery_rl import resolve_torch_device
from fishery_sim.fishery_rl import save_rl_checkpoint
from fishery_sim.fishery_rl import train_self_play_policy

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-play PPO baseline in Fishery Commons.")
    parser.add_argument("--config", default="experiments/configs/base.yaml")
    parser.add_argument(
        "--condition",
        choices=["none", "monitoring", "monitoring_sanctions", "adaptive_quota", "temporary_closure"],
        default="none",
    )
    parser.add_argument("--train-regen-rate", type=float, default=None)
    parser.add_argument("--train-obs-noise-std", type=float, default=None)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--rollout-steps", type=int, default=1_024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--action-bins", type=int, default=11)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--train-eval-episodes", type=int, default=16)
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
    parser.add_argument("--output-prefix", default="results/runs/rl_fishery/fishery_rl")
    parser.add_argument("--experiment-tag", default="fishery_rl")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _safe_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_manifest(
    path: str,
    *,
    args: argparse.Namespace,
    checkpoint_path: str,
    history_path: str,
    device: str,
) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "script": "experiments/train_fishery_rl.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _safe_git_hash(),
        "experiment_tag": args.experiment_tag,
        "condition": args.condition,
        "run_seed": int(args.run_seed),
        "device": device,
        "outputs": {
            "checkpoint": checkpoint_path,
            "train_history_csv": history_path,
        },
        "params": vars(args),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    cfg = apply_rl_condition(load_config(args.config), args.condition)
    if args.train_regen_rate is not None:
        cfg.regen_rate = float(args.train_regen_rate)
    if args.train_obs_noise_std is not None:
        cfg.obs_noise_std = float(args.train_obs_noise_std)

    device = resolve_torch_device(args.device)
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

    progress_bar = None
    progress_callback = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(total=train_cfg.total_timesteps, desc="Fishery RL timesteps")

        def _callback(done: int, total: int) -> None:
            del total
            progress_bar.n = int(done)
            progress_bar.refresh()

        progress_callback = _callback
    elif not args.no_progress:
        print(f"Progress: 0/{train_cfg.total_timesteps} timesteps")

        def _callback(done: int, total: int) -> None:
            if done == total or done % max(1, total // 10) == 0:
                print(f"Progress: {done}/{total} timesteps")

        progress_callback = _callback

    try:
        policy, action_bins, history_df = train_self_play_policy(
            cfg=cfg,
            train_cfg=train_cfg,
            run_seed=int(args.run_seed),
            device=device,
            progress_callback=progress_callback,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    checkpoint_path = f"{args.output_prefix}_checkpoint.pt"
    history_path = f"{args.output_prefix}_train_history.csv"

    save_rl_checkpoint(
        checkpoint_path,
        policy=policy,
        action_bins=action_bins,
        cfg=cfg,
        train_cfg=train_cfg,
        metadata={
            "condition": args.condition,
            "run_seed": int(args.run_seed),
            "device": device,
            "experiment_tag": args.experiment_tag,
        },
    )
    history_df.to_csv(history_path, index=False)
    print(f"Saved: {checkpoint_path}")
    print(f"Saved: {history_path}")

    if args.manifest_out:
        _write_manifest(
            args.manifest_out,
            args=args,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            device=device,
        )
        print(f"Saved: {args.manifest_out}")


if __name__ == "__main__":
    main()
