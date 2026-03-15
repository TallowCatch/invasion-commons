import argparse
import json
import os
import subprocess
from datetime import datetime
from datetime import timezone

from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_tier
from fishery_sim.harvest_rl import HarvestPPOTrainConfig
from fishery_sim.harvest_rl import resolve_torch_device
from fishery_sim.harvest_rl import save_rl_checkpoint
from fishery_sim.harvest_rl import train_self_play_policy

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-play PPO baseline in Harvest Commons.")
    parser.add_argument("--tier", choices=["easy_h1", "medium_h1", "hard_h1"], default="medium_h1")
    parser.add_argument("--condition", choices=["none", "top_down_only", "bottom_up_only", "hybrid"], default="hybrid")
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--train-regen-rate", type=float, default=None)
    parser.add_argument("--train-weather-noise-std", type=float, default=None)
    parser.add_argument("--train-neighbor-externality", type=float, default=None)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--rollout-steps", type=int, default=960)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--harvest-action-bins", type=int, default=9)
    parser.add_argument("--communication-bins", type=int, default=5)
    parser.add_argument("--credit-offer-bins", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--train-eval-episodes", type=int, default=16)
    parser.add_argument("--garden-failure-penalty", type=float, default=25.0)
    parser.add_argument("--patch-health-reward-weight", type=float, default=2.0)
    parser.add_argument("--local-aggression-penalty-weight", type=float, default=0.0)
    parser.add_argument("--neighborhood-overharvest-penalty-weight", type=float, default=0.0)
    parser.add_argument("--government-trigger", type=float, default=16.0)
    parser.add_argument("--strict-cap-frac", type=float, default=0.18)
    parser.add_argument("--relaxed-cap-frac", type=float, default=0.35)
    parser.add_argument("--soft-trigger", type=float, default=18.0)
    parser.add_argument("--deterioration-threshold", type=float, default=0.35)
    parser.add_argument("--activation-warmup", type=int, default=3)
    parser.add_argument("--aggressive-request-threshold", type=float, default=0.75)
    parser.add_argument("--aggressive-agent-fraction-trigger", type=float, default=0.34)
    parser.add_argument("--local-neighborhood-trigger", type=float, default=0.67)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-prefix", default="results/runs/rl_harvest/harvest_rl")
    parser.add_argument("--experiment-tag", default="harvest_rl")
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
        "script": "experiments/train_harvest_rl.py",
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
    cfg = make_harvest_cfg_for_tier(args.tier, n_agents=int(args.n_agents))
    if args.train_regen_rate is not None:
        cfg.regen_rate = float(args.train_regen_rate)
    if args.train_weather_noise_std is not None:
        cfg.weather_noise_std = float(args.train_weather_noise_std)
    if args.train_neighbor_externality is not None:
        cfg.neighbor_externality = float(args.train_neighbor_externality)

    government_params = {
        "trigger": float(args.government_trigger),
        "strict_cap_frac": float(args.strict_cap_frac),
        "relaxed_cap_frac": float(args.relaxed_cap_frac),
        "soft_trigger": float(args.soft_trigger),
        "deterioration_threshold": float(args.deterioration_threshold),
        "activation_warmup": int(args.activation_warmup),
        "aggressive_request_threshold": float(args.aggressive_request_threshold),
        "aggressive_agent_fraction_trigger": float(args.aggressive_agent_fraction_trigger),
        "local_neighborhood_trigger": float(args.local_neighborhood_trigger),
    }
    device = resolve_torch_device(args.device)
    train_cfg = HarvestPPOTrainConfig(
        total_timesteps=int(args.total_timesteps),
        rollout_steps=int(args.rollout_steps),
        update_epochs=int(args.update_epochs),
        minibatch_size=int(args.minibatch_size),
        learning_rate=float(args.learning_rate),
        hidden_size=int(args.hidden_size),
        harvest_action_bins=int(args.harvest_action_bins),
        communication_bins=int(args.communication_bins),
        credit_offer_bins=int(args.credit_offer_bins),
        eval_every=int(args.eval_every),
        train_eval_episodes=int(args.train_eval_episodes),
        garden_failure_penalty=float(args.garden_failure_penalty),
        patch_health_reward_weight=float(args.patch_health_reward_weight),
        local_aggression_penalty_weight=float(args.local_aggression_penalty_weight),
        neighborhood_overharvest_penalty_weight=float(args.neighborhood_overharvest_penalty_weight),
    )

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    progress_bar = None
    progress_callback = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(total=train_cfg.total_timesteps, desc="Harvest RL timesteps")

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
            condition=args.condition,
            train_cfg=train_cfg,
            run_seed=int(args.run_seed),
            government_params=government_params,
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
            "government_params": government_params,
            "tier": args.tier,
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
