import copy
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from fishery_sim.benchmarks import get_benchmark_pack
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_rl import PPOTrainConfig
from fishery_sim.fishery_rl import FisheryPPOPolicy
from fishery_sim.fishery_rl import TorchPolicyFisheryAgent
from fishery_sim.fishery_rl import apply_rl_condition
from fishery_sim.fishery_rl import build_action_bins
from fishery_sim.fishery_rl import build_rl_observation
from fishery_sim.fishery_rl import evaluate_self_play_policy
from fishery_sim.fishery_rl import load_rl_checkpoint
from fishery_sim.fishery_rl import save_rl_checkpoint
from fishery_sim.fishery_rl import train_self_play_policy
from fishery_sim.simulation import run_episode


def test_build_rl_observation_and_action_bins_are_bounded() -> None:
    cfg = FisheryConfig(horizon=20, stock_max=200.0, max_harvest_per_agent=10.0)
    obs = build_rl_observation(obs_stock=250.0, t=25, cfg=cfg)
    bins = build_action_bins(cfg.max_harvest_per_agent, 5)
    assert obs.shape == (2,)
    assert float(obs[0]) == 1.0
    assert float(obs[1]) == 1.0
    assert bins.tolist() == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_evaluate_self_play_policy_matches_run_episode_for_deterministic_policy() -> None:
    cfg = FisheryConfig(
        n_agents=4,
        horizon=12,
        stock_init=100.0,
        stock_max=200.0,
        regen_rate=1.5,
        collapse_threshold=10.0,
        collapse_patience=5,
        obs_noise_std=0.0,
        max_harvest_per_agent=10.0,
        seed=7,
    )
    action_bins = build_action_bins(cfg.max_harvest_per_agent, 5)
    policy = FisheryPPOPolicy(obs_dim=2, hidden_size=8, n_actions=len(action_bins))
    for param in policy.parameters():
        torch.nn.init.constant_(param, 0.0)
    policy.eval()

    agents = [
        TorchPolicyFisheryAgent(policy=policy, cfg=cfg, action_bins=action_bins, deterministic=True, device="cpu")
        for _ in range(cfg.n_agents)
    ]
    direct = run_episode(cfg, agents)
    episode_df, summary = evaluate_self_play_policy(
        cfg=cfg,
        policy=policy,
        action_bins=action_bins,
        n_eval_episodes=1,
        seed=cfg.seed,
        benchmark_pack=None,
        deterministic=True,
        prefix="test",
        device="cpu",
    )

    assert len(episode_df) == 1
    row = episode_df.iloc[0]
    assert float(row["mean_stock"]) == float(direct["mean_stock"])
    assert float(row["welfare"]) == float(direct["payoffs"].sum())
    assert bool(row["collapsed"]) == bool(direct["collapsed"])
    assert float(summary["test_collapse_mean"]) == float(direct["collapsed"])


def test_train_self_play_policy_smoke_and_checkpoint_reload(tmp_path: Path) -> None:
    cfg = apply_rl_condition(
        FisheryConfig(
            n_agents=4,
            horizon=18,
            stock_init=120.0,
            stock_max=200.0,
            regen_rate=1.6,
            collapse_threshold=10.0,
            collapse_patience=4,
            obs_noise_std=0.0,
            max_harvest_per_agent=10.0,
            seed=3,
        ),
        "monitoring_sanctions",
    )
    train_cfg = PPOTrainConfig(
        total_timesteps=128,
        rollout_steps=64,
        update_epochs=2,
        minibatch_size=32,
        hidden_size=16,
        eval_every=64,
        train_eval_episodes=2,
    )
    policy, action_bins, history_df = train_self_play_policy(
        cfg=copy.deepcopy(cfg),
        train_cfg=train_cfg,
        run_seed=11,
        device="cpu",
    )
    assert not history_df.empty
    assert "train_collapse_mean" in history_df.columns

    checkpoint_path = tmp_path / "fishery_rl_checkpoint.pt"
    save_rl_checkpoint(
        str(checkpoint_path),
        policy=policy,
        action_bins=action_bins,
        cfg=cfg,
        train_cfg=train_cfg,
        metadata={"condition": "monitoring_sanctions", "run_seed": 11},
    )
    loaded_policy, loaded_bins, meta = load_rl_checkpoint(str(checkpoint_path), device="cpu")
    assert meta["metadata"]["condition"] == "monitoring_sanctions"

    benchmark_pack = get_benchmark_pack("medium_v1")[:1]
    _, summary_a = evaluate_self_play_policy(
        cfg=cfg,
        policy=loaded_policy,
        action_bins=loaded_bins,
        n_eval_episodes=2,
        seed=21,
        benchmark_pack=benchmark_pack,
        deterministic=True,
        prefix="test",
        device="cpu",
    )
    _, summary_b = evaluate_self_play_policy(
        cfg=cfg,
        policy=loaded_policy,
        action_bins=loaded_bins,
        n_eval_episodes=2,
        seed=21,
        benchmark_pack=benchmark_pack,
        deterministic=True,
        prefix="test",
        device="cpu",
    )
    assert summary_a == summary_b
    assert "test_mean_stock_mean" in summary_a
    assert "per_regime_survival_over_generations_mean" in summary_a
