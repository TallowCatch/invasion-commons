import copy
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.harvest import HarvestMessage
from fishery_sim.harvest import HarvestObservation
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.harvest_benchmarks import get_harvest_regime_pack
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_tier
from fishery_sim.harvest_rl import HarvestPPOPolicy
from fishery_sim.harvest_rl import HarvestPPOTrainConfig
from fishery_sim.harvest_rl import TorchPolicyHarvestAgent
from fishery_sim.harvest_rl import _build_training_rewards
from fishery_sim.harvest_rl import build_harvest_action_bins
from fishery_sim.harvest_rl import build_harvest_rl_observation
from fishery_sim.harvest_rl import evaluate_self_play_policy
from fishery_sim.harvest_rl import load_rl_checkpoint
from fishery_sim.harvest_rl import make_harvest_governor
from fishery_sim.harvest_rl import save_rl_checkpoint
from fishery_sim.harvest_rl import train_self_play_policy


def test_build_harvest_observation_and_action_bins_are_bounded() -> None:
    cfg = HarvestCommonsConfig(horizon=20, patch_max=20.0, credit_cap=1.0)
    obs = build_harvest_rl_observation(
        observation=HarvestObservation(
            local_patch=30.0,
            neighbor_mean=25.0,
            last_credit_received=2.0,
            government_cap_frac=None,
        ),
        t=25,
        cfg=cfg,
        inbox=[
            HarvestMessage(announced_restraint=0.8, requested_credit=0.4),
            HarvestMessage(announced_restraint=0.4, requested_credit=0.0),
        ],
    )
    bins = build_harvest_action_bins(cfg, HarvestPPOTrainConfig(harvest_action_bins=5, communication_bins=3, credit_offer_bins=4))
    assert obs.shape == (10,)
    assert float(obs[0]) == 1.0
    assert float(obs[1]) == 1.0
    assert float(obs[2]) == 1.0
    assert float(obs[5]) == 1.0
    assert np.isclose(float(obs[6]), 0.6)
    assert np.isclose(float(obs[7]), 0.2)
    assert np.isclose(float(obs[8]), 0.5)
    assert np.isclose(float(obs[9]), 0.5)
    assert bins["harvest_frac"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert bins["credit_offer"].tolist() == [0.0, 0.3333333432674408, 0.6666666865348816, 1.0]


def test_evaluate_harvest_self_play_matches_direct_episode_for_deterministic_policy() -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=4, seed=7)
    cfg.communication_enabled = True
    cfg.side_payments_enabled = True
    train_cfg = HarvestPPOTrainConfig(harvest_action_bins=5, communication_bins=3, credit_offer_bins=3)
    action_bins = build_harvest_action_bins(cfg, train_cfg)
    policy = HarvestPPOPolicy(
        obs_dim=10,
        hidden_size=8,
        n_action_bins={key: len(values) for key, values in action_bins.items()},
    )
    for param in policy.parameters():
        torch.nn.init.constant_(param, 0.0)
    policy.eval()

    governor = make_harvest_governor("hybrid")
    agents = [
        TorchPolicyHarvestAgent(policy=policy, cfg=cfg, action_bins=action_bins, deterministic=True, device="cpu")
        for _ in range(cfg.n_agents)
    ]
    direct = run_harvest_episode(cfg, agents, governor=governor)
    episode_df, summary = evaluate_self_play_policy(
        cfg=cfg,
        condition="hybrid",
        policy=policy,
        action_bins=action_bins,
        n_eval_episodes=1,
        seed=cfg.seed,
        benchmark_pack=None,
        deterministic=True,
        prefix="test",
        government_params=None,
        device="cpu",
    )

    assert len(episode_df) == 1
    row = episode_df.iloc[0]
    assert float(row["mean_patch_health"]) == float(direct["mean_patch_health"])
    assert float(row["mean_welfare"]) == float(direct["mean_welfare"])
    assert bool(row["garden_failure_event"]) == bool(direct["garden_failure_event"])
    assert float(summary["test_garden_failure_rate"]) == float(direct["garden_failure_event"])


def test_train_harvest_self_play_smoke_and_checkpoint_reload(tmp_path: Path) -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=4, seed=3)
    train_cfg = HarvestPPOTrainConfig(
        total_timesteps=128,
        rollout_steps=64,
        update_epochs=2,
        minibatch_size=32,
        hidden_size=16,
        eval_every=64,
        train_eval_episodes=2,
        harvest_action_bins=5,
        communication_bins=3,
        credit_offer_bins=3,
    )
    policy, action_bins, history_df = train_self_play_policy(
        cfg=copy.deepcopy(cfg),
        condition="hybrid",
        train_cfg=train_cfg,
        run_seed=11,
        government_params=None,
        device="cpu",
    )
    assert not history_df.empty
    assert "train_garden_failure_rate" in history_df.columns

    checkpoint_path = tmp_path / "harvest_rl_checkpoint.pt"
    save_rl_checkpoint(
        str(checkpoint_path),
        policy=policy,
        action_bins=action_bins,
        cfg=cfg,
        train_cfg=train_cfg,
        metadata={"condition": "hybrid", "run_seed": 11},
    )
    loaded_policy, loaded_bins, meta = load_rl_checkpoint(str(checkpoint_path), device="cpu")
    assert meta["metadata"]["condition"] == "hybrid"

    benchmark_pack = get_harvest_regime_pack("medium_h1")[:1]
    _, summary_a = evaluate_self_play_policy(
        cfg=cfg,
        condition="hybrid",
        policy=loaded_policy,
        action_bins=loaded_bins,
        n_eval_episodes=2,
        seed=21,
        benchmark_pack=benchmark_pack,
        deterministic=True,
        prefix="test",
        government_params=None,
        device="cpu",
    )
    _, summary_b = evaluate_self_play_policy(
        cfg=cfg,
        condition="hybrid",
        policy=loaded_policy,
        action_bins=loaded_bins,
        n_eval_episodes=2,
        seed=21,
        benchmark_pack=benchmark_pack,
        deterministic=True,
        prefix="test",
        government_params=None,
        device="cpu",
    )
    assert summary_a == summary_b
    assert "test_mean_patch_health" in summary_a
    assert "per_regime_survival_over_generations_mean" in summary_a


def test_build_harvest_training_rewards_penalizes_aggression_and_overharvest() -> None:
    cfg = HarvestCommonsConfig(
        patch_max=20.0,
        max_harvest_per_agent=6.0,
        sustainable_harvest_frac=0.35,
    )
    train_cfg = HarvestPPOTrainConfig(
        patch_health_reward_weight=2.0,
        local_aggression_penalty_weight=1.5,
        neighborhood_overharvest_penalty_weight=2.0,
        garden_failure_penalty=10.0,
    )
    rewards = _build_training_rewards(
        payoffs=np.asarray([2.0, 3.0], dtype=np.float32),
        patch_health=np.asarray([10.0, 12.0], dtype=np.float32),
        max_local_aggression=0.8,
        neighborhood_overharvest=5.85,
        garden_failure_event=False,
        cfg=cfg,
        train_cfg=train_cfg,
    )
    # Base rewards [2, 3], patch bonus 1.1, aggression penalty 1.2, overharvest penalty 1.0.
    assert np.allclose(rewards, np.asarray([0.9, 1.9], dtype=np.float32))
