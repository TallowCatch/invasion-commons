from __future__ import annotations

import copy
import os
import re
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from .agents import BaseAgent
from .config import FisheryConfig
from .env import FisheryEnv
from .metrics import gini
from .simulation import run_episode


DEFAULT_MONITORING_PROB = 0.9
DEFAULT_QUOTA_FRACTION = 0.07
DEFAULT_SANCTION_BASE_FINE_RATE = 2.0
DEFAULT_SANCTION_FINE_GROWTH = 0.8
DEFAULT_ADAPTIVE_QUOTA_MIN_SCALE = 0.35
DEFAULT_ADAPTIVE_QUOTA_SENSITIVITY = 0.9
DEFAULT_TEMPORARY_CLOSURE_TRIGGER = 15.0
DEFAULT_TEMPORARY_CLOSURE_QUOTA_FRACTION = 0.01


@dataclass(frozen=True)
class PPOTrainConfig:
    total_timesteps: int = 250_000
    rollout_steps: int = 1_024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 2.5e-4
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 64
    action_bins: int = 11
    eval_every: int = 25_000
    train_eval_episodes: int = 16
    max_grad_norm: float = 0.5


@dataclass
class _EpisodeRollout:
    obs: np.ndarray
    actions: np.ndarray
    logprobs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    metrics: dict[str, float | int | bool]


class FisheryPPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.n_actions = int(n_actions)
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(self.hidden_size, self.n_actions)
        self.critic = nn.Linear(self.hidden_size, 1)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features(obs)
        return self.actor(features), self.critic(features).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        logprobs = dist.log_prob(actions)
        return actions, logprobs, values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


class TorchPolicyFisheryAgent(BaseAgent):
    name = "torch_policy_fishery"

    def __init__(
        self,
        policy: FisheryPPOPolicy,
        cfg: FisheryConfig,
        action_bins: np.ndarray,
        *,
        deterministic: bool = True,
        device: str = "cpu",
    ):
        self.policy = policy
        self.cfg = cfg
        self.action_bins = np.asarray(action_bins, dtype=np.float32)
        self.deterministic = bool(deterministic)
        self.device = device

    def act(self, obs_stock: float, t: int, n_agents: int) -> float:
        del n_agents
        obs = torch.as_tensor(
            build_rl_observation(obs_stock=obs_stock, t=t, cfg=self.cfg)[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            action, _, _ = self.policy.act(obs, deterministic=self.deterministic)
        return float(self.action_bins[int(action.item())])


def resolve_torch_device(requested: str = "auto") -> str:
    text = requested.strip().lower()
    if text != "auto":
        return text
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_action_bins(max_harvest_per_agent: float, n_bins: int) -> np.ndarray:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    return np.linspace(0.0, float(max_harvest_per_agent), int(n_bins), dtype=np.float32)


def build_rl_observation(obs_stock: float, t: int, cfg: FisheryConfig) -> np.ndarray:
    stock_scale = max(float(cfg.stock_max), 1e-9)
    horizon_scale = max(int(cfg.horizon) - 1, 1)
    return np.asarray(
        [
            float(np.clip(obs_stock / stock_scale, 0.0, 1.0)),
            float(np.clip(t / horizon_scale, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def apply_rl_condition(
    cfg: FisheryConfig,
    condition: str,
) -> FisheryConfig:
    out = copy.deepcopy(cfg)
    name = condition.strip().lower()
    if name == "none":
        out.monitoring_prob = 0.0
        out.quota_fraction = 0.0
        out.base_fine_rate = 0.0
        out.fine_growth = 0.0
        out.governance_variant = "static"
    elif name == "monitoring":
        out.monitoring_prob = DEFAULT_MONITORING_PROB
        out.quota_fraction = DEFAULT_QUOTA_FRACTION
        out.base_fine_rate = 0.0
        out.fine_growth = 0.0
        out.governance_variant = "static"
    elif name == "monitoring_sanctions":
        out.monitoring_prob = DEFAULT_MONITORING_PROB
        out.quota_fraction = DEFAULT_QUOTA_FRACTION
        out.base_fine_rate = DEFAULT_SANCTION_BASE_FINE_RATE
        out.fine_growth = DEFAULT_SANCTION_FINE_GROWTH
        out.governance_variant = "static"
    elif name == "adaptive_quota":
        out.monitoring_prob = DEFAULT_MONITORING_PROB
        out.quota_fraction = DEFAULT_QUOTA_FRACTION
        out.base_fine_rate = DEFAULT_SANCTION_BASE_FINE_RATE
        out.fine_growth = DEFAULT_SANCTION_FINE_GROWTH
        out.governance_variant = "adaptive_quota"
        out.adaptive_quota_min_scale = DEFAULT_ADAPTIVE_QUOTA_MIN_SCALE
        out.adaptive_quota_sensitivity = DEFAULT_ADAPTIVE_QUOTA_SENSITIVITY
    elif name == "temporary_closure":
        out.monitoring_prob = 1.0
        out.quota_fraction = max(DEFAULT_QUOTA_FRACTION, DEFAULT_TEMPORARY_CLOSURE_QUOTA_FRACTION)
        out.base_fine_rate = DEFAULT_SANCTION_BASE_FINE_RATE
        out.fine_growth = DEFAULT_SANCTION_FINE_GROWTH
        out.governance_variant = "temporary_closure"
        out.temporary_closure_trigger = DEFAULT_TEMPORARY_CLOSURE_TRIGGER
        out.temporary_closure_quota_fraction = DEFAULT_TEMPORARY_CLOSURE_QUOTA_FRACTION
    else:
        raise ValueError(f"Unknown RL condition: {condition}")
    return out


def make_fishery_env(
    cfg: FisheryConfig,
    *,
    rng: np.random.Generator | None = None,
) -> FisheryEnv:
    return FisheryEnv(
        n_agents=cfg.n_agents,
        stock_init=cfg.stock_init,
        stock_max=cfg.stock_max,
        regen_rate=cfg.regen_rate,
        collapse_threshold=cfg.collapse_threshold,
        collapse_patience=cfg.collapse_patience,
        max_harvest_per_agent=cfg.max_harvest_per_agent,
        obs_noise_std=cfg.obs_noise_std,
        monitoring_prob=cfg.monitoring_prob,
        quota_fraction=cfg.quota_fraction,
        base_fine_rate=cfg.base_fine_rate,
        fine_growth=cfg.fine_growth,
        governance_variant=cfg.governance_variant,
        adaptive_quota_min_scale=cfg.adaptive_quota_min_scale,
        adaptive_quota_sensitivity=cfg.adaptive_quota_sensitivity,
        temporary_closure_trigger=cfg.temporary_closure_trigger,
        temporary_closure_quota_fraction=cfg.temporary_closure_quota_fraction,
        rng=rng,
    )


def save_rl_checkpoint(
    path: str,
    policy: FisheryPPOPolicy,
    action_bins: np.ndarray,
    cfg: FisheryConfig,
    train_cfg: PPOTrainConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "state_dict": policy.state_dict(),
        "action_bins": np.asarray(action_bins, dtype=np.float32),
        "fishery_config": asdict(cfg),
        "train_config": asdict(train_cfg),
        "policy_config": {
            "obs_dim": policy.obs_dim,
            "hidden_size": policy.hidden_size,
            "n_actions": policy.n_actions,
        },
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)


def load_rl_checkpoint(
    path: str,
    *,
    device: str = "cpu",
) -> tuple[FisheryPPOPolicy, np.ndarray, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    policy_cfg = payload["policy_config"]
    policy = FisheryPPOPolicy(
        obs_dim=int(policy_cfg["obs_dim"]),
        hidden_size=int(policy_cfg["hidden_size"]),
        n_actions=int(policy_cfg["n_actions"]),
    )
    policy.load_state_dict(payload["state_dict"])
    policy.to(device)
    policy.eval()
    action_bins = np.asarray(payload["action_bins"], dtype=np.float32)
    metadata = {
        "fishery_config": dict(payload.get("fishery_config", {})),
        "train_config": dict(payload.get("train_config", {})),
        "metadata": dict(payload.get("metadata", {})),
    }
    return policy, action_bins, metadata


def train_self_play_policy(
    cfg: FisheryConfig,
    train_cfg: PPOTrainConfig,
    *,
    run_seed: int,
    device: str = "cpu",
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[FisheryPPOPolicy, np.ndarray, pd.DataFrame]:
    torch.manual_seed(int(run_seed))
    np.random.seed(int(run_seed))

    action_bins = build_action_bins(cfg.max_harvest_per_agent, train_cfg.action_bins)
    policy = FisheryPPOPolicy(obs_dim=2, hidden_size=train_cfg.hidden_size, n_actions=len(action_bins)).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.learning_rate)

    total_timesteps = 0
    update_idx = 0
    next_eval = min(train_cfg.eval_every, train_cfg.total_timesteps)
    best_score = (-float("inf"), -float("inf"))
    best_state = copy.deepcopy(policy.state_dict())
    history_rows: list[dict[str, float | int]] = []

    while total_timesteps < train_cfg.total_timesteps:
        rollout = _collect_rollout_batch(
            cfg=cfg,
            policy=policy,
            action_bins=action_bins,
            rollout_steps=min(train_cfg.rollout_steps, train_cfg.total_timesteps - total_timesteps),
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            device=device,
            seed=run_seed + update_idx * 1_000,
        )
        total_timesteps += int(len(rollout["actions"]))
        _ppo_update(
            policy=policy,
            optimizer=optimizer,
            batch=rollout,
            train_cfg=train_cfg,
            device=device,
        )
        update_idx += 1

        row: dict[str, float | int] = {
            "update": update_idx,
            "timesteps": total_timesteps,
            "rollout_reward_mean": float(rollout["reward_mean"]),
            "rollout_collapse_rate": float(rollout["collapse_rate"]),
            "rollout_episode_length_mean": float(rollout["episode_length_mean"]),
        }

        should_eval = total_timesteps >= next_eval or total_timesteps >= train_cfg.total_timesteps
        if should_eval:
            _, eval_summary = evaluate_self_play_policy(
                cfg=cfg,
                policy=policy,
                action_bins=action_bins,
                n_eval_episodes=train_cfg.train_eval_episodes,
                seed=run_seed + 50_000 + update_idx * 17,
                benchmark_pack=None,
                deterministic=True,
                prefix="train",
                device=device,
            )
            row.update(eval_summary)
            candidate = (
                -float(eval_summary["train_collapse_mean"]),
                float(eval_summary["train_mean_welfare_mean"]),
            )
            if candidate > best_score:
                best_score = candidate
                best_state = copy.deepcopy(policy.state_dict())
            next_eval += train_cfg.eval_every

        history_rows.append(row)
        if progress_callback is not None:
            progress_callback(total_timesteps, train_cfg.total_timesteps)

    policy.load_state_dict(best_state)
    policy.eval()
    return policy, action_bins, pd.DataFrame(history_rows)


def evaluate_self_play_policy(
    cfg: FisheryConfig,
    policy: FisheryPPOPolicy,
    action_bins: np.ndarray,
    *,
    n_eval_episodes: int,
    seed: int,
    benchmark_pack: list[dict[str, Any]] | None,
    deterministic: bool,
    prefix: str,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if n_eval_episodes < 1:
        raise ValueError("n_eval_episodes must be >= 1")

    regimes = benchmark_pack or [{"name": "default", "overrides": {}}]
    rows: list[dict[str, Any]] = []
    for regime_idx, regime in enumerate(regimes):
        regime_name = str(regime.get("name", f"regime_{regime_idx}"))
        overrides = dict(regime.get("overrides", {}))
        regime_cfg = copy.deepcopy(cfg)
        for key, value in overrides.items():
            if hasattr(regime_cfg, key):
                setattr(regime_cfg, key, value)

        for ep in range(n_eval_episodes):
            episode_cfg = copy.deepcopy(regime_cfg)
            episode_cfg.seed = int(seed + regime_idx * 1_000_000 + ep)
            agents = [
                TorchPolicyFisheryAgent(
                    policy=policy,
                    cfg=episode_cfg,
                    action_bins=action_bins,
                    deterministic=deterministic,
                    device=device,
                )
                for _ in range(episode_cfg.n_agents)
            ]
            out = run_episode(episode_cfg, agents)
            rows.append(
                {
                    "seed": episode_cfg.seed,
                    "regime": regime_name,
                    "collapsed": bool(out["collapsed"]),
                    "t_end": int(out["t_end"]),
                    "final_stock": float(out["final_stock"]),
                    "mean_stock": float(out["mean_stock"]),
                    "welfare": float(np.sum(out["payoffs"])),
                    "payoff_gini": float(gini(np.asarray(out["payoffs"], dtype=float))),
                    "sanction_total": float(out["sanction_total"]),
                    "violation_events": float(out["violation_events"]),
                    "mean_requested_harvest": float(out["mean_requested_harvest"]),
                    "mean_realized_harvest": float(out["mean_realized_harvest"]),
                    "mean_audit_rate": float(out["mean_audit_rate"]),
                    "mean_quota": float(out["mean_quota"]),
                    "mean_quota_clipped_total": float(out["mean_quota_clipped_total"]),
                    "mean_repeat_offender_rate": float(out["mean_repeat_offender_rate"]),
                    "closure_active_fraction": float(out["closure_active_fraction"]),
                    "mean_stock_recovery_lag": float(out["mean_stock_recovery_lag"]),
                }
            )

    episode_df = pd.DataFrame(rows)
    summary = _build_eval_summary(episode_df=episode_df, prefix=prefix)
    summary[f"{prefix}_regime_count"] = int(len(regimes))
    return episode_df, summary


def load_benchmark_pack_from_args(
    *,
    benchmark_pack: str | None,
    benchmark_pack_file: str | None,
    benchmark_pack_file_name: str | None,
) -> list[dict[str, Any]] | None:
    if benchmark_pack_file:
        from .benchmarks import load_benchmark_pack_file

        return load_benchmark_pack_file(benchmark_pack_file, pack_name=benchmark_pack_file_name)
    if benchmark_pack:
        from .benchmarks import get_benchmark_pack

        return get_benchmark_pack(benchmark_pack)
    return None


def _safe_name(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    out = out.strip("_")
    return out or "regime"


def _build_eval_summary(episode_df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if episode_df.empty:
        return {}

    summary: dict[str, Any] = {
        f"{prefix}_collapse_mean": float(episode_df["collapsed"].mean()),
        f"{prefix}_mean_stock_mean": float(episode_df["mean_stock"].mean()),
        f"{prefix}_mean_final_stock_mean": float(episode_df["final_stock"].mean()),
        f"{prefix}_mean_welfare_mean": float(episode_df["welfare"].mean()),
        f"{prefix}_mean_payoff_gini_mean": float(episode_df["payoff_gini"].mean()),
        f"{prefix}_mean_sanction_total_mean": float(episode_df["sanction_total"].mean()),
        f"{prefix}_mean_violation_events_mean": float(episode_df["violation_events"].mean()),
        f"{prefix}_mean_requested_harvest_mean": float(episode_df["mean_requested_harvest"].mean()),
        f"{prefix}_mean_realized_harvest_mean": float(episode_df["mean_realized_harvest"].mean()),
        f"{prefix}_mean_audit_rate_mean": float(episode_df["mean_audit_rate"].mean()),
        f"{prefix}_mean_quota_mean": float(episode_df["mean_quota"].mean()),
        f"{prefix}_mean_quota_clipped_total_mean": float(episode_df["mean_quota_clipped_total"].mean()),
        f"{prefix}_mean_repeat_offender_rate_mean": float(episode_df["mean_repeat_offender_rate"].mean()),
        f"{prefix}_closure_active_fraction_mean": float(episode_df["closure_active_fraction"].mean()),
        f"{prefix}_mean_stock_recovery_lag_mean": float(episode_df["mean_stock_recovery_lag"].mean()),
    }
    per_regime_survival: list[float] = []
    for regime_name, rdf in episode_df.groupby("regime", sort=True):
        safe = _safe_name(str(regime_name))
        val = float(1.0 - rdf["collapsed"].mean())
        summary[f"per_regime_survival_over_generations__{safe}"] = val
        per_regime_survival.append(val)
    summary["per_regime_survival_over_generations_mean"] = (
        float(np.mean(per_regime_survival)) if per_regime_survival else 0.0
    )
    return summary


def _collect_rollout_batch(
    cfg: FisheryConfig,
    policy: FisheryPPOPolicy,
    action_bins: np.ndarray,
    *,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    device: str,
    seed: int,
) -> dict[str, Any]:
    episodes: list[_EpisodeRollout] = []
    total_steps = 0
    episode_seed = int(seed)
    while total_steps < rollout_steps:
        episode = _run_training_episode(
            cfg=cfg,
            policy=policy,
            action_bins=action_bins,
            seed=episode_seed,
            device=device,
        )
        episodes.append(episode)
        total_steps += int(episode.obs.shape[0] * episode.obs.shape[1])
        episode_seed += 1

    obs_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    logprob_batches: list[np.ndarray] = []
    value_batches: list[np.ndarray] = []
    reward_batches: list[np.ndarray] = []
    advantage_batches: list[np.ndarray] = []
    return_batches: list[np.ndarray] = []
    collapse_flags: list[float] = []
    lengths: list[int] = []
    reward_means: list[float] = []

    for episode in episodes:
        advantages, returns = _gae_from_episode(
            rewards=episode.rewards,
            values=episode.values,
            dones=episode.dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        obs_batches.append(episode.obs.reshape(-1, episode.obs.shape[-1]))
        action_batches.append(episode.actions.reshape(-1))
        logprob_batches.append(episode.logprobs.reshape(-1))
        value_batches.append(episode.values.reshape(-1))
        reward_batches.append(episode.rewards.reshape(-1))
        advantage_batches.append(advantages.reshape(-1))
        return_batches.append(returns.reshape(-1))
        collapse_flags.append(float(bool(episode.metrics["collapsed"])))
        lengths.append(int(episode.obs.shape[0]))
        reward_means.append(float(np.mean(episode.rewards)))

    advantages = np.concatenate(advantage_batches).astype(np.float32)
    advantages = (advantages - advantages.mean()) / max(advantages.std(ddof=0), 1e-8)
    return {
        "obs": np.concatenate(obs_batches).astype(np.float32),
        "actions": np.concatenate(action_batches).astype(np.int64),
        "logprobs": np.concatenate(logprob_batches).astype(np.float32),
        "values": np.concatenate(value_batches).astype(np.float32),
        "rewards": np.concatenate(reward_batches).astype(np.float32),
        "advantages": advantages.astype(np.float32),
        "returns": np.concatenate(return_batches).astype(np.float32),
        "collapse_rate": float(np.mean(collapse_flags)) if collapse_flags else 0.0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "reward_mean": float(np.mean(reward_means)) if reward_means else 0.0,
    }


def _run_training_episode(
    cfg: FisheryConfig,
    policy: FisheryPPOPolicy,
    action_bins: np.ndarray,
    *,
    seed: int,
    device: str,
) -> _EpisodeRollout:
    local_cfg = copy.deepcopy(cfg)
    local_cfg.seed = int(seed)
    env = make_fishery_env(local_cfg, rng=np.random.default_rng(local_cfg.seed))
    env.reset()

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    logprob_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    done_rows: list[np.ndarray] = []

    stock_trace: list[float] = []
    payoff_total = np.zeros(local_cfg.n_agents, dtype=np.float32)
    sanction_total = 0.0
    violation_events = 0.0
    requested_trace: list[float] = []
    realized_trace: list[float] = []
    audit_rate_trace: list[float] = []
    quota_trace: list[float] = []
    quota_clipped_trace: list[float] = []
    repeat_offender_rate_trace: list[float] = []
    closure_trace: list[float] = []
    recovery_lags: list[int] = []
    low_stock_start: int | None = None
    collapsed = False
    t_end = -1

    for t in range(local_cfg.horizon):
        t_end = t
        obs_stock = env.observe_stock()
        base_obs = build_rl_observation(obs_stock=obs_stock, t=t, cfg=local_cfg)
        obs_batch = np.repeat(base_obs[None, :], local_cfg.n_agents, axis=0).astype(np.float32)
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, logprobs, values = policy.act(obs_tensor, deterministic=False)
        action_idx = actions.detach().cpu().numpy().astype(np.int64)
        harvests = action_bins[action_idx]
        step = env.step(harvests)

        obs_rows.append(obs_batch)
        action_rows.append(action_idx.astype(np.int64))
        logprob_rows.append(logprobs.detach().cpu().numpy().astype(np.float32))
        value_rows.append(values.detach().cpu().numpy().astype(np.float32))
        reward_rows.append(np.asarray(step.payoffs, dtype=np.float32))
        done_flag = bool(step.collapsed or t == local_cfg.horizon - 1)
        done_rows.append(np.full(local_cfg.n_agents, 1.0 if done_flag else 0.0, dtype=np.float32))

        stock_trace.append(float(step.stock))
        payoff_total += np.asarray(step.payoffs, dtype=np.float32)
        sanction_total += float(step.sanction_total)
        violation_events += float(step.num_violations)
        requested_trace.append(float(step.requested_harvest_total))
        realized_trace.append(float(step.realized_harvest_total))
        audit_rate_trace.append(float(step.audit_count / max(local_cfg.n_agents, 1)))
        quota_trace.append(float(step.quota))
        quota_clipped_trace.append(float(step.quota_clipped_total))
        repeat_offender_rate_trace.append(float(step.repeat_offender_count / max(local_cfg.n_agents, 1)))
        closure_trace.append(1.0 if step.closure_active else 0.0)

        if step.stock < local_cfg.collapse_threshold:
            if low_stock_start is None:
                low_stock_start = t
        elif low_stock_start is not None:
            recovery_lags.append(t - low_stock_start)
            low_stock_start = None

        if step.collapsed:
            collapsed = True
            break

    metrics = {
        "seed": local_cfg.seed,
        "collapsed": collapsed,
        "t_end": t_end,
        "final_stock": float(stock_trace[-1]) if stock_trace else float(env.stock),
        "mean_stock": float(np.mean(stock_trace)) if stock_trace else 0.0,
        "welfare": float(payoff_total.sum()),
        "sanction_total": float(sanction_total),
        "violation_events": float(violation_events),
        "mean_requested_harvest": float(np.mean(requested_trace)) if requested_trace else 0.0,
        "mean_realized_harvest": float(np.mean(realized_trace)) if realized_trace else 0.0,
        "mean_audit_rate": float(np.mean(audit_rate_trace)) if audit_rate_trace else 0.0,
        "mean_quota": float(np.mean(quota_trace)) if quota_trace else 0.0,
        "mean_quota_clipped_total": float(np.mean(quota_clipped_trace)) if quota_clipped_trace else 0.0,
        "mean_repeat_offender_rate": float(np.mean(repeat_offender_rate_trace)) if repeat_offender_rate_trace else 0.0,
        "closure_active_fraction": float(np.mean(closure_trace)) if closure_trace else 0.0,
        "mean_stock_recovery_lag": float(np.mean(recovery_lags)) if recovery_lags else 0.0,
    }

    return _EpisodeRollout(
        obs=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.int64),
        logprobs=np.asarray(logprob_rows, dtype=np.float32),
        values=np.asarray(value_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.float32),
        metrics=metrics,
    )


def _gae_from_episode(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_values = np.zeros(rewards.shape[1], dtype=np.float32)
            next_nonterminal = 1.0 - dones[t]
        else:
            next_values = values[t + 1]
            next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _ppo_update(
    policy: FisheryPPOPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    train_cfg: PPOTrainConfig,
    *,
    device: str,
) -> None:
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
    old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)

    n_samples = int(obs.shape[0])
    batch_size = min(train_cfg.minibatch_size, n_samples)
    indices = np.arange(n_samples)
    policy.train()
    for _ in range(train_cfg.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            idx = indices[start : start + batch_size]
            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_old_logprobs = old_logprobs[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]

            new_logprobs, entropy, values = policy.evaluate_actions(mb_obs, mb_actions)
            ratio = torch.exp(new_logprobs - mb_old_logprobs)
            pg_loss_1 = ratio * mb_advantages
            pg_loss_2 = torch.clamp(ratio, 1.0 - train_cfg.clip_coef, 1.0 + train_cfg.clip_coef) * mb_advantages
            policy_loss = -torch.min(pg_loss_1, pg_loss_2).mean()
            value_loss = 0.5 * torch.mean((values - mb_returns) ** 2)
            entropy_loss = entropy.mean()
            loss = policy_loss + train_cfg.value_coef * value_loss - train_cfg.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
    policy.eval()
