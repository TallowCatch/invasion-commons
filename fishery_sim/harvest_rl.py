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

from .harvest import BaseHarvestAgent
from .harvest import GovernmentAgent
from .harvest import HarvestAction
from .harvest import HarvestCommonsConfig
from .harvest import HarvestMessage
from .harvest import HarvestObservation
from .harvest import run_harvest_episode
from .harvest_benchmarks import get_harvest_regime_pack
from .harvest_evolution import DEFAULT_GOVERNMENT_PARAMS
from .metrics import gini


ACTION_HEADS = (
    "harvest_frac",
    "announced_restraint",
    "requested_credit",
    "credit_offer",
)


@dataclass(frozen=True)
class HarvestPPOTrainConfig:
    total_timesteps: int = 250_000
    rollout_steps: int = 960
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 2.5e-4
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 64
    harvest_action_bins: int = 9
    communication_bins: int = 5
    credit_offer_bins: int = 5
    eval_every: int = 25_000
    train_eval_episodes: int = 16
    max_grad_norm: float = 0.5
    garden_failure_penalty: float = 25.0
    patch_health_reward_weight: float = 2.0
    local_aggression_penalty_weight: float = 0.0
    neighborhood_overharvest_penalty_weight: float = 0.0


@dataclass
class _EpisodeRollout:
    communication_obs: np.ndarray
    action_obs: np.ndarray
    actions: np.ndarray
    logprobs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    metrics: dict[str, float | int | bool]


class HarvestPPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, n_action_bins: dict[str, int]):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.n_action_bins = {str(key): int(value) for key, value in n_action_bins.items()}
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        )
        self.communication_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.hidden_size, self.n_action_bins[name])
                for name in ("announced_restraint", "requested_credit")
            }
        )
        self.action_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.hidden_size, self.n_action_bins[name])
                for name in ("harvest_frac", "credit_offer")
            }
        )
        self.critic = nn.Linear(self.hidden_size, 1)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._features(obs)

    def _communication_logits(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward(obs)
        return {name: head(features) for name, head in self.communication_heads.items()}

    def _action_logits(
        self,
        obs: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        features = self._features(obs)
        logits = {name: head(features) for name, head in self.action_heads.items()}
        return logits, self.critic(features).squeeze(-1)

    def act_messages(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._communication_logits(obs)
        actions: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        for name in ("announced_restraint", "requested_credit"):
            dist = Categorical(logits=logits[name])
            action = torch.argmax(logits[name], dim=-1) if deterministic else dist.sample()
            actions.append(action)
            logprobs.append(dist.log_prob(action))
        action_tensor = torch.stack(actions, dim=-1)
        total_logprob = torch.stack(logprobs, dim=-1).sum(dim=-1)
        return action_tensor, total_logprob

    def act_actions(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self._action_logits(obs)
        actions: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        for name in ("harvest_frac", "credit_offer"):
            dist = Categorical(logits=logits[name])
            action = torch.argmax(logits[name], dim=-1) if deterministic else dist.sample()
            actions.append(action)
            logprobs.append(dist.log_prob(action))
        action_tensor = torch.stack(actions, dim=-1)
        total_logprob = torch.stack(logprobs, dim=-1).sum(dim=-1)
        return action_tensor, total_logprob, values

    def act(
        self,
        communication_obs: torch.Tensor,
        action_obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        comm_actions, comm_logprob = self.act_messages(
            communication_obs,
            deterministic=deterministic,
        )
        act_actions, act_logprob, values = self.act_actions(
            action_obs,
            deterministic=deterministic,
        )
        action_tensor = torch.cat([act_actions[:, :0], comm_actions, act_actions], dim=-1)
        total_logprob = comm_logprob + act_logprob
        return action_tensor, total_logprob, values

    def evaluate_actions(
        self,
        communication_obs: torch.Tensor,
        action_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        comm_logits = self._communication_logits(communication_obs)
        act_logits, values = self._action_logits(action_obs)
        logprobs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for idx, name in enumerate(("announced_restraint", "requested_credit")):
            dist = Categorical(logits=comm_logits[name])
            logprobs.append(dist.log_prob(actions[:, idx]))
            entropies.append(dist.entropy())
        for offset, name in enumerate(("harvest_frac", "credit_offer"), start=2):
            dist = Categorical(logits=act_logits[name])
            logprobs.append(dist.log_prob(actions[:, offset]))
            entropies.append(dist.entropy())
        total_logprob = torch.stack(logprobs, dim=-1).sum(dim=-1)
        total_entropy = torch.stack(entropies, dim=-1).sum(dim=-1)
        return total_logprob, total_entropy, values


class TorchPolicyHarvestAgent(BaseHarvestAgent):
    name = "torch_policy_harvest"

    def __init__(
        self,
        policy: HarvestPPOPolicy,
        cfg: HarvestCommonsConfig,
        action_bins: dict[str, np.ndarray],
        *,
        deterministic: bool = True,
        device: str = "cpu",
    ):
        self.policy = policy
        self.cfg = cfg
        self.action_bins = {
            key: np.asarray(values, dtype=np.float32)
            for key, values in action_bins.items()
        }
        self.deterministic = bool(deterministic)
        self.device = device
        self._cached_t: int | None = None
        self._cached_message_payload: dict[str, float] | None = None

    def reset(self) -> None:
        self._cached_t = None
        self._cached_message_payload = None

    def observe(self, observation: HarvestObservation, t: int) -> HarvestObservation:
        del t
        return observation

    def communicate(
        self,
        observation: HarvestObservation,
        t: int,
        neighbor_ids: list[int],
    ) -> HarvestMessage:
        del neighbor_ids
        if self._cached_t != int(t):
            obs = torch.as_tensor(
                build_harvest_rl_observation(
                    observation=observation,
                    t=t,
                    cfg=self.cfg,
                    inbox=None,
                )[None, :],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                message_actions, _ = self.policy.act_messages(obs, deterministic=self.deterministic)
            payload = _decode_message_row(
                message_actions[0].detach().cpu().numpy(),
                self.action_bins,
            )
            self._cached_t = int(t)
            self._cached_message_payload = payload
        payload = dict(self._cached_message_payload or {})
        return HarvestMessage(
            announced_restraint=float(payload.get("announced_restraint", 0.0)),
            requested_credit=float(payload.get("requested_credit", 0.0)),
        )

    def act(
        self,
        observation: HarvestObservation,
        inbox: list[HarvestMessage],
        t: int,
    ) -> HarvestAction:
        action_obs = torch.as_tensor(
            build_harvest_rl_observation(
                observation=observation,
                t=t,
                cfg=self.cfg,
                inbox=inbox,
            )[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            action_actions, _, _ = self.policy.act_actions(
                action_obs,
                deterministic=self.deterministic,
            )
        payload = _decode_action_only_row(
            action_actions[0].detach().cpu().numpy(),
            self.action_bins,
        )
        return HarvestAction(
            harvest_frac=float(payload.get("harvest_frac", 0.0)),
            credit_offer=float(payload.get("credit_offer", 0.0)),
        )


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


def build_harvest_action_bins(
    cfg: HarvestCommonsConfig,
    train_cfg: HarvestPPOTrainConfig,
) -> dict[str, np.ndarray]:
    if int(train_cfg.harvest_action_bins) < 2:
        raise ValueError("harvest_action_bins must be >= 2")
    if int(train_cfg.communication_bins) < 2:
        raise ValueError("communication_bins must be >= 2")
    if int(train_cfg.credit_offer_bins) < 2:
        raise ValueError("credit_offer_bins must be >= 2")
    return {
        "harvest_frac": np.linspace(0.0, 1.0, int(train_cfg.harvest_action_bins), dtype=np.float32),
        "announced_restraint": np.linspace(0.0, 1.0, int(train_cfg.communication_bins), dtype=np.float32),
        "requested_credit": np.linspace(0.0, 1.0, int(train_cfg.communication_bins), dtype=np.float32),
        "credit_offer": np.linspace(0.0, float(cfg.credit_cap), int(train_cfg.credit_offer_bins), dtype=np.float32),
    }


def build_harvest_rl_observation(
    *,
    observation: HarvestObservation,
    t: int,
    cfg: HarvestCommonsConfig,
    inbox: list[HarvestMessage] | None = None,
) -> np.ndarray:
    patch_scale = max(float(cfg.patch_max), 1e-9)
    credit_scale = max(float(cfg.credit_cap), 1e-9)
    horizon_scale = max(int(cfg.horizon) - 1, 1)
    cap_frac = (
        0.0
        if observation.government_cap_frac is None
        else float(np.clip(observation.government_cap_frac, 0.0, 1.0))
    )
    cap_active = 0.0 if observation.government_cap_frac is None else 1.0
    inbox_restraint_mean = 0.0
    inbox_credit_request_mean = 0.0
    inbox_requesting_fraction = 0.0
    inbox_cooperative_fraction = 0.0
    if inbox:
        restraints = np.asarray([msg.announced_restraint for msg in inbox], dtype=float)
        requests = np.asarray([msg.requested_credit for msg in inbox], dtype=float)
        inbox_restraint_mean = float(np.clip(restraints.mean(), 0.0, 1.0))
        inbox_credit_request_mean = float(np.clip(requests.mean(), 0.0, 1.0))
        inbox_requesting_fraction = float(np.mean(requests > 0.05))
        inbox_cooperative_fraction = float(np.mean(restraints >= 0.6))
    return np.asarray(
        [
            float(np.clip(observation.local_patch / patch_scale, 0.0, 1.0)),
            float(np.clip(observation.neighbor_mean / patch_scale, 0.0, 1.0)),
            float(np.clip(observation.last_credit_received / credit_scale, 0.0, 1.0)),
            cap_frac,
            cap_active,
            float(np.clip(t / horizon_scale, 0.0, 1.0)),
            inbox_restraint_mean,
            inbox_credit_request_mean,
            inbox_requesting_fraction,
            inbox_cooperative_fraction,
        ],
        dtype=np.float32,
    )


def apply_harvest_rl_condition(
    cfg: HarvestCommonsConfig,
    condition: str,
) -> HarvestCommonsConfig:
    out = copy.deepcopy(cfg)
    name = condition.strip().lower()
    if name == "none":
        out.communication_enabled = False
        out.side_payments_enabled = False
    elif name == "top_down_only":
        out.communication_enabled = False
        out.side_payments_enabled = False
    elif name == "bottom_up_only":
        out.communication_enabled = True
        out.side_payments_enabled = True
    elif name == "hybrid":
        out.communication_enabled = True
        out.side_payments_enabled = True
    else:
        raise ValueError(f"Unknown Harvest RL condition: {condition}")
    return out


def make_harvest_governor(
    condition: str,
    government_params: dict[str, float | int | bool] | None = None,
) -> GovernmentAgent | None:
    params = dict(DEFAULT_GOVERNMENT_PARAMS)
    if government_params:
        params.update(government_params)
    name = condition.strip().lower()
    if name == "top_down_only":
        return GovernmentAgent(**params, enforcement_scope="global", expand_target_neighbors=False)
    if name == "hybrid":
        return GovernmentAgent(**params, enforcement_scope="local", expand_target_neighbors=True)
    if name in {"none", "bottom_up_only"}:
        return None
    raise ValueError(f"Unknown Harvest RL condition: {condition}")


def save_rl_checkpoint(
    path: str,
    policy: HarvestPPOPolicy,
    action_bins: dict[str, np.ndarray],
    cfg: HarvestCommonsConfig,
    train_cfg: HarvestPPOTrainConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "state_dict": policy.state_dict(),
        "action_bins": {
            key: np.asarray(values, dtype=np.float32)
            for key, values in action_bins.items()
        },
        "harvest_config": asdict(cfg),
        "train_config": asdict(train_cfg),
        "policy_config": {
            "obs_dim": policy.obs_dim,
            "hidden_size": policy.hidden_size,
            "n_action_bins": dict(policy.n_action_bins),
        },
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)


def load_rl_checkpoint(
    path: str,
    *,
    device: str = "cpu",
) -> tuple[HarvestPPOPolicy, dict[str, np.ndarray], dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    policy_cfg = dict(payload["policy_config"])
    policy = HarvestPPOPolicy(
        obs_dim=int(policy_cfg["obs_dim"]),
        hidden_size=int(policy_cfg["hidden_size"]),
        n_action_bins={str(k): int(v) for k, v in dict(policy_cfg["n_action_bins"]).items()},
    )
    policy.load_state_dict(payload["state_dict"])
    policy.to(device)
    policy.eval()
    action_bins = {
        str(key): np.asarray(values, dtype=np.float32)
        for key, values in dict(payload["action_bins"]).items()
    }
    metadata = {
        "harvest_config": dict(payload.get("harvest_config", {})),
        "train_config": dict(payload.get("train_config", {})),
        "metadata": dict(payload.get("metadata", {})),
    }
    return policy, action_bins, metadata


def train_self_play_policy(
    cfg: HarvestCommonsConfig,
    condition: str,
    train_cfg: HarvestPPOTrainConfig,
    *,
    run_seed: int,
    government_params: dict[str, float | int | bool] | None = None,
    device: str = "cpu",
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[HarvestPPOPolicy, dict[str, np.ndarray], pd.DataFrame]:
    torch.manual_seed(int(run_seed))
    np.random.seed(int(run_seed))

    conditioned_cfg = apply_harvest_rl_condition(cfg, condition)
    action_bins = build_harvest_action_bins(conditioned_cfg, train_cfg)
    policy = HarvestPPOPolicy(
        obs_dim=10,
        hidden_size=train_cfg.hidden_size,
        n_action_bins={key: len(values) for key, values in action_bins.items()},
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.learning_rate)

    total_timesteps = 0
    update_idx = 0
    next_eval = min(train_cfg.eval_every, train_cfg.total_timesteps)
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    best_state = copy.deepcopy(policy.state_dict())
    history_rows: list[dict[str, float | int]] = []

    while total_timesteps < train_cfg.total_timesteps:
        rollout = _collect_rollout_batch(
            cfg=conditioned_cfg,
            condition=condition,
            policy=policy,
            action_bins=action_bins,
            train_cfg=train_cfg,
            rollout_steps=min(train_cfg.rollout_steps, train_cfg.total_timesteps - total_timesteps),
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            device=device,
            seed=run_seed + update_idx * 1_000,
            government_params=government_params,
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
            "rollout_garden_failure_rate": float(rollout["garden_failure_rate"]),
            "rollout_episode_length_mean": float(rollout["episode_length_mean"]),
        }

        should_eval = total_timesteps >= next_eval or total_timesteps >= train_cfg.total_timesteps
        if should_eval:
            _, eval_summary = evaluate_self_play_policy(
                cfg=conditioned_cfg,
                condition=condition,
                policy=policy,
                action_bins=action_bins,
                n_eval_episodes=train_cfg.train_eval_episodes,
                seed=run_seed + 50_000 + update_idx * 17,
                benchmark_pack=None,
                deterministic=True,
                prefix="train",
                government_params=government_params,
                device=device,
            )
            row.update(eval_summary)
            candidate = (
                -float(eval_summary["train_garden_failure_rate"]),
                float(eval_summary["train_mean_patch_health"]),
                float(eval_summary["train_mean_welfare"]),
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
    cfg: HarvestCommonsConfig,
    condition: str,
    policy: HarvestPPOPolicy,
    action_bins: dict[str, np.ndarray],
    *,
    n_eval_episodes: int,
    seed: int,
    benchmark_pack: list[dict[str, Any]] | None,
    deterministic: bool,
    prefix: str,
    government_params: dict[str, float | int | bool] | None = None,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if n_eval_episodes < 1:
        raise ValueError("n_eval_episodes must be >= 1")

    regimes = benchmark_pack or [{"name": "default", "overrides": {}}]
    rows: list[dict[str, Any]] = []
    for regime_idx, regime in enumerate(regimes):
        regime_name = str(regime.get("name", f"regime_{regime_idx}"))
        overrides = dict(regime.get("overrides", {}))
        regime_cfg = apply_harvest_rl_condition(copy.deepcopy(cfg), condition)
        for key, value in overrides.items():
            if hasattr(regime_cfg, key):
                setattr(regime_cfg, key, value)

        for ep in range(n_eval_episodes):
            episode_cfg = copy.deepcopy(regime_cfg)
            episode_cfg.seed = int(seed + regime_idx * 1_000_000 + ep)
            governor = make_harvest_governor(condition, government_params)
            agents = [
                TorchPolicyHarvestAgent(
                    policy=policy,
                    cfg=episode_cfg,
                    action_bins=action_bins,
                    deterministic=deterministic,
                    device=device,
                )
                for _ in range(episode_cfg.n_agents)
            ]
            out = run_harvest_episode(episode_cfg, agents, governor=governor)
            rows.append(
                {
                    "seed": episode_cfg.seed,
                    "regime": regime_name,
                    "garden_failure_event": bool(out["garden_failure_event"]),
                    "time_to_garden_failure": float(out["time_to_garden_failure"]),
                    "final_patch_health": float(out["final_patch_health"]),
                    "mean_patch_health": float(out["mean_patch_health"]),
                    "mean_welfare": float(out["mean_welfare"]),
                    "payoff_gini": float(out["payoff_gini"]),
                    "mean_credit_transferred": float(out["mean_credit_transferred"]),
                    "mean_government_cap": float(out["mean_government_cap"]),
                    "mean_aggressive_request_fraction": float(out["mean_aggressive_request_fraction"]),
                    "mean_max_local_aggression": float(out["mean_max_local_aggression"]),
                    "mean_neighborhood_overharvest": float(out["mean_neighborhood_overharvest"]),
                    "mean_capped_action_fraction": float(out["mean_capped_action_fraction"]),
                    "mean_targeted_agent_fraction": float(out["mean_targeted_agent_fraction"]),
                    "mean_prevented_harvest": float(out["mean_prevented_harvest"]),
                    "mean_patch_variance": float(out["mean_patch_variance"]),
                    "mean_requested_harvest": float(out["mean_requested_harvest"]),
                    "mean_realized_harvest": float(out["mean_realized_harvest"]),
                }
            )

    episode_df = pd.DataFrame(rows)
    summary = _build_eval_summary(episode_df=episode_df, prefix=prefix)
    summary[f"{prefix}_regime_count"] = int(len(regimes))
    return episode_df, summary


def load_harvest_benchmark_pack_from_args(benchmark_pack: str | None) -> list[dict[str, Any]] | None:
    if benchmark_pack:
        return get_harvest_regime_pack(benchmark_pack)
    return None


def _safe_name(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    out = out.strip("_")
    return out or "regime"


def _build_eval_summary(episode_df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if episode_df.empty:
        return {}

    summary: dict[str, Any] = {
        f"{prefix}_garden_failure_rate": float(episode_df["garden_failure_event"].mean()),
        f"{prefix}_mean_patch_health": float(episode_df["mean_patch_health"].mean()),
        f"{prefix}_mean_final_patch_health": float(episode_df["final_patch_health"].mean()),
        f"{prefix}_mean_welfare": float(episode_df["mean_welfare"].mean()),
        f"{prefix}_mean_payoff_gini": float(episode_df["payoff_gini"].mean()),
        f"{prefix}_mean_credit_transferred": float(episode_df["mean_credit_transferred"].mean()),
        f"{prefix}_mean_government_cap": float(episode_df["mean_government_cap"].mean()),
        f"{prefix}_mean_aggressive_request_fraction": float(episode_df["mean_aggressive_request_fraction"].mean()),
        f"{prefix}_mean_max_local_aggression": float(episode_df["mean_max_local_aggression"].mean()),
        f"{prefix}_mean_neighborhood_overharvest": float(episode_df["mean_neighborhood_overharvest"].mean()),
        f"{prefix}_mean_capped_action_fraction": float(episode_df["mean_capped_action_fraction"].mean()),
        f"{prefix}_mean_targeted_agent_fraction": float(episode_df["mean_targeted_agent_fraction"].mean()),
        f"{prefix}_mean_prevented_harvest": float(episode_df["mean_prevented_harvest"].mean()),
        f"{prefix}_mean_patch_variance": float(episode_df["mean_patch_variance"].mean()),
        f"{prefix}_mean_requested_harvest": float(episode_df["mean_requested_harvest"].mean()),
        f"{prefix}_mean_realized_harvest": float(episode_df["mean_realized_harvest"].mean()),
        f"{prefix}_mean_time_to_garden_failure": float(episode_df["time_to_garden_failure"].mean()),
    }
    per_regime_survival: list[float] = []
    for regime_name, rdf in episode_df.groupby("regime", sort=True):
        safe = _safe_name(str(regime_name))
        val = float(1.0 - rdf["garden_failure_event"].mean())
        summary[f"per_regime_survival_over_generations__{safe}"] = val
        per_regime_survival.append(val)
    summary["per_regime_survival_over_generations_mean"] = (
        float(np.mean(per_regime_survival)) if per_regime_survival else 0.0
    )
    return summary


def _decode_action_row(
    action_row: np.ndarray,
    action_bins: dict[str, np.ndarray],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for idx, name in enumerate(ACTION_HEADS):
        bins = action_bins[name]
        action_idx = int(np.clip(action_row[idx], 0, len(bins) - 1))
        values[name] = float(bins[action_idx])
    return values


def _decode_message_row(
    action_row: np.ndarray,
    action_bins: dict[str, np.ndarray],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for idx, name in enumerate(("announced_restraint", "requested_credit")):
        bins = action_bins[name]
        action_idx = int(np.clip(action_row[idx], 0, len(bins) - 1))
        values[name] = float(bins[action_idx])
    return values


def _decode_action_only_row(
    action_row: np.ndarray,
    action_bins: dict[str, np.ndarray],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for idx, name in enumerate(("harvest_frac", "credit_offer")):
        bins = action_bins[name]
        action_idx = int(np.clip(action_row[idx], 0, len(bins) - 1))
        values[name] = float(bins[action_idx])
    return values


def _build_training_rewards(
    *,
    payoffs: np.ndarray,
    patch_health: np.ndarray,
    max_local_aggression: float,
    neighborhood_overharvest: float,
    garden_failure_event: bool,
    cfg: HarvestCommonsConfig,
    train_cfg: HarvestPPOTrainConfig,
) -> np.ndarray:
    rewards = np.asarray(payoffs, dtype=np.float32)

    if float(train_cfg.patch_health_reward_weight) != 0.0:
        rewards = rewards + float(train_cfg.patch_health_reward_weight) * float(
            np.clip(patch_health.mean() / max(cfg.patch_max, 1e-9), 0.0, 1.0)
        )

    if float(train_cfg.local_aggression_penalty_weight) != 0.0:
        rewards = rewards - float(train_cfg.local_aggression_penalty_weight) * float(
            np.clip(max_local_aggression, 0.0, 1.0)
        )

    if float(train_cfg.neighborhood_overharvest_penalty_weight) != 0.0:
        max_overharvest = (
            3.0
            * max(0.0, 1.0 - float(cfg.sustainable_harvest_frac))
            * float(cfg.max_harvest_per_agent)
        )
        norm_overharvest = float(
            np.clip(neighborhood_overharvest / max(max_overharvest, 1e-9), 0.0, 1.0)
        )
        rewards = rewards - float(train_cfg.neighborhood_overharvest_penalty_weight) * norm_overharvest

    if garden_failure_event and float(train_cfg.garden_failure_penalty) > 0.0:
        rewards = rewards - float(train_cfg.garden_failure_penalty)

    return rewards.astype(np.float32)


def _collect_rollout_batch(
    cfg: HarvestCommonsConfig,
    condition: str,
    policy: HarvestPPOPolicy,
    action_bins: dict[str, np.ndarray],
    *,
    train_cfg: HarvestPPOTrainConfig,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    device: str,
    seed: int,
    government_params: dict[str, float | int | bool] | None,
) -> dict[str, Any]:
    episodes: list[_EpisodeRollout] = []
    total_steps = 0
    episode_seed = int(seed)
    while total_steps < rollout_steps:
        episode = _run_training_episode(
            cfg=cfg,
            condition=condition,
            policy=policy,
            action_bins=action_bins,
            train_cfg=train_cfg,
            seed=episode_seed,
            device=device,
            government_params=government_params,
        )
        episodes.append(episode)
        total_steps += int(episode.communication_obs.shape[0] * episode.communication_obs.shape[1])
        episode_seed += 1

    communication_obs_batches: list[np.ndarray] = []
    action_obs_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    logprob_batches: list[np.ndarray] = []
    value_batches: list[np.ndarray] = []
    reward_batches: list[np.ndarray] = []
    advantage_batches: list[np.ndarray] = []
    return_batches: list[np.ndarray] = []
    failure_flags: list[float] = []
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
        communication_obs_batches.append(
            episode.communication_obs.reshape(-1, episode.communication_obs.shape[-1])
        )
        action_obs_batches.append(
            episode.action_obs.reshape(-1, episode.action_obs.shape[-1])
        )
        action_batches.append(episode.actions.reshape(-1, episode.actions.shape[-1]))
        logprob_batches.append(episode.logprobs.reshape(-1))
        value_batches.append(episode.values.reshape(-1))
        reward_batches.append(episode.rewards.reshape(-1))
        advantage_batches.append(advantages.reshape(-1))
        return_batches.append(returns.reshape(-1))
        failure_flags.append(float(bool(episode.metrics["garden_failure_event"])))
        lengths.append(int(episode.communication_obs.shape[0]))
        reward_means.append(float(np.mean(episode.rewards)))

    advantages = np.concatenate(advantage_batches).astype(np.float32)
    advantages = (advantages - advantages.mean()) / max(advantages.std(ddof=0), 1e-8)
    return {
        "communication_obs": np.concatenate(communication_obs_batches).astype(np.float32),
        "action_obs": np.concatenate(action_obs_batches).astype(np.float32),
        "actions": np.concatenate(action_batches).astype(np.int64),
        "logprobs": np.concatenate(logprob_batches).astype(np.float32),
        "values": np.concatenate(value_batches).astype(np.float32),
        "rewards": np.concatenate(reward_batches).astype(np.float32),
        "advantages": advantages.astype(np.float32),
        "returns": np.concatenate(return_batches).astype(np.float32),
        "garden_failure_rate": float(np.mean(failure_flags)) if failure_flags else 0.0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "reward_mean": float(np.mean(reward_means)) if reward_means else 0.0,
    }


def _run_training_episode(
    cfg: HarvestCommonsConfig,
    condition: str,
    policy: HarvestPPOPolicy,
    action_bins: dict[str, np.ndarray],
    *,
    train_cfg: HarvestPPOTrainConfig,
    seed: int,
    device: str,
    government_params: dict[str, float | int | bool] | None,
) -> _EpisodeRollout:
    local_cfg = apply_harvest_rl_condition(copy.deepcopy(cfg), condition)
    local_cfg.seed = int(seed)
    rng = np.random.default_rng(local_cfg.seed)
    patch_health = np.full(local_cfg.n_agents, local_cfg.patch_init, dtype=float)
    last_credit_received = np.zeros(local_cfg.n_agents, dtype=float)
    failure_streak = 0
    garden_failure_event = 0
    failure_step = local_cfg.horizon
    governor = make_harvest_governor(condition, government_params)
    if governor is not None:
        governor.reset()

    communication_obs_rows: list[np.ndarray] = []
    action_obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    logprob_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    done_rows: list[np.ndarray] = []

    patch_mean_trace: list[float] = []
    welfare_trace: list[float] = []
    credit_trace: list[float] = []
    government_cap_trace: list[float] = []
    aggressive_request_trace: list[float] = []
    max_local_aggression_trace: list[float] = []
    capped_action_trace: list[float] = []
    prevented_harvest_trace: list[float] = []
    patch_variance_trace: list[float] = []
    neighborhood_overharvest_trace: list[float] = []
    targeted_agent_trace: list[float] = []
    requested_harvest_trace: list[float] = []
    realized_harvest_trace: list[float] = []

    final_payoffs = np.zeros(local_cfg.n_agents, dtype=np.float32)
    aggressive_threshold = (
        governor.aggressive_request_threshold if governor is not None else 0.75
    )
    t_end = local_cfg.horizon

    for t in range(local_cfg.horizon):
        government_cap_fracs = None
        if governor is not None:
            government_cap_fracs = governor.act(float(patch_health.mean()), t, local_cfg.n_agents)
        if government_cap_fracs is None:
            government_cap_trace.append(-1.0)
            targeted_agent_trace.append(0.0)
        else:
            active_caps = government_cap_fracs[~np.isnan(government_cap_fracs)]
            government_cap_trace.append(float(np.mean(active_caps)) if active_caps.size else -1.0)
            targeted_agent_trace.append(float(np.mean(~np.isnan(government_cap_fracs))))

        observations: list[HarvestObservation] = []
        for i in range(local_cfg.n_agents):
            neighbor_mean = float(
                np.mean([patch_health[j] for j in _neighbors(i, local_cfg.n_agents)])
            )
            cap_frac_i = None
            if government_cap_fracs is not None and not np.isnan(government_cap_fracs[i]):
                cap_frac_i = float(government_cap_fracs[i])
            observations.append(
                HarvestObservation(
                    local_patch=float(patch_health[i]),
                    neighbor_mean=neighbor_mean,
                    last_credit_received=float(last_credit_received[i]),
                    government_cap_frac=cap_frac_i,
                )
            )

        communication_obs_batch = np.asarray(
            [
                build_harvest_rl_observation(
                    observation=obs,
                    t=t,
                    cfg=local_cfg,
                    inbox=None,
                )
                for obs in observations
            ],
            dtype=np.float32,
        )
        communication_obs_tensor = torch.as_tensor(
            communication_obs_batch,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            message_actions, message_logprobs = policy.act_messages(
                communication_obs_tensor,
                deterministic=False,
            )
        message_action_idx = message_actions.detach().cpu().numpy().astype(np.int64)

        message_rows = [
            _decode_message_row(message_action_idx[i], action_bins)
            for i in range(local_cfg.n_agents)
        ]
        messages = [HarvestMessage() for _ in range(local_cfg.n_agents)]
        if local_cfg.communication_enabled:
            for i, payload in enumerate(message_rows):
                messages[i] = HarvestMessage(
                    announced_restraint=float(payload["announced_restraint"]),
                    requested_credit=float(payload["requested_credit"]),
                )

        action_obs_batch = np.asarray(
            [
                build_harvest_rl_observation(
                    observation=observations[i],
                    t=t,
                    cfg=local_cfg,
                    inbox=[messages[j] for j in _neighbors(i, local_cfg.n_agents)]
                    if local_cfg.communication_enabled
                    else [],
                )
                for i in range(local_cfg.n_agents)
            ],
            dtype=np.float32,
        )
        action_obs_tensor = torch.as_tensor(
            action_obs_batch,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            action_actions, action_logprobs, values = policy.act_actions(
                action_obs_tensor,
                deterministic=False,
            )
        action_action_idx = action_actions.detach().cpu().numpy().astype(np.int64)
        action_rows_decoded = [
            _decode_action_only_row(action_action_idx[i], action_bins)
            for i in range(local_cfg.n_agents)
        ]
        action_idx = np.concatenate([message_action_idx, action_action_idx], axis=-1)
        logprobs = message_logprobs + action_logprobs

        requested_fracs_arr = np.asarray(
            [float(payload["harvest_frac"]) for payload in action_rows_decoded],
            dtype=float,
        )
        if governor is not None:
            capped_fracs_arr, targeted_mask = governor.apply_cap(requested_fracs_arr, government_cap_fracs)
        else:
            capped_fracs_arr = requested_fracs_arr.copy()
            targeted_mask = np.zeros(local_cfg.n_agents, dtype=bool)
        aggressive_request_trace.append(
            float(np.mean(requested_fracs_arr > aggressive_threshold))
        )
        local_aggression = []
        local_overharvest = []
        overharvest_fracs = np.maximum(0.0, requested_fracs_arr - local_cfg.sustainable_harvest_frac)
        for i in range(local_cfg.n_agents):
            neighborhood = [i, *_neighbors(i, local_cfg.n_agents)]
            local_aggression.append(
                float(np.mean(requested_fracs_arr[neighborhood] > aggressive_threshold))
            )
            local_overharvest.append(
                float(np.sum(overharvest_fracs[neighborhood]) * local_cfg.max_harvest_per_agent)
            )
        max_local_aggression_trace.append(float(max(local_aggression, default=0.0)))
        neighborhood_overharvest_trace.append(
            float(np.mean(local_overharvest)) if local_overharvest else 0.0
        )
        capped_action_trace.append(
            float(np.mean(capped_fracs_arr + 1e-9 < requested_fracs_arr))
        )
        prevented_harvest_trace.append(
            float(
                np.sum(np.maximum(0.0, requested_fracs_arr - capped_fracs_arr))
                * local_cfg.max_harvest_per_agent
            )
        )
        if governor is not None:
            governor.observe(requested_fracs=requested_fracs_arr)

        harvests = np.zeros(local_cfg.n_agents, dtype=float)
        credit_offer_arr = np.asarray(
            [float(payload["credit_offer"]) for payload in action_rows_decoded],
            dtype=float,
        )
        for i in range(local_cfg.n_agents):
            requested = float(capped_fracs_arr[i]) * local_cfg.max_harvest_per_agent
            harvests[i] = min(requested, patch_health[i])

        credits_received = np.zeros(local_cfg.n_agents, dtype=float)
        credit_costs = np.zeros(local_cfg.n_agents, dtype=float)
        if local_cfg.communication_enabled and local_cfg.side_payments_enabled:
            for i in range(local_cfg.n_agents):
                eligible_neighbors: list[int] = []
                for j in _neighbors(i, local_cfg.n_agents):
                    if (
                        messages[j].announced_restraint >= 0.6
                        and harvests[j] <= local_cfg.max_harvest_per_agent * 0.6
                    ):
                        eligible_neighbors.append(j)
                if eligible_neighbors and credit_offer_arr[i] > 0.0:
                    per_neighbor = min(float(credit_offer_arr[i]), local_cfg.credit_cap) / len(eligible_neighbors)
                    credits_received[eligible_neighbors] += per_neighbor
                    credit_costs[i] += per_neighbor * len(eligible_neighbors)

        overharvest = np.maximum(
            0.0,
            harvests - local_cfg.sustainable_harvest_frac * local_cfg.max_harvest_per_agent,
        )
        next_health = np.zeros_like(patch_health)
        for i in range(local_cfg.n_agents):
            spillover_loss = local_cfg.neighbor_externality * sum(
                overharvest[j] for j in _neighbors(i, local_cfg.n_agents)
            )
            remaining = max(0.0, patch_health[i] - harvests[i] - spillover_loss)
            growth = local_cfg.regen_rate * remaining * (1.0 - remaining / local_cfg.patch_max)
            weather = rng.normal(0.0, local_cfg.weather_noise_std)
            next_health[i] = float(
                np.clip(remaining + max(0.0, growth) + weather, 0.0, local_cfg.patch_max)
            )

        payoffs = harvests + credits_received - credit_costs
        final_payoffs += np.asarray(payoffs, dtype=np.float32)
        patch_health = next_health
        last_credit_received = credits_received

        patch_mean_trace.append(float(patch_health.mean()))
        patch_variance_trace.append(float(np.var(patch_health)))
        welfare_trace.append(float(payoffs.sum()))
        credit_trace.append(float(credits_received.sum()))
        requested_harvest_trace.append(
            float(np.sum(requested_fracs_arr) * local_cfg.max_harvest_per_agent)
        )
        realized_harvest_trace.append(float(np.sum(harvests)))
        t_end = t + 1

        failed_fraction = float(
            np.mean(patch_health < local_cfg.local_patch_failure_threshold)
        )
        if failed_fraction >= local_cfg.failure_fraction_threshold:
            failure_streak += 1
        else:
            failure_streak = 0
        if failure_streak >= max(1, local_cfg.failure_patience):
            garden_failure_event = 1
            failure_step = t + 1

        rewards = _build_training_rewards(
            payoffs=np.asarray(payoffs, dtype=np.float32),
            patch_health=patch_health,
            max_local_aggression=float(max_local_aggression_trace[-1]) if max_local_aggression_trace else 0.0,
            neighborhood_overharvest=float(neighborhood_overharvest_trace[-1]) if neighborhood_overharvest_trace else 0.0,
            garden_failure_event=bool(garden_failure_event),
            cfg=local_cfg,
            train_cfg=train_cfg,
        )

        communication_obs_rows.append(communication_obs_batch)
        action_obs_rows.append(action_obs_batch)
        action_rows.append(action_idx.astype(np.int64))
        logprob_rows.append(logprobs.detach().cpu().numpy().astype(np.float32))
        value_rows.append(values.detach().cpu().numpy().astype(np.float32))
        reward_rows.append(rewards.astype(np.float32))
        done_flag = bool(garden_failure_event or t == local_cfg.horizon - 1)
        done_rows.append(
            np.full(local_cfg.n_agents, 1.0 if done_flag else 0.0, dtype=np.float32)
        )

        if garden_failure_event:
            break

    metrics = {
        "seed": local_cfg.seed,
        "garden_failure_event": bool(garden_failure_event),
        "failure_step": int(failure_step),
        "time_to_garden_failure": float(failure_step if garden_failure_event else t_end),
        "t_end": int(t_end),
        "mean_patch_health": float(np.mean(patch_mean_trace)) if patch_mean_trace else 0.0,
        "final_patch_health": float(patch_health.mean()),
        "mean_welfare": float(np.mean(welfare_trace)) if welfare_trace else 0.0,
        "payoff_gini": float(gini(final_payoffs)),
        "mean_credit_transferred": float(np.mean(credit_trace)) if credit_trace else 0.0,
        "mean_government_cap": float(
            np.mean([x for x in government_cap_trace if x >= 0.0])
        )
        if any(x >= 0.0 for x in government_cap_trace)
        else 0.0,
        "mean_aggressive_request_fraction": float(np.mean(aggressive_request_trace))
        if aggressive_request_trace
        else 0.0,
        "mean_max_local_aggression": float(np.mean(max_local_aggression_trace))
        if max_local_aggression_trace
        else 0.0,
        "mean_neighborhood_overharvest": float(np.mean(neighborhood_overharvest_trace))
        if neighborhood_overharvest_trace
        else 0.0,
        "mean_capped_action_fraction": float(np.mean(capped_action_trace))
        if capped_action_trace
        else 0.0,
        "mean_targeted_agent_fraction": float(np.mean(targeted_agent_trace))
        if targeted_agent_trace
        else 0.0,
        "mean_prevented_harvest": float(np.mean(prevented_harvest_trace))
        if prevented_harvest_trace
        else 0.0,
        "mean_patch_variance": float(np.mean(patch_variance_trace))
        if patch_variance_trace
        else 0.0,
        "mean_requested_harvest": float(np.mean(requested_harvest_trace))
        if requested_harvest_trace
        else 0.0,
        "mean_realized_harvest": float(np.mean(realized_harvest_trace))
        if realized_harvest_trace
        else 0.0,
    }

    return _EpisodeRollout(
        communication_obs=np.asarray(communication_obs_rows, dtype=np.float32),
        action_obs=np.asarray(action_obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.int64),
        logprobs=np.asarray(logprob_rows, dtype=np.float32),
        values=np.asarray(value_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.float32),
        metrics=metrics,
    )


def _neighbors(i: int, n_agents: int) -> list[int]:
    return [((i - 1) % n_agents), ((i + 1) % n_agents)]


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
    policy: HarvestPPOPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    train_cfg: HarvestPPOTrainConfig,
    *,
    device: str,
) -> None:
    communication_obs = torch.as_tensor(
        batch["communication_obs"],
        dtype=torch.float32,
        device=device,
    )
    action_obs = torch.as_tensor(
        batch["action_obs"],
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
    old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)

    n_samples = int(communication_obs.shape[0])
    batch_size = min(train_cfg.minibatch_size, n_samples)
    indices = np.arange(n_samples)
    policy.train()
    for _ in range(train_cfg.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            idx = indices[start : start + batch_size]
            mb_communication_obs = communication_obs[idx]
            mb_action_obs = action_obs[idx]
            mb_actions = actions[idx]
            mb_old_logprobs = old_logprobs[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]

            new_logprobs, entropy, values = policy.evaluate_actions(
                mb_communication_obs,
                mb_action_obs,
                mb_actions,
            )
            ratio = torch.exp(new_logprobs - mb_old_logprobs)
            pg_loss_1 = ratio * mb_advantages
            pg_loss_2 = torch.clamp(
                ratio,
                1.0 - train_cfg.clip_coef,
                1.0 + train_cfg.clip_coef,
            ) * mb_advantages
            policy_loss = -torch.min(pg_loss_1, pg_loss_2).mean()
            value_loss = 0.5 * torch.mean((values - mb_returns) ** 2)
            entropy_loss = entropy.mean()
            loss = policy_loss + train_cfg.value_coef * value_loss - train_cfg.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
    policy.eval()
