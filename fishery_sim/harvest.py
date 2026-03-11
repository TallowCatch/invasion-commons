from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import gini


@dataclass
class HarvestCommonsConfig:
    n_agents: int = 6
    horizon: int = 80
    patch_init: float = 14.0
    patch_max: float = 20.0
    regen_rate: float = 0.7
    weather_noise_std: float = 0.3
    neighbor_externality: float = 0.12
    sustainable_harvest_frac: float = 0.35
    max_harvest_per_agent: float = 6.0
    credit_cap: float = 1.0
    communication_enabled: bool = True
    side_payments_enabled: bool = True
    seed: int = 0


@dataclass
class HarvestObservation:
    local_patch: float
    neighbor_mean: float
    last_credit_received: float
    government_cap_frac: float | None


@dataclass
class HarvestMessage:
    announced_restraint: float = 0.0
    requested_credit: float = 0.0


@dataclass
class HarvestAction:
    harvest_frac: float
    credit_offer: float = 0.0


@dataclass
class HarvestStepResult:
    patch_health: np.ndarray
    harvests: np.ndarray
    payoffs: np.ndarray
    credits_received: np.ndarray
    total_credit_transferred: float
    government_cap_frac: float | None


class BaseHarvestAgent:
    name: str = "harvest_base"

    def observe(self, observation: HarvestObservation, t: int) -> HarvestObservation:
        del t
        return observation

    def communicate(
        self,
        observation: HarvestObservation,
        t: int,
        neighbor_ids: list[int],
    ) -> HarvestMessage:
        del observation, t, neighbor_ids
        return HarvestMessage()

    def act(
        self,
        observation: HarvestObservation,
        inbox: list[HarvestMessage],
        t: int,
    ) -> HarvestAction:
        del observation, inbox, t
        return HarvestAction(harvest_frac=0.0, credit_offer=0.0)


class SelfInterestedHarvestAgent(BaseHarvestAgent):
    name = "selfish_harvest"

    def communicate(self, observation: HarvestObservation, t: int, neighbor_ids: list[int]) -> HarvestMessage:
        del t, neighbor_ids
        return HarvestMessage(announced_restraint=0.1 if observation.local_patch < 8.0 else 0.2, requested_credit=0.0)

    def act(self, observation: HarvestObservation, inbox: list[HarvestMessage], t: int) -> HarvestAction:
        del inbox, t
        frac = 0.9 if observation.local_patch > 6.0 else 0.65
        return HarvestAction(harvest_frac=frac, credit_offer=0.0)


class ReciprocalHarvestAgent(BaseHarvestAgent):
    name = "reciprocal_harvest"

    def communicate(self, observation: HarvestObservation, t: int, neighbor_ids: list[int]) -> HarvestMessage:
        del t, neighbor_ids
        restraint = 0.8 if observation.local_patch < 10.0 else 0.6
        requested_credit = 0.4 if observation.local_patch < 8.0 else 0.0
        return HarvestMessage(announced_restraint=restraint, requested_credit=requested_credit)

    def act(self, observation: HarvestObservation, inbox: list[HarvestMessage], t: int) -> HarvestAction:
        del t
        cooperative_neighbors = sum(msg.announced_restraint >= 0.6 for msg in inbox)
        frac = 0.45 if observation.local_patch > 8.0 else 0.25
        if cooperative_neighbors == 0:
            frac += 0.1
        return HarvestAction(harvest_frac=min(frac, 0.7), credit_offer=0.2 if observation.local_patch > 10.0 else 0.0)


class CreditSharingHarvestAgent(BaseHarvestAgent):
    name = "credit_sharing_harvest"

    def communicate(self, observation: HarvestObservation, t: int, neighbor_ids: list[int]) -> HarvestMessage:
        del t, neighbor_ids
        return HarvestMessage(
            announced_restraint=0.75 if observation.local_patch < 11.0 else 0.5,
            requested_credit=0.2 if observation.local_patch < 7.0 else 0.0,
        )

    def act(self, observation: HarvestObservation, inbox: list[HarvestMessage], t: int) -> HarvestAction:
        del t
        requested_credit = sum(msg.requested_credit for msg in inbox)
        base_frac = 0.4 if observation.local_patch > 8.0 else 0.22
        offer = 0.4 if requested_credit > 0.0 and observation.local_patch > 9.0 else 0.15
        return HarvestAction(harvest_frac=base_frac, credit_offer=offer)


class GovernmentAgent:
    def __init__(
        self,
        trigger: float = 8.0,
        strict_cap_frac: float = 0.2,
        relaxed_cap_frac: float = 0.6,
        soft_trigger: float | None = None,
        deterioration_threshold: float = 0.35,
        activation_warmup: int = 3,
    ):
        self.trigger = float(trigger)
        self.strict_cap_frac = float(strict_cap_frac)
        self.relaxed_cap_frac = float(relaxed_cap_frac)
        self.soft_trigger = float(soft_trigger) if soft_trigger is not None else float(trigger * 1.15)
        self.deterioration_threshold = float(max(0.0, deterioration_threshold))
        self.activation_warmup = int(max(0, activation_warmup))
        self._prev_mean_patch_health: float | None = None

    def set_cap(self, mean_patch_health: float, t: int) -> float | None:
        trend = None if self._prev_mean_patch_health is None else mean_patch_health - self._prev_mean_patch_health
        self._prev_mean_patch_health = float(mean_patch_health)
        if mean_patch_health < self.trigger:
            return self.strict_cap_frac
        if t < self.activation_warmup:
            return None
        if mean_patch_health < self.soft_trigger and trend is not None and trend < -self.deterioration_threshold:
            return self.relaxed_cap_frac
        return None


def _neighbors(i: int, n_agents: int) -> list[int]:
    return [((i - 1) % n_agents), ((i + 1) % n_agents)]


def run_harvest_episode(
    cfg: HarvestCommonsConfig,
    agents: list[BaseHarvestAgent],
    governor: GovernmentAgent | None = None,
) -> dict:
    if len(agents) != cfg.n_agents:
        raise ValueError("agents list must match cfg.n_agents")

    rng = np.random.default_rng(cfg.seed)
    patch_health = np.full(cfg.n_agents, cfg.patch_init, dtype=float)
    last_credit_received = np.zeros(cfg.n_agents, dtype=float)

    patch_mean_trace = []
    welfare_trace = []
    government_cap_trace = []
    credit_trace = []

    final_payoffs = np.zeros(cfg.n_agents, dtype=float)

    for t in range(cfg.horizon):
        government_cap_frac = None
        if governor is not None:
            cap = governor.set_cap(float(patch_health.mean()), t)
            government_cap_frac = None if cap is None else float(cap)
        government_cap_trace.append(-1.0 if government_cap_frac is None else government_cap_frac)

        observations = []
        for i in range(cfg.n_agents):
            neighbor_mean = float(np.mean([patch_health[j] for j in _neighbors(i, cfg.n_agents)]))
            obs = HarvestObservation(
                local_patch=float(patch_health[i]),
                neighbor_mean=neighbor_mean,
                last_credit_received=float(last_credit_received[i]),
                government_cap_frac=government_cap_frac,
            )
            observations.append(agents[i].observe(obs, t))

        messages = [HarvestMessage() for _ in range(cfg.n_agents)]
        if cfg.communication_enabled:
            for i, agent in enumerate(agents):
                messages[i] = agent.communicate(observations[i], t, _neighbors(i, cfg.n_agents))

        actions = []
        for i, agent in enumerate(agents):
            inbox = [messages[j] for j in _neighbors(i, cfg.n_agents)] if cfg.communication_enabled else []
            action = agent.act(observations[i], inbox, t)
            capped_frac = float(np.clip(action.harvest_frac, 0.0, 1.0))
            if government_cap_frac is not None:
                capped_frac = min(capped_frac, government_cap_frac)
            actions.append(
                HarvestAction(
                    harvest_frac=capped_frac,
                    credit_offer=float(np.clip(action.credit_offer, 0.0, cfg.credit_cap)),
                )
            )

        harvests = np.zeros(cfg.n_agents, dtype=float)
        for i, action in enumerate(actions):
            requested = action.harvest_frac * cfg.max_harvest_per_agent
            harvests[i] = min(requested, patch_health[i])

        credits_received = np.zeros(cfg.n_agents, dtype=float)
        credit_costs = np.zeros(cfg.n_agents, dtype=float)
        if cfg.communication_enabled and cfg.side_payments_enabled:
            for i, action in enumerate(actions):
                eligible_neighbors = []
                for j in _neighbors(i, cfg.n_agents):
                    if messages[j].announced_restraint >= 0.6 and harvests[j] <= cfg.max_harvest_per_agent * 0.6:
                        eligible_neighbors.append(j)
                if eligible_neighbors and action.credit_offer > 0.0:
                    per_neighbor = min(action.credit_offer, cfg.credit_cap) / len(eligible_neighbors)
                    credits_received[eligible_neighbors] += per_neighbor
                    credit_costs[i] += per_neighbor * len(eligible_neighbors)

        overharvest = np.maximum(0.0, harvests - cfg.sustainable_harvest_frac * cfg.max_harvest_per_agent)
        next_health = np.zeros_like(patch_health)
        for i in range(cfg.n_agents):
            spillover_loss = cfg.neighbor_externality * sum(overharvest[j] for j in _neighbors(i, cfg.n_agents))
            remaining = max(0.0, patch_health[i] - harvests[i] - spillover_loss)
            growth = cfg.regen_rate * remaining * (1.0 - remaining / cfg.patch_max)
            weather = rng.normal(0.0, cfg.weather_noise_std)
            next_health[i] = float(np.clip(remaining + max(0.0, growth) + weather, 0.0, cfg.patch_max))

        payoffs = harvests + credits_received - credit_costs
        final_payoffs += payoffs
        patch_health = next_health
        last_credit_received = credits_received

        patch_mean_trace.append(float(patch_health.mean()))
        welfare_trace.append(float(payoffs.sum()))
        credit_trace.append(float(credits_received.sum()))

    return {
        "seed": cfg.seed,
        "mean_patch_health": float(np.mean(patch_mean_trace)) if patch_mean_trace else 0.0,
        "final_patch_health": float(patch_health.mean()),
        "total_welfare": float(np.sum(welfare_trace)),
        "mean_welfare": float(np.mean(welfare_trace)) if welfare_trace else 0.0,
        "payoff_gini": float(gini(final_payoffs)),
        "total_credit_transferred": float(np.sum(credit_trace)),
        "mean_credit_transferred": float(np.mean(credit_trace)) if credit_trace else 0.0,
        "mean_government_cap": float(
            np.mean([x for x in government_cap_trace if x >= 0.0])
        )
        if any(x >= 0.0 for x in government_cap_trace)
        else 0.0,
        "final_payoffs": final_payoffs,
    }


# Backward-compatible aliases for the earlier Orchard naming.
OrchardConfig = HarvestCommonsConfig
OrchardObservation = HarvestObservation
OrchardMessage = HarvestMessage
OrchardAction = HarvestAction
OrchardStepResult = HarvestStepResult
BaseOrchardAgent = BaseHarvestAgent
SelfInterestedOrchardAgent = SelfInterestedHarvestAgent
ReciprocalOrchardAgent = ReciprocalHarvestAgent
CreditSharingOrchardAgent = CreditSharingHarvestAgent
run_orchard_episode = run_harvest_episode
