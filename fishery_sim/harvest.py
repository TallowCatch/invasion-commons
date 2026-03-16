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
    local_patch_failure_threshold: float = 4.0
    failure_fraction_threshold: float = 0.5
    failure_patience: int = 5
    garden_failure_penalty: float = 25.0
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
    aggressive_request_fraction: float
    max_local_aggression: float
    capped_action_fraction: float
    prevented_harvest_total: float
    neighborhood_overharvest: float
    targeted_agent_fraction: float
    patch_variance: float


@dataclass
class HarvestStrategySpec:
    strategy_id: str
    low_patch_threshold: float
    high_patch_threshold: float
    low_harvest_frac: float
    mid_harvest_frac: float
    high_harvest_frac: float
    restraint_low: float
    restraint_high: float
    credit_request_low: float
    credit_request_high: float
    credit_offer_threshold: float
    credit_offer_amount: float
    neighbor_reciprocity_weight: float
    credit_response_weight: float
    cap_compliance_margin: float
    origin: str = "seed"
    rationale: str = ""
    llm_parse_status: str = ""
    llm_parse_error_type: str = ""

    def to_agent(self) -> "HarvestThresholdAgent":
        return HarvestThresholdAgent(spec=self)


class BaseHarvestAgent:
    name: str = "harvest_base"

    def reset(self) -> None:
        return None

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
        del t
        frac = 0.9 if observation.local_patch > 6.0 else 0.65
        cooperative_neighbors = sum(msg.announced_restraint >= 0.6 for msg in inbox)
        if cooperative_neighbors > 0:
            frac -= 0.05
        if observation.last_credit_received > 0.05:
            frac -= min(0.2, 0.1 + 0.2 * observation.last_credit_received)
        if observation.government_cap_frac is not None:
            frac = min(frac, observation.government_cap_frac + 0.03)
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
        if observation.last_credit_received > 0.05:
            frac -= min(0.12, 0.06 + 0.12 * observation.last_credit_received)
        if observation.government_cap_frac is not None:
            frac = min(frac, observation.government_cap_frac + 0.02)
        return HarvestAction(harvest_frac=min(max(frac, 0.05), 0.7), credit_offer=0.2 if observation.local_patch > 10.0 else 0.0)


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
        if observation.last_credit_received > 0.05:
            base_frac -= min(0.12, 0.05 + 0.15 * observation.last_credit_received)
        if observation.government_cap_frac is not None:
            base_frac = min(base_frac, observation.government_cap_frac + 0.02)
        offer = 0.4 if requested_credit > 0.0 and observation.local_patch > 9.0 else 0.15
        return HarvestAction(harvest_frac=max(base_frac, 0.05), credit_offer=offer)


class HarvestThresholdAgent(BaseHarvestAgent):
    def __init__(self, spec: HarvestStrategySpec):
        self.spec = spec
        self.name = spec.strategy_id

    def communicate(self, observation: HarvestObservation, t: int, neighbor_ids: list[int]) -> HarvestMessage:
        del t, neighbor_ids
        if observation.local_patch < self.spec.low_patch_threshold:
            restraint = self.spec.restraint_low
            requested_credit = self.spec.credit_request_low
        else:
            restraint = self.spec.restraint_high
            requested_credit = self.spec.credit_request_high
        return HarvestMessage(
            announced_restraint=float(np.clip(restraint, 0.0, 1.0)),
            requested_credit=float(np.clip(requested_credit, 0.0, 1.0)),
        )

    def act(self, observation: HarvestObservation, inbox: list[HarvestMessage], t: int) -> HarvestAction:
        del t
        if observation.local_patch < self.spec.low_patch_threshold:
            frac = self.spec.low_harvest_frac
        elif observation.local_patch < self.spec.high_patch_threshold:
            frac = self.spec.mid_harvest_frac
        else:
            frac = self.spec.high_harvest_frac

        if inbox:
            mean_restraint = float(np.mean([msg.announced_restraint for msg in inbox]))
            frac -= self.spec.neighbor_reciprocity_weight * mean_restraint * 0.35
            if any(msg.requested_credit > 0.0 for msg in inbox) and observation.local_patch > self.spec.credit_offer_threshold:
                offer = self.spec.credit_offer_amount
            else:
                offer = 0.0
        else:
            offer = 0.0

        if observation.last_credit_received > 0.0:
            frac -= self.spec.credit_response_weight * min(
                0.25, 0.08 + 0.18 * observation.last_credit_received
            )
        if observation.government_cap_frac is not None:
            frac = min(frac, observation.government_cap_frac + self.spec.cap_compliance_margin)
        return HarvestAction(
            harvest_frac=float(np.clip(frac, 0.0, 1.0)),
            credit_offer=float(np.clip(offer, 0.0, 1.0)),
        )


class GovernmentAgent:
    def __init__(
        self,
        trigger: float = 8.0,
        strict_cap_frac: float = 0.2,
        relaxed_cap_frac: float = 0.6,
        soft_trigger: float | None = None,
        deterioration_threshold: float = 0.35,
        activation_warmup: int = 3,
        aggressive_request_threshold: float = 0.75,
        aggressive_agent_fraction_trigger: float = 0.34,
        local_neighborhood_trigger: float = 0.67,
        enforcement_scope: str = "global",
        expand_target_neighbors: bool = False,
    ):
        self.trigger = float(trigger)
        self.strict_cap_frac = float(strict_cap_frac)
        self.relaxed_cap_frac = float(relaxed_cap_frac)
        self.soft_trigger = float(soft_trigger) if soft_trigger is not None else float(trigger * 1.15)
        self.deterioration_threshold = float(max(0.0, deterioration_threshold))
        self.activation_warmup = int(max(0, activation_warmup))
        self.aggressive_request_threshold = float(np.clip(aggressive_request_threshold, 0.0, 1.0))
        self.aggressive_agent_fraction_trigger = float(np.clip(aggressive_agent_fraction_trigger, 0.0, 1.0))
        self.local_neighborhood_trigger = float(np.clip(local_neighborhood_trigger, 0.0, 1.0))
        if enforcement_scope not in {"global", "local"}:
            raise ValueError("enforcement_scope must be 'global' or 'local'")
        self.enforcement_scope = enforcement_scope
        self.expand_target_neighbors = bool(expand_target_neighbors)
        self.reset()

    def reset(self) -> None:
        self._prev_mean_patch_health: float | None = None
        self._prev_aggressive_fraction: float = 0.0
        self._prev_max_local_aggression: float = 0.0
        self._prev_local_aggression: np.ndarray | None = None
        self._prev_requested_fracs: np.ndarray | None = None

    def _build_cap_array(self, cap_frac: float, n_agents: int) -> np.ndarray:
        if self.enforcement_scope == "global":
            return np.full(n_agents, float(cap_frac), dtype=float)
        target_mask = np.zeros(n_agents, dtype=bool)
        if self._prev_requested_fracs is not None:
            target_mask |= self._prev_requested_fracs > self.aggressive_request_threshold
        if self._prev_local_aggression is not None:
            target_mask |= self._prev_local_aggression >= self.local_neighborhood_trigger
        if self.expand_target_neighbors and np.any(target_mask):
            expanded = target_mask.copy()
            for idx in np.flatnonzero(target_mask):
                for neighbor_idx in _neighbors(int(idx), n_agents):
                    expanded[neighbor_idx] = True
            target_mask = expanded
        if not np.any(target_mask):
            return np.full(n_agents, float(cap_frac), dtype=float)
        cap_arr = np.full(n_agents, np.nan, dtype=float)
        cap_arr[target_mask] = float(cap_frac)
        return cap_arr

    def observe(
        self,
        *,
        requested_fracs: np.ndarray | None = None,
    ) -> None:
        if requested_fracs is None:
            return
        self._prev_requested_fracs = requested_fracs.copy()
        self._prev_aggressive_fraction = float(np.mean(requested_fracs > self.aggressive_request_threshold))
        if requested_fracs.size == 0:
            self._prev_max_local_aggression = 0.0
            self._prev_local_aggression = np.zeros(0, dtype=float)
            return
        local_aggression = []
        n_agents = requested_fracs.size
        for i in range(n_agents):
            neighborhood = [i, *_neighbors(i, n_agents)]
            local_aggression.append(
                float(np.mean(requested_fracs[neighborhood] > self.aggressive_request_threshold))
            )
        self._prev_local_aggression = np.asarray(local_aggression, dtype=float)
        self._prev_max_local_aggression = float(max(local_aggression, default=0.0))

    def act(self, mean_patch_health: float, t: int, n_agents: int) -> np.ndarray | None:
        trend = None if self._prev_mean_patch_health is None else mean_patch_health - self._prev_mean_patch_health
        self._prev_mean_patch_health = float(mean_patch_health)
        if mean_patch_health < self.trigger:
            return self._build_cap_array(self.strict_cap_frac, n_agents)
        if t < self.activation_warmup:
            return None
        if (
            mean_patch_health < self.soft_trigger
            and trend is not None
            and trend < -self.deterioration_threshold
            and (
                self._prev_aggressive_fraction >= self.aggressive_agent_fraction_trigger
                or self._prev_max_local_aggression >= self.local_neighborhood_trigger
            )
        ):
            return self._build_cap_array(self.relaxed_cap_frac, n_agents)
        return None

    def apply_cap(
        self,
        requested_fracs: np.ndarray,
        cap_fracs: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if cap_fracs is None:
            return requested_fracs.copy(), np.zeros_like(requested_fracs, dtype=bool)
        targeted = ~np.isnan(cap_fracs)
        capped = requested_fracs.copy()
        capped[targeted] = np.minimum(capped[targeted], cap_fracs[targeted])
        return capped, targeted

    # Backward-compatible wrappers.
    def set_cap(self, mean_patch_health: float, t: int, n_agents: int) -> np.ndarray | None:
        return self.act(mean_patch_health=mean_patch_health, t=t, n_agents=n_agents)

    def observe_step(self, requested_fracs: np.ndarray) -> None:
        self.observe(requested_fracs=requested_fracs)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


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
    failure_streak = 0
    garden_failure_event = 0
    failure_step = cfg.horizon

    for agent in agents:
        reset = getattr(agent, "reset", None)
        if callable(reset):
            reset()
    if governor is not None:
        governor.reset()

    patch_mean_trace: list[float] = []
    welfare_trace: list[float] = []
    government_cap_trace: list[float] = []
    credit_trace: list[float] = []
    aggressive_request_trace: list[float] = []
    max_local_aggression_trace: list[float] = []
    capped_action_trace: list[float] = []
    prevented_harvest_trace: list[float] = []
    patch_variance_trace: list[float] = []
    neighborhood_overharvest_trace: list[float] = []
    targeted_agent_trace: list[float] = []
    requested_harvest_trace: list[float] = []
    realized_harvest_trace: list[float] = []

    final_payoffs = np.zeros(cfg.n_agents, dtype=float)
    aggressive_threshold = governor.aggressive_request_threshold if governor is not None else 0.75
    t_end = cfg.horizon

    for t in range(cfg.horizon):
        government_cap_fracs = None
        if governor is not None:
            government_cap_fracs = governor.act(float(patch_health.mean()), t, cfg.n_agents)
        if government_cap_fracs is None:
            government_cap_trace.append(-1.0)
            targeted_agent_trace.append(0.0)
        else:
            active_caps = government_cap_fracs[~np.isnan(government_cap_fracs)]
            government_cap_trace.append(float(np.mean(active_caps)) if active_caps.size else -1.0)
            targeted_agent_trace.append(float(np.mean(~np.isnan(government_cap_fracs))))

        observations = []
        for i in range(cfg.n_agents):
            neighbor_mean = float(np.mean([patch_health[j] for j in _neighbors(i, cfg.n_agents)]))
            cap_frac_i = None
            if government_cap_fracs is not None and not np.isnan(government_cap_fracs[i]):
                cap_frac_i = float(government_cap_fracs[i])
            obs = HarvestObservation(
                local_patch=float(patch_health[i]),
                neighbor_mean=neighbor_mean,
                last_credit_received=float(last_credit_received[i]),
                government_cap_frac=cap_frac_i,
            )
            observations.append(agents[i].observe(obs, t))

        messages = [HarvestMessage() for _ in range(cfg.n_agents)]
        if cfg.communication_enabled:
            for i, agent in enumerate(agents):
                messages[i] = agent.communicate(observations[i], t, _neighbors(i, cfg.n_agents))

        raw_actions: list[HarvestAction] = []
        requested_fracs = []
        for i, agent in enumerate(agents):
            inbox = [messages[j] for j in _neighbors(i, cfg.n_agents)] if cfg.communication_enabled else []
            action = agent.act(observations[i], inbox, t)
            requested_frac = _clip01(action.harvest_frac)
            requested_fracs.append(requested_frac)
            raw_actions.append(
                HarvestAction(
                    harvest_frac=requested_frac,
                    credit_offer=float(np.clip(action.credit_offer, 0.0, cfg.credit_cap)),
                )
            )

        requested_fracs_arr = np.asarray(requested_fracs, dtype=float)
        if governor is not None:
            capped_fracs_arr, targeted_mask = governor.apply_cap(requested_fracs_arr, government_cap_fracs)
        else:
            capped_fracs_arr = requested_fracs_arr.copy()
            targeted_mask = np.zeros(cfg.n_agents, dtype=bool)
        actions = [
            HarvestAction(harvest_frac=float(capped_fracs_arr[i]), credit_offer=raw_actions[i].credit_offer)
            for i in range(cfg.n_agents)
        ]

        aggressive_request_trace.append(float(np.mean(requested_fracs_arr > aggressive_threshold)))
        local_aggression = []
        local_overharvest = []
        overharvest_fracs = np.maximum(0.0, requested_fracs_arr - cfg.sustainable_harvest_frac)
        for i in range(cfg.n_agents):
            neighborhood = [i, *_neighbors(i, cfg.n_agents)]
            local_aggression.append(float(np.mean(requested_fracs_arr[neighborhood] > aggressive_threshold)))
            local_overharvest.append(float(np.sum(overharvest_fracs[neighborhood]) * cfg.max_harvest_per_agent))
        max_local_aggression_trace.append(float(max(local_aggression, default=0.0)))
        neighborhood_overharvest_trace.append(float(np.mean(local_overharvest)) if local_overharvest else 0.0)
        capped_action_trace.append(float(np.mean(capped_fracs_arr + 1e-9 < requested_fracs_arr)))
        prevented_harvest_trace.append(
            float(np.sum(np.maximum(0.0, requested_fracs_arr - capped_fracs_arr)) * cfg.max_harvest_per_agent)
        )
        if governor is not None:
            governor.observe(requested_fracs=requested_fracs_arr)

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
        patch_variance_trace.append(float(np.var(patch_health)))
        welfare_trace.append(float(payoffs.sum()))
        credit_trace.append(float(credits_received.sum()))
        requested_harvest_trace.append(float(np.sum(requested_fracs_arr) * cfg.max_harvest_per_agent))
        realized_harvest_trace.append(float(np.sum(harvests)))
        t_end = t + 1

        failed_fraction = float(np.mean(patch_health < cfg.local_patch_failure_threshold))
        if failed_fraction >= cfg.failure_fraction_threshold:
            failure_streak += 1
        else:
            failure_streak = 0
        if failure_streak >= max(1, cfg.failure_patience):
            garden_failure_event = 1
            failure_step = t + 1
            break

    return {
        "seed": cfg.seed,
        "garden_failure_event": int(garden_failure_event),
        "failure_step": int(failure_step),
        "time_to_garden_failure": float(failure_step if garden_failure_event else t_end),
        "t_end": int(t_end),
        "mean_patch_health": float(np.mean(patch_mean_trace)) if patch_mean_trace else 0.0,
        "final_patch_health": float(patch_health.mean()),
        "total_welfare": float(np.sum(welfare_trace)),
        "mean_welfare": float(np.mean(welfare_trace)) if welfare_trace else 0.0,
        "payoff_gini": float(gini(final_payoffs)),
        "total_credit_transferred": float(np.sum(credit_trace)),
        "mean_credit_transferred": float(np.mean(credit_trace)) if credit_trace else 0.0,
        "mean_government_cap": float(np.mean([x for x in government_cap_trace if x >= 0.0]))
        if any(x >= 0.0 for x in government_cap_trace)
        else 0.0,
        "mean_aggressive_request_fraction": float(np.mean(aggressive_request_trace)) if aggressive_request_trace else 0.0,
        "mean_max_local_aggression": float(np.mean(max_local_aggression_trace)) if max_local_aggression_trace else 0.0,
        "mean_capped_action_fraction": float(np.mean(capped_action_trace)) if capped_action_trace else 0.0,
        "mean_prevented_harvest": float(np.mean(prevented_harvest_trace)) if prevented_harvest_trace else 0.0,
        "mean_neighborhood_overharvest": float(np.mean(neighborhood_overharvest_trace)) if neighborhood_overharvest_trace else 0.0,
        "mean_targeted_agent_fraction": float(np.mean(targeted_agent_trace)) if targeted_agent_trace else 0.0,
        "mean_patch_variance": float(np.mean(patch_variance_trace)) if patch_variance_trace else 0.0,
        "mean_requested_harvest": float(np.mean(requested_harvest_trace)) if requested_harvest_trace else 0.0,
        "mean_realized_harvest": float(np.mean(realized_harvest_trace)) if realized_harvest_trace else 0.0,
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
