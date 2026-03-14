from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

import numpy as np
import pandas as pd

from .harvest import GovernmentAgent
from .harvest import HarvestCommonsConfig
from .harvest import HarvestStrategySpec
from .harvest import run_harvest_episode
from .harvest_benchmarks import get_harvest_partner_mix_preset
from .llm_adapter import (
    NullPolicyLLMClient,
    PolicyLLMClient,
    extract_json_object,
)


DEFAULT_GOVERNMENT_PARAMS: dict[str, float | int | bool] = {
    "trigger": 16.0,
    "strict_cap_frac": 0.18,
    "relaxed_cap_frac": 0.35,
    "soft_trigger": 18.0,
    "deterioration_threshold": 0.35,
    "activation_warmup": 3,
    "aggressive_request_threshold": 0.75,
    "aggressive_agent_fraction_trigger": 0.34,
    "local_neighborhood_trigger": 0.67,
}


@dataclass
class HarvestPolicyJSON:
    rationale: str
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "rationale": self.rationale,
            "low_patch_threshold": float(self.low_patch_threshold),
            "high_patch_threshold": float(self.high_patch_threshold),
            "low_harvest_frac": float(self.low_harvest_frac),
            "mid_harvest_frac": float(self.mid_harvest_frac),
            "high_harvest_frac": float(self.high_harvest_frac),
            "restraint_low": float(self.restraint_low),
            "restraint_high": float(self.restraint_high),
            "credit_request_low": float(self.credit_request_low),
            "credit_request_high": float(self.credit_request_high),
            "credit_offer_threshold": float(self.credit_offer_threshold),
            "credit_offer_amount": float(self.credit_offer_amount),
            "neighbor_reciprocity_weight": float(self.neighbor_reciprocity_weight),
            "credit_response_weight": float(self.credit_response_weight),
            "cap_compliance_margin": float(self.cap_compliance_margin),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HarvestPolicyJSON":
        required = [
            "rationale",
            "low_patch_threshold",
            "high_patch_threshold",
            "low_harvest_frac",
            "mid_harvest_frac",
            "high_harvest_frac",
            "restraint_low",
            "restraint_high",
            "credit_request_low",
            "credit_request_high",
            "credit_offer_threshold",
            "credit_offer_amount",
            "neighbor_reciprocity_weight",
            "credit_response_weight",
            "cap_compliance_margin",
        ]
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Missing harvest policy keys: {missing}")
        return cls(
            rationale=str(raw["rationale"]),
            low_patch_threshold=float(raw["low_patch_threshold"]),
            high_patch_threshold=float(raw["high_patch_threshold"]),
            low_harvest_frac=float(raw["low_harvest_frac"]),
            mid_harvest_frac=float(raw["mid_harvest_frac"]),
            high_harvest_frac=float(raw["high_harvest_frac"]),
            restraint_low=float(raw["restraint_low"]),
            restraint_high=float(raw["restraint_high"]),
            credit_request_low=float(raw["credit_request_low"]),
            credit_request_high=float(raw["credit_request_high"]),
            credit_offer_threshold=float(raw["credit_offer_threshold"]),
            credit_offer_amount=float(raw["credit_offer_amount"]),
            neighbor_reciprocity_weight=float(raw["neighbor_reciprocity_weight"]),
            credit_response_weight=float(raw["credit_response_weight"]),
            cap_compliance_margin=float(raw["cap_compliance_margin"]),
        )


def harvest_strategy_spec_to_policy_json(spec: HarvestStrategySpec) -> HarvestPolicyJSON:
    return HarvestPolicyJSON(
        rationale=spec.rationale or spec.origin,
        low_patch_threshold=spec.low_patch_threshold,
        high_patch_threshold=spec.high_patch_threshold,
        low_harvest_frac=spec.low_harvest_frac,
        mid_harvest_frac=spec.mid_harvest_frac,
        high_harvest_frac=spec.high_harvest_frac,
        restraint_low=spec.restraint_low,
        restraint_high=spec.restraint_high,
        credit_request_low=spec.credit_request_low,
        credit_request_high=spec.credit_request_high,
        credit_offer_threshold=spec.credit_offer_threshold,
        credit_offer_amount=spec.credit_offer_amount,
        neighbor_reciprocity_weight=spec.neighbor_reciprocity_weight,
        credit_response_weight=spec.credit_response_weight,
        cap_compliance_margin=spec.cap_compliance_margin,
    )


def clamp_harvest_policy(policy: HarvestPolicyJSON, patch_max: float) -> HarvestPolicyJSON:
    low_patch_threshold = float(np.clip(policy.low_patch_threshold, 0.0, max(0.0, patch_max - 1.0)))
    high_patch_threshold = float(np.clip(policy.high_patch_threshold, low_patch_threshold + 1.0, patch_max))
    low_h = _clip01(policy.low_harvest_frac)
    mid_h = _clip01(max(low_h, policy.mid_harvest_frac))
    high_h = _clip01(max(mid_h, policy.high_harvest_frac))
    return HarvestPolicyJSON(
        rationale=policy.rationale,
        low_patch_threshold=low_patch_threshold,
        high_patch_threshold=high_patch_threshold,
        low_harvest_frac=low_h,
        mid_harvest_frac=mid_h,
        high_harvest_frac=high_h,
        restraint_low=_clip01(policy.restraint_low),
        restraint_high=_clip01(policy.restraint_high),
        credit_request_low=_clip01(policy.credit_request_low),
        credit_request_high=_clip01(policy.credit_request_high),
        credit_offer_threshold=float(np.clip(policy.credit_offer_threshold, 0.0, patch_max)),
        credit_offer_amount=_clip01(policy.credit_offer_amount),
        neighbor_reciprocity_weight=_clip01(policy.neighbor_reciprocity_weight),
        credit_response_weight=_clip01(policy.credit_response_weight),
        cap_compliance_margin=float(np.clip(policy.cap_compliance_margin, 0.0, 0.25)),
    )


def harvest_policy_json_to_strategy_spec(
    policy: HarvestPolicyJSON,
    strategy_id: str,
    origin: str,
) -> HarvestStrategySpec:
    return HarvestStrategySpec(
        strategy_id=strategy_id,
        low_patch_threshold=policy.low_patch_threshold,
        high_patch_threshold=policy.high_patch_threshold,
        low_harvest_frac=policy.low_harvest_frac,
        mid_harvest_frac=policy.mid_harvest_frac,
        high_harvest_frac=policy.high_harvest_frac,
        restraint_low=policy.restraint_low,
        restraint_high=policy.restraint_high,
        credit_request_low=policy.credit_request_low,
        credit_request_high=policy.credit_request_high,
        credit_offer_threshold=policy.credit_offer_threshold,
        credit_offer_amount=policy.credit_offer_amount,
        neighbor_reciprocity_weight=policy.neighbor_reciprocity_weight,
        credit_response_weight=policy.credit_response_weight,
        cap_compliance_margin=policy.cap_compliance_margin,
        origin=origin,
        rationale=policy.rationale,
    )


def build_harvest_policy_prompt(
    parent_policy: HarvestPolicyJSON,
    parent_fitness: float,
    patch_max: float,
    adversarial_pressure: float,
) -> str:
    target_style = "aggressive harvest invader" if adversarial_pressure >= 0.5 else "balanced harvest contender"
    schema = {
        "rationale": "short explanation (<=30 words)",
        "low_patch_threshold": "float in [0, patch_max]",
        "high_patch_threshold": "float in [low_patch_threshold, patch_max]",
        "low_harvest_frac": "float in [0,1]",
        "mid_harvest_frac": "float in [0,1]",
        "high_harvest_frac": "float in [0,1]",
        "restraint_low": "float in [0,1]",
        "restraint_high": "float in [0,1]",
        "credit_request_low": "float in [0,1]",
        "credit_request_high": "float in [0,1]",
        "credit_offer_threshold": "float in [0, patch_max]",
        "credit_offer_amount": "float in [0,1]",
        "neighbor_reciprocity_weight": "float in [0,1]",
        "credit_response_weight": "float in [0,1]",
        "cap_compliance_margin": "float in [0,0.25]",
    }
    return (
        "You are generating a Harvest Commons strategy as strict JSON.\n"
        f"Parent fitness: {parent_fitness:.4f}\n"
        f"Patch max: {patch_max:.4f}\n"
        f"Target style: {target_style}\n"
        "The strategy controls local harvest, restraint signals, credit requests/offers, reciprocity, and cap compliance.\n"
        "Parent policy:\n"
        f"{json.dumps(parent_policy.to_dict(), indent=2)}\n\n"
        "Output exactly one JSON object using this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "No markdown. No commentary. No extra keys."
    )


def _safe_name(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    out = out.strip("_")
    return out or "regime"


def _build_seed_schedule(seed_start: int, n_seeds: int) -> list[int]:
    return [seed_start + i for i in range(n_seeds)]


def _resolve_test_regimes(test_regimes: list[dict[str, dict[str, float]]] | None) -> list[dict[str, dict[str, float]]]:
    if not test_regimes:
        return [{"name": "default", "overrides": {}}]
    resolved: list[dict[str, dict[str, float]]] = []
    for i, item in enumerate(test_regimes):
        name = str(item.get("name", f"regime_{i}"))
        overrides = item.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"test_regimes[{i}].overrides must be a mapping.")
        resolved.append({"name": name, "overrides": dict(overrides)})
    return resolved


def _apply_cfg_overrides(base_cfg: HarvestCommonsConfig, overrides: dict[str, float] | None) -> HarvestCommonsConfig:
    cfg = copy.deepcopy(base_cfg)
    if overrides is None:
        return cfg
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown harvest config field in overrides: {key}")
        setattr(cfg, key, value)
    return cfg


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _sample_strategy(
    strategy_id: str,
    patch_max: float,
    rng: np.random.Generator,
    *,
    threshold_ranges: tuple[tuple[float, float], tuple[float, float]],
    harvest_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    restraint_ranges: tuple[tuple[float, float], tuple[float, float]],
    credit_request_ranges: tuple[tuple[float, float], tuple[float, float]],
    credit_offer_threshold_range: tuple[float, float],
    credit_offer_amount_range: tuple[float, float],
    reciprocity_range: tuple[float, float],
    credit_response_range: tuple[float, float],
    cap_compliance_margin_range: tuple[float, float],
    origin: str,
    rationale: str,
) -> HarvestStrategySpec:
    low_patch_threshold = float(rng.uniform(*threshold_ranges[0]))
    high_patch_threshold = float(rng.uniform(*threshold_ranges[1]))
    low_patch_threshold = float(np.clip(low_patch_threshold, 0.0, patch_max - 1.5))
    high_patch_threshold = float(np.clip(high_patch_threshold, low_patch_threshold + 1.0, patch_max))

    low_harvest_frac = float(rng.uniform(*harvest_ranges[0]))
    mid_harvest_frac = float(rng.uniform(*harvest_ranges[1]))
    high_harvest_frac = float(rng.uniform(*harvest_ranges[2]))
    mid_harvest_frac = max(low_harvest_frac, mid_harvest_frac)
    high_harvest_frac = max(mid_harvest_frac, high_harvest_frac)

    restraint_low = float(rng.uniform(*restraint_ranges[0]))
    restraint_high = float(rng.uniform(*restraint_ranges[1]))
    credit_request_low = float(rng.uniform(*credit_request_ranges[0]))
    credit_request_high = float(rng.uniform(*credit_request_ranges[1]))
    credit_offer_threshold = float(rng.uniform(*credit_offer_threshold_range))
    credit_offer_amount = float(rng.uniform(*credit_offer_amount_range))
    neighbor_reciprocity_weight = float(rng.uniform(*reciprocity_range))
    credit_response_weight = float(rng.uniform(*credit_response_range))
    cap_compliance_margin = float(rng.uniform(*cap_compliance_margin_range))

    return HarvestStrategySpec(
        strategy_id=strategy_id,
        low_patch_threshold=low_patch_threshold,
        high_patch_threshold=high_patch_threshold,
        low_harvest_frac=_clip01(low_harvest_frac),
        mid_harvest_frac=_clip01(mid_harvest_frac),
        high_harvest_frac=_clip01(high_harvest_frac),
        restraint_low=_clip01(restraint_low),
        restraint_high=_clip01(restraint_high),
        credit_request_low=_clip01(credit_request_low),
        credit_request_high=_clip01(credit_request_high),
        credit_offer_threshold=float(np.clip(credit_offer_threshold, 0.0, patch_max)),
        credit_offer_amount=_clip01(credit_offer_amount),
        neighbor_reciprocity_weight=_clip01(neighbor_reciprocity_weight),
        credit_response_weight=_clip01(credit_response_weight),
        cap_compliance_margin=float(np.clip(cap_compliance_margin, 0.0, 0.25)),
        origin=origin,
        rationale=rationale,
    )


def cooperative_harvest_strategy(strategy_id: str, patch_max: float, rng: np.random.Generator) -> HarvestStrategySpec:
    return _sample_strategy(
        strategy_id,
        patch_max,
        rng,
        threshold_ranges=((6.5, 10.5), (11.0, 17.5)),
        harvest_ranges=((0.04, 0.15), (0.12, 0.34), (0.26, 0.58)),
        restraint_ranges=((0.72, 0.95), (0.45, 0.78)),
        credit_request_ranges=((0.25, 0.60), (0.0, 0.25)),
        credit_offer_threshold_range=(9.0, 14.0),
        credit_offer_amount_range=(0.25, 0.85),
        reciprocity_range=(0.45, 0.90),
        credit_response_range=(0.40, 0.85),
        cap_compliance_margin_range=(0.0, 0.03),
        origin="cooperative_seed",
        rationale="cooperative-heavy harvest seed",
    )


def balanced_harvest_strategy(strategy_id: str, patch_max: float, rng: np.random.Generator) -> HarvestStrategySpec:
    return _sample_strategy(
        strategy_id,
        patch_max,
        rng,
        threshold_ranges=((4.5, 9.0), (9.5, 15.5)),
        harvest_ranges=((0.08, 0.28), (0.24, 0.52), (0.45, 0.82)),
        restraint_ranges=((0.30, 0.75), (0.15, 0.55)),
        credit_request_ranges=((0.10, 0.45), (0.0, 0.20)),
        credit_offer_threshold_range=(7.5, 12.5),
        credit_offer_amount_range=(0.10, 0.55),
        reciprocity_range=(0.20, 0.65),
        credit_response_range=(0.15, 0.60),
        cap_compliance_margin_range=(0.0, 0.08),
        origin="balanced_seed",
        rationale="balanced harvest seed",
    )


def adversarial_harvest_strategy(strategy_id: str, patch_max: float, rng: np.random.Generator) -> HarvestStrategySpec:
    return _sample_strategy(
        strategy_id,
        patch_max,
        rng,
        threshold_ranges=((2.5, 6.5), (6.5, 12.0)),
        harvest_ranges=((0.18, 0.50), (0.48, 0.82), (0.75, 1.0)),
        restraint_ranges=((0.0, 0.25), (0.0, 0.20)),
        credit_request_ranges=((0.0, 0.15), (0.0, 0.05)),
        credit_offer_threshold_range=(5.5, 9.5),
        credit_offer_amount_range=(0.0, 0.15),
        reciprocity_range=(0.0, 0.25),
        credit_response_range=(0.0, 0.20),
        cap_compliance_margin_range=(0.03, 0.18),
        origin="adversarial_seed",
        rationale="aggressive harvest seed",
    )


def random_harvest_strategy(strategy_id: str, patch_max: float, rng: np.random.Generator) -> HarvestStrategySpec:
    return _sample_strategy(
        strategy_id,
        patch_max,
        rng,
        threshold_ranges=((2.5, 10.5), (7.0, 18.5)),
        harvest_ranges=((0.03, 0.45), (0.08, 0.75), (0.20, 1.0)),
        restraint_ranges=((0.0, 0.95), (0.0, 0.80)),
        credit_request_ranges=((0.0, 0.65), (0.0, 0.30)),
        credit_offer_threshold_range=(5.0, 15.5),
        credit_offer_amount_range=(0.0, 1.0),
        reciprocity_range=(0.0, 1.0),
        credit_response_range=(0.0, 1.0),
        cap_compliance_margin_range=(0.0, 0.20),
        origin="random_init",
        rationale="random harvest seed",
    )


def build_initial_harvest_population(
    population_size: int,
    patch_max: float,
    rng: np.random.Generator,
    partner_mix_preset: str,
) -> list[HarvestStrategySpec]:
    weights = get_harvest_partner_mix_preset(partner_mix_preset)
    counts = [
        int(round(population_size * float(weights["cooperative_weight"]))),
        int(round(population_size * float(weights["balanced_weight"]))),
        int(round(population_size * float(weights["adversarial_weight"]))),
    ]
    while sum(counts) < population_size:
        counts[int(np.argmin(counts))] += 1
    while sum(counts) > population_size:
        counts[int(np.argmax(counts))] -= 1

    population: list[HarvestStrategySpec] = []
    for _ in range(counts[0]):
        population.append(cooperative_harvest_strategy(f"g0_s{len(population)}", patch_max=patch_max, rng=rng))
    for _ in range(counts[1]):
        population.append(balanced_harvest_strategy(f"g0_s{len(population)}", patch_max=patch_max, rng=rng))
    for _ in range(counts[2]):
        population.append(adversarial_harvest_strategy(f"g0_s{len(population)}", patch_max=patch_max, rng=rng))
    rng.shuffle(population)
    for i, spec in enumerate(population):
        spec.strategy_id = f"g0_s{i}"
    return population


def mutate_harvest_strategy(
    parent: HarvestStrategySpec,
    strategy_id: str,
    patch_max: float,
    rng: np.random.Generator,
    adversarial_pressure: float,
) -> HarvestStrategySpec:
    low_thr = float(np.clip(parent.low_patch_threshold + rng.normal(0.0, 1.6), 0.0, patch_max - 1.5))
    high_thr = float(np.clip(parent.high_patch_threshold + rng.normal(0.0, 2.0), low_thr + 1.0, patch_max))

    low_h = _clip01(parent.low_harvest_frac + rng.normal(0.0, 0.06) + adversarial_pressure * 0.03)
    mid_h = _clip01(parent.mid_harvest_frac + rng.normal(0.0, 0.08) + adversarial_pressure * 0.06)
    high_h = _clip01(parent.high_harvest_frac + rng.normal(0.0, 0.10) + adversarial_pressure * 0.10)
    mid_h = max(low_h, mid_h)
    high_h = max(mid_h, high_h)

    restraint_low = _clip01(parent.restraint_low + rng.normal(0.0, 0.10) - adversarial_pressure * 0.08)
    restraint_high = _clip01(parent.restraint_high + rng.normal(0.0, 0.08) - adversarial_pressure * 0.05)
    credit_request_low = _clip01(parent.credit_request_low + rng.normal(0.0, 0.10) - adversarial_pressure * 0.05)
    credit_request_high = _clip01(parent.credit_request_high + rng.normal(0.0, 0.06) - adversarial_pressure * 0.03)
    credit_offer_threshold = float(np.clip(parent.credit_offer_threshold + rng.normal(0.0, 1.2), 0.0, patch_max))
    credit_offer_amount = _clip01(parent.credit_offer_amount + rng.normal(0.0, 0.10) - adversarial_pressure * 0.08)
    neighbor_reciprocity_weight = _clip01(parent.neighbor_reciprocity_weight + rng.normal(0.0, 0.10) - adversarial_pressure * 0.08)
    credit_response_weight = _clip01(parent.credit_response_weight + rng.normal(0.0, 0.10) - adversarial_pressure * 0.08)
    cap_margin = float(np.clip(parent.cap_compliance_margin + rng.normal(0.0, 0.03) + adversarial_pressure * 0.03, 0.0, 0.25))

    if rng.random() < adversarial_pressure:
        high_h = _clip01(high_h + rng.uniform(0.02, 0.12))
        mid_h = max(mid_h, _clip01(mid_h + rng.uniform(0.01, 0.08)))
        restraint_low = _clip01(restraint_low - rng.uniform(0.02, 0.10))
        credit_offer_amount = _clip01(credit_offer_amount - rng.uniform(0.02, 0.12))
        cap_margin = float(np.clip(cap_margin + rng.uniform(0.01, 0.05), 0.0, 0.25))

    return HarvestStrategySpec(
        strategy_id=strategy_id,
        low_patch_threshold=low_thr,
        high_patch_threshold=high_thr,
        low_harvest_frac=low_h,
        mid_harvest_frac=mid_h,
        high_harvest_frac=high_h,
        restraint_low=restraint_low,
        restraint_high=restraint_high,
        credit_request_low=credit_request_low,
        credit_request_high=credit_request_high,
        credit_offer_threshold=credit_offer_threshold,
        credit_offer_amount=credit_offer_amount,
        neighbor_reciprocity_weight=neighbor_reciprocity_weight,
        credit_response_weight=credit_response_weight,
        cap_compliance_margin=cap_margin,
        origin="mutation",
        rationale=f"mutated from {parent.strategy_id}",
    )


class HarvestStrategyInjector(Protocol):
    def inject(
        self,
        parent: HarvestStrategySpec,
        parent_fitness: float,
        strategy_id: str,
        patch_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> HarvestStrategySpec: ...


class MutationHarvestStrategyInjector:
    def inject(
        self,
        parent: HarvestStrategySpec,
        parent_fitness: float,
        strategy_id: str,
        patch_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> HarvestStrategySpec:
        del parent_fitness
        return mutate_harvest_strategy(
            parent=parent,
            strategy_id=strategy_id,
            patch_max=patch_max,
            rng=rng,
            adversarial_pressure=adversarial_pressure,
        )


class RandomHarvestStrategyInjector:
    def inject(
        self,
        parent: HarvestStrategySpec,
        parent_fitness: float,
        strategy_id: str,
        patch_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> HarvestStrategySpec:
        del parent, parent_fitness, adversarial_pressure
        child = random_harvest_strategy(strategy_id=strategy_id, patch_max=patch_max, rng=rng)
        child.origin = "random_injector"
        child.rationale = "uninformed random harvest child"
        return child


class AdversarialHeuristicHarvestStrategyInjector:
    def inject(
        self,
        parent: HarvestStrategySpec,
        parent_fitness: float,
        strategy_id: str,
        patch_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> HarvestStrategySpec:
        del parent, parent_fitness, adversarial_pressure
        child = adversarial_harvest_strategy(strategy_id=strategy_id, patch_max=patch_max, rng=rng)
        child.origin = "adversarial_heuristic"
        child.rationale = "heuristic harvest invader biased to high extraction"
        return child


class SearchMutationHarvestStrategyInjector:
    def __init__(self, n_candidates: int = 6, eval_horizon: int = 30):
        self.n_candidates = max(2, int(n_candidates))
        self.eval_horizon = max(5, int(eval_horizon))
        self._context_cfg: HarvestCommonsConfig | None = None
        self._context_population: list[HarvestStrategySpec] = []
        self._context_seed: int = 0
        self._context_condition: str = "hybrid"
        self._context_government_params: dict[str, float | int | bool] | None = None

    def prepare_generation(
        self,
        base_cfg: HarvestCommonsConfig,
        condition: str,
        parent_pool: list[HarvestStrategySpec],
        generation: int,
        government_params: dict[str, float | int | bool] | None,
        rng: np.random.Generator,
    ) -> None:
        del rng
        self._context_cfg = copy.deepcopy(base_cfg)
        self._context_cfg.horizon = min(self._context_cfg.horizon, self.eval_horizon)
        self._context_population = [copy.deepcopy(spec) for spec in parent_pool]
        self._context_seed = int(base_cfg.seed + 1_000_000 + generation)
        self._context_condition = condition
        self._context_government_params = dict(government_params or {})

    def inject(
        self,
        parent: HarvestStrategySpec,
        parent_fitness: float,
        strategy_id: str,
        patch_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> HarvestStrategySpec:
        del parent_fitness
        candidates = [
            mutate_harvest_strategy(
                parent=parent,
                strategy_id=f"{strategy_id}_cand{i}",
                patch_max=patch_max,
                rng=rng,
                adversarial_pressure=adversarial_pressure,
            )
            for i in range(self.n_candidates)
        ]
        if not self._context_population or self._context_cfg is None:
            best = max(candidates, key=lambda spec: spec.high_harvest_frac + 0.5 * spec.mid_harvest_frac - 0.25 * spec.restraint_low)
        else:
            best = max(candidates, key=self._attack_score)
        best.strategy_id = strategy_id
        best.origin = "search_mutation"
        best.rationale = f"search-over-mutations from {parent.strategy_id}"
        return best

    def _attack_score(self, candidate: HarvestStrategySpec) -> float:
        assert self._context_cfg is not None
        cfg = copy.deepcopy(self._context_cfg)
        partners = [copy.deepcopy(spec) for spec in self._context_population]
        if partners:
            partners = partners[: max(1, cfg.n_agents - 1)]
        population = [candidate] + partners
        cfg.seed = self._context_seed
        cfg.n_agents = len(population)
        cfg, governor = _make_condition_setup(
            cfg,
            condition=self._context_condition,
            government_params=self._context_government_params,
        )
        agents = [spec.to_agent() for spec in population]
        out = run_harvest_episode(cfg, agents, governor=governor)
        collapse_cost = cfg.garden_failure_penalty if out["garden_failure_event"] else 0.0
        return float(out["final_payoffs"][0] - collapse_cost)


class LLMJSONHarvestStrategyInjector:
    """
    Harvest adapter path:
    prompt -> JSON policy -> HarvestStrategySpec
    """

    def __init__(
        self,
        llm_client: PolicyLLMClient | None = None,
        fallback_injector: HarvestStrategyInjector | None = None,
        jitter_std: float = 0.04,
    ):
        self.llm_client = llm_client if llm_client is not None else NullPolicyLLMClient()
        self.fallback_injector = fallback_injector if fallback_injector is not None else MutationHarvestStrategyInjector()
        self.jitter_std = float(max(0.0, jitter_std))

    def inject(
        self,
        parent: HarvestStrategySpec,
        parent_fitness: float,
        strategy_id: str,
        patch_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> HarvestStrategySpec:
        parent_policy = harvest_strategy_spec_to_policy_json(parent)
        prompt = build_harvest_policy_prompt(
            parent_policy=parent_policy,
            parent_fitness=parent_fitness,
            patch_max=patch_max,
            adversarial_pressure=adversarial_pressure,
        )
        try:
            raw_response = self.llm_client.complete(prompt)
            parsed = extract_json_object(raw_response)
            policy = HarvestPolicyJSON.from_dict(parsed)
            policy = clamp_harvest_policy(policy, patch_max=patch_max)
            policy = self._perturb_policy(policy, rng=rng, adversarial_pressure=adversarial_pressure, patch_max=patch_max)
            return harvest_policy_json_to_strategy_spec(policy=policy, strategy_id=strategy_id, origin="llm_json")
        except Exception:
            child = self.fallback_injector.inject(
                parent=parent,
                parent_fitness=parent_fitness,
                strategy_id=strategy_id,
                patch_max=patch_max,
                rng=rng,
                adversarial_pressure=adversarial_pressure,
            )
            child.origin = "llm_fallback_mutation"
            if not child.rationale:
                child.rationale = f"fallback from {parent.strategy_id}"
            return child

    def _perturb_policy(
        self,
        policy: HarvestPolicyJSON,
        rng: np.random.Generator,
        adversarial_pressure: float,
        patch_max: float,
    ) -> HarvestPolicyJSON:
        low_thr = float(np.clip(policy.low_patch_threshold + rng.normal(0.0, 1.2), 0.0, max(0.0, patch_max - 1.0)))
        high_thr = float(np.clip(policy.high_patch_threshold + rng.normal(0.0, 1.6), low_thr + 1.0, patch_max))

        low_h = _clip01(policy.low_harvest_frac + rng.normal(0.0, self.jitter_std * 0.7))
        mid_h = _clip01(policy.mid_harvest_frac + rng.normal(0.0, self.jitter_std) + adversarial_pressure * 0.03)
        high_h = _clip01(policy.high_harvest_frac + rng.normal(0.0, self.jitter_std * 1.2) + adversarial_pressure * 0.07)
        mid_h = max(low_h, mid_h)
        high_h = max(mid_h, high_h)

        restraint_low = _clip01(policy.restraint_low + rng.normal(0.0, self.jitter_std) - adversarial_pressure * 0.05)
        restraint_high = _clip01(policy.restraint_high + rng.normal(0.0, self.jitter_std * 0.8) - adversarial_pressure * 0.03)
        credit_request_low = _clip01(policy.credit_request_low + rng.normal(0.0, self.jitter_std * 0.8) - adversarial_pressure * 0.02)
        credit_request_high = _clip01(policy.credit_request_high + rng.normal(0.0, self.jitter_std * 0.6) - adversarial_pressure * 0.01)
        credit_offer_threshold = float(np.clip(policy.credit_offer_threshold + rng.normal(0.0, 0.9), 0.0, patch_max))
        credit_offer_amount = _clip01(policy.credit_offer_amount + rng.normal(0.0, self.jitter_std) - adversarial_pressure * 0.05)
        reciprocity = _clip01(policy.neighbor_reciprocity_weight + rng.normal(0.0, self.jitter_std) - adversarial_pressure * 0.05)
        credit_response = _clip01(policy.credit_response_weight + rng.normal(0.0, self.jitter_std) - adversarial_pressure * 0.04)
        cap_margin = float(
            np.clip(policy.cap_compliance_margin + rng.normal(0.0, self.jitter_std * 0.5) + adversarial_pressure * 0.02, 0.0, 0.25)
        )

        return HarvestPolicyJSON(
            rationale=policy.rationale,
            low_patch_threshold=low_thr,
            high_patch_threshold=high_thr,
            low_harvest_frac=low_h,
            mid_harvest_frac=mid_h,
            high_harvest_frac=high_h,
            restraint_low=restraint_low,
            restraint_high=restraint_high,
            credit_request_low=credit_request_low,
            credit_request_high=credit_request_high,
            credit_offer_threshold=credit_offer_threshold,
            credit_offer_amount=credit_offer_amount,
            neighbor_reciprocity_weight=reciprocity,
            credit_response_weight=credit_response,
            cap_compliance_margin=cap_margin,
        )


def make_harvest_strategy_injector(
    injector_mode: str,
    llm_client: PolicyLLMClient | None = None,
) -> HarvestStrategyInjector:
    mode = injector_mode.strip().lower()
    if mode == "random":
        return RandomHarvestStrategyInjector()
    if mode == "mutation":
        return MutationHarvestStrategyInjector()
    if mode == "adversarial_heuristic":
        return AdversarialHeuristicHarvestStrategyInjector()
    if mode == "search_mutation":
        return SearchMutationHarvestStrategyInjector()
    if mode == "llm_json":
        return LLMJSONHarvestStrategyInjector(llm_client=llm_client)
    raise ValueError(f"Unknown Harvest injector_mode: {injector_mode}")


def _population_diversity(population: Iterable[HarvestStrategySpec]) -> float:
    pop = list(population)
    if len(pop) < 2:
        return 0.0
    matrix = np.array(
        [
            [
                p.low_patch_threshold,
                p.high_patch_threshold,
                p.low_harvest_frac,
                p.mid_harvest_frac,
                p.high_harvest_frac,
                p.restraint_low,
                p.restraint_high,
                p.credit_request_low,
                p.credit_request_high,
                p.credit_offer_threshold,
                p.credit_offer_amount,
                p.neighbor_reciprocity_weight,
                p.credit_response_weight,
                p.cap_compliance_margin,
            ]
            for p in pop
        ],
        dtype=float,
    )
    return float(matrix.std(axis=0).mean())


def _origin_fraction(strategy_df: pd.DataFrame, origin: str) -> float:
    if strategy_df.empty:
        return 0.0
    return float((strategy_df["origin"] == origin).mean())


def _make_condition_setup(
    base_cfg: HarvestCommonsConfig,
    condition: str,
    government_params: dict[str, float | int | bool] | None,
) -> tuple[HarvestCommonsConfig, GovernmentAgent | None]:
    cfg = copy.deepcopy(base_cfg)
    params = dict(DEFAULT_GOVERNMENT_PARAMS)
    if government_params:
        params.update(government_params)
    if condition == "none":
        cfg.communication_enabled = False
        cfg.side_payments_enabled = False
        return cfg, None
    if condition == "top_down_only":
        cfg.communication_enabled = False
        cfg.side_payments_enabled = False
        return cfg, GovernmentAgent(**params, enforcement_scope="global", expand_target_neighbors=False)
    if condition == "bottom_up_only":
        cfg.communication_enabled = True
        cfg.side_payments_enabled = True
        return cfg, None
    if condition == "hybrid":
        cfg.communication_enabled = True
        cfg.side_payments_enabled = True
        return cfg, GovernmentAgent(**params, enforcement_scope="local", expand_target_neighbors=True)
    raise ValueError(f"Unknown Harvest governance condition: {condition}")


def _summarize_episode_df(episode_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
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


def evaluate_harvest_population(
    base_cfg: HarvestCommonsConfig,
    condition: str,
    population: list[HarvestStrategySpec],
    seeds: list[int],
    government_params: dict[str, float | int | bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not population:
        raise ValueError("Population cannot be empty.")

    payoff_sum = np.zeros(len(population))
    fitness_sum = np.zeros(len(population))
    rows: list[dict] = []

    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed
        cfg.n_agents = len(population)
        cfg, governor = _make_condition_setup(cfg, condition=condition, government_params=government_params)
        agents = [spec.to_agent() for spec in population]
        out = run_harvest_episode(cfg, agents, governor=governor)
        failure = bool(out["garden_failure_event"])
        collapse_cost = cfg.garden_failure_penalty if failure else 0.0

        payoff_sum += out["final_payoffs"]
        fitness_sum += out["final_payoffs"] - collapse_cost

        rows.append(
            {
                "seed": seed,
                "garden_failure_event": failure,
                "t_end": out["t_end"],
                "time_to_garden_failure": out["time_to_garden_failure"],
                "final_patch_health": out["final_patch_health"],
                "mean_patch_health": out["mean_patch_health"],
                "mean_welfare": out["mean_welfare"],
                "payoff_gini": out["payoff_gini"],
                "mean_credit_transferred": out["mean_credit_transferred"],
                "mean_government_cap": out["mean_government_cap"],
                "mean_aggressive_request_fraction": out["mean_aggressive_request_fraction"],
                "mean_max_local_aggression": out["mean_max_local_aggression"],
                "mean_neighborhood_overharvest": out["mean_neighborhood_overharvest"],
                "mean_capped_action_fraction": out["mean_capped_action_fraction"],
                "mean_targeted_agent_fraction": out["mean_targeted_agent_fraction"],
                "mean_prevented_harvest": out["mean_prevented_harvest"],
                "mean_patch_variance": out["mean_patch_variance"],
                "mean_requested_harvest": out["mean_requested_harvest"],
                "mean_realized_harvest": out["mean_realized_harvest"],
            }
        )

    n_seeds = max(1, len(seeds))
    score_rows: list[dict] = []
    for i, spec in enumerate(population):
        score_rows.append(
            {
                "strategy_id": spec.strategy_id,
                "origin": spec.origin,
                "rationale": spec.rationale,
                "mean_payoff": float(payoff_sum[i] / n_seeds),
                "fitness": float(fitness_sum[i] / n_seeds),
                "low_patch_threshold": spec.low_patch_threshold,
                "high_patch_threshold": spec.high_patch_threshold,
                "low_harvest_frac": spec.low_harvest_frac,
                "mid_harvest_frac": spec.mid_harvest_frac,
                "high_harvest_frac": spec.high_harvest_frac,
                "restraint_low": spec.restraint_low,
                "restraint_high": spec.restraint_high,
                "credit_request_low": spec.credit_request_low,
                "credit_request_high": spec.credit_request_high,
                "credit_offer_threshold": spec.credit_offer_threshold,
                "credit_offer_amount": spec.credit_offer_amount,
                "neighbor_reciprocity_weight": spec.neighbor_reciprocity_weight,
                "credit_response_weight": spec.credit_response_weight,
                "cap_compliance_margin": spec.cap_compliance_margin,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def run_harvest_invasion(
    base_cfg: HarvestCommonsConfig,
    condition: str,
    generations: int = 15,
    population_size: int = 6,
    seeds_per_generation: int = 32,
    test_seeds_per_generation: int | None = None,
    replacement_fraction: float = 0.2,
    adversarial_pressure: float = 0.3,
    rng_seed: int = 0,
    partner_mix_preset: str = "balanced",
    injector: HarvestStrategyInjector | None = None,
    test_regimes: list[dict[str, dict[str, float]]] | None = None,
    government_params: dict[str, float | int | bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    if generations < 1:
        raise ValueError("generations must be >= 1")
    if not 0.0 < replacement_fraction < 1.0:
        raise ValueError("replacement_fraction must be in (0, 1)")

    rng = np.random.default_rng(rng_seed)
    strategy_injector = injector if injector is not None else MutationHarvestStrategyInjector()
    resolved_test_regimes = _resolve_test_regimes(test_regimes)
    test_seed_count = int(test_seeds_per_generation or seeds_per_generation)

    population = build_initial_harvest_population(
        population_size=population_size,
        patch_max=base_cfg.patch_max,
        rng=rng,
        partner_mix_preset=partner_mix_preset,
    )

    generation_rows: list[dict] = []
    strategy_rows: list[dict] = []

    keep_count = max(2, int(round(population_size * (1.0 - replacement_fraction))))
    replace_count = population_size - keep_count

    for generation in range(generations):
        train_seed_start = base_cfg.seed + generation * seeds_per_generation
        test_seed_start = base_cfg.seed + 10_000_000 + generation * test_seed_count
        train_seeds = _build_seed_schedule(train_seed_start, seeds_per_generation)
        test_seeds = _build_seed_schedule(test_seed_start, test_seed_count)

        train_episode_df, train_score_df = evaluate_harvest_population(
            base_cfg=base_cfg,
            condition=condition,
            population=population,
            seeds=train_seeds,
            government_params=government_params,
        )
        train_score_df = train_score_df.sort_values("fitness", ascending=False).reset_index(drop=True)
        train_score_df["rank"] = np.arange(1, len(train_score_df) + 1)
        train_score_df["generation"] = generation
        for _, row in train_score_df.iterrows():
            strategy_rows.append(row.to_dict())

        all_test_episode_dfs: list[pd.DataFrame] = []
        per_regime_summaries: dict[str, float] = {}
        for regime in resolved_test_regimes:
            regime_name = str(regime["name"])
            regime_cfg = _apply_cfg_overrides(base_cfg, regime["overrides"])
            regime_episode_df, _ = evaluate_harvest_population(
                base_cfg=regime_cfg,
                condition=condition,
                population=population,
                seeds=test_seeds,
                government_params=government_params,
            )
            safe = _safe_name(regime_name)
            per_regime_summaries.update(_summarize_episode_df(regime_episode_df, prefix=f"test_{safe}"))
            all_test_episode_dfs.append(regime_episode_df.assign(regime=regime_name))
        test_episode_df = pd.concat(all_test_episode_dfs, ignore_index=True)

        row = {
            "generation": generation,
            "test_regime_count": len(resolved_test_regimes),
            "population_diversity": _population_diversity(population),
            "mean_high_harvest_frac": float(train_score_df["high_harvest_frac"].mean()),
            "mean_mid_harvest_frac": float(train_score_df["mid_harvest_frac"].mean()),
            "mean_restraint_low": float(train_score_df["restraint_low"].mean()),
            "mean_credit_offer_amount": float(train_score_df["credit_offer_amount"].mean()),
            "best_fitness": float(train_score_df["fitness"].iloc[0]),
            "worst_fitness": float(train_score_df["fitness"].iloc[-1]),
            "injector_mode": type(strategy_injector).__name__,
            "partner_mix_preset": partner_mix_preset,
            "condition": condition,
            "adversarial_pressure": float(adversarial_pressure),
            "llm_json_fraction": _origin_fraction(train_score_df, "llm_json"),
            "llm_fallback_fraction": _origin_fraction(train_score_df, "llm_fallback_mutation"),
        }
        row.update(_summarize_episode_df(train_episode_df, prefix="train"))
        row.update(_summarize_episode_df(test_episode_df, prefix="test"))
        row.update(per_regime_summaries)
        generation_rows.append(row)

        if progress_callback is not None:
            progress_callback(generation + 1, generations)

        if generation == generations - 1:
            break

        survivors = train_score_df.head(keep_count)["strategy_id"].tolist()
        fitness_by_id = {
            row["strategy_id"]: float(row["fitness"])
            for _, row in train_score_df[["strategy_id", "fitness"]].iterrows()
        }
        survivor_map = {spec.strategy_id: spec for spec in population}
        parent_pool = [survivor_map[sid] for sid in survivors]
        parent_pool_top = parent_pool[: max(2, len(parent_pool) // 2)]
        prepare_generation = getattr(strategy_injector, "prepare_generation", None)
        if callable(prepare_generation):
            prepare_generation(
                base_cfg=base_cfg,
                condition=condition,
                parent_pool=parent_pool,
                generation=generation,
                government_params=government_params,
                rng=rng,
            )

        injected: list[HarvestStrategySpec] = []
        for i in range(replace_count):
            parent = parent_pool_top[i % len(parent_pool_top)]
            child = strategy_injector.inject(
                parent=parent,
                parent_fitness=fitness_by_id.get(parent.strategy_id, 0.0),
                strategy_id=f"g{generation + 1}_s{i}",
                patch_max=base_cfg.patch_max,
                rng=rng,
                adversarial_pressure=adversarial_pressure,
            )
            child.strategy_id = f"g{generation + 1}_s{i}"
            injected.append(child)

        population = parent_pool + injected

    return pd.DataFrame(generation_rows), pd.DataFrame(strategy_rows)
