from __future__ import annotations

from dataclasses import dataclass
import copy
import re
from typing import Callable, Iterable, Protocol

import numpy as np
import pandas as pd

from .agents import BaseAgent
from .config import FisheryConfig
from .llm_adapter import (
    NullPolicyLLMClient,
    PolicyJSON,
    PolicyLLMClient,
    build_policy_prompt,
    clamp_policy,
    extract_json_object,
)
from .metrics import gini
from .simulation import run_episode


@dataclass
class StrategySpec:
    """
    Interpretable threshold strategy used for population turnover experiments.
    """

    strategy_id: str
    low_stock_threshold: float
    high_stock_threshold: float
    low_harvest_frac: float
    mid_harvest_frac: float
    high_harvest_frac: float
    origin: str = "seed"
    rationale: str = ""

    def to_agent(self, max_harvest_per_agent: float) -> "ThresholdStrategy":
        return ThresholdStrategy(
            name=self.strategy_id,
            max_h=max_harvest_per_agent,
            low_stock_threshold=self.low_stock_threshold,
            high_stock_threshold=self.high_stock_threshold,
            low_harvest_frac=self.low_harvest_frac,
            mid_harvest_frac=self.mid_harvest_frac,
            high_harvest_frac=self.high_harvest_frac,
        )


class ThresholdStrategy(BaseAgent):
    def __init__(
        self,
        name: str,
        max_h: float,
        low_stock_threshold: float,
        high_stock_threshold: float,
        low_harvest_frac: float,
        mid_harvest_frac: float,
        high_harvest_frac: float,
    ):
        self.name = name
        self.max_h = max_h
        self.low_stock_threshold = low_stock_threshold
        self.high_stock_threshold = high_stock_threshold
        self.low_harvest_frac = low_harvest_frac
        self.mid_harvest_frac = mid_harvest_frac
        self.high_harvest_frac = high_harvest_frac

    def act(self, obs_stock: float, t: int, n_agents: int) -> float:
        if obs_stock < self.low_stock_threshold:
            frac = self.low_harvest_frac
        elif obs_stock < self.high_stock_threshold:
            frac = self.mid_harvest_frac
        else:
            frac = self.high_harvest_frac
        return float(np.clip(self.max_h * frac, 0.0, self.max_h))


def strategy_spec_to_policy_json(spec: StrategySpec) -> PolicyJSON:
    return PolicyJSON(
        rationale=spec.rationale or f"{spec.origin} strategy",
        low_stock_threshold=spec.low_stock_threshold,
        high_stock_threshold=spec.high_stock_threshold,
        low_harvest_frac=spec.low_harvest_frac,
        mid_harvest_frac=spec.mid_harvest_frac,
        high_harvest_frac=spec.high_harvest_frac,
    )


def policy_json_to_strategy_spec(policy: PolicyJSON, strategy_id: str, origin: str) -> StrategySpec:
    return StrategySpec(
        strategy_id=strategy_id,
        low_stock_threshold=policy.low_stock_threshold,
        high_stock_threshold=policy.high_stock_threshold,
        low_harvest_frac=policy.low_harvest_frac,
        mid_harvest_frac=policy.mid_harvest_frac,
        high_harvest_frac=policy.high_harvest_frac,
        origin=origin,
        rationale=policy.rationale,
    )


def random_strategy(strategy_id: str, stock_max: float, rng: np.random.Generator) -> StrategySpec:
    return _sample_strategy(
        strategy_id=strategy_id,
        stock_max=stock_max,
        rng=rng,
        threshold_ranges=((0.05, 0.4), (0.45, 0.95)),
        harvest_ranges=((0.02, 0.25), (0.1, 0.7), (0.25, 1.0)),
        origin="random_init",
        rationale="random seed strategy",
    )


def _sample_strategy(
    strategy_id: str,
    stock_max: float,
    rng: np.random.Generator,
    threshold_ranges: tuple[tuple[float, float], tuple[float, float]],
    harvest_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    origin: str,
    rationale: str,
) -> StrategySpec:
    low_stock_threshold = float(rng.uniform(*threshold_ranges[0]) * stock_max)
    high_stock_threshold = float(rng.uniform(*threshold_ranges[1]) * stock_max)
    if high_stock_threshold <= low_stock_threshold:
        high_stock_threshold = min(stock_max, low_stock_threshold + 5.0)

    low_harvest_frac = float(rng.uniform(*harvest_ranges[0]))
    mid_harvest_frac = float(rng.uniform(*harvest_ranges[1]))
    high_harvest_frac = float(rng.uniform(*harvest_ranges[2]))

    return StrategySpec(
        strategy_id=strategy_id,
        low_stock_threshold=low_stock_threshold,
        high_stock_threshold=high_stock_threshold,
        low_harvest_frac=low_harvest_frac,
        mid_harvest_frac=mid_harvest_frac,
        high_harvest_frac=high_harvest_frac,
        origin=origin,
        rationale=rationale,
    )


def cooperative_strategy(strategy_id: str, stock_max: float, rng: np.random.Generator) -> StrategySpec:
    return _sample_strategy(
        strategy_id=strategy_id,
        stock_max=stock_max,
        rng=rng,
        threshold_ranges=((0.18, 0.45), (0.60, 0.95)),
        harvest_ranges=((0.01, 0.10), (0.08, 0.28), (0.20, 0.50)),
        origin="cooperative_seed",
        rationale="cooperative-heavy seed strategy",
    )


def adversarial_strategy(strategy_id: str, stock_max: float, rng: np.random.Generator) -> StrategySpec:
    return _sample_strategy(
        strategy_id=strategy_id,
        stock_max=stock_max,
        rng=rng,
        threshold_ranges=((0.01, 0.18), (0.18, 0.55)),
        harvest_ranges=((0.12, 0.45), (0.45, 0.85), (0.75, 1.0)),
        origin="adversarial_seed",
        rationale="aggressive seed strategy",
    )


def build_initial_population(
    population_size: int,
    stock_max: float,
    rng: np.random.Generator,
    partner_mix_preset: str,
) -> list[StrategySpec]:
    mix = partner_mix_preset.strip().lower()
    presets = {
        "cooperative_heavy": (0.6, 0.3, 0.1),
        "balanced": (0.34, 0.33, 0.33),
        "adversarial_heavy": (0.1, 0.3, 0.6),
    }
    if mix not in presets:
        raise ValueError(f"Unknown partner_mix_preset: {partner_mix_preset}")

    coop_frac, neutral_frac, adv_frac = presets[mix]
    counts = [
        int(round(population_size * coop_frac)),
        int(round(population_size * neutral_frac)),
        int(round(population_size * adv_frac)),
    ]
    while sum(counts) < population_size:
        counts[np.argmin(counts)] += 1
    while sum(counts) > population_size:
        counts[np.argmax(counts)] -= 1

    population: list[StrategySpec] = []
    for i in range(counts[0]):
        population.append(cooperative_strategy(f"g0_s{len(population)}", stock_max=stock_max, rng=rng))
    for _ in range(counts[1]):
        population.append(random_strategy(f"g0_s{len(population)}", stock_max=stock_max, rng=rng))
    for _ in range(counts[2]):
        population.append(adversarial_strategy(f"g0_s{len(population)}", stock_max=stock_max, rng=rng))

    rng.shuffle(population)
    for i, spec in enumerate(population):
        spec.strategy_id = f"g0_s{i}"
    return population


def mutate_strategy(
    parent: StrategySpec,
    strategy_id: str,
    stock_max: float,
    rng: np.random.Generator,
    adversarial_pressure: float,
) -> StrategySpec:
    """
    Mutation operator biased toward exploitative invaders as pressure rises.
    """

    low = parent.low_stock_threshold + rng.normal(0.0, 6.0)
    high = parent.high_stock_threshold + rng.normal(0.0, 8.0)
    low = float(np.clip(low, 0.0, stock_max - 1.0))
    high = float(np.clip(high, low + 1.0, stock_max))

    low_frac = float(np.clip(parent.low_harvest_frac + rng.normal(0.0, 0.06), 0.0, 1.0))
    mid_frac = float(
        np.clip(parent.mid_harvest_frac + rng.normal(0.0, 0.08) + adversarial_pressure * 0.05, 0.0, 1.0)
    )
    high_frac = float(
        np.clip(parent.high_harvest_frac + rng.normal(0.0, 0.12) + adversarial_pressure * 0.12, 0.0, 1.0)
    )

    if rng.random() < adversarial_pressure:
        high_frac = float(np.clip(high_frac + rng.uniform(0.04, 0.16), 0.0, 1.0))
        mid_frac = float(np.clip(mid_frac + rng.uniform(0.02, 0.08), 0.0, 1.0))

    return StrategySpec(
        strategy_id=strategy_id,
        low_stock_threshold=low,
        high_stock_threshold=high,
        low_harvest_frac=low_frac,
        mid_harvest_frac=mid_frac,
        high_harvest_frac=high_frac,
        origin="mutation",
        rationale=f"mutated from {parent.strategy_id}",
    )


class StrategyInjector(Protocol):
    def inject(
        self,
        parent: StrategySpec,
        parent_fitness: float,
        strategy_id: str,
        stock_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> StrategySpec: ...


class MutationStrategyInjector:
    def inject(
        self,
        parent: StrategySpec,
        parent_fitness: float,
        strategy_id: str,
        stock_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> StrategySpec:
        del parent_fitness
        return mutate_strategy(
            parent=parent,
            strategy_id=strategy_id,
            stock_max=stock_max,
            rng=rng,
            adversarial_pressure=adversarial_pressure,
        )


class RandomStrategyInjector:
    def inject(
        self,
        parent: StrategySpec,
        parent_fitness: float,
        strategy_id: str,
        stock_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> StrategySpec:
        del parent, parent_fitness, adversarial_pressure
        child = random_strategy(strategy_id=strategy_id, stock_max=stock_max, rng=rng)
        child.origin = "random_injector"
        child.rationale = "uninformed random child"
        return child


class AdversarialHeuristicStrategyInjector:
    def inject(
        self,
        parent: StrategySpec,
        parent_fitness: float,
        strategy_id: str,
        stock_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> StrategySpec:
        del parent, parent_fitness, adversarial_pressure
        child = adversarial_strategy(strategy_id=strategy_id, stock_max=stock_max, rng=rng)
        child.origin = "adversarial_heuristic"
        child.rationale = "heuristic invader biased to high extraction"
        return child


class SearchMutationStrategyInjector:
    def __init__(self, n_candidates: int = 6, eval_horizon: int = 40):
        self.n_candidates = max(2, int(n_candidates))
        self.eval_horizon = max(5, int(eval_horizon))
        self._context_cfg: FisheryConfig | None = None
        self._context_population: list[StrategySpec] = []
        self._context_seed: int = 0

    def prepare_generation(
        self,
        base_cfg: FisheryConfig,
        parent_pool: list[StrategySpec],
        generation: int,
        rng: np.random.Generator,
    ) -> None:
        del rng
        self._context_cfg = copy.deepcopy(base_cfg)
        self._context_cfg.horizon = min(self._context_cfg.horizon, self.eval_horizon)
        self._context_population = [copy.deepcopy(spec) for spec in parent_pool]
        self._context_seed = int(base_cfg.seed + 1_000_000 + generation)

    def inject(
        self,
        parent: StrategySpec,
        parent_fitness: float,
        strategy_id: str,
        stock_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> StrategySpec:
        del parent_fitness
        candidates = [
            mutate_strategy(
                parent=parent,
                strategy_id=f"{strategy_id}_cand{i}",
                stock_max=stock_max,
                rng=rng,
                adversarial_pressure=adversarial_pressure,
            )
            for i in range(self.n_candidates)
        ]
        if not self._context_population or self._context_cfg is None:
            best = max(candidates, key=lambda spec: spec.high_harvest_frac + 0.5 * spec.mid_harvest_frac)
        else:
            best = max(candidates, key=self._attack_score)
        best.strategy_id = strategy_id
        best.origin = "search_mutation"
        best.rationale = f"search-over-mutations from {parent.strategy_id}"
        return best

    def _attack_score(self, candidate: StrategySpec) -> float:
        assert self._context_cfg is not None
        cfg = copy.deepcopy(self._context_cfg)
        partners = [copy.deepcopy(spec) for spec in self._context_population]
        if partners:
            partners = partners[: max(1, cfg.n_agents - 1)]
        population = [candidate] + partners
        cfg.seed = self._context_seed
        cfg.n_agents = len(population)
        agents: list[BaseAgent] = [spec.to_agent(cfg.max_harvest_per_agent) for spec in population]
        out = run_episode(cfg, agents)
        return float(out["payoffs"][0])


class LLMJSONStrategyInjector:
    """
    Adapter path:
    prompt -> JSON policy -> StrategySpec
    """

    def __init__(
        self,
        llm_client: PolicyLLMClient | None = None,
        fallback_injector: StrategyInjector | None = None,
        jitter_std: float = 0.04,
    ):
        self.llm_client = llm_client if llm_client is not None else NullPolicyLLMClient()
        self.fallback_injector = fallback_injector if fallback_injector is not None else MutationStrategyInjector()
        self.jitter_std = float(max(0.0, jitter_std))

    def inject(
        self,
        parent: StrategySpec,
        parent_fitness: float,
        strategy_id: str,
        stock_max: float,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> StrategySpec:
        parent_policy = strategy_spec_to_policy_json(parent)
        prompt = build_policy_prompt(
            parent_policy=parent_policy,
            parent_fitness=parent_fitness,
            stock_max=stock_max,
            adversarial_pressure=adversarial_pressure,
        )
        try:
            raw_response = self.llm_client.complete(prompt)
            parsed = extract_json_object(raw_response)
            policy = PolicyJSON.from_dict(parsed)
            policy = clamp_policy(policy, stock_max=stock_max)
            policy = self._perturb_policy(policy, rng=rng, adversarial_pressure=adversarial_pressure)
            child = policy_json_to_strategy_spec(policy=policy, strategy_id=strategy_id, origin="llm_json")
            return child
        except Exception:
            child = self.fallback_injector.inject(
                parent=parent,
                parent_fitness=parent_fitness,
                strategy_id=strategy_id,
                stock_max=stock_max,
                rng=rng,
                adversarial_pressure=adversarial_pressure,
            )
            child.origin = "llm_fallback_mutation"
            if not child.rationale:
                child.rationale = f"fallback from {parent.strategy_id}"
            return child

    def _perturb_policy(
        self,
        policy: PolicyJSON,
        rng: np.random.Generator,
        adversarial_pressure: float,
    ) -> PolicyJSON:
        low_thr = float(
            np.clip(policy.low_stock_threshold + rng.normal(0.0, 4.0), 0.0, policy.high_stock_threshold - 1.0)
        )
        high_thr = float(np.clip(policy.high_stock_threshold + rng.normal(0.0, 6.0), low_thr + 1.0, np.inf))

        low_frac = policy.low_harvest_frac + rng.normal(0.0, self.jitter_std * 0.8)
        mid_frac = policy.mid_harvest_frac + rng.normal(0.0, self.jitter_std) + adversarial_pressure * 0.03
        high_frac = policy.high_harvest_frac + rng.normal(0.0, self.jitter_std * 1.3) + adversarial_pressure * 0.08

        low_frac = float(np.clip(low_frac, 0.0, 1.0))
        mid_frac = float(np.clip(max(low_frac, mid_frac), 0.0, 1.0))
        high_frac = float(np.clip(max(mid_frac, high_frac), 0.0, 1.0))

        return PolicyJSON(
            rationale=policy.rationale,
            low_stock_threshold=low_thr,
            high_stock_threshold=high_thr,
            low_harvest_frac=low_frac,
            mid_harvest_frac=mid_frac,
            high_harvest_frac=high_frac,
        )


def make_strategy_injector(
    injector_mode: str,
    llm_client: PolicyLLMClient | None = None,
) -> StrategyInjector:
    mode = injector_mode.strip().lower()
    if mode == "mutation":
        return MutationStrategyInjector()
    if mode == "random":
        return RandomStrategyInjector()
    if mode == "adversarial_heuristic":
        return AdversarialHeuristicStrategyInjector()
    if mode == "search_mutation":
        return SearchMutationStrategyInjector()
    if mode == "llm_json":
        return LLMJSONStrategyInjector(llm_client=llm_client)
    raise ValueError(f"Unknown injector_mode: {injector_mode}")


def _build_seed_schedule(seed_start: int, n_seeds: int) -> list[int]:
    return [seed_start + i for i in range(n_seeds)]


def _population_diversity(population: Iterable[StrategySpec]) -> float:
    pop = list(population)
    if len(pop) < 2:
        return 0.0
    matrix = np.array(
        [
            [
                p.low_stock_threshold,
                p.high_stock_threshold,
                p.low_harvest_frac,
                p.mid_harvest_frac,
                p.high_harvest_frac,
            ]
            for p in pop
        ],
        dtype=float,
    )
    return float(matrix.std(axis=0).mean())


def _apply_cfg_overrides(
    base_cfg: FisheryConfig,
    overrides: dict[str, float] | None,
) -> FisheryConfig:
    cfg = copy.deepcopy(base_cfg)
    if overrides is None:
        return cfg
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config field in overrides: {key}")
        setattr(cfg, key, value)
    return cfg


def _summarize_episode_df(episode_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_collapse_rate": float(episode_df["collapsed"].mean()),
        f"{prefix}_mean_stock": float(episode_df["mean_stock"].mean()),
        f"{prefix}_mean_final_stock": float(episode_df["final_stock"].mean()),
        f"{prefix}_mean_welfare": float(episode_df["welfare"].mean()),
        f"{prefix}_mean_payoff_gini": float(episode_df["payoff_gini"].mean()),
        f"{prefix}_mean_sanction_total": float(episode_df["sanction_total"].mean()),
        f"{prefix}_mean_violation_events": float(episode_df["violation_events"].mean()),
        f"{prefix}_mean_requested_harvest": float(episode_df["mean_requested_harvest"].mean()),
        f"{prefix}_mean_realized_harvest": float(episode_df["mean_realized_harvest"].mean()),
        f"{prefix}_mean_audit_rate": float(episode_df["mean_audit_rate"].mean()),
        f"{prefix}_mean_quota": float(episode_df["mean_quota"].mean()),
        f"{prefix}_mean_quota_clipped_total": float(episode_df["mean_quota_clipped_total"].mean()),
        f"{prefix}_mean_repeat_offender_rate": float(episode_df["mean_repeat_offender_rate"].mean()),
        f"{prefix}_closure_active_fraction": float(episode_df["closure_active_fraction"].mean()),
        f"{prefix}_mean_stock_recovery_lag": float(episode_df["mean_stock_recovery_lag"].mean()),
    }


def _safe_name(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    out = out.strip("_")
    return out or "regime"


def _resolve_test_regimes(
    test_overrides: dict[str, float] | None,
    test_regimes: list[dict[str, dict[str, float]]] | None,
) -> list[dict[str, dict[str, float]]]:
    if test_regimes:
        resolved: list[dict[str, dict[str, float]]] = []
        for i, item in enumerate(test_regimes):
            name = str(item.get("name", f"regime_{i}"))
            overrides = item.get("overrides", {})
            if not isinstance(overrides, dict):
                raise ValueError(f"test_regimes[{i}].overrides must be a mapping.")
            resolved.append({"name": name, "overrides": dict(overrides)})
        return resolved
    return [{"name": "default", "overrides": dict(test_overrides or {})}]


def evaluate_population(
    base_cfg: FisheryConfig,
    population: list[StrategySpec],
    seeds: list[int],
    collapse_penalty: float,
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
        agents: list[BaseAgent] = [spec.to_agent(cfg.max_harvest_per_agent) for spec in population]

        out = run_episode(cfg, agents)
        collapsed = bool(out["collapsed"])
        welfare = float(out["payoffs"].sum())
        payoff_gini = gini(out["payoffs"])
        collapse_cost = collapse_penalty if collapsed else 0.0

        payoff_sum += out["payoffs"]
        fitness_sum += out["payoffs"] - collapse_cost

        rows.append(
            {
                "seed": seed,
                "collapsed": collapsed,
                "t_end": out["t_end"],
                "final_stock": out["final_stock"],
                "mean_stock": out["mean_stock"],
                "welfare": welfare,
                "payoff_gini": payoff_gini,
                "sanction_total": out["sanction_total"],
                "violation_events": out["violation_events"],
                "mean_requested_harvest": out["mean_requested_harvest"],
                "mean_realized_harvest": out["mean_realized_harvest"],
                "mean_audit_rate": out["mean_audit_rate"],
                "mean_quota": out["mean_quota"],
                "mean_quota_clipped_total": out["mean_quota_clipped_total"],
                "mean_repeat_offender_rate": out["mean_repeat_offender_rate"],
                "closure_active_fraction": out["closure_active_fraction"],
                "mean_stock_recovery_lag": out["mean_stock_recovery_lag"],
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
                "low_stock_threshold": spec.low_stock_threshold,
                "high_stock_threshold": spec.high_stock_threshold,
                "low_harvest_frac": spec.low_harvest_frac,
                "mid_harvest_frac": spec.mid_harvest_frac,
                "high_harvest_frac": spec.high_harvest_frac,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def run_evolutionary_invasion(
    base_cfg: FisheryConfig,
    generations: int = 30,
    population_size: int = 12,
    seeds_per_generation: int = 64,
    test_seeds_per_generation: int | None = None,
    replacement_fraction: float = 0.3,
    collapse_penalty: float = 50.0,
    adversarial_pressure: float = 0.7,
    rng_seed: int = 0,
    train_overrides: dict[str, float] | None = None,
    test_overrides: dict[str, float] | None = None,
    test_regimes: list[dict[str, dict[str, float]]] | None = None,
    partner_mix_preset: str = "balanced",
    injector: StrategyInjector | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repeatedly removes weak strategies and injects invaders under selection pressure.
    Selection is based on train regime fitness; report includes train and held-out test regime metrics.
    Returns:
    - generation-level summary dataframe
    - per-strategy fitness history dataframe
    """

    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    if generations < 1:
        raise ValueError("generations must be >= 1")
    if not 0.0 < replacement_fraction < 1.0:
        raise ValueError("replacement_fraction must be in (0, 1)")

    rng = np.random.default_rng(rng_seed)
    train_cfg = _apply_cfg_overrides(base_cfg, train_overrides)
    resolved_test_regimes = _resolve_test_regimes(test_overrides=test_overrides, test_regimes=test_regimes)
    test_seed_count = int(test_seeds_per_generation or seeds_per_generation)
    strategy_injector = injector if injector is not None else MutationStrategyInjector()

    population = build_initial_population(
        population_size=population_size,
        stock_max=base_cfg.stock_max,
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
        train_seeds = _build_seed_schedule(seed_start=train_seed_start, n_seeds=seeds_per_generation)
        test_seeds = _build_seed_schedule(seed_start=test_seed_start, n_seeds=test_seed_count)

        train_episode_df, train_score_df = evaluate_population(
            base_cfg=train_cfg,
            population=population,
            seeds=train_seeds,
            collapse_penalty=collapse_penalty,
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
            regime_episode_df, _ = evaluate_population(
                base_cfg=regime_cfg,
                population=population,
                seeds=test_seeds,
                collapse_penalty=collapse_penalty,
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
            "best_fitness": float(train_score_df["fitness"].iloc[0]),
            "worst_fitness": float(train_score_df["fitness"].iloc[-1]),
            "injector_mode": type(strategy_injector).__name__,
            "partner_mix_preset": partner_mix_preset,
            "adversarial_pressure": float(adversarial_pressure),
        }
        row.update(_summarize_episode_df(train_episode_df, prefix="train"))
        row.update(_summarize_episode_df(test_episode_df, prefix="test"))
        row.update(per_regime_summaries)
        # Backward-compatible aliases.
        row["collapse_rate"] = row["test_collapse_rate"]
        row["mean_stock"] = row["test_mean_stock"]
        row["mean_final_stock"] = row["test_mean_final_stock"]
        row["mean_welfare"] = row["test_mean_welfare"]
        row["mean_payoff_gini"] = row["test_mean_payoff_gini"]
        row["mean_sanction_total"] = row["test_mean_sanction_total"]
        row["mean_violation_events"] = row["test_mean_violation_events"]
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
                base_cfg=train_cfg,
                parent_pool=parent_pool,
                generation=generation,
                rng=rng,
            )

        injected: list[StrategySpec] = []
        for i in range(replace_count):
            parent = parent_pool_top[i % len(parent_pool_top)]
            parent_fitness = fitness_by_id.get(parent.strategy_id, 0.0)
            child = strategy_injector.inject(
                parent=parent,
                parent_fitness=parent_fitness,
                strategy_id=f"g{generation + 1}_s{i}",
                stock_max=base_cfg.stock_max,
                rng=rng,
                adversarial_pressure=adversarial_pressure,
            )
            child.strategy_id = f"g{generation + 1}_s{i}"
            injected.append(child)

        population = parent_pool + injected

    return pd.DataFrame(generation_rows), pd.DataFrame(strategy_rows)
