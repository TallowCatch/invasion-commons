"""Small, resumable CPU checks with frozen policies. Never calls a language model.

Each gzip block is atomic and independently reproducible. A manifest pins parameters,
source hashes and implementation before results can be resumed or analyzed.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.extract_harvest_oversight_case import _matching_strategy_rows, _strategy_from_row
from fishery_sim.harvest import GovernmentAgent, HarvestStrategySpec, harvest_global_safe, run_harvest_episode
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_scenario, get_harvest_scenario_preset, get_harvest_regime_pack
from fishery_sim.harvest_evolution import DEFAULT_GOVERNMENT_PARAMS, build_initial_harvest_population, mutate_harvest_strategy

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_strategy_history.csv"
AGENTS = HISTORY.with_name(HISTORY.name.replace("strategy_history.csv", "agent_history.csv.gz"))
SCENARIOS = ["community_irrigation", "forest_co_management"]
MECHANISMS = ["none", "communication", "uniform_off", "uniform_on", "neighborhood_off",
              "neighborhood_on", "local_cutoff", "local_state", "signal_only", "joint_reference"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value, compressed=False):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if compressed else open
    with opener(tmp, "wt", encoding="utf8") as handle:
        json.dump(value, handle, allow_nan=False)
    tmp.replace(path)


def read_json(path, compressed=False):
    with (gzip.open if compressed else open)(path, "rt", encoding="utf8") as handle:
        return json.load(handle)


def pin_manifest(directory, settings, inputs):
    directory.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "fishery_sim/harvest.py", ROOT / "fishery_sim/harvest_evolution.py",
               ROOT / "fishery_sim/harvest_benchmarks.py", ROOT / "experiments/extract_harvest_oversight_case.py"]
    manifest = {"protocol": "harvest_checks_v1", "settings": settings,
                "input_sha256": {str(Path(p).relative_to(ROOT)): digest(p) for p in inputs + sources}}
    target = directory / "manifest.json"
    if target.exists() and read_json(target) != manifest:
        raise ValueError("Manifest mismatch: use a new output directory; do not mix protocols")
    write_json(target, manifest)


def predict_health(cfg, health, requests):
    harvest = np.minimum(requests * cfg.max_harvest_per_agent, health)
    excess = np.maximum(0, harvest - cfg.sustainable_harvest_frac * cfg.max_harvest_per_agent)
    remaining = np.maximum(0, health - harvest - cfg.neighbor_externality * (np.roll(excess, 1) + np.roll(excess, -1)))
    return np.clip(remaining + np.maximum(0, cfg.regen_rate * remaining * (1 - remaining / cfg.patch_max)), 0, cfg.patch_max)


def local_cutoff(cfg, health, requests):
    return np.minimum(requests, cfg.sustainable_harvest_frac + cfg.local_safety_margin)


def local_state(cfg, health, requests):
    # Invert own-patch, zero-weather regrowth. Unknown neighbour actions are omitted.
    # This is an optimistic local one-step model, not a safety guarantee.
    if not 0 <= cfg.regen_rate <= 1:
        raise ValueError("Local inverse assumes monotone regrowth (0 <= rate <= 1)")
    r, target = cfg.regen_rate, cfg.global_min_mean_patch_health + 1e-8
    disc = (1 + r)**2 - 4 * r * target / cfg.patch_max
    if disc < 0:
        return np.zeros_like(requests)
    residual = 2 * target / (1 + r + np.sqrt(disc))
    return np.minimum(requests, np.maximum(0, health - residual) / cfg.max_harvest_per_agent)


def joint_reference(cfg, health, requests):
    if not 0 <= cfg.regen_rate <= 1:
        raise ValueError("Joint scaling assumes monotone regrowth")
    def safe(scale):
        return harvest_global_safe(predict_health(cfg, health, requests * scale),
            min_mean_patch_health=cfg.global_min_mean_patch_health + 1e-8,
            local_patch_failure_threshold=cfg.local_patch_failure_threshold,
            failure_fraction_threshold=cfg.failure_fraction_threshold)
    if safe(1):
        return requests.copy()
    if not safe(0):
        return np.zeros_like(requests)
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if safe(mid):
            lo = mid
        else:
            hi = mid
    return requests * lo


class SignalOnly(GovernmentAgent):
    def apply_cap(self, requested_fracs, cap_fracs):
        # Keep announcement and observation timing; perform no enforcement or charge.
        self._last_intended_target_count = int(np.sum(~np.isnan(cap_fracs))) if cap_fracs is not None else 0
        self._last_actual_target_count = 0
        self._last_missed_target_count = self._last_intended_target_count
        self._last_governance_budget_spent = 0.0
        return requested_fracs.copy(), np.zeros_like(requested_fracs, dtype=bool)


def setup(cfg, mechanism):
    cfg = copy.deepcopy(cfg)
    cfg.side_payments_enabled = False
    cfg.communication_enabled = mechanism in {"communication", "uniform_on", "neighborhood_on"}
    filters = {"local_cutoff": local_cutoff, "local_state": local_state, "joint_reference": joint_reference}
    governor = None
    if mechanism.startswith(("uniform_", "neighborhood_")) or mechanism == "signal_only":
        cls = SignalOnly if mechanism == "signal_only" else GovernmentAgent
        params = dict(DEFAULT_GOVERNMENT_PARAMS)
        params.update(detection_recall=0.7, enforcement_delay_rounds=1, max_target_share=0.5,
                      governance_budget_cost=0.0, capacity_rule="floor")
        local = mechanism.startswith("neighborhood_")
        governor = cls(**params, enforcement_scope="local" if local else "global", expand_target_neighbors=local)
    if mechanism not in MECHANISMS:
        raise ValueError(mechanism)
    return cfg, governor, filters.get(mechanism)


def serial_episode(cfg, specs, mechanism, trace=True):
    configured, governor, action_filter = setup(cfg, mechanism)
    result = run_harvest_episode(configured, [spec.to_agent() for spec in specs], governor,
                                record_trace=trace, action_filter=action_filter)
    rows = result.pop("episode_trace_rows")
    metrics = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in result.items()
               if isinstance(v, (int, float, np.number))}
    metrics = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in metrics.items()}
    return {"metrics": metrics, "trace": rows, "config": asdict(configured)}


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(HISTORY).fillna("")
    base = df[df.condition.eq("none") & df.overseer_capability_level.eq("strong_overseer")
              & df.actor_capability_level.isin(["low_actor", "high_actor"])]
    populations, expected = [], set()
    for (scenario, actor, run), group in base.groupby(["scenario_preset", "actor_capability_level", "run_id"]):
        generation = int(group.generation.max())
        row = group[group.generation.eq(generation)].iloc[0]
        ordered = _matching_strategy_rows(df, row)
        specs = [_strategy_from_row(r) for _, r in ordered.iterrows()]
        if len(specs) != 6 or len(set(s.strategy_id for s in specs)) != 6:
            raise ValueError("Expected six unique policy artifacts")
        populations.append({"scenario": scenario, "actor": actor, "run_id": int(run), "generation": generation,
                            "strategies": [asdict(s) for s in specs]})
        expected.update((scenario, actor, int(run), generation, i, spec.strategy_id) for i, spec in enumerate(specs))
    print("Checking restored spatial order against original per-agent archive", flush=True)
    cols = ["scenario_preset", "actor_capability_level", "run_id", "generation", "agent_index", "strategy_id",
            "condition", "overseer_capability_level"]
    observed = set()
    for chunk in pd.read_csv(AGENTS, usecols=cols, chunksize=250_000):
        selected = chunk[chunk.condition.eq("none") & chunk.overseer_capability_level.eq("strong_overseer")
                         & chunk.generation.eq(14) & chunk.actor_capability_level.isin(["low_actor", "high_actor"])]
        observed.update(selected[cols[:6]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(f"Spatial-order validation failed: missing={len(expected-observed)}, extra={len(observed-expected)}")
    pin_manifest(directory, {"kind": "source_snapshots", "phase": "final_generation", "ordering_verified": True}, [HISTORY, AGENTS])
    write_json(directory / "populations.json", populations)
    print(f"Verified {len(populations)} populations / {len(expected)} agent positions", flush=True)


def run_mechanisms(directory, sources, evaluation_regime="base"):
    source = sources / "populations.json"
    settings = {"kind": "frozen_mechanisms", "evaluation_regime": evaluation_regime, "seeds": list(range(8)), "base_seed": 80_000_000,
                "mechanisms": MECHANISMS, "controls": ["base", "no_weather", "no_spillover", "neither"],
                "control_mechanisms": ["none", "local_cutoff", "local_state", "joint_reference"],
                "cost": 0, "credits": False, "q": 0.7, "delay": 1, "target_share": 0.5}
    pin_manifest(directory, settings, [source, sources / "manifest.json"])
    populations = read_json(source)
    count = 0
    started = time.monotonic()
    for pop in populations:
        specs = [HarvestStrategySpec(**s) for s in pop["strategies"]]
        for control in settings["controls"]:
            modes = MECHANISMS if control == "base" else settings["control_mechanisms"]
            for mechanism in modes:
                key = f"{pop['scenario']}__{pop['actor']}__{pop['run_id']}__{control}__{mechanism}"
                path = directory / f"{key}.json.gz"
                if path.exists():
                    continue
                episodes = []
                for offset in settings["seeds"]:
                    seed = settings["base_seed"] + pop["run_id"] * 100 + offset
                    cfg = make_harvest_cfg_for_scenario(pop["scenario"], seed=seed)
                    if evaluation_regime != "base":
                        tier = get_harvest_scenario_preset(pop["scenario"])["tier"]
                        regimes = {r["name"]: r["overrides"] for r in get_harvest_regime_pack(tier)}
                        if evaluation_regime not in regimes:
                            raise ValueError(f"Unknown held-out regime: {evaluation_regime}")
                        for field, value in regimes[evaluation_regime].items():
                            setattr(cfg, field, value)
                    if control in {"no_weather", "neither"}:
                        cfg.weather_noise_std = 0
                    if control in {"no_spillover", "neither"}:
                        cfg.neighbor_externality = 0
                    out = serial_episode(cfg, specs, mechanism)
                    out["metadata"] = {k: v for k, v in pop.items() if k != "strategies"}
                    out["metadata"].update(control=control, mechanism=mechanism, seed=seed)
                    episodes.append(out)
                write_json(path, episodes, compressed=True)
                count += len(episodes)
        print(f"Completed {pop['scenario']} / {pop['actor']} / run {pop['run_id']} ({count} new episodes)", flush=True)
    write_json(directory / "completion.json", {"expected_blocks": len(populations) * 22,
        "actual_blocks": len(list(directory.glob("*.json.gz"))), "new_episodes": count, "seconds": time.monotonic()-started})


def raw_score(cfg, population, entrant_index):
    out = run_harvest_episode(cfg, [p.to_agent() for p in population])
    return {"score": float(out["final_payoffs"][entrant_index] - cfg.garden_failure_penalty * out["garden_failure_event"]),
            "entrant_payoff": float(out["final_payoffs"][entrant_index]), "total_welfare": out["total_welfare"],
            "unsafe_rate": out["global_unsafe_rate"], "patch_health": out["mean_patch_health"]}


def run_calibration(directory):
    pin_manifest(directory, {"kind": "generator_calibration", "contexts": 12, "candidate_counts": [1, 6, 12],
        "horizons": [30, 60], "test_seeds": 16, "partner_mix": "balanced", "pressure": 0.3,
        "parent_seed_base": 90_000_000, "selection_seed_base": 91_000_000, "test_seed_base": 92_000_000}, [])
    started = time.monotonic()
    for scenario in SCENARIOS:
        for context in range(12):
            path = directory / f"{scenario}__{context}.json.gz"
            if path.exists():
                continue
            rng = np.random.default_rng(90_000_000 + context)
            cfg = make_harvest_cfg_for_scenario(scenario, seed=91_000_000 + context,
                                               communication_enabled=False, side_payments_enabled=False)
            population = build_initial_harvest_population(6, cfg.patch_max, rng, "balanced")
            index = context % 6
            parent = population[index]
            candidates = [mutate_harvest_strategy(parent, f"candidate_{i}", cfg.patch_max, rng, 0.3) for i in range(12)]
            selection_rows, tests = [], []
            for horizon in [30, 60]:
                scores = []
                for i, candidate in enumerate(candidates):
                    trial = list(population)
                    trial[index] = candidate
                    short = copy.deepcopy(cfg)
                    short.horizon = horizon
                    score = raw_score(short, trial, index)["score"]
                    scores.append(score)
                    selection_rows.append({"horizon": horizon, "candidate": i, "score": score})
                for k in [1, 6, 12]:
                    selected = int(np.argmax(scores[:k]))
                    trial = list(population)
                    trial[index] = candidates[selected]
                    for offset in range(16):
                        held = copy.deepcopy(cfg)
                        held.seed = 92_000_000 + context * 100 + offset
                        tests.append({"scenario": scenario, "context": context, "position": index,
                            "candidate_count": k, "search_horizon": horizon, "selected_candidate": selected,
                            "seed": held.seed, **raw_score(held, trial, index)})
            write_json(path, {"population": [asdict(s) for s in population], "candidates": [asdict(s) for s in candidates],
                             "config": asdict(cfg), "selection": selection_rows, "tests": tests}, compressed=True)
            print(f"Calibration {scenario} / parent context {context} complete", flush=True)
    write_json(directory / "completion.json", {"expected_blocks": 24, "actual_blocks": len(list(directory.glob("*.json.gz"))),
                                               "seconds": time.monotonic()-started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "mechanisms", "calibration"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=ROOT / "results/runs/validation_v1/sources")
    parser.add_argument("--evaluation-regime", default="base")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output_dir)
    elif args.mode == "mechanisms":
        run_mechanisms(args.output_dir, args.sources, args.evaluation_regime)
    else:
        run_calibration(args.output_dir)


if __name__ == "__main__":
    main()
