"""Nested actor-search budgets evaluated against unchanged Harvest reviewers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from experiments.common.run_matched_oversight import ROOT, digest, read_json, write_json
from fishery_sim.budgeted_oversight import MODES, decide_budgeted_harvest, mask_requests
from fishery_sim.harvest import HarvestCommonsConfig, HarvestStrategySpec, run_harvest_episode
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_scenario
from fishery_sim.harvest_evolution import (
    adversarial_harvest_strategy, balanced_harvest_strategy,
    cooperative_harvest_strategy, mutate_harvest_strategy,
)
from fishery_sim.oversight_protocol import (
    MonitorSettings, harvest_reference_risk, harvest_safe, score_verdict,
)


SOURCE_FILES = (
    "notes/research_review/ACTOR_PRESSURE_PILOT_PROTOCOL.md",
    "experiments/archive/harvest_2026q2/run_actor_pressure_pilot.py",
    "experiments/archive/harvest_2026q2/analyze_actor_pressure_pilot.py",
    "fishery_sim/budgeted_oversight.py",
    "fishery_sim/oversight_protocol.py",
    "fishery_sim/harvest.py",
    "fishery_sim/harvest_evolution.py",
    "fishery_sim/harvest_benchmarks.py",
)


def plan(profile: str) -> dict:
    if profile not in {"smoke", "pilot"}:
        raise ValueError("Unknown profile")
    smoke = profile == "smoke"
    return dict(
        version="actor_pressure_pilot_v1", profile=profile,
        contexts=1 if smoke else 4, weather_streams=1 if smoke else 2,
        train_horizon=30, test_horizon=12 if smoke else 80,
        max_candidates=8 if smoke else 32,
        actor_budgets=[1, 8] if smoke else [1, 8, 32],
        inspection_budgets=[0, 6] if smoke else [0, 3, 6],
        modes=list(MODES), train_weather_streams=2, adversarial_pressure=0.3,
        reference_draws=128, settings=asdict(MonitorSettings()),
        candidate_seed_base=600_000_000, train_partner_seed_base=601_000_000,
        test_partner_seed_base=602_000_000,
        train_weather_seed_base=603_000_000,
        test_weather_seed_base=604_000_000,
        reference_seed_base=605_000_000,
        purpose="bounded development; no population inference",
    )


def game_config(cfg: dict, seed: int, horizon: int) -> HarvestCommonsConfig:
    game = make_harvest_cfg_for_scenario(
        "forest_co_management", seed=seed, horizon=horizon,
        communication_enabled=False, side_payments_enabled=False,
    )
    game.regen_rate *= 0.85
    return game


def partners_for(cfg: dict, context: int, *, test: bool) -> list[HarvestStrategySpec]:
    base = cfg["test_partner_seed_base" if test else "train_partner_seed_base"]
    game = game_config(cfg, 0, cfg["train_horizon"])
    rng = np.random.default_rng(base + context)
    stressed = set(rng.permutation(np.arange(1, game.n_agents))[:2].tolist())
    return [(adversarial_harvest_strategy if i in stressed else cooperative_harvest_strategy)(
        f"partner_{i}", game.patch_max, rng) for i in range(1, game.n_agents)]


def candidates_for(cfg: dict, context: int) -> list[HarvestStrategySpec]:
    game = game_config(cfg, 0, cfg["train_horizon"])
    rng = np.random.default_rng(cfg["candidate_seed_base"] + context)
    parent = balanced_harvest_strategy("parent", game.patch_max, rng)
    return [mutate_harvest_strategy(
        parent=parent, strategy_id=f"candidate_{i}", patch_max=game.patch_max,
        rng=rng, adversarial_pressure=cfg["adversarial_pressure"],
    ) for i in range(cfg["max_candidates"])]


def choose_candidates(cfg: dict, context: int) -> dict:
    candidates = candidates_for(cfg, context)
    partners = partners_for(cfg, context, test=False)
    train_seeds = [cfg["train_weather_seed_base"] + 1000 * context + weather
                   for weather in range(cfg["train_weather_streams"])]
    scores = []
    for candidate in candidates:
        outcomes = []
        for seed in train_seeds:
            game = game_config(cfg, seed, cfg["train_horizon"])
            result = run_harvest_episode(game, [candidate.to_agent()] + [p.to_agent() for p in partners])
            outcomes.append(float(result["final_payoffs"][0]) -
                            game.garden_failure_penalty * int(result["garden_failure_event"]))
        scores.append(float(np.mean(outcomes)))
    selected = {str(budget): int(np.argmax(scores[:budget])) for budget in cfg["actor_budgets"]}
    return dict(context=context, candidates=[asdict(s) for s in candidates],
                train_partner_policies=[asdict(s) for s in partners],
                train_weather_seeds=train_seeds, train_scores=scores, selected=selected)


def block_name(context: int, actor_budget: int, weather: int, mode: str, inspection_budget: int) -> str:
    return f"c{context:03d}__a{actor_budget:02d}__w{weather}__{mode}__k{inspection_budget}"


def all_jobs(cfg: dict) -> list[tuple[int, int, int, str, int]]:
    jobs = []
    for context in range(cfg["contexts"]):
        for actor_budget in cfg["actor_budgets"]:
            for weather in range(cfg["weather_streams"]):
                jobs.append((context, actor_budget, weather, "none", 0))
                for inspection_budget in cfg["inspection_budgets"]:
                    for mode in cfg["modes"]:
                        jobs.append((context, actor_budget, weather, mode, inspection_budget))
    return jobs


def mask_identity(context: int, weather: int, step: int) -> str:
    # Omit actor budget so all search arms inspect the same positions.
    return f"harvest_actor_pressure__c{context}__w{weather}__t{step}"


def episode_block(cfg: dict, job: tuple, selection: dict) -> dict:
    context, actor_budget, weather, mode, inspection_budget = job
    entrant = HarvestStrategySpec(**selection["candidates"][selection["selected"][str(actor_budget)]])
    partners = partners_for(cfg, context, test=True)
    policies = [entrant] + partners
    seed = cfg["test_weather_seed_base"] + 1000 * context + weather
    game = game_config(cfg, seed, cfg["test_horizon"])
    decisions = []
    settings = MonitorSettings(**cfg["settings"])

    def monitor(_, health, requests):
        visible = mask_requests(requests, inspection_budget,
                                mask_identity(context, weather, len(decisions)))
        decision = decide_budgeted_harvest(game, health, visible, mode, settings)
        decisions.append(dict(**decision.record(),
                              inspected_indices=np.flatnonzero(np.isfinite(visible)).tolist()))
        return requests * decision.scale

    result = run_harvest_episode(game, [p.to_agent() for p in policies], record_trace=True,
                                 action_filter=None if mode == "none" else monitor)
    trace = result["episode_trace_rows"]
    if mode != "none":
        if len(trace) != len(decisions):
            raise ValueError("Monitor and trace lengths differ")
        for row, decision in zip(trace, decisions, strict=True):
            row.update(decision)
    metrics = {key: float(result[key]) for key in (
        "global_unsafe_rate", "total_welfare", "mean_patch_health", "t_end",
        "garden_failure_event", "mean_realized_harvest", "mean_prevented_harvest",
    )}
    metrics["entrant_payoff"] = float(result["final_payoffs"][0])
    metrics["request_inspections"] = int(sum(x["request_inspections"] for x in decisions))
    metrics["candidate_evaluations"] = int(sum(x["candidate_evaluations"] for x in decisions))
    metrics["component_evaluations"] = int(sum(x["component_evaluations"] for x in decisions))
    metrics["transmitted_scalars"] = int(sum(x["transmitted_scalars"] for x in decisions))
    unsafe_steps = metrics["global_unsafe_rate"] * metrics["t_end"]
    if metrics["garden_failure_event"]:
        unsafe_steps += cfg["test_horizon"] - metrics["t_end"]
    metrics["fixed_horizon_unsafe_rate"] = float(unsafe_steps / cfg["test_horizon"])
    return dict(context=context, actor_budget=actor_budget, weather=weather,
                mode=mode, inspection_budget=inspection_budget,
                config=asdict(game), policies=[asdict(p) for p in policies],
                metrics=metrics, trace=trace)


def frozen_cases(cfg: dict, output: Path) -> list[dict]:
    cases = []
    for context, actor_budget, weather, mode, inspection_budget in all_jobs(cfg):
        if mode != "none":
            continue
        block = read_json(output / "blocks" /
                          f"{block_name(context, actor_budget, weather, mode, inspection_budget)}.json.gz")
        for row in block["trace"]:
            if not row["pre_global_safe"]:
                continue
            cases.append(dict(
                case_id=f"c{context}__a{actor_budget}__w{weather}__t{row['step']}",
                context=context, actor_budget=actor_budget, weather=weather,
                step=row["step"], config=block["config"],
                state=json.loads(row["patch_health_before_json"]),
                proposals=json.loads(row["requested_fracs_json"]),
            ))
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate frozen case")
    return cases


def judge_case(cfg: dict, case: dict) -> tuple[dict, list[dict]]:
    game = HarvestCommonsConfig(**case["config"])
    state, proposals = np.asarray(case["state"], dtype=float), np.asarray(case["proposals"], dtype=float)
    if not harvest_safe(game, state):
        raise ValueError("Frozen case is not initially safe")
    seed = cfg["reference_seed_base"] + 1000 * case["context"] + 100 * case["weather"] + case["step"]
    reference = harvest_reference_risk(game, state, proposals, seed, cfg["reference_draws"],
                                      cfg["settings"]["risk_tolerance"])
    label = {key: case[key] for key in ("case_id", "context", "actor_budget", "weather", "step")}
    label.update(reference)
    decisions = []
    settings = MonitorSettings(**cfg["settings"])
    for inspection_budget in cfg["inspection_budgets"]:
        visible = mask_requests(proposals, inspection_budget,
                                mask_identity(case["context"], case["weather"], case["step"]))
        inspected = np.flatnonzero(np.isfinite(visible)).tolist()
        for mode in cfg["modes"]:
            decision = decide_budgeted_harvest(game, state, visible, mode, settings)
            decisions.append(dict(case_id=case["case_id"], context=case["context"],
                                  actor_budget=case["actor_budget"], weather=case["weather"],
                                  step=case["step"], mode=mode, inspection_budget=inspection_budget,
                                  inspected_indices=inspected, **decision.record(),
                                  **score_verdict(decision.verdict, reference["reference_label"])))
    return label, decisions


def manifest_for(cfg: dict, output: Path) -> dict:
    cfg = dict(cfg, source_sha256={name: digest(ROOT / name) for name in SOURCE_FILES},
               python=platform.python_version())
    cfg = json.loads(json.dumps(cfg, sort_keys=True))
    target = output / "manifest.json"
    if target.exists():
        old = read_json(target)
        if old["protocol"] != cfg:
            raise ValueError("Protocol/source changed; use a new output directory")
        return old
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    manifest = dict(protocol=cfg, git_head=head,
                    tracked_diff_sha256=hashlib.sha256(diff).hexdigest())
    write_json(target, manifest)
    for name in SOURCE_FILES:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    return manifest


def run(output: Path, profile: str, max_seconds: float, max_bytes: int) -> dict:
    if max_seconds <= 0 or max_bytes <= 0:
        raise ValueError("Resource caps must be positive")
    output.mkdir(parents=True, exist_ok=True)
    cfg = manifest_for(plan(profile), output)["protocol"]
    start, cpu_start = time.monotonic(), time.process_time()

    def check_cap():
        if time.monotonic() - start > max_seconds or time.process_time() - cpu_start > max_seconds:
            raise TimeoutError("Pilot time cap; completed blocks remain for inspection")
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > max_bytes:
            raise RuntimeError("Pilot artifact cap")

    selections = {}
    for context in range(cfg["contexts"]):
        check_cap()
        path = output / "selection" / f"context_{context:03d}.json.gz"
        if not path.exists():
            write_json(path, choose_candidates(cfg, context))
        selection = read_json(path)
        if selection["context"] != context or len(selection["candidates"]) != cfg["max_candidates"]:
            raise ValueError("Selection identity mismatch")
        selections[context] = selection

    jobs = all_jobs(cfg)
    for job in jobs:
        check_cap()
        path = output / "blocks" / f"{block_name(*job)}.json.gz"
        if not path.exists():
            write_json(path, episode_block(cfg, job, selections[job[0]]))
        block = read_json(path)
        if tuple(block[key] for key in ("context", "actor_budget", "weather", "mode", "inspection_budget")) != job:
            raise ValueError("Episode block identity mismatch")
    cases = frozen_cases(cfg, output)
    write_json(output / "frozen_cases.json.gz", cases)
    labels, decisions = [], []
    for case in cases:
        check_cap()
        label, judged = judge_case(cfg, case)
        labels.append(label)
        decisions.extend(judged)
    write_json(output / "labels.json.gz", labels)
    write_json(output / "decisions.json.gz", decisions)
    from experiments.archive.harvest_2026q2.analyze_actor_pressure_pilot import analyze
    summary = analyze(output, cfg)
    check_cap()
    protected = ["manifest.json", "frozen_cases.json.gz", "labels.json.gz", "decisions.json.gz",
                 "analysis/context_summary.csv", "analysis/candidate_summary.csv"]
    protected.extend(str(path.relative_to(output)) for path in sorted((output / "selection").glob("*.gz")))
    protected.extend(str(path.relative_to(output)) for path in sorted((output / "blocks").glob("*.gz")))
    completed = dict(jobs=len(jobs), frozen_cases=len(cases), decisions=len(decisions),
                     wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu_start,
                     output_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
                     summary=summary, sha256={name: digest(output / name) for name in protected})
    write_json(output / "completion.json", completed)
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-bytes", type=int, default=150_000_000)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.profile, args.max_seconds, args.max_bytes), indent=2))
