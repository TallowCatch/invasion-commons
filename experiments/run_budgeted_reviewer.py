"""Bounded local study of request inspections and reviewer information."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from experiments.run_matched_oversight import (
    digest, fishery_block, harvest_block, read_json, write_json,
)
from experiments.run_heldout_oversight import reference_seed
from fishery_sim.budgeted_oversight import (
    MODES, decide_budgeted_fishery, decide_budgeted_harvest, mask_requests,
)
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_oversight import FisherySnapshot, reference_risk, safe, transition
from fishery_sim.harvest import HarvestCommonsConfig, HarvestStrategySpec, run_harvest_episode
from fishery_sim.oversight_protocol import (
    MonitorSettings, harvest_reference_risk, harvest_safe, score_verdict,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "notes/research_review/BUDGETED_REVIEWER_PILOT_PROTOCOL.md",
    "experiments/run_budgeted_reviewer.py", "experiments/analyze_budgeted_reviewer.py",
    "experiments/run_matched_oversight.py", "experiments/run_heldout_oversight.py",
    "fishery_sim/budgeted_oversight.py", "fishery_sim/oversight_protocol.py",
    "fishery_sim/fishery_oversight.py", "fishery_sim/harvest.py",
    "fishery_sim/harvest_evolution.py", "fishery_sim/harvest_benchmarks.py",
    "fishery_sim/config.py", "fishery_sim/env.py",
)


def plan(profile):
    if profile not in {"smoke", "pilot"}:
        raise ValueError("Unknown profile")
    smoke = profile == "smoke"
    return dict(version="budgeted_reviewer_v1", profile=profile,
        policy_groups=["mix2"] if smoke else ["mix2", "mix4"],
        harvest_regimes=["base"] if smoke else ["base", "slow_regen"],
        contexts=1 if smoke else 4, weather_streams=1 if smoke else 2,
        horizon=12 if smoke else 80, inspection_budgets=[0, 6] if smoke else [0, 3, 6],
        modes=list(MODES), settings=asdict(MonitorSettings()), reference_draws=128,
        population_seed_bases={"mix2": 300_000_000, "mix4": 310_000_000},
        weather_seed_base=320_000_000, reference_seed_base=330_000_000,
        tracking_backend="local-files", purpose="development; no scalar capability inference")


def manifest_for(output, profile):
    cfg = plan(profile)
    cfg["source_sha256"] = {path: digest(ROOT / path) for path in SOURCE_FILES}
    cfg["packages"] = {pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "scipy", "pandas")}
    cfg["python"] = platform.python_version()
    cfg = json.loads(json.dumps(cfg, sort_keys=True))
    target = output / "manifest.json"
    if target.exists():
        previous = read_json(target)
        if previous["protocol"] != cfg:
            raise ValueError("Manifest differs; use a new directory")
        return previous
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    manifest = dict(protocol=cfg, git_head=head,
                    tracked_diff_sha256=hashlib.sha256(diff).hexdigest())
    write_json(target, manifest)
    for relative in SOURCE_FILES:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return manifest


def jobs_for(cfg):
    jobs = []
    for group in cfg["policy_groups"]:
        for regime in cfg["harvest_regimes"]:
            for context in range(cfg["contexts"]):
                for weather in range(cfg["weather_streams"]):
                    jobs.append(("harvest", group, regime, context, weather, "none", 0))
                    for budget in cfg["inspection_budgets"]:
                        for mode in MODES:
                            jobs.append(("harvest", group, regime, context, weather, mode, budget))
        for context in range(cfg["contexts"]):
            jobs.append(("fishery", group, "deterministic", context, 0, "none", 0))
            for budget in cfg["inspection_budgets"]:
                for mode in MODES:
                    jobs.append(("fishery", group, "deterministic", context, 0, mode, budget))
    return jobs


def name_for(job):
    return "__".join(map(str, job))


def source_config(cfg, group):
    return dict(cfg, policy_group=group,
                population_seed_base=cfg["population_seed_bases"][group])


def case_identity(game, group, regime, context, weather, step):
    return f"{game}__{group}__{regime}__{context}__{weather}__{step}"


def harvest_monitored_block(job, cfg, baseline):
    _, group, regime, context, weather, mode, budget = job
    game_cfg = HarvestCommonsConfig(**baseline["config"])
    agents = [HarvestStrategySpec(**policy).to_agent() for policy in baseline["policies"]]
    settings = MonitorSettings(**cfg["settings"])
    decisions = []

    def monitor(_, health, requests):
        step = len(decisions)
        identity = case_identity("harvest", group, regime, context, weather, step)
        visible = mask_requests(requests, budget, identity)
        decision = decide_budgeted_harvest(game_cfg, health, visible, mode, settings)
        decisions.append(dict(**decision.record(), inspected_indices=np.flatnonzero(np.isfinite(visible)).tolist()))
        return requests * decision.scale

    result = run_harvest_episode(game_cfg, agents, record_trace=True, action_filter=monitor)
    trace = result["episode_trace_rows"]
    for row, decision in zip(trace, decisions, strict=True):
        row.update(decision)
    metrics = {key: result[key] for key in ("global_unsafe_rate", "total_welfare", "mean_patch_health",
        "t_end", "garden_failure_event", "mean_realized_harvest", "mean_prevented_harvest")}
    metrics.update(onset_count=sum(row["pre_global_safe"] and row["global_unsafe"] for row in trace),
        candidate_evaluations=sum(x["candidate_evaluations"] for x in decisions),
        component_evaluations=sum(x["component_evaluations"] for x in decisions),
        request_inspections=sum(x["request_inspections"] for x in decisions),
        transmitted_scalars=sum(x["transmitted_scalars"] for x in decisions),
        infeasible_steps=sum(x["status"] == "infeasible" for x in decisions))
    return dict(game="harvest", policy_group=group, regime=regime, context=context, weather=weather,
                mode=mode, inspection_budget=budget, config=baseline["config"],
                policies=baseline["policies"], metrics=metrics, trace=trace)


def fishery_monitored_block(job, cfg, baseline):
    _, group, regime, context, weather, mode, budget = job
    game_cfg = FisheryConfig(**baseline["config"])
    settings = MonitorSettings(**cfg["settings"])
    policies = baseline["policies"]
    low, high = np.asarray(policies["low"]), np.asarray(policies["high"])
    thresholds = np.asarray(policies["thresholds"])
    state = FisherySnapshot(game_cfg.stock_init)
    rows = []
    for step in range(game_cfg.horizon):
        requests = np.where(state.stock < thresholds, low, high)
        identity = case_identity("fishery", group, regime, context, weather, step)
        visible = mask_requests(requests, budget, identity)
        decision = decide_budgeted_fishery(game_cfg, state, visible, mode, settings)
        allowed = requests * decision.scale
        future, rewards, harvest = transition(game_cfg, state, allowed)
        _, _, unmodified_harvest = transition(game_cfg, state, requests)
        rows.append(dict(step=step, pre_global_safe=int(safe(game_cfg, state)),
            global_unsafe=int(not safe(game_cfg, future)), state=asdict(state),
            next_state=asdict(future), requested_fracs_json=json.dumps(requests.tolist()),
            allowed_fracs_json=json.dumps(allowed.tolist()),
            mean_patch_health_after=future.stock, realized_harvest=harvest,
            prevented_harvest=unmodified_harvest-harvest, welfare=float(rewards.sum()),
            inspected_indices=np.flatnonzero(np.isfinite(visible)).tolist(), **decision.record()))
        state = future
        if state.collapsed:
            break
    metrics = dict(global_unsafe_rate=float(np.mean([row["global_unsafe"] for row in rows])),
        total_welfare=float(sum(row["welfare"] for row in rows)),
        mean_patch_health=float(np.mean([row["mean_patch_health_after"] for row in rows])),
        t_end=len(rows), garden_failure_event=int(state.collapsed),
        mean_realized_harvest=float(np.mean([row["realized_harvest"] for row in rows])),
        mean_prevented_harvest=float(np.mean([row["prevented_harvest"] for row in rows])),
        onset_count=sum(row["pre_global_safe"] and row["global_unsafe"] for row in rows),
        candidate_evaluations=sum(row["candidate_evaluations"] for row in rows),
        component_evaluations=sum(row["component_evaluations"] for row in rows),
        request_inspections=sum(row["request_inspections"] for row in rows),
        transmitted_scalars=sum(row["transmitted_scalars"] for row in rows),
        infeasible_steps=sum(row["status"] == "infeasible" for row in rows))
    return dict(game="fishery", policy_group=group, regime=regime, context=context, weather=weather,
                mode=mode, inspection_budget=budget, config=baseline["config"],
                policies=policies, metrics=metrics, trace=rows)


def execute(job, cfg, baseline=None):
    game, group, regime, context, weather, mode, budget = job
    if mode == "none":
        source = source_config(cfg, group)
        block = (harvest_block(context, weather, regime, "none", source) if game == "harvest"
                 else fishery_block(context, "none", source))
        block.update(mode="none", inspection_budget=0, policy_group=group)
        return block
    if baseline is None:
        raise ValueError("Monitored episode requires its paired no-intervention baseline")
    return (harvest_monitored_block(job, cfg, baseline) if game == "harvest"
            else fishery_monitored_block(job, cfg, baseline))


def check_blocks(output, jobs):
    checksums = read_json(output / "block_checksums.json")
    if set(checksums) != {name_for(job) for job in jobs}:
        raise ValueError("Incomplete block inventory")
    for job in jobs:
        name = name_for(job)
        path = output / "blocks" / f"{name}.json.gz"
        if digest(path) != checksums[name]:
            raise ValueError(f"Corrupt block {name}")
        block = read_json(path)
        identity = tuple(block[k] for k in ("game", "policy_group", "regime", "context",
                                             "weather", "mode", "inspection_budget"))
        if identity != job:
            raise ValueError(f"Block identity mismatch {name}")
        baseline = read_json(output / "blocks" / f"{name_for((*job[:5], 'none', 0))}.json.gz")
        if block["policies"] != baseline["policies"] or block["config"] != baseline["config"]:
            raise ValueError(f"Unpaired policies or game settings {name}")
    return checksums


def freeze_cases(output, jobs):
    cases = []
    for job in jobs:
        game, group, regime, context, weather, mode, _ = job
        if mode != "none":
            continue
        block = read_json(output / "blocks" / f"{name_for(job)}.json.gz")
        for trace in block["trace"]:
            cases.append(dict(case_id=case_identity(game, group, regime, context, weather, trace["step"]),
                game=game, policy_group=group, regime=regime, context=context, weather=weather,
                step=trace["step"], config=block["config"],
                pre_global_safe=int(trace["pre_global_safe"]),
                state=json.loads(trace["patch_health_before_json"]) if game == "harvest" else trace["state"],
                proposals=json.loads(trace["requested_fracs_json"])))
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate frozen case identity")
    return cases


def judge_case(case, cfg):
    values = np.asarray(case["proposals"], dtype=float)
    if case["game"] == "harvest":
        game_cfg = HarvestCommonsConfig(**case["config"])
        state = np.asarray(case["state"], dtype=float)
        if int(harvest_safe(game_cfg, state)) != case["pre_global_safe"]:
            raise ValueError("Frozen Harvest state safety mismatch")
        reference = harvest_reference_risk(game_cfg, state, values,
            reference_seed(case, cfg["reference_seed_base"]), cfg["reference_draws"],
            cfg["settings"]["risk_tolerance"])
        decide = lambda masked, mode: decide_budgeted_harvest(game_cfg, state, masked, mode,
                                                               MonitorSettings(**cfg["settings"]))
    else:
        game_cfg = FisheryConfig(**case["config"])
        state = FisherySnapshot(**case["state"])
        if int(safe(game_cfg, state)) != case["pre_global_safe"]:
            raise ValueError("Frozen Fishery state safety mismatch")
        reference = reference_risk(game_cfg, state, values)
        decide = lambda masked, mode: decide_budgeted_fishery(game_cfg, state, masked, mode,
                                                               MonitorSettings(**cfg["settings"]))
    label = {k: case[k] for k in ("case_id", "game", "policy_group", "regime", "context",
                                  "weather", "step", "pre_global_safe")}
    label.update(reference)
    decisions = []
    for budget in cfg["inspection_budgets"]:
        visible = mask_requests(values, budget, case["case_id"])
        for mode in MODES:
            decision = decide(visible, mode)
            decisions.append(dict(case_id=case["case_id"], inspection_budget=budget,
                inspected_indices=np.flatnonzero(np.isfinite(visible)).tolist(),
                **decision.record(), **score_verdict(decision.verdict, reference["reference_label"])))
    return label, decisions


def verify_completion(output, jobs):
    path = output / "completion.json"
    if not path.exists():
        return False
    record = read_json(path)
    check_blocks(output, jobs)
    for relative, expected in record["sha256"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Corrupt completed artifact: {relative}")
    if record["episodes"] != len(jobs):
        raise ValueError("Wrong episode inventory")
    return True


def run(output, profile="smoke", max_seconds=900, max_bytes=250_000_000):
    if max_seconds <= 0 or max_bytes <= 0:
        raise ValueError("Resource caps must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = manifest_for(output, profile)["protocol"]
    jobs = jobs_for(cfg)
    if verify_completion(output, jobs):
        print(json.dumps(read_json(output / "completion.json"), indent=2))
        return
    start, cpu = time.monotonic(), time.process_time()

    def cap():
        if time.monotonic()-start > max_seconds or time.process_time()-cpu > max_seconds:
            raise TimeoutError("Reviewer pilot time cap reached; atomic blocks remain resumable")
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > max_bytes:
            raise RuntimeError("Reviewer pilot artifact cap reached")

    (output / "blocks").mkdir(exist_ok=True)
    for job in jobs:
        cap()
        path = output / "blocks" / f"{name_for(job)}.json.gz"
        if path.exists():
            block = read_json(path)
            if tuple(block[k] for k in ("game", "policy_group", "regime", "context",
                                         "weather", "mode", "inspection_budget")) != job:
                raise ValueError("Existing block identity mismatch")
            continue
        baseline = None
        if job[5] != "none":
            baseline = read_json(output / "blocks" / f"{name_for((*job[:5], 'none', 0))}.json.gz")
        write_json(path, execute(job, cfg, baseline))
    write_json(output / "block_checksums.json", {
        name_for(job): digest(output / "blocks" / f"{name_for(job)}.json.gz") for job in jobs})
    check_blocks(output, jobs)
    cap()
    frozen_path = output / "frozen_cases.json.gz"
    frozen = freeze_cases(output, jobs)
    if frozen_path.exists():
        if read_json(frozen_path) != frozen:
            raise ValueError("Frozen cases differ from source episodes")
    else:
        write_json(frozen_path, frozen)
    labels, decisions = [], []
    for case in frozen:
        cap()
        label, verdicts = judge_case(case, cfg)
        labels.append(label)
        decisions.extend(verdicts)
    for name, rows in (("labels.json.gz", labels), ("decisions.json.gz", decisions)):
        target = output / name
        if target.exists():
            if read_json(target) != rows:
                raise ValueError(f"Existing {name} differs from frozen inputs")
        else:
            write_json(target, rows)
    from experiments.analyze_budgeted_reviewer import analyze
    analyze(output, jobs)
    cap()
    protected = ["manifest.json", "block_checksums.json", "frozen_cases.json.gz",
                 "labels.json.gz", "decisions.json.gz"]
    protected += [str(path.relative_to(output)) for path in sorted((output / "analysis").iterdir()) if path.is_file()]
    write_json(output / "completion.json", dict(episodes=len(jobs), frozen_cases=len(frozen),
        decisions=len(decisions), wall_seconds=time.monotonic()-start,
        cpu_seconds=time.process_time()-cpu,
        output_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        sha256={name: digest(output / name) for name in protected}))
    print(json.dumps(read_json(output / "completion.json"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--max-bytes", type=int, default=250_000_000)
    args = parser.parse_args()
    run(args.output_dir, args.profile, args.max_seconds, args.max_bytes)
