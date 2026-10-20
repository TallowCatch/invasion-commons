"""Bounded held-out policy development pilot; no training or remote execution."""
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

from experiments.common.run_matched_oversight import (
    digest, fishery_block, harvest_block, read_json, write_json,
)
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_oversight import FisherySnapshot, decide_fishery, reference_risk, safe
from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.oversight_protocol import (
    MonitorSettings, decide_harvest, harvest_reference_risk, harvest_safe, score_verdict,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY_GROUPS = ("useful", "stress")
HARVEST_METHODS = ("none", "local_uncertain", "local_conservative_uncertain", "joint_uncertain")
FISHERY_METHODS = ("none", "local_nominal", "local_conservative", "joint_nominal")
SOURCES = (
    "notes/research_review/HELDOUT_POLICY_PILOT_PROTOCOL.md",
    "experiments/common/run_heldout_oversight.py", "experiments/common/analyze_heldout_oversight.py",
    "experiments/common/run_matched_oversight.py", "fishery_sim/oversight_protocol.py",
    "fishery_sim/fishery_oversight.py", "fishery_sim/harvest.py",
    "fishery_sim/harvest_evolution.py", "fishery_sim/harvest_benchmarks.py",
    "fishery_sim/config.py", "fishery_sim/env.py",
)


def plan(profile):
    if profile not in {"smoke", "pilot", "mixed_pilot"}:
        raise ValueError("Unknown profile")
    smoke = profile == "smoke"
    mixed = profile == "mixed_pilot"
    return dict(version="mixed_policy_repair_v1" if mixed else "heldout_policy_pilot_v1", profile=profile,
        policy_groups=["mix2", "mix4"] if mixed else (["useful"] if smoke else list(POLICY_GROUPS)),
        harvest_regimes=["base"] if smoke else ["base", "slow_regen"],
        harvest_methods=list(HARVEST_METHODS), fishery_methods=list(FISHERY_METHODS),
        contexts=1 if smoke else 4, weather_streams=1 if smoke else 2,
        horizon=12 if smoke else 80, reference_draws=128,
        settings=asdict(MonitorSettings()),
        population_seed_bases={"mix2": 250_000_000, "mix4": 260_000_000} if mixed else
            {"useful": 210_000_000, "stress": 220_000_000},
        weather_seed_base=270_000_000 if mixed else 230_000_000,
        reference_seed_base=280_000_000 if mixed else 240_000_000,
        purpose="held-out development, no capability ordering or confirmatory inference")


def manifest_for(output, profile):
    protocol = plan(profile)
    sources = SOURCES + (("notes/research_review/MIXED_POLICY_REPAIR_PROTOCOL.md",) if profile == "mixed_pilot" else ())
    protocol["source_sha256"] = {p: digest(ROOT / p) for p in sources}
    protocol["packages"] = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas")}
    protocol["python"] = platform.python_version()
    protocol = json.loads(json.dumps(protocol, sort_keys=True))
    target = output / "manifest.json"
    if target.exists():
        old = read_json(target)
        if old["protocol"] != protocol:
            raise ValueError("Manifest differs; use a new output directory")
        return old
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    manifest = dict(protocol=protocol, git_head=head,
                    tracked_diff_sha256=hashlib.sha256(diff).hexdigest())
    write_json(target, manifest)
    for relative in sources:
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
                    for method in HARVEST_METHODS:
                        jobs.append(("harvest", group, regime, context, weather, method))
        for context in range(cfg["contexts"]):
            for method in FISHERY_METHODS:
                jobs.append(("fishery", group, "deterministic", context, 0, method))
    return jobs


def job_name(job):
    return "__".join(map(str, job))


def job_config(cfg, group):
    return dict(cfg, policy_group=group,
                population_seed_base=cfg["population_seed_bases"][group])


def execute(job, cfg):
    game, group, regime, context, weather, method = job
    settings = job_config(cfg, group)
    if game == "harvest":
        block = harvest_block(context, weather, regime, method, settings)
    else:
        block = fishery_block(context, method, settings)
    block["policy_group"] = group
    if tuple(block[k] for k in ("game", "regime", "context", "weather", "method")) != (
            game, regime, context, weather, method):
        raise ValueError("Block identity does not match job")
    return block


def verify_blocks(output, jobs):
    checksums = read_json(output / "block_checksums.json")
    if set(checksums) != {job_name(job) for job in jobs}:
        raise ValueError("Incomplete block inventory")
    for job in jobs:
        name = job_name(job)
        path = output / "blocks" / f"{name}.json.gz"
        if digest(path) != checksums[name]:
            raise ValueError(f"Corrupt block: {name}")
        block = read_json(path)
        if tuple(block[k] for k in ("game", "policy_group", "regime", "context", "weather", "method")) != job:
            raise ValueError(f"Block identity mismatch: {name}")
    return checksums


def assert_paired_policies(output, jobs):
    grouped = {}
    for game, group, regime, context, weather, method in jobs:
        name = job_name((game, group, regime, context, weather, method))
        block = read_json(output / "blocks" / f"{name}.json.gz")
        key = (game, group, regime, context, weather)
        pairing = (block["policies"], block["config"])
        if key in grouped and grouped[key] != pairing:
            raise ValueError(f"Monitor arms have different policies or transition settings: {key}")
        grouped[key] = pairing


def freeze_cases(output, jobs):
    cases = []
    for job in jobs:
        game, group, regime, context, weather, method = job
        if method != "none":
            continue
        block = read_json(output / "blocks" / f"{job_name(job)}.json.gz")
        for row in block["trace"]:
            case = dict(case_id=f"{game}__{group}__{regime}__{context}__{weather}__{row['step']}",
                        game=game, policy_group=group, regime=regime, context=context,
                        weather=weather, step=row["step"], config=block["config"],
                        pre_global_safe=int(row["pre_global_safe"]),
                        state=json.loads(row["patch_health_before_json"]) if game == "harvest" else row["state"],
                        proposals=json.loads(row["requested_fracs_json"]))
            cases.append(case)
    if len({x["case_id"] for x in cases}) != len(cases):
        raise ValueError("Duplicate frozen case ID")
    return cases


def reference_seed(case, base):
    # Identical physical states and proposed actions must share reference draws,
    # even if they were observed in different weather streams or policy contexts.
    transition_cfg = dict(case["config"])
    transition_cfg.pop("seed", None)
    content = json.dumps([case["game"], transition_cfg, case["state"], case["proposals"]],
                         sort_keys=True, separators=(",", ":"))
    return base + int(hashlib.sha256(content.encode()).hexdigest()[:8], 16)


def evaluate_case(case, cfg):
    requests = np.asarray(case["proposals"], dtype=float)
    if case["game"] == "harvest":
        game_cfg = HarvestCommonsConfig(**case["config"])
        state = np.asarray(case["state"], dtype=float)
        if int(harvest_safe(game_cfg, state)) != case["pre_global_safe"]:
            raise ValueError("Frozen Harvest safety flag mismatch")
        seed = reference_seed(case, cfg["reference_seed_base"])
        reference = harvest_reference_risk(game_cfg, state, requests, seed,
                                          cfg["reference_draws"], cfg["settings"]["risk_tolerance"])
        methods = HARVEST_METHODS
        decide = lambda method: decide_harvest(game_cfg, state, requests, method,
                                               MonitorSettings(**cfg["settings"]))
    else:
        game_cfg = FisheryConfig(**case["config"])
        state = FisherySnapshot(**case["state"])
        if int(safe(game_cfg, state)) != case["pre_global_safe"]:
            raise ValueError("Frozen Fishery safety flag mismatch")
        reference = reference_risk(game_cfg, state, requests)
        methods = FISHERY_METHODS
        decide = lambda method: decide_fishery(game_cfg, state, requests, method,
                                               MonitorSettings(**cfg["settings"]))
    label = {k: case[k] for k in ("case_id", "game", "policy_group", "regime", "context",
                                  "weather", "step", "pre_global_safe")}
    label.update(reference)
    decisions = []
    for method in methods:
        decision = decide(method)
        decisions.append(dict(case_id=case["case_id"], **decision.record(),
                              **score_verdict(decision.verdict, reference["reference_label"])))
    return label, decisions


def verify_completion(output, jobs):
    path = output / "completion.json"
    if not path.exists():
        return False
    complete = read_json(path)
    verify_blocks(output, jobs)
    for relative, expected in complete["sha256"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Corrupt completed artifact: {relative}")
    if complete["episodes"] != len(jobs):
        raise ValueError("Wrong episode count")
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
    started_wall, started_cpu = time.monotonic(), time.process_time()

    def check_cap():
        if time.monotonic() - started_wall > max_seconds or time.process_time() - started_cpu > max_seconds:
            raise TimeoutError("Held-out pilot time cap reached; completed blocks remain resumable")
        size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if size > max_bytes:
            raise RuntimeError("Held-out pilot artifact cap reached")

    (output / "blocks").mkdir(exist_ok=True)
    for job in jobs:
        check_cap()
        path = output / "blocks" / f"{job_name(job)}.json.gz"
        if path.exists():
            block = read_json(path)
            if tuple(block[k] for k in ("game", "policy_group", "regime", "context", "weather", "method")) != job:
                raise ValueError("Existing block identity mismatch")
            continue
        write_json(path, execute(job, cfg))
    write_json(output / "block_checksums.json", {
        job_name(job): digest(output / "blocks" / f"{job_name(job)}.json.gz") for job in jobs})
    verify_blocks(output, jobs)
    assert_paired_policies(output, jobs)
    check_cap()
    frozen = output / "frozen_cases.json.gz"
    if not frozen.exists():
        write_json(frozen, freeze_cases(output, jobs))
    cases = read_json(frozen)
    if cases != freeze_cases(output, jobs):
        raise ValueError("Frozen cases differ from no-intervention traces")
    labels, decisions = [], []
    for case in cases:
        check_cap()
        label, verdicts = evaluate_case(case, cfg)
        labels.append(label)
        decisions.extend(verdicts)
    # These two files are atomic and are not changed by the analyzer.
    for name, rows in (("labels.json.gz", labels), ("decisions.json.gz", decisions)):
        path = output / name
        if path.exists() and read_json(path) != rows:
            raise ValueError(f"Existing {name} differs from frozen inputs")
        if not path.exists():
            write_json(path, rows)
    from experiments.common.analyze_heldout_oversight import analyze
    analyze(output, jobs)
    check_cap()
    protected = ["manifest.json", "block_checksums.json", "frozen_cases.json.gz",
                 "labels.json.gz", "decisions.json.gz"]
    protected += [str(path.relative_to(output)) for path in sorted((output / "analysis").iterdir()) if path.is_file()]
    write_json(output / "completion.json", dict(episodes=len(jobs), frozen_cases=len(cases),
        decisions=len(decisions), wall_seconds=time.monotonic()-started_wall,
        cpu_seconds=time.process_time()-started_cpu,
        output_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        sha256={name: digest(output / name) for name in protected}))
    print(json.dumps(read_json(output / "completion.json"), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot", "mixed_pilot"), default="smoke")
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--max-bytes", type=int, default=250_000_000)
    args = parser.parse_args()
    run(args.output_dir, args.profile, args.max_seconds, args.max_bytes)


if __name__ == "__main__":
    main()
