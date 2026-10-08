"""Fresh-seed, bounded confirmation of the budgeted reviewer comparison."""
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

from experiments.run_budgeted_reviewer import (
    ROOT, check_blocks, execute, freeze_cases, judge_case, name_for,
    verify_completion,
)
from experiments.run_matched_oversight import digest, read_json, write_json
from fishery_sim.budgeted_oversight import MODES
from fishery_sim.oversight_protocol import MonitorSettings


SOURCE_FILES = (
    "notes/research_review/BUDGETED_REVIEWER_CONFIRMATION_PROTOCOL.md",
    "experiments/run_budgeted_reviewer_confirmation.py",
    "experiments/analyze_budgeted_reviewer_confirmation.py",
    "experiments/run_budgeted_reviewer.py",
    "experiments/run_matched_oversight.py",
    "experiments/run_heldout_oversight.py",
    "fishery_sim/budgeted_oversight.py",
    "fishery_sim/oversight_protocol.py",
    "fishery_sim/fishery_oversight.py",
    "fishery_sim/harvest.py",
    "fishery_sim/harvest_evolution.py",
    "fishery_sim/harvest_benchmarks.py",
    "fishery_sim/config.py",
    "fishery_sim/env.py",
)


def plan(profile: str) -> dict:
    if profile not in {"smoke", "confirm"}:
        raise ValueError("Unknown profile")
    smoke = profile == "smoke"
    return dict(
        version="budgeted_reviewer_confirmation_v1", profile=profile,
        cells=[dict(game="fishery", policy_group="mix4", regime="deterministic"),
               dict(game="harvest", policy_group="mix2", regime="slow_regen")],
        contexts=1 if smoke else 64, weather_streams=1 if smoke else 2,
        horizon=12 if smoke else 80,
        inspection_budgets=[0, 6] if smoke else [0, 3, 6],
        modes=list(MODES), settings=asdict(MonitorSettings()), reference_draws=128,
        population_seed_bases={"mix4": 500_000_000, "mix2": 510_000_000},
        weather_seed_base=520_000_000, reference_seed_base=530_000_000,
        tracking_backend="local-files", purpose="fresh-seed confirmation; two selected strata",
    )


def jobs_for(cfg: dict) -> list[tuple]:
    jobs = []
    for cell in cfg["cells"]:
        game, group, regime = cell["game"], cell["policy_group"], cell["regime"]
        weather_count = cfg["weather_streams"] if game == "harvest" else 1
        for context in range(cfg["contexts"]):
            for weather in range(weather_count):
                jobs.append((game, group, regime, context, weather, "none", 0))
                for budget in cfg["inspection_budgets"]:
                    for mode in MODES:
                        jobs.append((game, group, regime, context, weather, mode, budget))
    return jobs


def manifest_for(output: Path, profile: str) -> dict:
    cfg = plan(profile)
    cfg["source_sha256"] = {name: digest(ROOT / name) for name in SOURCE_FILES}
    cfg["packages"] = {name: importlib.metadata.version(name)
                       for name in ("numpy", "scipy", "pandas")}
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


def run(output: Path, profile: str, max_seconds: float, max_bytes: int) -> dict:
    if max_seconds <= 0 or max_bytes <= 0:
        raise ValueError("Resource caps must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = manifest_for(output, profile)["protocol"]
    jobs = jobs_for(cfg)
    if verify_completion(output, jobs):
        return read_json(output / "completion.json")
    start, cpu = time.monotonic(), time.process_time()

    def cap() -> None:
        if time.monotonic() - start > max_seconds or time.process_time() - cpu > max_seconds:
            raise TimeoutError("Confirmation time cap; atomic blocks remain resumable")
        size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if size > max_bytes:
            raise RuntimeError("Confirmation artifact cap")

    (output / "blocks").mkdir(exist_ok=True)
    for job in jobs:
        cap()
        path = output / "blocks" / f"{name_for(job)}.json.gz"
        if path.exists():
            block = read_json(path)
            actual = tuple(block[key] for key in ("game", "policy_group", "regime", "context",
                                                  "weather", "mode", "inspection_budget"))
            if actual != job:
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
    frozen = freeze_cases(output, jobs)
    frozen_path = output / "frozen_cases.json.gz"
    if frozen_path.exists():
        if read_json(frozen_path) != frozen:
            raise ValueError("Frozen cases differ from episode sources")
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
                raise ValueError(f"Existing {name} differs")
        else:
            write_json(target, rows)
    from experiments.analyze_budgeted_reviewer_confirmation import analyze
    analyze(output, jobs)
    cap()
    protected = ["manifest.json", "block_checksums.json", "frozen_cases.json.gz",
                 "labels.json.gz", "decisions.json.gz"]
    protected.extend(str(path.relative_to(output)) for path in sorted((output / "analysis").iterdir())
                     if path.is_file())
    write_json(output / "completion.json", dict(
        episodes=len(jobs), frozen_cases=len(frozen), decisions=len(decisions),
        wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu,
        output_bytes=sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        sha256={name: digest(output / name) for name in protected}))
    if not verify_completion(output, jobs):
        raise ValueError("Completion verification failed")
    return read_json(output / "completion.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "confirm"), default="smoke")
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--max-bytes", type=int, default=250_000_000)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.profile, args.max_seconds, args.max_bytes), indent=2))
