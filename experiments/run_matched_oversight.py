"""Bounded, resumable development experiment; no LLM calls or remote execution.

Run --profile smoke before --profile pilot. The latter runs 512 Harvest episodes
and 16 deterministic Fishery episodes; it is not a confirmatory publication run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_oversight import FisherySnapshot, decide_fishery, reference_risk, safe, transition
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_scenario
from fishery_sim.harvest_evolution import (
    adversarial_harvest_strategy, build_initial_harvest_population,
    cooperative_harvest_strategy,
)
from fishery_sim.oversight_protocol import (
    METHODS, MonitorSettings, decide_harvest, harvest_reference_risk, harvest_safe, score_verdict,
)

ROOT = Path(__file__).resolve().parents[1]
FISHERY_METHODS = ("none", "local_nominal", "local_conservative", "joint_nominal")
SOURCE_FILES = (
    "experiments/run_matched_oversight.py", "fishery_sim/oversight_protocol.py",
    "fishery_sim/fishery_oversight.py", "fishery_sim/harvest.py", "fishery_sim/env.py",
    "fishery_sim/harvest_evolution.py", "fishery_sim/harvest_benchmarks.py",
    "fishery_sim/config.py", "fishery_sim/metrics.py", "experiments/analyze_matched_oversight.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf8") as f:
        json.dump(data, f, allow_nan=False, sort_keys=True)
    temporary.replace(path)


def read_json(path):
    with (gzip.open if Path(path).suffix == ".gz" else open)(path, "rt", encoding="utf8") as f:
        return json.load(f)


def verify_completion(output, expected_names):
    """Completed runs are immutable; do not replace their original timing record."""
    if not (output / "completion.json").exists():
        return False
    completion = read_json(output / "completion.json")
    checksums = read_json(output / "block_checksums.json")
    if (set(checksums) != set(expected_names)
            or completion["expected_blocks"] != len(expected_names)
            or completion["completed_blocks"] != len(expected_names)):
        raise ValueError("Completed run has an incomplete block inventory")
    for name, checksum in checksums.items():
        if digest(output / "blocks" / f"{name}.json.gz") != checksum:
            raise ValueError(f"Corrupt completed block: {name}")
    frozen_path = output / "frozen_judgments.json.gz"
    if ("frozen_sha256" in completion
            and digest(frozen_path) != completion["frozen_sha256"]):
        raise ValueError("Corrupt completed frozen judgments")
    # Older pilot records have a row count but no frozen-file checksum.
    if len(read_json(frozen_path)) != completion["frozen_judgment_rows"]:
        raise ValueError("Completed run has incomplete frozen judgments")
    return True


def make_manifest(output, profile, reference_draws, settings):
    protocol = dict(version="matched_oversight_v1", profile=profile, reference_draws=reference_draws,
                    settings=asdict(settings), contexts=1 if profile == "smoke" else 4,
                    weather_seeds=1 if profile == "smoke" else 8,
                    horizon=12 if profile == "smoke" else 80,
                    scenarios=["forest_co_management"], regimes=["base", "slow_regen"],
                    harvest_methods=list(METHODS), fishery_methods=list(FISHERY_METHODS),
                    population_seed_base=130_000_000, weather_seed_base=140_000_000,
                    reference_seed_base=150_000_000, frozen_steps=[0, 5, 15, 30, 50, 70],
                    frozen_proposal_scales=[1.0, .5, .25, 0.0],
                    purpose="development, no significance or universal capability claim",
                    source_sha256={p: digest(ROOT / p) for p in SOURCE_FILES},
                    packages={p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas"]},
                    python=platform.python_version())
    # JSON normalizes tuples; compare canonical data rather than Python types.
    protocol = json.loads(json.dumps(protocol))
    target = output / "manifest.json"
    if target.exists():
        previous = read_json(target)
        if previous["protocol"] != protocol:
            raise ValueError("Manifest differs: use a new directory, never overwrite old evidence")
        return previous
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked_diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    manifest = dict(protocol=protocol, git_head=commit, tracked_diff_sha256=hashlib.sha256(tracked_diff).hexdigest())
    write_json(target, manifest)
    for relative in SOURCE_FILES:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return manifest


def harvest_block(context, weather, regime, method, config):
    cfg = make_harvest_cfg_for_scenario("forest_co_management", horizon=config["horizon"],
        seed=config["weather_seed_base"] + 1000 * context + weather,
        communication_enabled=False, side_payments_enabled=False)
    if regime == "slow_regen":
        cfg.regen_rate *= .85
    rng = np.random.default_rng(config["population_seed_base"] + context)
    policy_group = config.get("policy_group", "balanced")
    if policy_group == "balanced":
        specs = build_initial_harvest_population(cfg.n_agents, cfg.patch_max, rng, "balanced")
    elif policy_group in {"useful", "stress"}:
        generator = cooperative_harvest_strategy if policy_group == "useful" else adversarial_harvest_strategy
        specs = [generator(f"heldout_{i}", cfg.patch_max, rng) for i in range(cfg.n_agents)]
    elif policy_group in {"mix2", "mix4"}:
        stressed = set(rng.permutation(cfg.n_agents)[:int(policy_group[-1])].tolist())
        specs = [(adversarial_harvest_strategy if i in stressed else cooperative_harvest_strategy)(
            f"heldout_{i}", cfg.patch_max, rng) for i in range(cfg.n_agents)]
    else:
        raise ValueError(f"Unknown policy group: {policy_group}")
    settings = MonitorSettings(**{**config["settings"], "scales": tuple(config["settings"]["scales"])})
    decisions = []

    def monitor(c, health, requests):
        decision = decide_harvest(c, health, requests, method, settings)
        decisions.append(decision.record())
        return requests * decision.scale

    result = run_harvest_episode(cfg, [s.to_agent() for s in specs], record_trace=True, action_filter=monitor)
    trace = result["episode_trace_rows"]
    for row, decision in zip(trace, decisions, strict=True):
        row.update(decision)
    # Costs are computational quantities, not arbitrary money subtracted from welfare.
    metrics = {k: result[k] for k in ["global_unsafe_rate", "total_welfare", "mean_patch_health", "t_end",
                "garden_failure_event", "mean_realized_harvest", "mean_prevented_harvest"]}
    metrics.update(onset_count=sum(r["pre_global_safe"] and r["global_unsafe"] for r in trace),
        candidate_evaluations=sum(d["candidate_evaluations"] for d in decisions),
        component_evaluations=sum(d["component_evaluations"] for d in decisions),
        infeasible_steps=sum(d["status"] == "infeasible" for d in decisions),
        budget_exhausted_steps=sum(d["status"] == "budget_exhausted" for d in decisions))
    return dict(game="harvest", scenario="forest_co_management", regime=regime, context=context,
                weather=weather, method=method, config=asdict(cfg),
                policies=[asdict(s) for s in specs], metrics=metrics, trace=trace)


def fishery_block(context, method, config):
    cfg = FisheryConfig(n_agents=6, horizon=config["horizon"], stock_init=70, stock_max=100,
                        regen_rate=.7, obs_noise_std=0, max_harvest_per_agent=6)
    settings = MonitorSettings(**{**config["settings"], "scales": tuple(config["settings"]["scales"])})
    rng = np.random.default_rng(config["population_seed_base"] + context)
    # Inspectable heterogeneous threshold policies; no actor capability ordering.
    policy_group = config.get("policy_group", "balanced")
    ranges = {
        "balanced": ((.08, .35), (.35, .95)),
        "useful": ((.05, .20), (.20, .40)),
        "stress": ((.45, .65), (.75, .95)),
    }
    if policy_group in ranges:
        low = rng.uniform(*ranges[policy_group][0], cfg.n_agents)
        high = rng.uniform(*ranges[policy_group][1], cfg.n_agents)
    elif policy_group in {"mix2", "mix4"}:
        stressed = set(rng.permutation(cfg.n_agents)[:int(policy_group[-1])].tolist())
        low = np.array([rng.uniform(*ranges["stress" if i in stressed else "useful"][0])
                        for i in range(cfg.n_agents)])
        high = np.array([rng.uniform(*ranges["stress" if i in stressed else "useful"][1])
                         for i in range(cfg.n_agents)])
    else:
        raise ValueError(f"Unknown policy group: {policy_group}")
    thresholds = rng.uniform(25, 65, cfg.n_agents)
    state = FisherySnapshot(cfg.stock_init)
    rows = []
    for t in range(cfg.horizon):
        requests = np.where(state.stock < thresholds, low, high)
        decision = decide_fishery(cfg, state, requests, method, settings)
        allowed = requests * decision.scale
        future, payoffs, harvest = transition(cfg, state, allowed)
        unmodified, _, available_harvest = transition(cfg, state, requests)
        rows.append(dict(step=t, pre_global_safe=int(safe(cfg, state)), global_unsafe=int(not safe(cfg, future)),
            state=asdict(state), next_state=asdict(future), requested_fracs_json=json.dumps(requests.tolist()),
            allowed_fracs_json=json.dumps(allowed.tolist()), mean_patch_health_after=future.stock,
            realized_harvest=harvest, prevented_harvest=available_harvest-harvest,
            welfare=float(payoffs.sum()), **decision.record()))
        state = future
        if state.collapsed:
            break
    metrics = dict(global_unsafe_rate=float(np.mean([r["global_unsafe"] for r in rows])),
        total_welfare=sum(r["welfare"] for r in rows), mean_patch_health=float(np.mean([r["mean_patch_health_after"] for r in rows])),
        t_end=len(rows), garden_failure_event=int(state.collapsed), mean_realized_harvest=float(np.mean([r["realized_harvest"] for r in rows])),
        mean_prevented_harvest=float(np.mean([r["prevented_harvest"] for r in rows])),
        onset_count=sum(r["pre_global_safe"] and r["global_unsafe"] for r in rows),
        candidate_evaluations=sum(r["candidate_evaluations"] for r in rows),
        component_evaluations=sum(r["component_evaluations"] for r in rows),
        infeasible_steps=sum(r["status"] == "infeasible" for r in rows),
        budget_exhausted_steps=sum(r["status"] == "budget_exhausted" for r in rows))
    return dict(game="fishery", scenario="single_stock", regime="deterministic", context=context,
        weather=0, method=method, config=asdict(cfg), metrics=metrics, trace=rows,
        policies=dict(low=low.tolist(), high=high.tolist(), thresholds=thresholds.tolist()))


def frozen_judgments(block, config):
    """Every monitor judges the same no-intervention states and proposals.

    These are not independent episode trials. Stratify by initially safe versus
    already unsafe, and aggregate within population context in analysis.
    """
    settings = MonitorSettings(**{**config["settings"], "scales": tuple(config["settings"]["scales"])})
    rows = []
    for trace in block["trace"]:
        if trace["step"] not in config["frozen_steps"]:
            continue
        for proposal_scale in config["frozen_proposal_scales"]:
            rows.extend(judge_snapshot(block, trace, proposal_scale, config, settings))
    return rows


def judge_snapshot(block, trace, proposal_scale, config, settings):
    from fishery_sim.harvest import HarvestCommonsConfig
    requests = np.array(json.loads(trace["requested_fracs_json"])) * proposal_scale
    if block["game"] == "harvest":
        cfg = HarvestCommonsConfig(**block["config"])
        state = np.array(json.loads(trace["patch_health_before_json"]))
        seed = config["reference_seed_base"] + block["context"] * 10000 + block["weather"] * 100 + trace["step"]
        ref = harvest_reference_risk(cfg, state, requests, seed, config["reference_draws"], settings.risk_tolerance)
        pre_safe = harvest_safe(cfg, state)
        methods = METHODS
        decide = lambda m: decide_harvest(cfg, state, requests, m, settings)
        state_record = state.tolist()
    else:
        cfg = FisheryConfig(**block["config"])
        state = FisherySnapshot(**trace["state"])
        ref = reference_risk(cfg, state, requests)
        pre_safe = safe(cfg, state)
        methods = FISHERY_METHODS
        decide = lambda m: decide_fishery(cfg, state, requests, m, settings)
        state_record = asdict(state)
    rows = []
    for method in methods:
        decision = decide(method)
        rows.append(dict(game=block["game"], scenario=block["scenario"], regime=block["regime"],
            context=block["context"], weather=block["weather"], step=trace["step"],
            proposal_scale=proposal_scale,
            pre_global_safe=int(pre_safe), state=state_record, proposals=requests.tolist(),
            **decision.record(), **ref, **score_verdict(decision.verdict, ref["reference_label"])))
    return rows


def log_mlflow(output, manifest):
    # Local paths are explicit. Never inherit a remote tracking endpoint/token.
    import mlflow
    mlflow.set_tracking_uri("sqlite:///" + str((ROOT / "results/scientific_tracking/mlflow.db").resolve()))
    (ROOT / "results/scientific_tracking").mkdir(parents=True, exist_ok=True)
    name = "commons-matched-oversight"
    existing = mlflow.get_experiment_by_name(name)
    local_artifacts = (ROOT / "results/scientific_tracking/artifacts").resolve().as_uri()
    if existing and existing.artifact_location != local_artifacts:
        raise ValueError("Existing experiment artifact destination is not the intended local directory")
    experiment_id = existing.experiment_id if existing else mlflow.create_experiment(
        name, artifact_location=local_artifacts)
    with mlflow.start_run(experiment_id=experiment_id, run_name=output.name) as run:
        mlflow.log_params({k: manifest["protocol"][k] for k in ["version", "profile", "contexts", "horizon", "reference_draws"]})
        mlflow.set_tags({"study_stage": "development_pilot", "git_head": manifest["git_head"],
                         "source_hash_manifest": "manifest.json", "claims": "no_confirmatory_inference"})
        completion = read_json(output / "completion.json")
        mlflow.log_metrics({k: v for k, v in completion.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(output / "manifest.json"))
        mlflow.log_artifact(str(output / "completion.json"))
        mlflow.log_artifact(str(output / "block_checksums.json"))
        import pandas as pd
        for _, row in pd.read_csv(output / "analysis/outcomes.csv").iterrows():
            prefix = ".".join(str(row[k]) for k in ["game", "regime", "method"])
            for key in ["global_unsafe_rate", "total_welfare", "mean_patch_health", "checks_per_step"]:
                mlflow.log_metric(prefix + "." + key, float(row[key]))
        mlflow.log_artifacts(str(output / "analysis"), artifact_path="analysis")
        write_json(output / "mlflow_run.json", dict(run_id=run.info.run_id, experiment_id=experiment_id,
            tracking_uri=mlflow.get_tracking_uri()))


def run(output, profile="smoke", reference_draws=128, max_seconds=900, tracking=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(output, profile, reference_draws, MonitorSettings())
    cfg = manifest["protocol"]
    blocks = output / "blocks"
    blocks.mkdir(exist_ok=True)
    jobs = []
    for regime in cfg["regimes"]:
        for context in range(cfg["contexts"]):
            for weather in range(cfg["weather_seeds"]):
                for method in METHODS:
                    jobs.append((f"harvest__{regime}__{context}__{weather}__{method}",
                        lambda c=context, w=weather, r=regime, m=method: harvest_block(c, w, r, m, cfg)))
    for context in range(cfg["contexts"]):
        for method in FISHERY_METHODS:
            jobs.append((f"fishery__{context}__{method}", lambda c=context, m=method: fishery_block(c, m, cfg)))
    if verify_completion(output, [name for name, _ in jobs]):
        if not (output / "analysis/experiment.md").exists():
            from experiments.analyze_matched_oversight import analyze
            analyze(output)
        if tracking and not (output / "mlflow_run.json").exists():
            log_mlflow(output, manifest)
        print(json.dumps(read_json(output / "completion.json"), indent=2), flush=True)
        return
    started, new_blocks = time.monotonic(), 0
    for name, execute in jobs:
        target = blocks / f"{name}.json.gz"
        if target.exists():
            # A truncated/corrupt block must not silently count as complete.
            read_json(target)
            continue
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("Pilot time budget reached; completed atomic blocks are resumable")
        write_json(target, execute())
        new_blocks += 1
        if new_blocks % 32 == 0:
            print(f"Completed {new_blocks} new blocks in {time.monotonic()-started:.1f}s", flush=True)
    frozen = []
    for name, _ in jobs:
        if name.endswith("__none"):
            frozen.extend(frozen_judgments(read_json(blocks / f"{name}.json.gz"), cfg))
    write_json(output / "frozen_judgments.json.gz", frozen)
    checksums = {name: digest(blocks / f"{name}.json.gz") for name, _ in jobs}
    write_json(output / "block_checksums.json", checksums)
    write_json(output / "completion.json", dict(expected_blocks=len(jobs), completed_blocks=len(checksums),
        new_blocks=new_blocks, invocation_seconds=time.monotonic()-started, frozen_judgment_rows=len(frozen),
        frozen_sha256=digest(output / "frozen_judgments.json.gz"),
        output_bytes=sum(p.stat().st_size for p in blocks.glob("*.json.gz"))))
    from experiments.analyze_matched_oversight import analyze
    analyze(output)
    if tracking and not (output / "mlflow_run.json").exists():
        log_mlflow(output, manifest)
    print(json.dumps(read_json(output / "completion.json"), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--reference-draws", type=int, default=128)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--mlflow", action="store_true", help="Log to repo-local SQLite only")
    args = parser.parse_args()
    if args.reference_draws < 1 or args.max_seconds <= 0:
        parser.error("Draws and time budget must be positive")
    run(args.output_dir, args.profile, args.reference_draws, args.max_seconds, args.mlflow)


if __name__ == "__main__":
    main()
