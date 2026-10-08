"""Freeze and evaluate the bounded coverage repair; never rerun old episodes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.run_matched_oversight import (
    ROOT, FISHERY_METHODS, digest, read_json, write_json, verify_completion,
)
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_decision_cases import fishery_challenge_cases
from fishery_sim.fishery_oversight import FisherySnapshot, decide_fishery, reference_risk, safe, transition
from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.harvest_decision_cases import harvest_challenge_cases, validate_native_replay
from fishery_sim.oversight_protocol import METHODS, MonitorSettings, decide_harvest, harvest_reference_risk, harvest_safe, score_verdict

PROTOCOL = "notes/research_review/DECISION_CASE_COVERAGE_PROTOCOL.md"
SOURCES = [__file__, "experiments/analyze_oversight_decision_suite.py", "experiments/run_matched_oversight.py",
    "fishery_sim/harvest_decision_cases.py", "fishery_sim/fishery_decision_cases.py",
    "fishery_sim/oversight_protocol.py", "fishery_sim/fishery_oversight.py", "fishery_sim/harvest.py",
    "fishery_sim/env.py", "fishery_sim/config.py", "fishery_sim/metrics.py", PROTOCOL]


def fingerprint(case):
    cfg = {k: v for k, v in case["config"].items() if k not in {"seed", "horizon", "patch_init", "stock_init"}}
    data = [case["game"], case["regime"], cfg, case["state"], case["proposals"]]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def collect_cases(source):
    source = Path(source)
    manifest = read_json(source / "manifest.json")
    cfg = manifest["protocol"]
    expected = [f"harvest__{r}__{c}__{w}__{m}" for r in cfg["regimes"]
                for c in range(cfg["contexts"]) for w in range(cfg["weather_seeds"])
                for m in cfg["harvest_methods"]]
    expected += [f"fishery__{c}__{m}" for c in range(cfg["contexts"]) for m in cfg["fishery_methods"]]
    if not verify_completion(source, expected):
        raise ValueError("Source pilot is incomplete")
    cases, configurations = [], {}
    for name in sorted(expected):
        if not name.endswith("__none"):
            continue
        block = read_json(source / "blocks" / f"{name}.json.gz")
        parts = name.split("__")
        if parts[0] == "harvest":
            identity = (block["game"], block["regime"], str(block["context"]),
                        str(block["weather"]), block["method"])
        else:
            identity = (block["game"], str(block["context"]), block["method"])
        expected_scenario = cfg["scenarios"][0] if block["game"] == "harvest" else "single_stock"
        expected_regime = parts[1] if block["game"] == "harvest" else "deterministic"
        if (identity != tuple(parts) or block["game"] not in {"fishery", "harvest"}
                or block["scenario"] != expected_scenario or block["regime"] != expected_regime):
            raise ValueError(f"Source block identity differs from inventory: {name}")
        steps = [row["step"] for row in block["trace"]]
        if steps != list(range(len(steps))) or len(steps) > cfg["horizon"]:
            raise ValueError(f"Source trace has invalid step identity: {name}")
        key = (block["game"], block["regime"])
        configurations.setdefault(key, block["config"])
        for row in block["trace"]:
            if row["step"] not in cfg["frozen_steps"]:
                continue
            state = json.loads(row["patch_health_before_json"]) if block["game"] == "harvest" else row["state"]
            cases.append(dict(case_id=f"natural-{name}-{row['step']}", source="recorded",
                game=block["game"], regime=block["regime"], config=block["config"], state=state,
                proposals=json.loads(row["requested_fracs_json"]),
                design=dict(block=name, context=block["context"], weather=block["weather"], step=row["step"])))
    for (game, regime), config in sorted(configurations.items()):
        cfg_obj = HarvestCommonsConfig(**config) if game == "harvest" else FisheryConfig(**config)
        generated = harvest_challenge_cases(cfg_obj) if game == "harvest" else fishery_challenge_cases(cfg_obj)
        for case in generated:
            cases.append({**case, "case_id": f"structural-{regime}-{case['case_id']}",
                          "source": "structural", "game": game, "regime": regime, "config": config})
    if len({c["case_id"] for c in cases}) != len(cases) or len(cases) > 1600:
        raise ValueError("Duplicate case ID or case cap exceeded")
    for case in cases:
        content = fingerprint(case)
        case.update(content_sha256=content, reference_seed=160_000_000 + int(content[:12], 16),
                    parity_seed=170_000_000 + int(content[12:24], 16))
    return cases


def case_runtime(case):
    actions = np.asarray(case["proposals"], dtype=float)
    if case["game"] == "harvest":
        cfg, state = HarvestCommonsConfig(**case["config"]), np.array(case["state"])
    else:
        cfg, state = FisheryConfig(**case["config"]), FisherySnapshot(**case["state"])
    return cfg, state, actions


def reference(case, scale=1.):
    cfg, state, actions = case_runtime(case)
    if case["game"] == "harvest":
        return harvest_reference_risk(cfg, state, actions * scale, case["reference_seed"], 128, .05)
    return reference_risk(cfg, state, actions * scale)


def label_case(case):
    cfg, state, actions = case_runtime(case)
    if case["game"] == "harvest":
        parity = validate_native_replay(cfg, state, actions, case["parity_seed"])
        pre_safe = harvest_safe(cfg, state)
    else:
        future, _, extracted = transition(cfg, state, actions)
        parity = dict(next_state=asdict(future), extracted=extracted, native_parity=True)
        pre_safe = safe(cfg, state)
    return dict(case_id=case["case_id"], pre_global_safe=int(pre_safe), parity=parity, **reference(case))


def evaluate_case(case, label, shared_cache=None):
    cfg, state, actions = case_runtime(case)
    methods = METHODS if case["game"] == "harvest" else FISHERY_METHODS
    cache = {} if shared_cache is None else shared_cache
    cache.setdefault((case["content_sha256"], 1.), {k: label[k] for k in reference_keys()})
    original = label["parity"]["extracted"]
    rows = []
    for method in methods:
        decision = (decide_harvest if case["game"] == "harvest" else decide_fishery)(cfg, state, actions, method)
        key = (case["content_sha256"], decision.scale)
        if key not in cache:
            if case["game"] == "harvest":
                validate_native_replay(cfg, state, actions * decision.scale, case["parity_seed"])
            cache[key] = reference(case, decision.scale)
        executed = cache[key]
        if case["game"] == "harvest":
            retained = float(np.minimum(actions * decision.scale * cfg.max_harvest_per_agent, state).sum())
        else:
            _, _, retained = transition(cfg, state, actions * decision.scale)
        rows.append({k: case[k] for k in ("case_id", "game", "regime", "source", "content_sha256")}
                    | dict(pre_global_safe=label["pre_global_safe"], reference_label=label["reference_label"],
                        original_risk=label["risk"], **score_verdict(decision.verdict, label["reference_label"]),
                        **decision.record(), executed_risk=executed["risk"],
                        executed_native_parity=True,
                        executed_risk_lower=executed["risk_lower"], executed_risk_upper=executed["risk_upper"],
                        executed_label=executed["reference_label"], original_extraction=original,
                        retained_extraction=retained,
                        retained_fraction=retained/original if original > 0 else None))
    return rows


def reference_keys():
    return ("risk", "risk_lower", "risk_upper", "reference_label", "reference_draws", "reference_kind")


def track(output):
    """Import completed artifacts once; explicitly local, independent of shell defaults."""
    if (output / "mlflow_run.json").exists():
        return
    for key in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_TRACKING_URI"):
        os.environ.pop(key, None)
    import mlflow
    store = ROOT / "results/scientific_tracking"
    uri = "sqlite:///" + str(store / "mlflow.db")
    artifacts = (store / "artifacts").as_uri()
    mlflow.set_tracking_uri(uri)
    name = "commons-decision-coverage"
    experiment = mlflow.get_experiment_by_name(name)
    if experiment and experiment.artifact_location != artifacts:
        raise ValueError("Unexpected artifact destination")
    eid = experiment.experiment_id if experiment else mlflow.create_experiment(name, artifact_location=artifacts)
    with mlflow.start_run(experiment_id=eid, run_name=output.name) as run:
        if not run.info.artifact_uri.startswith(artifacts + "/"):
            raise ValueError("Run artifact destination is not the requested local directory")
        mlflow.set_tags({"study_stage": "coverage_development", "inference": "finite_suite_descriptive"})
        mlflow.log_params({"reference_draws": 128, "risk_tolerance": .05})
        completion = read_json(output / "completion.json")
        mlflow.log_metrics({k: v for k, v in completion.items() if isinstance(v, (int, float))})
        for file in ("manifest.json", "completion.json", "checksums.json"):
            mlflow.log_artifact(str(output / file))
        mlflow.log_artifacts(str(output / "analysis"), artifact_path="analysis")
        write_json(output / "mlflow_run.json", dict(run_id=run.info.run_id, experiment_id=eid, tracking_uri=uri))


def run(source, output, max_seconds=900, tracking=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new directory; completed cases must not be silently regenerated")
    started = time.monotonic()
    cpu = time.process_time()
    def guard():
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("Decision-suite time cap reached")
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > 250_000_000:
            raise RuntimeError("Decision-suite artifact cap reached")

    output.mkdir(parents=True)
    files = {str(Path(p).resolve().relative_to(ROOT)): digest(p) for p in SOURCES}
    manifest = dict(version="decision_coverage_v1", created_utc=datetime.now(timezone.utc).isoformat(),
        input_dir=str(source), input_manifest_sha256=digest(source / "manifest.json"),
        input_checksums_sha256=digest(source / "block_checksums.json"), source_sha256=files,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        python=sys.version, packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas")},
        reference_draws=128, risk_tolerance=.05, settings=asdict(MonitorSettings()), max_seconds=max_seconds)
    write_json(output / "manifest.json", manifest)
    for path in files:
        destination = output / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, destination)
    cases = collect_cases(source)
    write_json(output / "cases.json.gz", cases)
    labels, labeled_content = [], {}
    for case in cases:
        guard()
        content = case["content_sha256"]
        if content not in labeled_content:
            labeled_content[content] = label_case(case)
        labels.append({**labeled_content[content], "case_id": case["case_id"]})
    write_json(output / "labels.json.gz", labels)
    # Cases and reference labels are committed before any monitor is compared.
    write_json(output / "freeze.json", dict(cases_sha256=digest(output / "cases.json.gz"),
        labels_sha256=digest(output / "labels.json.gz"), frozen_utc=datetime.now(timezone.utc).isoformat()))
    decisions, shared_cache = [], {}
    for case, label in zip(cases, labels, strict=True):
        guard()
        decisions.extend(evaluate_case(case, label, shared_cache))
    if len(decisions) > 12800:
        raise RuntimeError("Judgment cap exceeded")
    write_json(output / "decisions.json.gz", decisions)
    files = ["cases.json.gz", "labels.json.gz", "decisions.json.gz", "freeze.json", "manifest.json"]
    write_json(output / "checksums.json", {f: digest(output / f) for f in files})
    write_json(output / "pending_analysis.json", dict(cases=len(cases), decisions=len(decisions),
        unique_labeled_content=len(labeled_content), unique_evaluated_content_scales=len(shared_cache)))
    from experiments.analyze_oversight_decision_suite import analyze
    analyze(output)
    guard()
    deliverables = [p for base in (output / "analysis", output / "source")
                    for p in base.rglob("*") if p.is_file()]
    write_json(output / "deliverable_checksums.json", {
        str(p.relative_to(output)): digest(p) for p in sorted(deliverables)})
    write_json(output / "completion.json", dict(cases=len(cases), decisions=len(decisions),
        simulation_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
        native_parity_cases=len(labels), unique_labeled_content=len(labeled_content),
        unique_evaluated_content_scales=len(shared_cache),
        deliverable_checksums_sha256=digest(output / "deliverable_checksums.json")))
    from experiments.analyze_oversight_decision_suite import verify_artifacts
    verify_artifacts(output)
    if tracking:
        track(output)
    print(json.dumps(read_json(output / "completion.json"), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("results/runs/matched_oversight_v1_pilot"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=900, choices=range(1, 901))
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--track-only", action="store_true")
    args = parser.parse_args()
    # Apply process-level caps as well as between-case guards, including native calls.
    resource.setrlimit(resource.RLIMIT_CPU, (args.max_seconds, args.max_seconds))
    signal.alarm(args.max_seconds)
    try:
        if args.track_only:
            from experiments.analyze_oversight_decision_suite import verify_artifacts
            verify_artifacts(args.output_dir)
            track(args.output_dir.resolve())
        else:
            run(args.source_dir, args.output_dir, args.max_seconds, args.mlflow)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
