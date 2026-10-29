"""Post-hoc replay of a coupled local reviewer on frozen original proposals."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.common.run_matched_oversight import read_json
from fishery_sim.budgeted_oversight import (
    decide_budgeted_fishery, decide_budgeted_harvest, mask_requests,
)
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_oversight import FisherySnapshot
from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.oversight_protocol import MonitorSettings


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_case(case: dict, budget: int, settings: MonitorSettings):
    visible = mask_requests(case["proposals"], budget, case["case_id"])
    if case["game"] == "fishery":
        cfg = FisheryConfig(**case["config"])
        state = FisherySnapshot(**case["state"])
        return decide_budgeted_fishery(cfg, state, visible, "local_coupled", settings)
    if case["game"] == "harvest":
        cfg = HarvestCommonsConfig(**case["config"])
        return decide_budgeted_harvest(cfg, case["state"], visible, "local_coupled", settings)
    raise ValueError(f"Unsupported game {case['game']}")


def run(input_dir: Path, output_dir: Path, max_seconds: float = 180) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Preserve existing replay: {output_dir}")
    source_files = ["completion.json", "manifest.json", "frozen_cases.json.gz",
                    "labels.json.gz", "decisions.json.gz"]
    for filename in source_files:
        if not (input_dir / filename).is_file():
            raise FileNotFoundError(input_dir / filename)
    source = read_json(input_dir / "manifest.json")
    if source["protocol"]["profile"] != "confirm":
        raise ValueError("Only the completed confirmation cohort is supported")
    cases = read_json(input_dir / "frozen_cases.json.gz")
    labels = {item["case_id"]: item for item in read_json(input_dir / "labels.json.gz")}
    joint = {(item["case_id"], item["inspection_budget"]): item
             for item in read_json(input_dir / "decisions.json.gz") if item["method"] == "joint"}
    budgets = source["protocol"]["inspection_budgets"]
    if len(labels) != len(cases) or len(joint) != len(cases) * len(budgets):
        raise ValueError("Incomplete or duplicated frozen reference/decision inventory")
    settings = MonitorSettings(**source["protocol"]["settings"])
    summary = {(game, budget): dict(game=game, inspection_budget=budget, cases=0,
        initially_safe=0, safe_resolved=0, risky_resolved=0, unresolved=0,
        disagreements=0, joint_transmitted_scalars=0, local_transmitted_scalars=0)
        for game in ("fishery", "harvest") for budget in budgets}
    disagreements = []
    started = time.monotonic()
    for index, case in enumerate(cases):
        if index % 100 == 0 and time.monotonic() - started > max_seconds:
            raise TimeoutError("Replay exceeded its prespecified wall-time cap")
        label = labels[case["case_id"]]
        for budget in budgets:
            observed = joint[(case["case_id"], budget)]
            local = replay_case(case, budget, settings)
            row = summary[(case["game"], budget)]
            row["cases"] += 1
            row["joint_transmitted_scalars"] += observed["transmitted_scalars"]
            row["local_transmitted_scalars"] += local.transmitted_scalars
            if case["pre_global_safe"]:
                row["initially_safe"] += 1
                category = label["reference_label"]
                if category not in {"safe", "risky", "unresolved"}:
                    raise ValueError(f"Unexpected reference label {category}")
                row[f"{category}_resolved" if category != "unresolved" else "unresolved"] += 1
            comparable = ("scale", "verdict", "status", "candidate_evaluations",
                          "component_evaluations", "request_inspections", "predicted_safe")
            if any(getattr(local, key) != observed[key] for key in comparable):
                row["disagreements"] += 1
                disagreements.append(dict(case_id=case["case_id"], budget=budget,
                                          local=asdict(local), joint=observed))
    result = dict(protocol="notes/research_review/COUPLED_LOCAL_REPLAY_PROTOCOL_20260924.md",
                  analysis_status="post_hoc_exploratory", source=str(input_dir),
                  source_sha256={name: digest(input_dir / name) for name in source_files},
                  elapsed_seconds=time.monotonic() - started,
                  total_cases=len(cases), paired_decisions=len(cases) * len(budgets),
                  total_disagreements=len(disagreements), rows=list(summary.values()),
                  disagreement_examples=disagreements[:10])
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if (output_dir / "summary.json").stat().st_size > 50_000_000:
        raise ValueError("Output exceeds prespecified 50 MB cap")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=180)
    args = parser.parse_args()
    outcome = run(args.input_dir, args.output_dir, args.max_seconds)
    print(json.dumps({key: outcome[key] for key in ("total_cases", "paired_decisions",
                                                    "total_disagreements", "elapsed_seconds")}, indent=2))
