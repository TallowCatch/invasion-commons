"""Export the small aggregate confirmation tables used by the paper figure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from experiments.paper_v5.analyze_reviewer_longrun import analyze


TABLES = ("context_decision_counts.csv", "primary_decision_quality.csv", "outcomes.csv", "episodes.csv")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source: Path, destination: Path) -> dict:
    if not (source / "completion.json").exists() or not (source / "manifest.json").exists():
        raise ValueError("Source must be a completed, manifested confirmation run")
    protocol = json.loads((source / "manifest.json").read_text())["protocol"]
    if protocol.get("profile") != "confirm" or protocol.get("contexts") != 64:
        raise ValueError("Unexpected source cohort")
    frames = {name: pd.read_csv(source / "analysis" / name) for name in TABLES}
    if len(frames["context_decision_counts.csv"]) != 2 * 64 * 3 * 3:
        raise ValueError("Incomplete context-level decision table")
    if len(frames["primary_decision_quality.csv"]) != 2 * 3 * 3:
        raise ValueError("Incomplete primary decision table")
    if len(frames["episodes.csv"]) != 1920:
        raise ValueError("Incomplete episode table")
    target = destination / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        shutil.copy2(source / "analysis" / name, target / name)
    analyze(target / "episodes.csv", target)
    derived = ("longrun_context_differences.csv", "longrun_paired_summary.csv")
    record = dict(status="curated_summary_and_episode_outcomes", independent_contexts_per_game=64,
                  source_completion_sha256=sha256(source / "completion.json"),
                  source_manifest_sha256=sha256(source / "manifest.json"),
                  tables={name: dict(sha256=sha256(target / name), rows=len(frames[name]))
                          for name in TABLES},
                  derived_tables={name: dict(sha256=sha256(target / name),
                                             rows=len(pd.read_csv(target / name)))
                                  for name in derived})
    (destination / "provenance.json").write_text(json.dumps(record, indent=2) + "\n",
                                                 encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.input_dir, args.output_dir), indent=2))
