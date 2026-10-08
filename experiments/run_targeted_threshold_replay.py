from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
from pathlib import Path

import pandas as pd


METRICS = [
    "test_global_unsafe_rate_mean",
    "test_local_pass_global_fail_rate_mean",
    "test_mean_patch_health_mean",
    "test_garden_failure_mean",
    "test_mean_welfare_mean",
    "test_governance_budget_spent_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a guarded targeted threshold sensitivity replay. By default this prints commands only; "
            "pass --execute to run the selected small grid."
        )
    )
    parser.add_argument("--scenarios", default="forest_co_management")
    parser.add_argument("--conditions", default="bottom_up_only,top_down_only,hybrid")
    parser.add_argument("--actor-capability-levels", default="high_actor")
    parser.add_argument("--overseer-capability-levels", default="weak_overseer")
    parser.add_argument("--local-margins", default="0.00,0.025,0.05,0.075,0.10")
    parser.add_argument("--global-thresholds", default="9.0,9.5,10.0,10.5,11.0")
    parser.add_argument("--limit-threshold-combinations", type=int, default=4)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--seeds-per-generation", type=int, default=8)
    parser.add_argument("--test-seeds-per-generation", type=int, default=8)
    parser.add_argument("--replacement-fraction", type=float, default=0.2)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--output-dir", default="results/runs/threshold_replay")
    parser.add_argument("--summary-csv", default="results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_targeted.csv")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--write-summary-each-cell", action="store_true", default=True)
    parser.add_argument("--no-write-summary-each-cell", dest="write_summary_each_cell", action="store_false")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_float_csv(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _slug(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _build_command(args: argparse.Namespace, scenario: str, local_margin: float, global_threshold: float) -> tuple[list[str], Path]:
    stem = f"{scenario}__lm{_slug(local_margin)}__gh{_slug(global_threshold)}"
    output_prefix = Path(args.output_dir) / stem
    command = [
        "python",
        "-m",
        "experiments.run_harvest_invasion_matrix",
        "--scenario-presets",
        scenario,
        "--conditions",
        args.conditions,
        "--actor-capability-levels",
        args.actor_capability_levels,
        "--overseer-capability-levels",
        args.overseer_capability_levels,
        "--n-runs",
        str(args.n_runs),
        "--generations",
        str(args.generations),
        "--population-size",
        str(args.population_size),
        "--seeds-per-generation",
        str(args.seeds_per_generation),
        "--test-seeds-per-generation",
        str(args.test_seeds_per_generation),
        "--replacement-fraction",
        str(args.replacement_fraction),
        "--local-safety-margin",
        str(local_margin),
        "--global-min-mean-patch-health",
        str(global_threshold),
        "--max-workers",
        str(args.max_workers),
        "--output-prefix",
        str(output_prefix),
        "--experiment-tag",
        "harvest_threshold_replay_targeted",
        "--no-progress",
    ]
    return command, output_prefix


def _summarise_run_outputs(records: list[dict], summary_csv: Path) -> None:
    rows = []
    for record in records:
        runs_csv = Path(str(record["output_prefix"]) + "_runs.csv")
        if not runs_csv.exists():
            continue
        runs = pd.read_csv(runs_csv)
        for _, row in runs.iterrows():
            out = dict(record)
            out["condition"] = row.get("condition")
            out["actor_capability_level"] = row.get("actor_capability_level")
            out["overseer_capability_level"] = row.get("overseer_capability_level")
            out["capability_gap"] = row.get("capability_gap")
            out["run_id"] = row.get("run_id")
            for metric in METRICS:
                out[metric] = row.get(metric)
            rows.append(out)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")


def main() -> None:
    args = parse_args()
    scenarios = _parse_csv(args.scenarios)
    local_margins = _parse_float_csv(args.local_margins)
    global_thresholds = _parse_float_csv(args.global_thresholds)
    threshold_pairs = list(itertools.product(local_margins, global_thresholds))
    if not args.full_grid and args.limit_threshold_combinations > 0:
        threshold_pairs = threshold_pairs[: args.limit_threshold_combinations]

    records = []
    print("Threshold replay plan:")
    print(f"  scenarios: {scenarios}")
    print(f"  threshold pairs: {len(threshold_pairs)}")
    print(f"  execute: {args.execute}")
    print(f"  resume: {args.resume}")
    for scenario, (local_margin, global_threshold) in itertools.product(scenarios, threshold_pairs):
        command, output_prefix = _build_command(args, scenario, local_margin, global_threshold)
        runs_csv = Path(str(output_prefix) + "_runs.csv")
        record = {
            "scenario_preset": scenario,
            "local_safety_margin": local_margin,
            "global_min_mean_patch_health": global_threshold,
            "output_prefix": str(output_prefix),
        }
        records.append(record)
        print(" ".join(shlex.quote(part) for part in command))
        if args.execute:
            if args.resume and runs_csv.exists():
                print(f"Skipping existing: {runs_csv}")
                if args.write_summary_each_cell:
                    _summarise_run_outputs(records, Path(args.summary_csv))
                continue
            output_prefix.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
            if args.write_summary_each_cell:
                _summarise_run_outputs(records, Path(args.summary_csv))

    if args.execute:
        _summarise_run_outputs(records, Path(args.summary_csv))
    else:
        print("Dry run only. Re-run with --execute to launch the targeted replay.")


if __name__ == "__main__":
    main()
