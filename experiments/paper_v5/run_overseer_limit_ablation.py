from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import pandas as pd


ABLATION_LEVELS = [
    "strong_overseer",
    "recall_limited_only",
    "delay_limited_only",
    "capacity_limited_only",
    "cost_limited_only",
    "limited_overseer",
    "weak_overseer",
]

METRICS = [
    "test_global_unsafe_rate_mean",
    "test_local_pass_global_fail_rate_mean",
    "test_mean_patch_health_mean",
    "test_mean_welfare_mean",
    "test_governance_budget_spent_mean",
    "test_missed_target_rate_mean",
    "test_delayed_intervention_count_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reduced overseer-limit ablation. By default this prints the command only; "
            "pass --execute to run it."
        )
    )
    parser.add_argument("--scenario", default="forest_co_management")
    parser.add_argument("--conditions", default="top_down_only,hybrid")
    parser.add_argument("--actor-capability-level", default="high_actor")
    parser.add_argument("--overseer-levels", default=",".join(ABLATION_LEVELS))
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--seeds-per-generation", type=int, default=16)
    parser.add_argument("--test-seeds-per-generation", type=int, default=16)
    parser.add_argument("--replacement-fraction", type=float, default=0.2)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--output-prefix", default="results/runs/overseer_limit_ablation/reduced_ablation")
    parser.add_argument("--summary-csv", default="results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced.csv")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _command(args: argparse.Namespace) -> list[str]:
    return [
        "python",
        "-m",
        "experiments.common.run_harvest_invasion_matrix",
        "--scenario-presets",
        args.scenario,
        "--conditions",
        args.conditions,
        "--actor-capability-levels",
        args.actor_capability_level,
        "--overseer-capability-levels",
        args.overseer_levels,
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
        "--max-workers",
        str(args.max_workers),
        "--output-prefix",
        args.output_prefix,
        "--experiment-tag",
        "harvest_overseer_limit_ablation",
        "--no-progress",
    ]


def _summarise(output_prefix: str, summary_csv: str) -> None:
    runs_csv = Path(output_prefix + "_runs.csv")
    if not runs_csv.exists():
        raise FileNotFoundError(f"Missing run output: {runs_csv}")
    runs = pd.read_csv(runs_csv)
    group_cols = ["condition", "actor_capability_level", "overseer_capability_level", "capability_gap"]
    if "scenario_preset" in runs:
        group_cols.insert(0, "scenario_preset")
    if runs.duplicated(group_cols + ["run_id"]).any():
        raise ValueError("Duplicate run keys; refusing to treat duplicate outputs as independent runs")
    rows = []
    for keys, group in runs.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(group["run_id"].nunique())
        for metric in METRICS:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_sem"] = float(group[metric].sem()) if len(group) > 1 else 0.0
        rows.append(row)
    out = Path(summary_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")


def main() -> None:
    args = parse_args()
    command = _command(args)
    print("Overseer-limit ablation command:")
    print(" ".join(shlex.quote(part) for part in command))
    if not args.execute:
        print("Dry run only. Re-run with --execute to launch the reduced ablation.")
        return
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    _summarise(args.output_prefix, args.summary_csv)


if __name__ == "__main__":
    main()
