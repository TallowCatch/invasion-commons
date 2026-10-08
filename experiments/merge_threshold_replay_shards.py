from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.merge_harvest_invasion_outputs import _collect_csvs
from experiments.merge_harvest_invasion_outputs import _merge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded threshold-replay outputs and emit a replay summary CSV.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--merged-prefix", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Merge only run-level outputs. Use this for full threshold robustness "
            "grids where generation, strategy, and agent histories are too large "
            "to concatenate on a GitHub Actions runner."
        ),
    )
    return parser.parse_args()


def _write_if_nonempty(df: pd.DataFrame, path: Path, sort_cols: list[str]) -> None:
    if df.empty:
        return
    cols = [col for col in sort_cols if col in df.columns]
    if cols:
        df = df.sort_values(cols).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    merged_prefix = Path(args.merged_prefix)
    summary_csv = Path(args.summary_csv)
    merged_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    runs_df = _merge(_collect_csvs(input_dir, "_runs.csv"))
    if args.summary_only:
        gen_df = pd.DataFrame()
        strat_df = pd.DataFrame()
        agent_df = pd.DataFrame()
    else:
        gen_df = _merge(_collect_csvs(input_dir, "_generation_history.csv"))
        strat_df = _merge(_collect_csvs(input_dir, "_strategy_history.csv"))
        agent_df = _merge(_collect_csvs(input_dir, "_agent_history.csv"))

    if runs_df.empty:
        raise FileNotFoundError(f"No threshold replay run shards found under {input_dir}")

    _write_if_nonempty(
        runs_df,
        merged_prefix.with_name(merged_prefix.name + "_runs.csv"),
        [
            "scenario_preset",
            "local_safety_margin",
            "global_min_mean_patch_health",
            "actor_capability_level",
            "overseer_capability_level",
            "condition",
            "run_id",
        ],
    )
    _write_if_nonempty(
        gen_df,
        merged_prefix.with_name(merged_prefix.name + "_generation_history.csv"),
        [
            "scenario_preset",
            "local_safety_margin",
            "global_min_mean_patch_health",
            "actor_capability_level",
            "overseer_capability_level",
            "condition",
            "run_id",
            "generation",
        ],
    )
    _write_if_nonempty(
        strat_df,
        merged_prefix.with_name(merged_prefix.name + "_strategy_history.csv"),
        [
            "scenario_preset",
            "local_safety_margin",
            "global_min_mean_patch_health",
            "actor_capability_level",
            "overseer_capability_level",
            "condition",
            "run_id",
            "generation",
            "strategy_id",
        ],
    )
    _write_if_nonempty(
        agent_df,
        merged_prefix.with_name(merged_prefix.name + "_agent_history.csv"),
        [
            "scenario_preset",
            "local_safety_margin",
            "global_min_mean_patch_health",
            "actor_capability_level",
            "overseer_capability_level",
            "condition",
            "run_id",
            "generation",
            "phase",
            "regime",
            "seed",
            "agent_index",
        ],
    )

    summary_cols = [
        "scenario_preset",
        "local_safety_margin",
        "global_min_mean_patch_health",
        "condition",
        "actor_capability_level",
        "overseer_capability_level",
        "capability_gap",
        "run_id",
        "test_global_unsafe_rate_mean",
        "test_local_pass_global_fail_rate_mean",
        "test_mean_patch_health_mean",
        "test_garden_failure_mean",
        "test_mean_welfare_mean",
        "test_governance_budget_spent_mean",
    ]
    available_cols = [col for col in summary_cols if col in runs_df.columns]
    summary_df = runs_df[available_cols].copy()
    summary_df = summary_df.sort_values(
        [
            col
            for col in [
                "scenario_preset",
                "local_safety_margin",
                "global_min_mean_patch_health",
                "actor_capability_level",
                "overseer_capability_level",
                "condition",
                "run_id",
            ]
            if col in summary_df.columns
        ]
    ).reset_index(drop=True)
    summary_df.to_csv(summary_csv, index=False)

    print(merged_prefix.with_name(merged_prefix.name + "_runs.csv"))
    print(merged_prefix.with_name(merged_prefix.name + "_generation_history.csv"))
    print(merged_prefix.with_name(merged_prefix.name + "_strategy_history.csv"))
    print(merged_prefix.with_name(merged_prefix.name + "_agent_history.csv"))
    print(summary_csv)


if __name__ == "__main__":
    main()
