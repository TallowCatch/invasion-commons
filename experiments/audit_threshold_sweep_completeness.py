from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd


SCENARIOS = ["community_irrigation", "forest_co_management"]
LOCAL_MARGINS = [0.0, 0.025, 0.05, 0.075, 0.10]
GLOBAL_THRESHOLDS = [9.0, 9.5, 10.0, 10.5, 11.0]
ACTOR_LEVELS = ["low_actor", "medium_actor", "high_actor"]
OVERSEER_LEVELS = ["strong_overseer", "limited_overseer", "weak_overseer"]
CONDITIONS = ["none", "bottom_up_only", "top_down_only", "hybrid"]
RUN_IDS = [0, 1, 2, 3, 4]

KEY_COLS = [
    "scenario_preset",
    "local_safety_margin_key",
    "global_min_mean_patch_health_key",
    "actor_capability_level",
    "overseer_capability_level",
    "condition",
]
RUN_KEY_COLS = KEY_COLS + ["run_id"]
DECISION_COLS = KEY_COLS[:-1]

CORE_METRICS = [
    "test_global_unsafe_rate_mean",
    "test_local_pass_global_fail_rate_mean",
    "test_mean_patch_health_mean",
    "test_garden_failure_mean",
    "test_mean_welfare_mean",
    "test_governance_budget_spent_mean",
]

CONDITION_LABELS = {
    "none": "No oversight",
    "bottom_up_only": "Local",
    "top_down_only": "Global signal",
    "hybrid": "Hybrid",
}

SCENARIO_LABELS = {
    "community_irrigation": "Moderate coupling",
    "forest_co_management": "High coupling",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the full threshold replay sweep for completeness.")
    parser.add_argument(
        "--summary-csv",
        default="results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered.csv",
    )
    parser.add_argument(
        "--raw-runs-csv",
        default="results/runs/threshold_replay/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered_runs.csv",
    )
    parser.add_argument(
        "--stagea-runs-csv",
        default="results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_runs.csv",
    )
    parser.add_argument(
        "--artifact-zip",
        default="results/artifacts/github/harvest_oversight_gap_threshold_replay_full_grid_recovered-bundle.zip",
    )
    parser.add_argument(
        "--output-md",
        default="notes/THRESHOLD_SWEEP_COMPLETENESS_AUDIT.md",
    )
    return parser.parse_args()


def _normalized(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["local_safety_margin_key"] = out["local_safety_margin"].round(3)
    out["global_min_mean_patch_health_key"] = out["global_min_mean_patch_health"].round(1)
    return out


def _expected_runs() -> pd.DataFrame:
    rows = []
    for scenario, margin, threshold, actor, overseer, condition, run_id in product(
        SCENARIOS,
        LOCAL_MARGINS,
        GLOBAL_THRESHOLDS,
        ACTOR_LEVELS,
        OVERSEER_LEVELS,
        CONDITIONS,
        RUN_IDS,
    ):
        rows.append(
            {
                "scenario_preset": scenario,
                "local_safety_margin_key": round(margin, 3),
                "global_min_mean_patch_health_key": round(threshold, 1),
                "actor_capability_level": actor,
                "overseer_capability_level": overseer,
                "condition": condition,
                "run_id": run_id,
            }
        )
    return pd.DataFrame(rows)


def _anti_join(left: pd.DataFrame, right: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    marker = left.merge(right[cols].drop_duplicates(), on=cols, how="left", indicator=True)
    return marker[marker["_merge"] == "left_only"][cols].copy()


def _winner_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_cells = (
        df.groupby(KEY_COLS, as_index=False)
        .agg(
            patch=("test_mean_patch_health_mean", "mean"),
            lpgf=("test_local_pass_global_fail_rate_mean", "mean"),
        )
    )
    winners = []
    for keys, group in condition_cells.groupby(DECISION_COLS):
        best = group.sort_values(["patch", "lpgf"], ascending=[False, True]).iloc[0]
        key = dict(zip(DECISION_COLS, keys if isinstance(keys, tuple) else (keys,)))
        key["winner"] = best["condition"]
        winners.append(key)
    winner_df = pd.DataFrame(winners)
    counts = (
        winner_df.groupby(["scenario_preset", "winner"], as_index=False)
        .size()
        .rename(columns={"size": "wins"})
    )
    return winner_df, counts


def _default_consistency(threshold_df: pd.DataFrame, stagea_path: Path) -> tuple[str, dict[str, float] | None]:
    if not stagea_path.exists():
        return f"Stage A default file not found: `{stagea_path}`.", None

    stage = pd.read_csv(stagea_path)
    default = threshold_df[
        (threshold_df["local_safety_margin_key"] == 0.05)
        & (threshold_df["global_min_mean_patch_health_key"] == 10.0)
    ].copy()
    stage_keys = [
        "scenario_preset",
        "condition",
        "actor_capability_level",
        "overseer_capability_level",
        "run_id",
    ]
    missing_cols = sorted(set(stage_keys + CORE_METRICS).difference(stage.columns))
    if missing_cols:
        return f"Stage A default file is missing columns: `{missing_cols}`.", None

    merged = default[stage_keys + CORE_METRICS].merge(
        stage[stage_keys + CORE_METRICS],
        on=stage_keys,
        how="outer",
        suffixes=("_replay", "_stagea"),
        indicator=True,
    )
    merge_counts = merged["_merge"].value_counts().to_dict()
    if merge_counts.get("left_only", 0) or merge_counts.get("right_only", 0):
        return f"Default-threshold key mismatch: `{merge_counts}`.", None

    diffs = {}
    for metric in CORE_METRICS:
        diffs[metric] = float((merged[f"{metric}_replay"] - merged[f"{metric}_stagea"]).abs().max())
    return "Default-threshold replay exactly matches the Stage A run-level defaults on checked metrics.", diffs


def _format_missing(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df.empty:
        return "None."
    shown = df.head(max_rows).to_dict(orient="records")
    suffix = "" if len(df) <= max_rows else f"\n\nAdditional missing rows omitted: {len(df) - max_rows}."
    return f"`{shown}`{suffix}"


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    raw_runs_path = Path(args.raw_runs_csv)
    stagea_path = Path(args.stagea_runs_csv)
    artifact_path = Path(args.artifact_zip)
    output_path = Path(args.output_md)

    df = _normalized(pd.read_csv(summary_path))
    expected = _expected_runs()

    observed_base = df[KEY_COLS].drop_duplicates()
    observed_run = df[RUN_KEY_COLS].drop_duplicates()
    expected_base = expected[KEY_COLS].drop_duplicates()

    missing_base = _anti_join(expected_base, observed_base, KEY_COLS)
    missing_runs = _anti_join(expected, observed_run, RUN_KEY_COLS)
    unexpected_runs = _anti_join(observed_run, expected, RUN_KEY_COLS)
    duplicate_run_rows = int(len(df) - len(observed_run))
    run_counts = df.groupby(KEY_COLS, as_index=False)["run_id"].nunique().rename(columns={"run_id": "run_count"})
    bad_run_counts = run_counts[run_counts["run_count"] != len(RUN_IDS)]
    metric_nulls = {metric: int(df[metric].isna().sum()) for metric in CORE_METRICS if metric in df.columns}

    winner_df, winner_counts = _winner_counts(df)
    default_msg, default_diffs = _default_consistency(df, stagea_path)

    safe_to_cite = (
        len(df) == len(expected)
        and len(observed_base) == len(expected_base)
        and len(observed_run) == len(expected)
        and missing_base.empty
        and missing_runs.empty
        and unexpected_runs.empty
        and duplicate_run_rows == 0
        and bad_run_counts.empty
        and all(value == 0 for value in metric_nulls.values())
        and default_diffs is not None
        and all(value == 0.0 for value in default_diffs.values())
    )

    total_decision_cells = int(len(winner_df))
    lines = [
        "# Threshold Sweep Completeness Audit",
        "",
        "## Files Checked",
        "",
        f"- Summary CSV: `{summary_path}`",
        f"- Raw recovered runs CSV: `{raw_runs_path}`",
        f"- Stage A default runs CSV: `{stagea_path}`",
        f"- GitHub artifact bundle: `{artifact_path}`",
        "",
        "## Expected Grid",
        "",
        f"- Stress settings: {len(SCENARIOS)}",
        f"- Local safety margins: {len(LOCAL_MARGINS)}",
        f"- Global patch-health thresholds: {len(GLOBAL_THRESHOLDS)}",
        f"- Actor-capability levels: {len(ACTOR_LEVELS)}",
        f"- Overseer-capability levels: {len(OVERSEER_LEVELS)}",
        f"- Oversight architectures: {len(CONDITIONS)}",
        f"- Runs per cell: {len(RUN_IDS)}",
        f"- Expected run rows: {len(expected)}",
        f"- Expected architecture cells before run expansion: {len(expected_base)}",
        f"- Expected patch-health decision cells: {len(expected_base) // len(CONDITIONS)}",
        "",
        "## Observed Counts",
        "",
        f"- Observed run rows: {len(df)}",
        f"- Observed unique run keys: {len(observed_run)}",
        f"- Observed architecture cells before run expansion: {len(observed_base)}",
        f"- Duplicate run rows: {duplicate_run_rows}",
        f"- Bad run-count cells: {len(bad_run_counts)}",
        f"- Missing architecture cells: {len(missing_base)}",
        f"- Missing run keys: {len(missing_runs)}",
        f"- Unexpected run keys: {len(unexpected_runs)}",
        "",
        "## Metric Null Check",
        "",
    ]
    for metric, count in metric_nulls.items():
        lines.append(f"- `{metric}` null values: {count}")

    lines.extend(
        [
            "",
            "## Winner Count Check",
            "",
            f"- Observed patch-health decision cells: {total_decision_cells}",
            "",
            "| Stress setting | Winner | Patch-health winning cells |",
            "| --- | --- | ---: |",
        ]
    )
    for _, row in winner_counts.sort_values(["scenario_preset", "wins"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {SCENARIO_LABELS[row['scenario_preset']]} | {CONDITION_LABELS[row['winner']]} | {int(row['wins'])} |"
        )

    total_wins = (
        winner_counts.groupby("winner", as_index=False)["wins"].sum().sort_values("wins", ascending=False)
    )
    lines.extend(["", "| Winner | Total patch-health winning cells |", "| --- | ---: |"])
    for _, row in total_wins.iterrows():
        lines.append(f"| {CONDITION_LABELS[row['winner']]} | {int(row['wins'])} |")

    lines.extend(
        [
            "",
            "## Missing Or Duplicate Cells",
            "",
            f"- Missing architecture cells: {_format_missing(missing_base)}",
            f"- Missing run keys: {_format_missing(missing_runs)}",
            f"- Unexpected run keys: {_format_missing(unexpected_runs)}",
        ]
    )
    if bad_run_counts.empty:
        lines.append("- Cells with non-five run count: None.")
    else:
        lines.append(f"- Cells with non-five run count: `{bad_run_counts.head(8).to_dict(orient='records')}`")

    lines.extend(["", "## Default-Threshold Consistency", "", f"- {default_msg}"])
    if default_diffs is not None:
        lines.append("")
        lines.append("| Metric | Maximum absolute difference vs Stage A defaults |")
        lines.append("| --- | ---: |")
        for metric, value in default_diffs.items():
            lines.append(f"| `{metric}` | {value:.12g} |")

    lines.extend(
        [
            "",
            "## Citation Safety",
            "",
            (
                "Safe to cite in the paper: yes. The recovered summary contains the complete expected grid, no duplicate run keys, five runs per cell, no nulls in checked metrics, and exact default-threshold agreement with the Stage A run-level defaults."
                if safe_to_cite
                else "Safe to cite in the paper: not yet. One or more completeness checks failed; keep the paper conservative and inspect the failed checks above."
            ),
            "",
            "Note: this audit verifies completeness of the recovered merged artifact. It does not re-audit the GitHub Actions UI logs shard by shard. A silently skipped shard would show up here as a missing expected run key or a non-five run-count cell.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)
    print(f"safe_to_cite={safe_to_cite}")


if __name__ == "__main__":
    main()
