from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd


DEFAULT_LEVELS = [
    "strong_overseer",
    "recall_limited_only",
    "delay_limited_only",
    "capacity_limited_only",
    "cost_limited_only",
    "limited_overseer",
    "weak_overseer",
]

CORE_METRICS = [
    "test_global_unsafe_rate_mean",
    "test_local_pass_global_fail_rate_mean",
    "test_mean_patch_health_mean",
    "test_garden_failure_mean",
    "test_mean_welfare_mean",
    "test_governance_budget_spent_mean",
    "test_missed_target_rate_mean",
    "test_delayed_intervention_count_mean",
]

KEY_COLS = [
    "scenario_preset",
    "condition",
    "actor_capability_level",
    "overseer_capability_level",
    "run_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit overseer-limit ablation run completeness.")
    parser.add_argument("--runs-csv", default="results/runs/overseer_limit_ablation/reduced_ablation_runs.csv")
    parser.add_argument("--scenarios", default="forest_co_management")
    parser.add_argument("--conditions", default="top_down_only,hybrid")
    parser.add_argument("--actor-capability-levels", default="high_actor")
    parser.add_argument("--overseer-levels", default=",".join(DEFAULT_LEVELS))
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument(
        "--output-md",
        default="results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced_audit.md",
    )
    return parser.parse_args()


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _expected(args: argparse.Namespace) -> pd.DataFrame:
    scenarios = _parse_csv(args.scenarios)
    conditions = _parse_csv(args.conditions)
    actor_levels = _parse_csv(args.actor_capability_levels)
    overseer_levels = _parse_csv(args.overseer_levels)
    rows = []
    for scenario, condition, actor, overseer, run_id in product(
        scenarios,
        conditions,
        actor_levels,
        overseer_levels,
        range(args.n_runs),
    ):
        rows.append(
            {
                "scenario_preset": scenario,
                "condition": condition,
                "actor_capability_level": actor,
                "overseer_capability_level": overseer,
                "run_id": run_id,
            }
        )
    return pd.DataFrame(rows)


def _anti_join(left: pd.DataFrame, right: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    marker = left.merge(right[cols].drop_duplicates(), on=cols, how="left", indicator=True)
    return marker[marker["_merge"] == "left_only"][cols].copy()


def _format_rows(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df.empty:
        return "None."
    shown = df.head(max_rows).to_dict(orient="records")
    suffix = "" if len(df) <= max_rows else f"\n\nAdditional rows omitted: {len(df) - max_rows}."
    return f"`{shown}`{suffix}"


def main() -> None:
    args = parse_args()
    runs_path = Path(args.runs_csv)
    df = pd.read_csv(runs_path)
    expected = _expected(args)

    missing_cols = sorted(set(KEY_COLS).difference(df.columns))
    if missing_cols:
        raise ValueError(f"Run CSV missing required columns: {missing_cols}")

    observed = df[KEY_COLS].drop_duplicates()
    missing = _anti_join(expected, observed, KEY_COLS)
    unexpected = _anti_join(observed, expected, KEY_COLS)
    duplicate_rows = int(len(df) - len(observed))
    metric_nulls = {metric: int(df[metric].isna().sum()) for metric in CORE_METRICS if metric in df.columns}

    expected_count = len(expected)
    safe = (
        len(observed) == expected_count
        and len(df) == expected_count
        and missing.empty
        and unexpected.empty
        and duplicate_rows == 0
        and all(count == 0 for count in metric_nulls.values())
    )

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Overseer-Limit Ablation Completeness Audit",
        "",
        f"- Run CSV: `{runs_path}`",
        f"- Expected run rows: {expected_count}",
        f"- Observed run rows: {len(df)}",
        f"- Observed unique run keys: {len(observed)}",
        f"- Duplicate run rows: {duplicate_rows}",
        f"- Missing run keys: {len(missing)}",
        f"- Unexpected run keys: {len(unexpected)}",
        "",
        "## Expected Scope",
        "",
        f"- Scenarios: `{_parse_csv(args.scenarios)}`",
        f"- Conditions: `{_parse_csv(args.conditions)}`",
        f"- Actor capability levels: `{_parse_csv(args.actor_capability_levels)}`",
        f"- Overseer levels: `{_parse_csv(args.overseer_levels)}`",
        f"- Runs per cell: {args.n_runs}",
        "",
        "## Metric Null Check",
        "",
    ]
    for metric, count in metric_nulls.items():
        lines.append(f"- `{metric}` null values: {count}")
    lines.extend(
        [
            "",
            "## Missing Or Unexpected Rows",
            "",
            f"- Missing run keys: {_format_rows(missing)}",
            f"- Unexpected run keys: {_format_rows(unexpected)}",
            "",
            "## Citation Safety",
            "",
            (
                "Safe to cite as completed for the declared scope: yes."
                if safe
                else "Safe to cite as completed for the declared scope: not yet. Inspect missing, unexpected, duplicate, or null rows above."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)
    print(f"safe_to_cite={safe}")


if __name__ == "__main__":
    main()
