from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_METRICS = [
    "garden_failure_rate",
    "mean_patch_health",
    "global_unsafe_rate",
    "local_pass_global_fail_rate",
    "mean_welfare",
    "governance_budget_spent",
]


STRESS_PREFIXES = {
    "held_out_average": "test",
    "noisy_weather": "test_noisy_weather",
    "strong_externality": "test_strong_externality",
    "low_init": "test_low_init",
    "slow_regen": "test_slow_regen",
}


CONDITION_LABELS = {
    "none": "No oversight",
    "bottom_up_only": "Local oversight",
    "top_down_only": "Global signal",
    "hybrid": "Hybrid oversight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage A held-out stress-regime readouts.")
    parser.add_argument(
        "--generation-history-csv",
        default="results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_generation_history.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/runs/showcase/curated/harvest_oversight_gap_stageA_stress_regimes",
    )
    parser.add_argument(
        "--final-generation-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the final generation from each run rather than all generations.",
    )
    return parser.parse_args()


def _metric_column(prefix: str, metric: str) -> str:
    return f"{prefix}_{metric}"


def _available_regimes(df: pd.DataFrame) -> dict[str, str]:
    available = {}
    for regime, prefix in STRESS_PREFIXES.items():
        if any(_metric_column(prefix, metric) in df.columns for metric in BASE_METRICS):
            available[regime] = prefix
    return available


def _run_level_long(df: pd.DataFrame) -> pd.DataFrame:
    regimes = _available_regimes(df)
    id_cols = [
        "scenario_preset",
        "condition",
        "actor_capability_level",
        "overseer_capability_level",
        "capability_gap",
        "run_id",
        "generation",
    ]
    rows = []
    for _, row in df.iterrows():
        base = {col: row[col] for col in id_cols if col in df.columns}
        base["condition_label"] = CONDITION_LABELS.get(str(row.get("condition")), str(row.get("condition")))
        for regime, prefix in regimes.items():
            out = dict(base)
            out["stress_regime"] = regime
            for metric in BASE_METRICS:
                col = _metric_column(prefix, metric)
                out[metric] = float(row[col]) if col in df.columns and pd.notna(row[col]) else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def _summarise(long_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "stress_regime",
        "scenario_preset",
        "condition",
        "condition_label",
        "actor_capability_level",
        "overseer_capability_level",
        "capability_gap",
    ]
    # Generations from one evolutionary run are repeated observations, not replicates.
    long_df = long_df.groupby(group_cols + ["run_id"], dropna=False)[BASE_METRICS].mean().reset_index()
    rows = []
    for keys, group in long_df.groupby(group_cols, dropna=False):
        out = dict(zip(group_cols, keys))
        out["n_runs"] = int(group["run_id"].nunique()) if "run_id" in group else int(len(group))
        for metric in BASE_METRICS:
            values = group[metric].astype(float)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_sem"] = float(values.sem()) if values.notna().sum() > 1 else 0.0
        rows.append(out)
    return pd.DataFrame(rows).sort_values(group_cols)


def _condition_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df = long_df.copy()
    inactive = long_df.condition.isin(["none", "bottom_up_only"])
    keys = ["stress_regime", "scenario_preset", "condition", "actor_capability_level", "run_id", "generation"]
    keys = [key for key in keys if key in long_df]
    if inactive.any():
        if (long_df[inactive].groupby(keys, dropna=False)[BASE_METRICS].nunique() > 1).any().any():
            raise ValueError("Inactive-overseer copies differ; check provenance before pooling")
        long_df = pd.concat([long_df[~inactive], long_df[inactive].drop_duplicates(keys)])
    # This descriptive overall estimand averages design cells within the same run ID.
    long_df = long_df.groupby(["stress_regime", "condition", "condition_label", "run_id"], dropna=False)[BASE_METRICS].mean().reset_index()
    group_cols = ["stress_regime", "condition", "condition_label"]
    rows = []
    for keys, group in long_df.groupby(group_cols, dropna=False):
        out = dict(zip(group_cols, keys))
        out["n_rows"] = int(len(group))
        for metric in BASE_METRICS:
            values = group[metric].astype(float)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_sem"] = float(values.sem()) if values.notna().sum() > 1 else 0.0
        rows.append(out)
    return pd.DataFrame(rows).sort_values(group_cols)


def _write_note(condition_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Stage A Stress-Regime Summary",
        "",
        "This analysis uses existing Stage A generation-history columns. It summarizes internal held-out stress labels and should not be presented as field-calibrated environmental validation.",
        "",
        "| Stress regime | Condition | Unsafe | Local/global fail | Patch health | Welfare | Burden |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in condition_df.iterrows():
        lines.append(
            f"| {row['stress_regime']} | {row['condition_label']} | "
            f"{row['global_unsafe_rate_mean']:.3f} | "
            f"{row['local_pass_global_fail_rate_mean']:.3f} | "
            f"{row['mean_patch_health_mean']:.2f} | "
            f"{row['mean_welfare_mean']:.2f} | "
            f"{row['governance_budget_spent_mean']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.generation_history_csv)
    if args.final_generation_only:
        group_cols = [
            "scenario_preset",
            "condition",
            "actor_capability_level",
            "overseer_capability_level",
            "capability_gap",
            "run_id",
        ]
        idx = df.groupby(group_cols, dropna=False)["generation"].idxmax()
        df = df.loc[idx].copy()
    long_df = _run_level_long(df)
    summary = _summarise(long_df)
    condition_summary = _condition_summary(long_df)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(prefix.with_name(prefix.name + "_run_long.csv"), index=False)
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    condition_summary.to_csv(prefix.with_name(prefix.name + "_condition_summary.csv"), index=False)
    _write_note(condition_summary, prefix.with_name(prefix.name + ".md"))
    print(f"Saved: {prefix.with_name(prefix.name + '_run_long.csv')}")
    print(f"Saved: {prefix.with_name(prefix.name + '_summary.csv')}")
    print(f"Saved: {prefix.with_name(prefix.name + '_condition_summary.csv')}")
    print(f"Saved: {prefix.with_name(prefix.name + '.md')}")


if __name__ == "__main__":
    main()
