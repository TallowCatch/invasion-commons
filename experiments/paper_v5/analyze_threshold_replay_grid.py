from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONDITION_LABELS = {
    "none": "No oversight",
    "bottom_up_only": "Local",
    "top_down_only": "Global",
    "hybrid": "Hybrid",
}
CONDITION_ORDER = ["none", "bottom_up_only", "top_down_only", "hybrid"]
SCENARIO_ORDER = ["community_irrigation", "forest_co_management"]
SCENARIO_LABELS = {
    "community_irrigation": "Moderate coupling",
    "forest_co_management": "High coupling",
}

CAPABILITY_COLS = ["actor_capability_level", "overseer_capability_level"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize threshold-replay grid outputs.")
    parser.add_argument(
        "--input-csv",
        default="results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_reduced_full_grid.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_reduced_full_grid",
    )
    return parser.parse_args()


def _load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "scenario_preset",
        "local_safety_margin",
        "global_min_mean_patch_health",
        "condition",
        "test_global_unsafe_rate_mean",
        "test_local_pass_global_fail_rate_mean",
        "test_mean_patch_health_mean",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.copy()


def _pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scenario_preset", "local_safety_margin", "global_min_mean_patch_health"]
    group_cols.extend([col for col in CAPABILITY_COLS if col in df.columns])
    group_cols.append("condition")
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            global_unsafe_rate=("test_global_unsafe_rate_mean", "mean"),
            local_pass_global_fail_rate=("test_local_pass_global_fail_rate_mean", "mean"),
            mean_patch_health=("test_mean_patch_health_mean", "mean"),
            garden_failure_rate=("test_garden_failure_mean", "mean"),
            mean_welfare=("test_mean_welfare_mean", "mean"),
            governance_burden=("test_governance_budget_spent_mean", "mean"),
        )
    )
    agg["condition_label"] = agg["condition"].map(CONDITION_LABELS)
    return agg


def _winner_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    winners = []
    group_cols = ["scenario_preset", "local_safety_margin", "global_min_mean_patch_health"]
    group_cols.extend([col for col in CAPABILITY_COLS if col in pair_df.columns])
    for keys, group in pair_df.groupby(group_cols, as_index=False):
        key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        best = group.sort_values(["mean_patch_health", "local_pass_global_fail_rate"], ascending=[False, True]).iloc[0]
        key_dict["winner"] = best["condition"]
        winners.append(key_dict)
    winner_df = pd.DataFrame(winners)
    out = (
        winner_df.groupby(["scenario_preset", "winner"], as_index=False)
        .size()
        .rename(columns={"size": "threshold_pair_wins"})
    )
    out["winner_label"] = out["winner"].map(CONDITION_LABELS)
    return out


def _condition_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    default = pair_df[
        (pair_df["local_safety_margin"] == 0.05) & (pair_df["global_min_mean_patch_health"] == 10.0)
    ].copy()
    threshold_pair_cols = ["local_safety_margin", "global_min_mean_patch_health"]
    capability_cols = [col for col in CAPABILITY_COLS if col in pair_df.columns]
    for (scenario, condition), group in pair_df.groupby(["scenario_preset", "condition"], as_index=False):
        row = {
            "scenario_preset": scenario,
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "mean_global_unsafe_rate": group["global_unsafe_rate"].mean(),
            "mean_local_pass_global_fail_rate": group["local_pass_global_fail_rate"].mean(),
            "positive_lpgf_cells": int((group["local_pass_global_fail_rate"] > 0).sum()),
            "threshold_cells": int(len(group)),
            "threshold_pairs": int(len(group[threshold_pair_cols].drop_duplicates())),
            "capability_cells": int(len(group[capability_cols].drop_duplicates())) if capability_cols else 1,
            "mean_patch_health": group["mean_patch_health"].mean(),
            "mean_garden_failure_rate": group["garden_failure_rate"].mean(),
        }
        default_row = default[(default["scenario_preset"] == scenario) & (default["condition"] == condition)]
        if not default_row.empty:
            row["default_lpgf_rate"] = float(default_row["local_pass_global_fail_rate"].iloc[0])
            row["default_patch_health"] = float(default_row["mean_patch_health"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(condition_df: pd.DataFrame, winner_df: pd.DataFrame, path: Path) -> None:
    is_full_grid = int(condition_df["capability_cells"].max()) > 1 if "capability_cells" in condition_df else False
    title = "Full Threshold Sweep Summary" if is_full_grid else "Reduced Threshold Sweep Summary"
    scope = (
        "This full sweep covers both stress settings, all 25 threshold pairs, all four oversight architectures, "
        "and all nine actor-overseer capability combinations."
        if is_full_grid
        else "This reduced sweep covers both stress settings, all 25 threshold pairs, all four oversight architectures, "
        "and the high-actor / weak-overseer slice."
    )
    lines = [
        f"# {title}",
        "",
        scope,
        "",
        "## Winner Counts By Patch Health",
        "",
        "| Scenario | Winner | Decision-cell wins |",
        "| --- | --- | ---: |",
    ]
    for _, row in winner_df.sort_values(["scenario_preset", "threshold_pair_wins"], ascending=[True, False]).iterrows():
        lines.append(f"| {row['scenario_preset']} | {row['winner_label']} | {int(row['threshold_pair_wins'])} |")
    lines.extend(
        [
            "",
            "## Condition Summary",
            "",
            "| Scenario | Condition | Mean unsafe | Mean local/global fail | Positive LPGF cells | Decision cells | Mean patch health | Mean garden failure | Default LPGF | Default patch |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in condition_df.sort_values(["scenario_preset", "condition"]).iterrows():
        lines.append(
            f"| {row['scenario_preset']} | {row['condition_label']} | "
            f"{row['mean_global_unsafe_rate']:.3f} | {row['mean_local_pass_global_fail_rate']:.3f} | "
            f"{int(row['positive_lpgf_cells'])} | {int(row['threshold_cells'])} | "
            f"{row['mean_patch_health']:.2f} | {row['mean_garden_failure_rate']:.3f} | "
            f"{row.get('default_lpgf_rate', float('nan')):.3f} | {row.get('default_patch_health', float('nan')):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(condition_df: pd.DataFrame, winner_df: pd.DataFrame, path: Path) -> None:
    is_full_grid = int(condition_df["capability_cells"].max()) > 1 if "capability_cells" in condition_df else False
    caption = (
        "Full threshold sweep over both stress settings, all 25 threshold pairs, and all nine actor-overseer capability combinations. Winners are defined by mean patch health within each scenario-threshold-capability cell. Positive LPGF cells are cells with nonzero local-pass/global-fail rate."
        if is_full_grid
        else "Reduced threshold sweep over both stress settings and all 25 threshold pairs for the high-actor / weak-overseer slice. Winners are defined by mean patch health within each scenario-threshold pair."
    )
    label = "tab:threshold-robustness-sweep" if is_full_grid else "tab:reduced-threshold-sweep"
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        "    \\resizebox{\\linewidth}{!}{%",
        "    \\begin{tabular}{llrrrrrr}",
        "        \\toprule",
        "        Stress setting & Architecture & Mean unsafe & Mean LPGF & Positive LPGF cells & Cell wins & Mean patch & Mean garden fail \\\\",
        "        \\midrule",
    ]
    pair_wins = {
        (row["scenario_preset"], row["winner"]): int(row["threshold_pair_wins"])
        for _, row in winner_df.iterrows()
    }
    for scenario_idx, scenario in enumerate(SCENARIO_ORDER):
        for condition in CONDITION_ORDER:
            row = condition_df[
                (condition_df["scenario_preset"] == scenario) & (condition_df["condition"] == condition)
            ].iloc[0]
            wins = pair_wins.get((row["scenario_preset"], row["condition"]), 0)
            lines.append(
                f"        {SCENARIO_LABELS.get(row['scenario_preset'], row['scenario_preset'].replace('_', ' '))} & {row['condition_label']} & "
                f"{row['mean_global_unsafe_rate']:.3f} & "
                f"{row['mean_local_pass_global_fail_rate']:.3f} & "
                f"{int(row['positive_lpgf_cells'])}/{int(row['threshold_cells'])} & "
                f"{wins} & "
                f"{row['mean_patch_health']:.2f} & "
                f"{row['mean_garden_failure_rate']:.3f} \\\\"
            )
        if scenario_idx < len(SCENARIO_ORDER) - 1:
            lines.append("        \\addlinespace")
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    }",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = _load(args.input_csv)
    pair_df = _pair_summary(df)
    winner_df = _winner_summary(pair_df)
    condition_df = _condition_summary(pair_df)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(prefix.with_name(prefix.name + "_pair_summary.csv"), index=False)
    winner_df.to_csv(prefix.with_name(prefix.name + "_winner_summary.csv"), index=False)
    condition_df.to_csv(prefix.with_name(prefix.name + "_condition_summary.csv"), index=False)
    _write_markdown(condition_df, winner_df, prefix.with_name(prefix.name + "_summary.md"))
    _write_latex(condition_df, winner_df, prefix.with_name(prefix.name + "_table.tex"))
    has_capability_dims = all(col in df.columns for col in CAPABILITY_COLS)
    paper_table_name = "table_threshold_robustness_sweep.tex" if has_capability_dims else "table_reduced_threshold_sweep.tex"
    paper_table = Path("paper/paper_v5_scalable_oversight_commons/tables") / paper_table_name
    paper_table.parent.mkdir(parents=True, exist_ok=True)
    _write_latex(condition_df, winner_df, paper_table)
    print(prefix.with_name(prefix.name + "_pair_summary.csv"))
    print(prefix.with_name(prefix.name + "_winner_summary.csv"))
    print(prefix.with_name(prefix.name + "_condition_summary.csv"))
    print(prefix.with_name(prefix.name + "_summary.md"))
    print(prefix.with_name(prefix.name + "_table.tex"))
    print(paper_table)


if __name__ == "__main__":
    main()
