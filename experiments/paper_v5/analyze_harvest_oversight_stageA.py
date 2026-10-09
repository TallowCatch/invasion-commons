from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONDITION_LABELS = {
    "none": "No oversight",
    "bottom_up_only": "Local oversight",
    "top_down_only": "Global signal",
    "hybrid": "Hybrid oversight",
}

CONDITION_ORDER = ["none", "bottom_up_only", "top_down_only", "hybrid"]
ACTOR_ORDER = ["low_actor", "medium_actor", "high_actor"]
ACTOR_LABELS = {
    "low_actor": "Low actor",
    "medium_actor": "Medium actor",
    "high_actor": "High actor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage A oversight analysis tables.")
    parser.add_argument(
        "--table-csv",
        default="results/runs/showcase/curated/harvest_oversight_gap_stageA_table.csv",
    )
    parser.add_argument(
        "--case-trace-csv",
        default="results/runs/showcase/curated/harvest_oversight_gap_stageA_oversight_case_trace.csv",
    )
    parser.add_argument(
        "--strategy-history-csv",
        default="results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_strategy_history.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/runs/showcase/curated/harvest_oversight_gap_stageA",
    )
    parser.add_argument("--local-margins", default="0.00,0.05,0.10")
    parser.add_argument("--global-thresholds", default="9.5,10.0,10.5")
    parser.add_argument("--failure-fraction-threshold", type=float, default=0.5)
    parser.add_argument("--sustainable-harvest-frac", type=float, default=0.35)
    return parser.parse_args()


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _read_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "condition" not in df.columns:
        raise ValueError("Stage A table is missing condition column.")
    return df


def _condition_means(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("condition")
        .agg(
            global_unsafe_rate=("test_global_unsafe_rate_mean_mean", "mean"),
            local_pass_global_fail_rate=("test_local_pass_global_fail_rate_mean_mean", "mean"),
            mean_patch_health=("test_mean_patch_health_mean_mean", "mean"),
            mean_welfare=("test_mean_welfare_mean_mean", "mean"),
            governance_burden=("test_governance_budget_spent_mean_mean", "mean"),
        )
        .reindex(CONDITION_ORDER)
        .reset_index()
    )
    out["condition_label"] = out["condition"].map(CONDITION_LABELS)
    return out[
        [
            "condition",
            "condition_label",
            "global_unsafe_rate",
            "local_pass_global_fail_rate",
            "mean_patch_health",
            "mean_welfare",
            "governance_burden",
        ]
    ]


def _write_condition_latex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Stage A mean outcomes by oversight architecture.}",
        "    \\label{tab:stagea-condition-means}",
        "    \\begin{tabular}{lrrrrr}",
        "        \\toprule",
        "        Architecture & Unsafe & Local/global fail & Patch health & Welfare & Burden \\\\",
        "        \\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            "        "
            f"{row['condition_label']} & "
            f"{row['global_unsafe_rate']:.3f} & "
            f"{row['local_pass_global_fail_rate']:.3f} & "
            f"{row['mean_patch_health']:.2f} & "
            f"{row['mean_welfare']:.2f} & "
            f"{row['governance_burden']:.2f} \\\\"
        )
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _actor_validation(df: pd.DataFrame) -> pd.DataFrame:
    no_oversight = df[df["condition"] == "none"].copy()
    out = (
        no_oversight.groupby("actor_capability_level")
        .agg(
            global_unsafe_rate=("test_global_unsafe_rate_mean_mean", "mean"),
            local_pass_global_fail_rate=("test_local_pass_global_fail_rate_mean_mean", "mean"),
            mean_requested_harvest=("test_mean_requested_harvest_mean_mean", "mean"),
            aggressive_request_fraction=("test_mean_aggressive_request_fraction_mean_mean", "mean"),
            max_local_aggression=("test_mean_max_local_aggression_mean_mean", "mean"),
            mean_patch_health=("test_mean_patch_health_mean_mean", "mean"),
        )
        .reindex(ACTOR_ORDER)
        .reset_index()
    )
    out["actor_label"] = out["actor_capability_level"].map(ACTOR_LABELS)
    return out[
        [
            "actor_capability_level",
            "actor_label",
            "global_unsafe_rate",
            "local_pass_global_fail_rate",
            "mean_requested_harvest",
            "aggressive_request_fraction",
            "max_local_aggression",
            "mean_patch_health",
        ]
    ]


def _actor_validation_extended(table_df: pd.DataFrame, strategy_df: pd.DataFrame | None) -> pd.DataFrame:
    behavior = _actor_validation(table_df).set_index("actor_capability_level")
    rows = []
    if strategy_df is not None and not strategy_df.empty:
        new_mask = strategy_df["is_new_in_generation"].astype(str).str.lower().isin({"true", "1"})
        entrants = strategy_df[new_mask & (strategy_df["generation"].astype(int) > 0)].copy()
        if not entrants.empty:
            if "condition" in entrants.columns:
                no_oversight_entrants = entrants[entrants["condition"] == "none"].copy()
                entrants_for_strength = no_oversight_entrants if not no_oversight_entrants.empty else entrants
            else:
                entrants_for_strength = entrants
            entrant_summary = entrants_for_strength.groupby("actor_capability_level").agg(
                new_entrant_payoff=("mean_payoff", "mean"),
                new_entrant_high_harvest=("high_harvest_frac", "mean"),
                new_entrant_cap_margin=("cap_compliance_margin", "mean"),
                search_candidates=("search_candidates", "max"),
                search_eval_horizon=("search_eval_horizon", "max"),
            )
        else:
            entrant_summary = pd.DataFrame()
    else:
        entrant_summary = pd.DataFrame()

    process_label = {
        "low_actor": "Mutation",
        "medium_actor": "Search mutation",
        "high_actor": "Search mutation",
    }
    for actor in ACTOR_ORDER:
        b = behavior.loc[actor]
        if actor in entrant_summary.index:
            e = entrant_summary.loc[actor]
            payoff = float(e["new_entrant_payoff"])
            entrant_high = float(e["new_entrant_high_harvest"])
            cap_margin = float(e["new_entrant_cap_margin"])
            candidates = int(e["search_candidates"])
            horizon = int(e["search_eval_horizon"])
        else:
            payoff = float("nan")
            entrant_high = float("nan")
            cap_margin = float("nan")
            candidates = 0
            horizon = 0
        rows.append(
            {
                "actor_capability_level": actor,
                "actor_label": ACTOR_LABELS[actor],
                "entrant_process": process_label[actor],
                "search_candidates": candidates,
                "search_eval_horizon": horizon,
                "new_entrant_payoff": payoff,
                "new_entrant_high_harvest": entrant_high,
                "new_entrant_cap_margin": cap_margin,
                "no_oversight_requested_harvest": float(b["mean_requested_harvest"]),
                "no_oversight_aggressive_request": float(b["aggressive_request_fraction"]),
                "no_oversight_max_local_aggression": float(b["max_local_aggression"]),
                "no_oversight_global_unsafe_rate": float(b["global_unsafe_rate"]),
            }
        )
    return pd.DataFrame(rows)


def _write_actor_validation_latex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Actor-capability validation. The first columns define the entrant-generation setting. The remaining columns report no-oversight entrant strength and no-oversight behavior.}",
        "    \\label{tab:actor-capability-validation}",
        "    \\resizebox{\\linewidth}{!}{%",
        "    \\begin{tabular}{llrrrrrrrr}",
        "        \\toprule",
        "        Actor setting & Entrant process & Candidates & Horizon & Entrant payoff & High harvest & Cap margin & Requested harvest & Aggressive request & Unsafe \\\\",
        "        \\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            "        "
            f"{row['actor_label']} & "
            f"{row['entrant_process']} & "
            f"{int(row['search_candidates'])} & "
            f"{int(row['search_eval_horizon'])} & "
            f"{row['new_entrant_payoff']:.1f} & "
            f"{row['new_entrant_high_harvest']:.2f} & "
            f"{row['new_entrant_cap_margin']:.2f} & "
            f"{row['no_oversight_requested_harvest']:.2f} & "
            f"{row['no_oversight_aggressive_request']:.3f} & "
            f"{row['no_oversight_global_unsafe_rate']:.3f} \\\\"
        )
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


def _write_actor_validation_note(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Stage A Actor-Capability Validation",
        "",
        "This check uses no-oversight cells so the entrant process is evaluated without governance intervention.",
        "",
        "| Actor level | Unsafe rate | Local/global fail | Requested harvest | Aggressive request | Max local aggression | Patch health |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['actor_label']} | "
            f"{row['global_unsafe_rate']:.3f} | "
            f"{row['local_pass_global_fail_rate']:.3f} | "
            f"{row['mean_requested_harvest']:.2f} | "
            f"{row['aggressive_request_fraction']:.3f} | "
            f"{row['max_local_aggression']:.3f} | "
            f"{row['mean_patch_health']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the no-oversight validation should be read as a behavioral check, not as a standalone result. In Stage A, higher actor settings increase requested harvest and aggression-related metrics, while global unsafe rate does not increase monotonically because the higher-capability search can also find strategies that preserve enough resource to keep earning payoff.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_actor_validation_extended_note(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Stage A Actor-Capability Validation, Extended",
        "",
        "This table separates the designed capability setting from observed behavioral pressure. The search columns define the capability setting. The payoff and action columns use no-oversight rows to check whether the setting actually produces stronger entrants or more aggressive behavior.",
        "",
        "| Actor setting | Entrant process | Candidates | Horizon | Entrant payoff | Entrant high harvest | Cap margin | Requested harvest | Aggressive request | Unsafe rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['actor_label']} | "
            f"{row['entrant_process']} | "
            f"{int(row['search_candidates'])} | "
            f"{int(row['search_eval_horizon'])} | "
            f"{row['new_entrant_payoff']:.1f} | "
            f"{row['new_entrant_high_harvest']:.2f} | "
            f"{row['new_entrant_cap_margin']:.2f} | "
            f"{row['no_oversight_requested_harvest']:.2f} | "
            f"{row['no_oversight_aggressive_request']:.3f} | "
            f"{row['no_oversight_global_unsafe_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: medium and high actor settings have explicit search budget and horizon increases. New entrant payoff rises from low to medium to high actor settings. Requested harvest and aggressive-request fraction also increase under no oversight. Global unsafe rate does not rise monotonically, which is why the paper treats capability as search/evaluation capacity plus observed behavioral pressure rather than as a direct synonym for collapse.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _threshold_sensitivity(
    trace_df: pd.DataFrame,
    *,
    sustainable_harvest_frac: float,
    local_margins: list[float],
    global_thresholds: list[float],
    failure_fraction_threshold: float,
) -> pd.DataFrame:
    if "max_requested_frac" not in trace_df.columns:
        raise ValueError(
            "Case trace is missing max_requested_frac. Regenerate the trace with the current code before running sensitivity."
        )
    rows = []
    for margin in local_margins:
        local_threshold = sustainable_harvest_frac + margin
        all_local_safe = trace_df["max_requested_frac"].astype(float) <= local_threshold + 1e-12
        for global_threshold in global_thresholds:
            global_unsafe = (
                (trace_df["mean_patch_health_after"].astype(float) < global_threshold)
                | (trace_df["failed_patch_fraction_after"].astype(float) >= failure_fraction_threshold)
            )
            local_pass_global_fail = all_local_safe & global_unsafe
            rows.append(
                {
                    "local_safety_margin": margin,
                    "global_min_mean_patch_health": global_threshold,
                    "all_local_safe_step_fraction": float(all_local_safe.mean()),
                    "global_unsafe_rate": float(global_unsafe.mean()),
                    "local_pass_global_fail_rate": float(local_pass_global_fail.mean()),
                    "local_pass_global_fail_steps": int(local_pass_global_fail.sum()),
                    "n_steps": int(len(trace_df)),
                }
            )
    return pd.DataFrame(rows)


def _write_threshold_note(df: pd.DataFrame, path: Path) -> None:
    survives = bool((df["local_pass_global_fail_steps"] > 0).any())
    lines = [
        "# Stage A Threshold Sensitivity",
        "",
        "This sensitivity check recomputes local-pass/global-fail on the extracted episode trace using nearby local-safety margins and global patch-health thresholds.",
        "",
        f"Local-pass/global-fail survives at least one threshold setting: {'yes' if survives else 'no'}.",
        "",
        "| Local margin | Global threshold | Global unsafe | Local/global fail | Fail steps |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['local_safety_margin']:.2f} | "
            f"{row['global_min_mean_patch_health']:.1f} | "
            f"{row['global_unsafe_rate']:.3f} | "
            f"{row['local_pass_global_fail_rate']:.3f} | "
            f"{int(row['local_pass_global_fail_steps'])} |"
        )
    lines.extend(
        [
            "",
            "This is a targeted episode-level sensitivity check, not a full matrix rerun. A paper-ready robustness appendix should repeat the same check on a small set of targeted cells if reviewers need cell-level sensitivity.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    table_df = _read_table(args.table_csv)
    condition_df = _condition_means(table_df)
    condition_df.to_csv(prefix.with_name(prefix.name + "_condition_means.csv"), index=False)
    _write_condition_latex(condition_df, prefix.with_name(prefix.name + "_condition_means.tex"))

    actor_df = _actor_validation(table_df)
    actor_df.to_csv(prefix.with_name(prefix.name + "_actor_capability_validation.csv"), index=False)
    _write_actor_validation_note(actor_df, prefix.with_name(prefix.name + "_actor_capability_validation.md"))

    strategy_path = Path(args.strategy_history_csv)
    strategy_df = pd.read_csv(strategy_path) if strategy_path.exists() else None
    actor_ext_df = _actor_validation_extended(table_df, strategy_df)
    actor_ext_df.to_csv(
        prefix.with_name(prefix.name + "_actor_capability_validation_extended.csv"),
        index=False,
    )
    _write_actor_validation_latex(
        actor_ext_df,
        prefix.with_name(prefix.name + "_actor_capability_validation_extended.tex"),
    )
    _write_actor_validation_extended_note(
        actor_ext_df,
        prefix.with_name(prefix.name + "_actor_capability_validation_extended.md"),
    )

    trace_path = Path(args.case_trace_csv)
    if trace_path.exists():
        trace_df = pd.read_csv(trace_path)
        sensitivity_df = _threshold_sensitivity(
            trace_df,
            sustainable_harvest_frac=float(args.sustainable_harvest_frac),
            local_margins=_parse_floats(args.local_margins),
            global_thresholds=_parse_floats(args.global_thresholds),
            failure_fraction_threshold=float(args.failure_fraction_threshold),
        )
        sensitivity_df.to_csv(prefix.with_name(prefix.name + "_threshold_sensitivity.csv"), index=False)
        _write_threshold_note(sensitivity_df, prefix.with_name(prefix.name + "_threshold_sensitivity.md"))

    print(f"Saved: {prefix.with_name(prefix.name + '_condition_means.csv')}")
    print(f"Saved: {prefix.with_name(prefix.name + '_condition_means.tex')}")
    print(f"Saved: {prefix.with_name(prefix.name + '_actor_capability_validation.csv')}")
    print(f"Saved: {prefix.with_name(prefix.name + '_actor_capability_validation.md')}")
    print(f"Saved: {prefix.with_name(prefix.name + '_actor_capability_validation_extended.csv')}")
    print(f"Saved: {prefix.with_name(prefix.name + '_actor_capability_validation_extended.tex')}")
    print(f"Saved: {prefix.with_name(prefix.name + '_actor_capability_validation_extended.md')}")
    if trace_path.exists():
        print(f"Saved: {prefix.with_name(prefix.name + '_threshold_sensitivity.csv')}")
        print(f"Saved: {prefix.with_name(prefix.name + '_threshold_sensitivity.md')}")


if __name__ == "__main__":
    main()
