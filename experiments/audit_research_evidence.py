"""Audit existing Stage A evidence without generating strategies or running episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fishery_sim.harvest_llm_population import _harvest_bank_variation_anchors


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/runs"
STAGE = RUNS / "harvest_invasion/curated/harvest_oversight_gap_stageA_runs.csv"
SWEEP = RUNS / "threshold_replay/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered_runs.csv"
CASE = RUNS / "showcase/curated/harvest_oversight_gap_stageA_oversight_case_trace.csv"
ACTORS = ["low_actor", "medium_actor", "high_actor"]
OVERSEERS = ["strong_overseer", "limited_overseer", "weak_overseer"]
CONTEXT = ["scenario_preset", "actor_capability_level", "overseer_capability_level"]
METRICS = [
    "test_global_unsafe_rate_mean",
    "test_local_pass_global_fail_rate_mean",
    "test_mean_patch_health_mean",
    "test_mean_welfare_mean",
    "test_governance_budget_spent_mean",
]


def within_group_ranges(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> dict[str, float]:
    grouped = df.groupby(keys, dropna=False)[metrics]
    return (grouped.max() - grouped.min()).max().to_dict()


def local_global_table(df: pd.DataFrame) -> pd.DataFrame:
    approval = df["test_all_local_safe_step_fraction_mean"]
    unsafe = df[METRICS[0]]
    lpgf = df[METRICS[1]]
    out = df[[*CONTEXT, "condition", "run_id"]].copy()
    out["all_pass_global_safe"] = approval - lpgf
    out["all_pass_global_unsafe"] = lpgf
    out["some_fail_global_safe"] = 1 - approval - unsafe + lpgf
    out["some_fail_global_unsafe"] = unsafe - lpgf
    values = out[["all_pass_global_safe", "all_pass_global_unsafe",
                  "some_fail_global_safe", "some_fail_global_unsafe"]]
    if (values.to_numpy() < -1e-8).any() or not np.allclose(values.sum(axis=1), 1):
        raise ValueError("Local/global joint probabilities are inconsistent.")
    return out


def trace_events(trace: pd.DataFrame) -> dict:
    # Exclude the first row: the original failed-patch fraction before it is not logged.
    previous_unsafe = trace["global_unsafe"].shift(1)
    lpgf = trace["local_pass_global_fail"].eq(1)
    known = previous_unsafe.notna()
    return {
        "steps": len(trace),
        "all_local_safe_steps": int(trace.all_local_safe.sum()),
        "global_unsafe_steps": int(trace.global_unsafe.sum()),
        "lpgf_steps": int(lpgf.sum()),
        "lpgf_safe_to_unsafe_known_previous_state": int((lpgf & previous_unsafe.eq(0) & known).sum()),
        "lpgf_already_unsafe_known_previous_state": int((lpgf & previous_unsafe.eq(1) & known).sum()),
        "lpgf_previous_state_unavailable": int((lpgf & ~known).sum()),
        "some_local_fail_global_safe_steps": int((trace.all_local_safe.eq(0) & trace.global_unsafe.eq(0)).sum()),
        "lpgf_rows_with_pre_mean_below_10": int((lpgf & trace.mean_patch_health_before.lt(10)).sum()),
    }


def paired_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context, group in df.groupby(CONTEXT):
        for left, right in [("hybrid", "top_down_only"), ("hybrid", "bottom_up_only"), ("top_down_only", "bottom_up_only")]:
            a = group[group.condition.eq(left)].set_index("run_id")
            b = group[group.condition.eq(right)].set_index("run_id")
            if set(a.index) != set(b.index):
                raise ValueError("Unpaired run IDs in architecture comparison.")
            for metric in METRICS:
                delta = a[metric].sort_index() - b[metric].sort_index()
                rows.append({**dict(zip(CONTEXT, context)), "contrast": f"{left}_minus_{right}",
                             "metric": metric, "n_pairs": len(delta), "mean_difference": delta.mean(),
                             "min_run_difference": delta.min(), "max_run_difference": delta.max()})
    return pd.DataFrame(rows)


def make_factor_figure(df: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                         "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    scenarios = [("community_irrigation", "Moderate stress"), ("forest_co_management", "High stress")]
    styles = [("#176B70", "o", "-", "Strong limits preset"),
              ("#A06022", "s", "--", "Limited preset"),
              ("#343B46", "^", ":", "Weak preset")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), sharex=True, sharey=True)
    for row, (scenario, title) in enumerate(scenarios):
        for col, condition in enumerate(["top_down_only", "hybrid"]):
            ax = axes[row, col]
            subset = df[df.scenario_preset.eq(scenario) & df.condition.eq(condition)]
            for overseer, (color, marker, line, label) in zip(OVERSEERS, styles):
                groups = subset[subset.overseer_capability_level.eq(overseer)].groupby("actor_capability_level")[METRICS[0]]
                points = [groups.get_group(actor).to_numpy() for actor in ACTORS]
                mean = [np.mean(values) for values in points]
                ax.plot(range(3), mean, color=color, marker=marker, linestyle=line, lw=1.3, ms=4, label=label)
                # Show actual independent-run values, without treating settings as replicates.
                for x, values in enumerate(points):
                    ax.scatter(x + np.linspace(-0.035, 0.035, len(values)), values,
                               s=9, color=color, alpha=0.45, edgecolors="none")
            architecture = "Global caps" if col == 0 else "Hybrid"
            ax.set_title(f"({chr(97 + row * 2 + col)}) {title} / {architecture}", loc="left")
            ax.set_xticks(range(3), ["Mutation", "Search\n6 candidates\n30-step horizon", "Search\n12 candidates\n60-step horizon"])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(-0.008, 0.24)
            ax.grid(axis="y", color="#d9d9d9", lw=0.4)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel("Global unsafe step rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels[0] = "Strong preset"
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.18, hspace=0.26, wspace=0.16)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output / f"actor_overseer_separated.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "notes/research_review/evidence")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage, sweep, case = [pd.read_csv(path) for path in [STAGE, SWEEP, CASE]]
    keys = [*CONTEXT, "condition", "run_id"]
    if stage.duplicated(keys).any() or sweep.duplicated(keys + ["local_safety_margin", "global_min_mean_patch_health"]).any():
        raise ValueError("Duplicate run keys found.")
    stage.groupby("condition")[METRICS].mean().to_csv(output / "condition_means.csv")
    cells = stage.groupby([*CONTEXT, "condition", "capability_gap"])[METRICS].mean().reset_index()
    cells.to_csv(output / "separate_actor_overseer_cells.csv", index=False)
    cells[cells.capability_gap.eq(0)].to_csv(output / "equal_gap_different_settings.csv", index=False)
    local_global_table(stage).to_csv(output / "local_global_joint_rates.csv", index=False)
    paired_contrasts(stage).to_csv(output / "paired_architecture_differences.csv", index=False)
    inputs = [STAGE, SWEEP, CASE]
    checks = {
        "stage_a_run_rows": len(stage),
        "stage_a_architecture_cells": len(cells),
        "full_threshold_run_rows": len(sweep),
        "threshold_variation_max_within_run": within_group_ranges(sweep, keys, METRICS[2:]),
        "inactive_overseer_max_within_run": {},
        "trace": trace_events(case),
    }
    for condition in ["none", "bottom_up_only"]:
        checks["inactive_overseer_max_within_run"][condition] = within_group_ranges(
            stage[stage.condition.eq(condition)],
            ["scenario_preset", "actor_capability_level", "condition", "run_id"], METRICS)
    bank_rows = []
    for model in ["qwen2_5_3b", "llama3_2_3b"]:
        path = RUNS / f"showcase/curated/harvest_llm_bridge_stageB32_v3_local_{model}_bank.csv"
        inputs.append(path)
        bank = pd.read_csv(path)
        if set(bank.bank_prompt_version) != {"harvest_bank_v3_attitude_anchors"}:
            raise ValueError("Anchor comparison only applies to the v3 prompt.")
        anchors = pd.DataFrame([_harvest_bank_variation_anchors(row.bank_attitude, 20.0, int(row.prompt_nonce))
                                for row in bank.itertuples()])
        errors = (bank[anchors.columns] - anchors).abs()
        bank_rows.append({"model": model, "strategies": len(bank), "numeric_fields": len(anchors.columns),
                          "exact_anchor_strategies": int(errors.max(axis=1).lt(1e-9).sum()),
                          "exact_anchor_field_fraction": float(errors.lt(1e-9).to_numpy().mean()),
                          "high_harvest_anchor_mae": float(errors.high_harvest_frac.mean())})
    pd.DataFrame(bank_rows).to_csv(output / "model_anchor_comparison.csv", index=False)
    checks["llm_anchor_comparison"] = bank_rows
    for relative in ["fishery_sim/harvest.py", "fishery_sim/harvest_evolution.py", "fishery_sim/harvest_llm_population.py",
                     "experiments/audit_research_evidence.py", "paper/paper_v5_scalable_oversight_commons/main.tex"]:
        inputs.append(ROOT / relative)
    checks["input_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    checks["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (output / "evidence_checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")
    make_factor_figure(stage, output)
    print(json.dumps({key: value for key, value in checks.items() if key != "input_sha256"}, indent=2))
    print(f"Saved evidence audit: {output}")


if __name__ == "__main__":
    main()
