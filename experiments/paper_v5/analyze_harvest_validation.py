"""Analyze completed validation blocks. Units of uncertainty are source runs/parents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from experiments.common.validate_harvest_mechanisms import read_json


def interval(values):
    values = pd.Series(values).dropna().astype(float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1) * values.sem()) if len(values) > 1 else float("nan")
    return mean, half


def require_complete(directory):
    completion = read_json(directory / "completion.json")
    if completion["expected_blocks"] != completion["actual_blocks"]:
        raise ValueError(f"Incomplete run: {directory}")
    if len(list(directory.glob("*.json.gz"))) != completion["expected_blocks"]:
        raise ValueError("Completion count does not match files")


def analyze_mechanisms(directory, output):
    require_complete(directory)
    rows = []
    for path in sorted(directory.glob("*.json.gz")):
        for episode in read_json(path, True):
            row = {**episode["metadata"], **episode["metrics"]}
            own_approval, own_mismatch, own_onset = 0, 0, 0
            cfg = episode["config"]
            for step in episode["trace"]:
                h = np.array(json.loads(step["patch_health_before_json"]))
                p = np.array(json.loads(step["allowed_fracs_json"]))
                residual = np.maximum(0, h - np.minimum(h, p * cfg["max_harvest_per_agent"]))
                predicted = residual + np.maximum(0, cfg["regen_rate"] * residual * (1-residual/cfg["patch_max"]))
                passed = bool((predicted >= cfg["global_min_mean_patch_health"] - 1e-10).all())
                own_approval += int(passed)
                own_mismatch += int(passed and step["global_unsafe"])
                own_onset += int(passed and step["pre_global_safe"] and step["global_unsafe"])
            row.update(own_model_approval_steps=own_approval, own_model_mismatch_steps=own_mismatch,
                       own_model_onset_count=own_onset)
            row["any_approved_onset"] = int(row["approved_onset_count"] > 0)
            row["any_clean_prefix_onset"] = int(row["clean_prefix_onset_count"] > 0)
            row["any_own_model_onset"] = int(own_onset > 0)
            rows.append(row)
    df = pd.DataFrame(rows)
    keys = ["scenario", "actor", "run_id", "control", "mechanism", "seed"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicated episode keys")
    df.to_csv(output / "mechanism_episodes.csv", index=False)
    metrics = ["global_unsafe_rate", "local_pass_global_fail_rate", "approval_coverage", "mean_patch_health",
               "total_welfare", "mean_welfare", "t_end", "mean_prevented_harvest", "any_approved_onset",
               "any_clean_prefix_onset", "any_own_model_onset"]
    groups = ["scenario", "actor", "control", "mechanism"]
    run_df = df.groupby(groups + ["run_id"])[metrics].mean().reset_index()
    run_df.to_csv(output / "mechanism_run_means.csv", index=False)
    summary = []
    for key, group in run_df.groupby(groups):
        row = dict(zip(groups, key))
        row["n_source_runs"] = len(group)
        for metric in metrics:
            row[metric], row[metric + "_ci_half"] = interval(group[metric])
        summary.append(row)
    pd.DataFrame(summary).to_csv(output / "mechanism_summary.csv", index=False)
    event_cols = ["approved_steps", "approved_onset_count", "approved_persistence_count", "clean_prefix_onset_count",
                  "own_model_approval_steps", "own_model_mismatch_steps", "own_model_onset_count", "t_end"]
    df.groupby(["scenario", "control", "mechanism"])[event_cols].sum().to_csv(output / "transition_counts.csv")
    pairs = []
    contrasts = [("uniform_on", "uniform_off"), ("neighborhood_on", "neighborhood_off"),
                 ("neighborhood_on", "uniform_on"), ("neighborhood_off", "uniform_off"),
                 ("uniform_off", "signal_only"), ("local_state", "local_cutoff"), ("joint_reference", "local_state")]
    for (scenario, actor), group in run_df[run_df.control.eq("base")].groupby(["scenario", "actor"]):
        for left, right in contrasts:
            a, b = [group[group.mechanism.eq(v)].set_index("run_id") for v in [left, right]]
            for metric in metrics:
                delta = a[metric] - b[metric]
                mean, half = interval(delta)
                pairs.append({"scenario": scenario, "actor": actor, "left": left, "right": right,
                    "metric": metric, "difference": mean, "ci_half": half, "n_pairs": len(delta)})
    pd.DataFrame(pairs).to_csv(output / "mechanism_paired_differences.csv", index=False)
    return df, run_df


def analyze_calibration(directory, output):
    require_complete(directory)
    rows = []
    for path in sorted(directory.glob("*.json.gz")):
        rows.extend(read_json(path, True)["tests"])
    df = pd.DataFrame(rows)
    if df.duplicated(["scenario", "context", "candidate_count", "search_horizon", "seed"]).any():
        raise ValueError("Duplicated calibration keys")
    if len(df) != 2304:
        raise ValueError("Calibration must contain 2304 held-out evaluations")
    df.to_csv(output / "calibration_episodes.csv", index=False)
    metrics = ["score", "entrant_payoff", "total_welfare", "unsafe_rate", "patch_health"]
    grouped = df.groupby(["scenario", "context", "candidate_count", "search_horizon"])[metrics].mean().reset_index()
    grouped.to_csv(output / "calibration_parent_means.csv", index=False)
    pairs = []
    for (scenario, horizon), group in grouped.groupby(["scenario", "search_horizon"]):
        baseline = group[group.candidate_count.eq(1)].set_index("context")
        for k in [6, 12]:
            stronger = group[group.candidate_count.eq(k)].set_index("context")
            for metric in metrics:
                mean, half = interval(stronger[metric] - baseline[metric])
                pairs.append({"scenario": scenario, "search_horizon": horizon, "candidate_count": k,
                    "metric": metric, "difference": mean, "ci_half": half, "n_pairs": len(stronger)})
    pd.DataFrame(pairs).to_csv(output / "calibration_paired_differences.csv", index=False)
    return grouped


def figures(runs, calibration, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    modes = ["none", "communication", "signal_only", "uniform_off", "uniform_on", "neighborhood_off",
             "neighborhood_on", "local_cutoff", "local_state", "joint_reference"]
    labels = ["None", "Messages only", "Announced cap only", "Uniform caps", "Uniform + messages",
              "Neighbourhood caps", "Neighbourhood + messages", "Fixed local cutoff", "Local state filter", "Joint reference"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    # Average the two actor treatments within source-run IDs before uncertainty.
    base = runs[runs.control.eq("base")].groupby(["scenario", "mechanism", "run_id"]).global_unsafe_rate.mean().reset_index()
    for ax, (scenario, title) in zip(axes, [("community_irrigation", "Moderate stress"), ("forest_co_management", "High stress")]):
        for y, mechanism in enumerate(modes):
            values = base[base.scenario.eq(scenario) & base.mechanism.eq(mechanism)].global_unsafe_rate
            ax.scatter(values * 100, np.full(len(values), y), s=10, color="#6c737a", alpha=.45)
            ax.scatter([values.mean()*100], [y], s=30, color="#126b70", marker="D", zorder=3)
        ax.set_title(title, loc="left")
        ax.set_yticks(range(len(modes)), labels)
        ax.set_xlabel("Globally unsafe steps (%)")
        ax.grid(axis="x", color=".9", lw=.5)
    axes[0].invert_yaxis()
    fig.tight_layout()
    for suffix in ["pdf", "png", "svg"]:
        fig.savefig(output / f"matched_mechanisms.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, (scenario, title) in zip(axes, [("community_irrigation", "Moderate stress"), ("forest_co_management", "High stress")]):
        for horizon, color, marker in [(30, "#126b70", "o"), (60, "#a06022", "s")]:
            sub = calibration[calibration.scenario.eq(scenario) & calibration.search_horizon.eq(horizon)]
            means, halves = zip(*(interval(sub[sub.candidate_count.eq(k)].score) for k in [1, 6, 12]))
            ax.errorbar([1, 6, 12], means, yerr=halves, color=color, marker=marker, lw=1.2,
                        capsize=2, label=f"{horizon}-step search evaluation")
        ax.set_title(title, loc="left")
        ax.set_xticks([1, 6, 12])
        ax.set_xlabel("Number of candidate strategies")
        ax.grid(axis="y", color=".9", lw=.5)
    axes[0].set_ylabel("Held-out entrant score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1))
    for suffix in ["pdf", "png", "svg"]:
        fig.savefig(output / f"generator_validation.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/runs/validation_v1"))
    parser.add_argument("--output", type=Path, default=Path("notes/research_review/completed_checks"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episodes, runs = analyze_mechanisms(args.root / "mechanisms", args.output)
    calibration = analyze_calibration(args.root / "calibration", args.output)
    figures(runs, calibration, args.output)
    if (args.root / "slow_regrowth/completion.json").exists():
        slow_output = args.output / "slow_regrowth"
        slow_output.mkdir(exist_ok=True)
        slow_episodes, slow_runs = analyze_mechanisms(args.root / "slow_regrowth", slow_output)
        figures(slow_runs, calibration, slow_output)
        table = []
        names = {"none": "None", "communication": "Messages only", "signal_only": "Announcement only",
            "uniform_off": "Uniform caps", "neighborhood_on": "Neighbourhood caps + messages",
            "local_cutoff": "Fixed local filter", "local_state": "Local state filter", "joint_reference": "Joint reference"}
        for mechanism, label in names.items():
            values = []
            for frame in [episodes, slow_episodes]:
                for scenario in ["community_irrigation", "forest_co_management"]:
                    subset = frame[frame.control.eq("base") & frame.mechanism.eq(mechanism) & frame.scenario.eq(scenario)]
                    values.append(100 * subset.global_unsafe_rate.mean())
            table.append(label + " & " + " & ".join(f"{v:.2f}" for v in values) + r" \\")
        tex = "\n".join([r"\begin{table}[t]\centering\small", r"\begin{tabular}{lrrrr}\toprule",
            r"& \multicolumn{2}{c}{Base regrowth} & \multicolumn{2}{c}{Slower regrowth}\\",
            r"Mechanism & Moderate & High & Moderate & High\\\midrule", *table,
            r"\bottomrule\end{tabular}",
            r"\caption{Unsafe step percentages in matched-policy checks. Each cell has ten fixed populations (five source runs at each generation treatment) and eight seeds. These are descriptive means; generation treatment and source-run variation are retained in the accompanying data. The slow-regrowth follow-up uses an existing held-out condition.}",
            r"\label{tab:matched}\end{table}", ""])
        (args.output / "table_validation_results.tex").write_text(tex)
    print(f"Analyzed {len(episodes)} base matched episodes and 2304 held-out calibration episodes")
    if (args.root / "slow_regrowth/completion.json").exists():
        print(f"Also analyzed {len(slow_episodes)} slow-regrowth matched episodes")
    print(args.output)


if __name__ == "__main__":
    main()
