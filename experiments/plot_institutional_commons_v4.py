from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot institutional commons v4 figures.")
    parser.add_argument("--scenario-table-csv", required=True)
    parser.add_argument("--architecture-contrast-csv", required=True)
    parser.add_argument("--llm-map-summary-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_scenario_map(scenario_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    cols = ["scenario_preset", "institutional_archetype", "preferred_governance_comparison"]
    display_df = scenario_df[cols].copy()
    table = ax.table(
        cellText=display_df.values,
        colLabels=["Scenario", "Archetype", "Focus comparison"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("Literature-backed Harvest scenario archetypes", fontsize=12)
    _savefig(output_path)


def plot_architecture_effects(contrast_df: pd.DataFrame, output_path: str) -> None:
    df = contrast_df.copy()
    df = df[df["contrast_name"] == "hybrid_minus_top_down_only"].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No hybrid-minus-top-down rows found.", ha="center", va="center")
        ax.axis("off")
        _savefig(output_path)
        return

    df["label"] = (
        df["scenario_preset"].fillna("").astype(str)
        + "\n"
        + df["governance_friction_regime"].fillna("").astype(str)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    patch_col = "delta__test_mean_patch_health_mean_mean"
    welfare_col = "delta__test_mean_welfare_mean_mean"
    axes[0].barh(df["label"], df[patch_col], color="#2a9d8f")
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Patch-health delta")
    axes[0].set_xlabel("Hybrid - top-down-only")
    axes[1].barh(df["label"], df[welfare_col], color="#e76f51")
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Welfare delta")
    axes[1].set_xlabel("Hybrid - top-down-only")
    fig.suptitle("Architecture performance under ideal and constrained oversight", fontsize=12)
    _savefig(output_path)


def plot_llm_governance_map(summary_df: pd.DataFrame, output_path: str) -> None:
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No LLM governance-map rows found.", ha="center", va="center")
        ax.axis("off")
        _savefig(output_path)
        return

    scenarios = sorted(summary_df["scenario_preset"].dropna().astype(str).unique().tolist())
    models = sorted(summary_df["bank_model_label"].dropna().astype(str).unique().tolist())
    fig, axes = plt.subplots(len(scenarios), len(models), figsize=(5.4 * len(models), 3.6 * len(scenarios)), squeeze=False)
    for i, scenario in enumerate(scenarios):
        for j, model in enumerate(models):
            ax = axes[i][j]
            sub = summary_df[(summary_df["scenario_preset"] == scenario) & (summary_df["bank_model_label"] == model)].copy()
            for condition, color in [("none", "#777777"), ("top_down_only", "#1d3557"), ("hybrid", "#2a9d8f")]:
                cur = sub[sub["condition"] == condition].sort_values("exploitative_share")
                if cur.empty:
                    continue
                ax.plot(cur["exploitative_share"], cur["mean_patch_health"], marker="o", label=condition, color=color)
            ax.set_title(f"{scenario}\n{model}")
            ax.set_xlabel("Exploitative-bank share")
            ax.set_ylabel("Mean patch health")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            if i == 0 and j == len(models) - 1:
                ax.legend(frameon=False, loc="best")
    fig.suptitle("LLM-population governance map", fontsize=12)
    _savefig(output_path)


def main() -> None:
    args = parse_args()
    scenario_df = pd.read_csv(args.scenario_table_csv)
    architecture_df = pd.read_csv(args.architecture_contrast_csv)
    llm_map_df = pd.read_csv(args.llm_map_summary_csv)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_scenario_map(scenario_df, str(output_prefix.with_name(output_prefix.name + "_scenario_map.png")))
    plot_architecture_effects(architecture_df, str(output_prefix.with_name(output_prefix.name + "_architecture_effects.png")))
    plot_llm_governance_map(llm_map_df, str(output_prefix.with_name(output_prefix.name + "_llm_governance_map.png")))


if __name__ == "__main__":
    main()
