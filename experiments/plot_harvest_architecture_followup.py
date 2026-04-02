import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_COLORS = {
    "none": "#d9d9d9",
    "bottom_up_only": "#66c2a5",
    "top_down_only": "#fc8d62",
    "hybrid": "#8da0cb",
}

CONTRAST_COLORS = {
    "hybrid_minus_top_down_only": "#8da0cb",
    "hybrid_minus_bottom_up_only": "#1b9e77",
    "top_down_only_minus_bottom_up_only": "#d95f02",
}

INJECTOR_LABELS = {
    "random": "Random",
    "mutation": "Mutation",
    "adversarial_heuristic": "Heuristic",
    "search_mutation": "Search",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Harvest architecture follow-up outputs.")
    parser.add_argument("--ranking-csv", required=True)
    parser.add_argument("--contrast-ci-csv", required=True)
    parser.add_argument("--capability-ladder-csv", default=None)
    parser.add_argument("--aggression-summary-csv", default=None)
    parser.add_argument("--targeting-summary-csv", default=None)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def _load_optional_csv(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    path_obj = Path(path)
    if not path_obj.exists():
        return pd.DataFrame()
    return pd.read_csv(path_obj)


def _architecture_heatmap(ranking_df: pd.DataFrame, output: Path) -> None:
    if ranking_df.empty:
        return
    winners = ranking_df[ranking_df["rank"] == 1].copy()
    winners["cell"] = winners.apply(
        lambda row: f"{row['tier']}\n{row['partner_mix']}\np={row['adversarial_pressure']}",
        axis=1,
    )
    winners["column"] = winners["injector_mode_requested"].astype(str)
    table = winners.pivot(index="cell", columns="column", values="condition")
    row_labels = list(table.index)
    col_labels = list(table.columns)
    color_grid = np.empty((len(row_labels), len(col_labels), 4))
    for i, row_label in enumerate(row_labels):
        for j, col_label in enumerate(col_labels):
            condition = table.loc[row_label, col_label]
            color_grid[i, j] = matplotlib.colors.to_rgba(CONDITION_COLORS.get(condition, "#ffffff"))
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 2.0), max(6, len(row_labels) * 0.8)))
    ax.imshow(color_grid, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i, row_label in enumerate(row_labels):
        for j, col_label in enumerate(col_labels):
            ax.text(j, i, table.loc[row_label, col_label], ha="center", va="center", fontsize=8)
    ax.set_title("Harvest architecture winners by cell and injector")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _contrast_bars(contrast_ci_df: pd.DataFrame, output: Path) -> None:
    if contrast_ci_df.empty:
        return
    focus = contrast_ci_df[
        contrast_ci_df["contrast_name"].isin(
            ["hybrid_minus_top_down_only", "hybrid_minus_bottom_up_only", "top_down_only_minus_bottom_up_only"]
        )
    ].copy()
    if focus.empty:
        return
    focus["cell"] = focus.apply(
        lambda row: f"{row['tier']} | {row['partner_mix']} | {row['injector_mode_requested']} | p={row['adversarial_pressure']}",
        axis=1,
    )
    metrics = [
        ("delta__test_mean_patch_health_mean_mean", "Patch health"),
        ("delta__test_mean_welfare_mean_mean", "Welfare"),
        ("delta__test_mean_neighborhood_overharvest_mean_mean", "Neighborhood overharvest"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 10), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (metric_col, title) in zip(axes, metrics):
        labels = np.arange(len(focus))
        colors = [CONTRAST_COLORS.get(name, "#457b9d") for name in focus["contrast_name"]]
        means = focus[metric_col].to_numpy(dtype=float)
        base = metric_col.removesuffix("_mean")
        low = means - focus[f"{base}_ci95_low"].to_numpy(dtype=float)
        high = focus[f"{base}_ci95_high"].to_numpy(dtype=float) - means
        ax.bar(labels, means, color=colors, alpha=0.85)
        ax.errorbar(labels, means, yerr=np.vstack([low, high]), fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xticks(labels)
        ax.set_xticklabels(focus["cell"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Harvest architecture pairwise deltas")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _capability_ladder_plot(ladder_df: pd.DataFrame, output: Path) -> None:
    if ladder_df.empty:
        return
    break_cols = [
        "first_ecological_break_injector",
        "first_control_break_injector",
        "first_costly_robustness_injector",
    ]
    rung_map = {"none": -0.2, "random": 0.0, "mutation": 1.0, "adversarial_heuristic": 2.0, "search_mutation": 3.0}
    fig, axes = plt.subplots(len(break_cols), 1, figsize=(14, 9), constrained_layout=True)
    axes = np.atleast_1d(axes)
    ladder_df = ladder_df.copy()
    ladder_df["cell"] = ladder_df.apply(
        lambda row: f"{row['contrast_name']} | {row['tier']} | {row['partner_mix']} | p={row['adversarial_pressure']}",
        axis=1,
    )
    for ax, col in zip(axes, break_cols):
        y = ladder_df[col].map(rung_map).to_numpy(dtype=float)
        ax.scatter(np.arange(len(ladder_df)), y, color="#2a9d8f", s=45)
        ax.set_xticks(np.arange(len(ladder_df)))
        ax.set_xticklabels(ladder_df["cell"], rotation=35, ha="right", fontsize=8)
        ax.set_yticks([-0.2, 0.0, 1.0, 2.0, 3.0])
        ax.set_yticklabels(["no break", "random", "mutation", "heuristic", "search"])
        ax.set_ylabel(col.replace("first_", "").replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Harvest capability ladder")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _incidence_plot(aggression_df: pd.DataFrame, targeting_df: pd.DataFrame, output: Path) -> None:
    if aggression_df.empty and targeting_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    if not aggression_df.empty:
        focus = aggression_df[aggression_df["contrast_name"] == "hybrid_minus_top_down_only"].copy()
        if focus.empty:
            focus = aggression_df.copy()
        labels = focus["aggression_group"].astype(str)
        values = focus["delta__mean_welfare_mean"].to_numpy(dtype=float)
        axes[0].bar(labels, values, color="#8da0cb")
        axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        axes[0].set_title("Welfare delta by aggression tercile")
        axes[0].set_ylabel("Mean welfare delta")
    else:
        axes[0].axis("off")
    if not targeting_df.empty:
        focus = targeting_df[targeting_df["contrast_name"] == "hybrid_minus_top_down_only"].copy()
        if focus.empty:
            focus = targeting_df.copy()
        labels = focus["target_group"].astype(str)
        values = focus["delta__mean_welfare_mean"].to_numpy(dtype=float)
        axes[1].bar(labels, values, color="#66c2a5")
        axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        axes[1].set_title("Welfare delta by targeted status")
        axes[1].set_ylabel("Mean welfare delta")
    else:
        axes[1].axis("off")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prefix = Path(args.output_prefix)
    ranking_df = pd.read_csv(args.ranking_csv)
    contrast_ci_df = pd.read_csv(args.contrast_ci_csv)
    ladder_df = _load_optional_csv(args.capability_ladder_csv)
    aggression_df = _load_optional_csv(args.aggression_summary_csv)
    targeting_df = _load_optional_csv(args.targeting_summary_csv)

    _architecture_heatmap(ranking_df, prefix.with_name(prefix.name + "_architecture.png"))
    _contrast_bars(contrast_ci_df, prefix.with_name(prefix.name + "_architecture_contrasts.png"))
    _capability_ladder_plot(ladder_df, prefix.with_name(prefix.name + "_capability_ladder.png"))
    _incidence_plot(aggression_df, targeting_df, prefix.with_name(prefix.name + "_welfare_incidence.png"))


if __name__ == "__main__":
    main()
