import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


FISHERY_COLORS = {
    "none": "#c9c9c9",
    "monitoring_sanctions": "#e9a03b",
    "adaptive_quota": "#2a9d8f",
    "temporary_closure": "#d55d47",
}

FISHERY_LABELS = {
    "none": "None",
    "monitoring_sanctions": "Sanctions",
    "adaptive_quota": "Adaptive quota",
    "temporary_closure": "Closure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate extra talk-friendly figures for Fishery and Harvest.")
    parser.add_argument(
        "--fishery-ranking-csvs",
        nargs="+",
        default=[
            "results/runs/ablation/curated/study1b_medium_hp_mutation_ranking.csv",
            "results/runs/ablation/curated/study1b_medium_hp_advheur_ranking.csv",
            "results/runs/ablation/curated/study1b_medium_hp_searchmut_ranking.csv",
        ],
    )
    parser.add_argument(
        "--harvest-ci-csv",
        default="results/runs/showcase/curated/harvest_invasion_search_stageC_ci.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/runs/showcase/curated/talk_figures",
    )
    return parser.parse_args()


def _load_fishery_winners(csv_paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df = df[df["rank"] == 1].copy()
        path_lower = Path(path).name.lower()
        if "advheur" in path_lower:
            injector = "Adversarial\nheuristic"
        elif "searchmut" in path_lower:
            injector = "Search\nmutation"
        else:
            injector = "Mutation"
        df["injector_label"] = injector
        frames.append(df)
    winners = pd.concat(frames, ignore_index=True)
    rows = []
    for (injector, mix, pressure), gdf in winners.groupby(
        ["injector_label", "partner_mix", "adversarial_pressure"], sort=True
    ):
        counts = gdf["condition"].value_counts()
        winner = counts.idxmax()
        rows.append(
            {
                "injector_label": injector,
                "partner_mix": mix,
                "adversarial_pressure": pressure,
                "winner": winner,
                "winner_count": int(counts.max()),
                "total_runs": int(len(gdf)),
            }
        )
    return pd.DataFrame(rows)


def _plot_fishery_winner_map(df: pd.DataFrame, output: Path) -> None:
    row_order = ["Mutation", "Adversarial\nheuristic", "Search\nmutation"]
    col_order = [
        ("balanced", 0.3),
        ("balanced", 0.5),
        ("adversarial_heavy", 0.3),
        ("adversarial_heavy", 0.5),
    ]
    col_labels = ["Bal\np=0.3", "Bal\np=0.5", "Adv\np=0.3", "Adv\np=0.5"]

    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    ax.set_xlim(0, len(col_order))
    ax.set_ylim(0, len(row_order))
    ax.invert_yaxis()

    for yi, row_label in enumerate(row_order):
        for xi, (mix, pressure) in enumerate(col_order):
            cell = df[
                (df["injector_label"] == row_label)
                & (df["partner_mix"] == mix)
                & (df["adversarial_pressure"] == pressure)
            ]
            if cell.empty:
                winner = "none"
                count = 0
                total = 0
            else:
                winner = cell.iloc[0]["winner"]
                count = int(cell.iloc[0]["winner_count"])
                total = int(cell.iloc[0]["total_runs"])
            ax.add_patch(
                plt.Rectangle(
                    (xi, yi),
                    1,
                    1,
                    facecolor=FISHERY_COLORS[winner],
                    edgecolor="white",
                    linewidth=2,
                )
            )
            ax.text(
                xi + 0.5,
                yi + 0.42,
                FISHERY_LABELS[winner],
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )
            ax.text(
                xi + 0.5,
                yi + 0.72,
                f"{count}/{total} runs",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    ax.set_xticks(np.arange(len(col_order)) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(row_order)) + 0.5)
    ax.set_yticklabels(row_order, fontsize=10)
    ax.set_title("Fishery Commons: top-down policy winners by hostile cell", fontsize=14, pad=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_items = [Patch(facecolor=color, label=label) for key, color in FISHERY_COLORS.items() for label in [FISHERY_LABELS[key]]]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_harvest_ci(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["cell_label"] = df.apply(
        lambda r: f"{'Hard' if r['tier']=='hard_h1' else 'Medium'} / "
        f"{'Adv' if r['partner_mix']=='adversarial_heavy' else 'Bal'} / "
        f"p={r['adversarial_pressure']:.1f}",
        axis=1,
    )
    df["short_label"] = df.apply(
        lambda r: f"{'H' if r['tier']=='hard_h1' else 'M'}/"
        f"{'A' if r['partner_mix']=='adversarial_heavy' else 'B'}/"
        f"{r['adversarial_pressure']:.1f}",
        axis=1,
    )
    order = {
        ("hard_h1", "adversarial_heavy", 0.3): 0,
        ("hard_h1", "adversarial_heavy", 0.5): 1,
        ("hard_h1", "balanced", 0.3): 2,
        ("hard_h1", "balanced", 0.5): 3,
        ("medium_h1", "adversarial_heavy", 0.3): 4,
        ("medium_h1", "adversarial_heavy", 0.5): 5,
        ("medium_h1", "balanced", 0.3): 6,
        ("medium_h1", "balanced", 0.5): 7,
    }
    df["plot_order"] = df.apply(lambda r: order[(r["tier"], r["partner_mix"], r["adversarial_pressure"])], axis=1)
    return df.sort_values("plot_order").reset_index(drop=True)


def _plot_harvest_tradeoff(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    colors = {"hard_h1": "#355070", "medium_h1": "#2a9d8f"}
    markers = {"balanced": "o", "adversarial_heavy": "s"}

    for _, row in df.iterrows():
        x = row["hybrid_minus_top__test_mean_welfare_mean_mean"]
        y = row["hybrid_minus_top__test_mean_patch_health_mean_mean"]
        ax.scatter(
            x,
            y,
            s=90,
            color=colors[row["tier"]],
            marker=markers[row["partner_mix"]],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.axvline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_xlabel("Hybrid minus top-down welfare")
    ax.set_ylabel("Hybrid minus top-down patch health")
    ax.set_title("Harvest Commons: ecological gain versus welfare cost", fontsize=14, pad=10)
    ax.grid(alpha=0.25)

    tier_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#355070", markeredgecolor="black", label="Hard tier", markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2a9d8f", markeredgecolor="black", label="Medium tier", markersize=6),
    ]
    mix_handles = [
        plt.Line2D([0], [0], marker="o", color="black", linestyle="None", label="Balanced", markersize=6),
        plt.Line2D([0], [0], marker="s", color="black", linestyle="None", label="Adversarial-heavy", markersize=6),
    ]
    legend_tier = ax.legend(
        handles=tier_handles,
        title="Tier",
        loc="upper left",
        bbox_to_anchor=(0.79, 0.98),
        frameon=False,
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=8,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    legend_tier._legend_box.align = "left"
    ax.add_artist(legend_tier)
    legend_mix = ax.legend(
        handles=mix_handles,
        title="Partner mix",
        loc="upper left",
        bbox_to_anchor=(0.79, 0.80),
        frameon=False,
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=8,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    legend_mix._legend_box.align = "left"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_harvest_forest(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), constrained_layout=True)
    y = np.arange(len(df))
    labels = df["cell_label"].tolist()

    specs = [
        (
            "hybrid_minus_top__test_mean_patch_health_mean",
            "Patch health delta",
            "#2a9d8f",
        ),
        (
            "hybrid_minus_top__test_mean_welfare_mean",
            "Welfare delta",
            "#457b9d",
        ),
    ]

    for ax, (base, title, color) in zip(axes, specs):
        means = df[f"{base}_mean"].to_numpy()
        lows = df[f"{base}_ci95_low"].to_numpy()
        highs = df[f"{base}_ci95_high"].to_numpy()
        xerr = np.vstack([means - lows, highs - means])
        ax.errorbar(means, y, xerr=xerr, fmt="o", color=color, ecolor="black", capsize=3, markersize=6)
        ax.axvline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
    axes[0].set_xlabel("Hybrid minus top-down")
    axes[1].set_xlabel("Hybrid minus top-down")
    fig.suptitle("Harvest Commons: cell-level confidence intervals", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prefix = Path(args.output_prefix)

    fishery_df = _load_fishery_winners(args.fishery_ranking_csvs)
    harvest_df = _load_harvest_ci(args.harvest_ci_csv)

    _plot_fishery_winner_map(fishery_df, prefix.with_name(prefix.name + "_fishery_winner_map.png"))
    _plot_harvest_tradeoff(harvest_df, prefix.with_name(prefix.name + "_harvest_tradeoff_scatter.png"))
    _plot_harvest_forest(harvest_df, prefix.with_name(prefix.name + "_harvest_cell_forest.png"))


if __name__ == "__main__":
    main()
