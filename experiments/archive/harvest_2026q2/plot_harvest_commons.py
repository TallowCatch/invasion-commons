import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = ["none", "top_down_only", "bottom_up_only", "hybrid"]
CONDITION_LABELS = {
    "none": "None",
    "top_down_only": "Top-down only",
    "bottom_up_only": "Bottom-up only",
    "hybrid": "Hybrid",
}
CONDITION_COLORS = {
    "none": "#8d99ae",
    "top_down_only": "#d1495b",
    "bottom_up_only": "#2a9d8f",
    "hybrid": "#264653",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Harvest Commons study outputs.")
    parser.add_argument("--table-csv", required=True)
    parser.add_argument(
        "--output-prefix",
        default="results/runs/showcase/curated/harvest_commons_batch",
    )
    return parser.parse_args()


def _plot_metric_grid(
    df: pd.DataFrame,
    metrics: list[tuple[str, str]],
    output_path: Path,
    suptitle: str,
) -> None:
    social_mixes = df["social_mix"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    width = 0.18
    x = np.arange(len(social_mixes))

    for ax, (metric, ylabel) in zip(axes, metrics):
        for idx, condition in enumerate(CONDITION_ORDER):
            subset = (
                df[df["condition"] == condition]
                .set_index("social_mix")
                .reindex(social_mixes)
            )
            values = subset[metric].to_numpy(dtype=float)
            ax.bar(
                x + (idx - 1.5) * width,
                values,
                width=width,
                label=CONDITION_LABELS[condition],
                color=CONDITION_COLORS[condition],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([mix.replace("_", "\n") for mix in social_mixes])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper left", ncols=2, fontsize=9)
    fig.suptitle(suptitle, fontsize=16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_hybrid_deltas(df: pd.DataFrame, output_path: Path) -> None:
    social_mixes = df["social_mix"].drop_duplicates().tolist()
    top = df[df["condition"] == "top_down_only"].set_index("social_mix").reindex(social_mixes)
    hybrid = df[df["condition"] == "hybrid"].set_index("social_mix").reindex(social_mixes)
    delta = pd.DataFrame(
        {
            "social_mix": social_mixes,
            "patch_health_delta": hybrid["mean_patch_health_mean"] - top["mean_patch_health_mean"],
            "welfare_delta": hybrid["mean_welfare_mean"] - top["mean_welfare_mean"],
            "local_aggression_delta": hybrid["mean_max_local_aggression_mean"] - top["mean_max_local_aggression_mean"],
            "neighborhood_overharvest_delta": hybrid["mean_neighborhood_overharvest_mean"]
            - top["mean_neighborhood_overharvest_mean"],
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()
    metric_specs = [
        ("patch_health_delta", "Hybrid - top-down patch health", "#2a9d8f"),
        ("welfare_delta", "Hybrid - top-down welfare", "#264653"),
        ("local_aggression_delta", "Hybrid - top-down max local aggression", "#d1495b"),
        ("neighborhood_overharvest_delta", "Hybrid - top-down neighborhood overharvest", "#e76f51"),
    ]
    x = np.arange(len(social_mixes))

    for ax, (metric, ylabel, color) in zip(axes, metric_specs):
        values = delta[metric].to_numpy(dtype=float)
        ax.bar(x, values, color=color, width=0.55)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([mix.replace("_", "\n") for mix in social_mixes])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Hybrid minus top-down deltas by social mix", fontsize=16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table_csv)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    health_metrics = [
        ("mean_patch_health_mean", "Mean patch health"),
        ("mean_welfare_mean", "Mean welfare"),
        ("payoff_gini_mean", "Payoff Gini"),
        ("mean_patch_variance_mean", "Patch variance"),
    ]
    mechanism_metrics = [
        ("mean_aggressive_request_fraction_mean", "Aggressive request fraction"),
        ("mean_max_local_aggression_mean", "Max local aggression"),
        ("mean_capped_action_fraction_mean", "Capped action fraction"),
        ("mean_neighborhood_overharvest_mean", "Neighborhood overharvest"),
    ]

    _plot_metric_grid(
        df=df,
        metrics=health_metrics,
        output_path=output_prefix.with_name(output_prefix.name + "_health_welfare.png"),
        suptitle="Harvest Commons Outcomes by Social Mix and Governance",
    )
    _plot_metric_grid(
        df=df,
        metrics=mechanism_metrics,
        output_path=output_prefix.with_name(output_prefix.name + "_mechanisms.png"),
        suptitle="Harvest Commons Mechanisms by Social Mix and Governance",
    )
    _plot_hybrid_deltas(
        df=df,
        output_path=output_prefix.with_name(output_prefix.name + "_hybrid_vs_topdown_deltas.png"),
    )


if __name__ == "__main__":
    main()
