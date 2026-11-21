import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_SPECS = [
    ("patch_health", "Hybrid - top-down patch health", "#2a9d8f"),
    ("welfare", "Hybrid - top-down welfare", "#264653"),
    ("max_local_aggression", "Hybrid - top-down local aggression", "#d1495b"),
    ("neighborhood_overharvest", "Hybrid - top-down neighborhood overharvest", "#e76f51"),
]

TIER_LABELS = {
    "easy_h1": "Easy",
    "medium_h1": "Medium",
    "hard_h1": "Hard",
}
TIER_ORDER = ["easy_h1", "medium_h1", "hard_h1"]
MIX_ORDER = ["adversarial_heavy", "cooperative_heavy", "mixed_pressure"]

MIX_LABELS = {
    "adversarial_heavy": "Adv",
    "cooperative_heavy": "Coop",
    "mixed_pressure": "Mix",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Harvest Commons high-power paired deltas with 95% bootstrap confidence intervals."
    )
    parser.add_argument("--ci-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.ci_csv)
    tier_order = [tier for tier in TIER_ORDER if tier in set(df["tier"])]
    social_mix_order = [mix for mix in MIX_ORDER if mix in set(df["social_mix"])]
    contexts = [(tier, social_mix) for tier in tier_order for social_mix in social_mix_order]
    x = np.arange(len(contexts))

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    axes = axes.flatten()

    for ax, (metric, ylabel, color) in zip(axes, METRIC_SPECS):
        metric_df = df[df["metric"] == metric].copy()
        metric_df["context"] = list(zip(metric_df["tier"], metric_df["social_mix"]))
        metric_df = metric_df.set_index("context").reindex(contexts)
        means = metric_df["delta_mean"].to_numpy(dtype=float)
        lows = metric_df["ci95_low"].to_numpy(dtype=float)
        highs = metric_df["ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([means - lows, highs - means])
        ax.bar(x, means, color=color, width=0.62, alpha=0.92)
        ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([MIX_LABELS[mix] for _, mix in contexts], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for split in range(1, len(tier_order)):
            ax.axvline(split * len(social_mix_order) - 0.5, color="gray", linewidth=0.8, alpha=0.4)
        y_text = -0.20
        for idx, tier in enumerate(tier_order):
            center = idx * len(social_mix_order) + (len(social_mix_order) - 1) / 2
            ax.text(
                center,
                y_text,
                TIER_LABELS.get(tier, tier),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )
        ax.text(
            -0.55,
            y_text,
            "Mix:",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=9,
        )

    fig.suptitle("Harvest Commons: hybrid minus top-down deltas (95% bootstrap CIs)", fontsize=16)
    fig.subplots_adjust(bottom=0.16, top=0.90, wspace=0.26, hspace=0.34)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
