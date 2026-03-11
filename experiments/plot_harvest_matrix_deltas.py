import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("hybrid_minus_top_patch_health", "Hybrid - top-down patch health", "#2a9d8f"),
    ("hybrid_minus_top_welfare", "Hybrid - top-down welfare", "#264653"),
    ("hybrid_minus_top_local_aggression", "Hybrid - top-down local aggression", "#d1495b"),
    ("hybrid_minus_top_neighborhood_overharvest", "Hybrid - top-down neighborhood overharvest", "#e76f51"),
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
    parser = argparse.ArgumentParser(description="Plot Harvest Commons hybrid-vs-top-down deltas.")
    parser.add_argument("--delta-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.delta_csv)
    tiers = [tier for tier in TIER_ORDER if tier in set(df["tier"])]
    social_mixes = [mix for mix in MIX_ORDER if mix in set(df["social_mix"])]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes = axes.flatten()
    width = 0.22
    x = np.arange(len(social_mixes))
    palette = ["#8ecae6", "#219ebc", "#023047", "#ffb703"]
    tier_colors = {tier: palette[idx % len(palette)] for idx, tier in enumerate(tiers)}

    for ax, (metric, ylabel, _) in zip(axes, METRICS):
        for idx, tier in enumerate(tiers):
            subset = df[df["tier"] == tier].set_index("social_mix").reindex(social_mixes)
            ax.bar(
                x + (idx - (len(tiers) - 1) / 2) * width,
                subset[metric].to_numpy(dtype=float),
                width=width,
                label=tier,
                color=tier_colors[tier],
            )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([MIX_LABELS.get(mix, mix) for mix in social_mixes], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(
        labels=[TIER_LABELS.get(tier, tier) for tier in tiers],
        loc="upper left",
        fontsize=9,
        title="Tier",
        title_fontsize=9,
    )
    fig.suptitle("Harvest Commons: hybrid minus top-down mean deltas", fontsize=16)
    fig.supxlabel("Social mix")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
