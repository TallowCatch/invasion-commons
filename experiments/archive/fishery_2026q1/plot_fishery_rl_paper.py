from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "none": "#c9c9c9",
    "monitoring_sanctions": "#e9a03b",
    "temporary_closure": "#d55d47",
    "adaptive_quota": "#2a9d8f",
}

LABELS = {
    "none": "None",
    "monitoring_sanctions": "Sanctions",
    "temporary_closure": "Closure",
    "adaptive_quota": "Adaptive quota",
}

ORDER = ["none", "monitoring_sanctions", "temporary_closure", "adaptive_quota"]

METRICS = [
    (
        "test_collapse_mean_mean",
        "test_collapse_mean_ci95_low",
        "test_collapse_mean_ci95_high",
        "Held-out collapse",
        "Mean collapse rate",
    ),
    (
        "test_mean_stock_mean_mean",
        "test_mean_stock_mean_ci95_low",
        "test_mean_stock_mean_ci95_high",
        "Held-out stock",
        "Mean stock",
    ),
    (
        "test_mean_welfare_mean_mean",
        "test_mean_welfare_mean_ci95_low",
        "test_mean_welfare_mean_ci95_high",
        "Held-out welfare",
        "Mean welfare",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Fishery RL summary figure for paper_v2.")
    parser.add_argument(
        "--governance-table",
        default="results/runs/rl_fishery/curated/fishery_rl_medium_v1_governance_followup_table.csv",
    )
    parser.add_argument(
        "--confirmatory-table",
        default="results/runs/rl_fishery/curated/fishery_rl_medium_v1_adaptive_quota_confirmatory_table.csv",
    )
    parser.add_argument(
        "--output",
        default="paper/paper_v2/figures/study1_rl_summary.png",
    )
    return parser.parse_args()


def load_frame(governance_path: Path, confirmatory_path: Path) -> pd.DataFrame:
    governance = pd.read_csv(governance_path)
    confirmatory = pd.read_csv(confirmatory_path)

    none_row = confirmatory[confirmatory["condition"] == "none"].copy()
    governed_rows = governance[
        governance["condition"].isin(["monitoring_sanctions", "temporary_closure", "adaptive_quota"])
    ].copy()
    frame = pd.concat([none_row, governed_rows], ignore_index=True)
    return frame.set_index("condition").loc[ORDER].reset_index()


def main() -> None:
    args = parse_args()
    frame = load_frame(Path(args.governance_table), Path(args.confirmatory_table))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), constrained_layout=True)
    x = np.arange(len(ORDER))

    for ax, (value_col, low_col, high_col, title, ylabel) in zip(axes, METRICS):
        values = frame[value_col].to_numpy(dtype=float)
        low = frame[low_col].to_numpy(dtype=float)
        high = frame[high_col].to_numpy(dtype=float)
        yerr = np.vstack([values - low, high - values])
        colors = [COLORS[c] for c in frame["condition"]]
        ax.bar(x, values, yerr=yerr, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in frame["condition"]], rotation=20, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylim(bottom=0.0)
    fig.suptitle("Fishery Commons PPO validation on medium_v1", fontsize=13)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
