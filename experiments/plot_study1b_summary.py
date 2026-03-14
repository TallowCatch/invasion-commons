from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INJECTOR_LABELS = {
    "mutation": "Mutation",
    "advheur": "Adversarial heuristic",
    "searchmut": "Search over mutations",
}

PARTNER_LABELS = {
    "balanced": "Balanced",
    "adversarial_heavy": "Adversarial-heavy",
}


def load_with_injector(path: Path, injector_key: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.copy()
    frame["injector_key"] = injector_key
    return frame


def collapse_reduction_table(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=["injector_key", "partner_mix", "adversarial_pressure"],
        columns="condition",
        values="test_collapse_mean_mean",
        aggfunc="first",
    ).reset_index()
    pivot["collapse_reduction"] = pivot["none"] - pivot["monitoring_sanctions"]
    return (
        pivot.groupby(["injector_key", "partner_mix"], as_index=False)["collapse_reduction"]
        .mean()
        .sort_values(["partner_mix", "injector_key"])
    )


def welfare_diff_table(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=["injector_key", "partner_mix", "adversarial_pressure"],
        columns="condition",
        values="test_mean_welfare_mean_mean",
        aggfunc="first",
    ).reset_index()
    pivot["adaptive_minus_closure_welfare"] = (
        pivot["adaptive_quota"] - pivot["temporary_closure"]
    )
    return (
        pivot.groupby(["injector_key", "partner_mix"], as_index=False)[
            "adaptive_minus_closure_welfare"
        ]
        .mean()
        .sort_values(["partner_mix", "injector_key"])
    )


def plot_grouped_bars(ax: plt.Axes, frame: pd.DataFrame, value_col: str, title: str, ylabel: str) -> None:
    partner_order = ["balanced", "adversarial_heavy"]
    injector_order = ["mutation", "advheur", "searchmut"]
    width = 0.24
    x = range(len(partner_order))
    for idx, injector in enumerate(injector_order):
        values = []
        for partner in partner_order:
            row = frame[
                (frame["injector_key"] == injector) & (frame["partner_mix"] == partner)
            ]
            values.append(float(row[value_col].iloc[0]))
        offsets = [pos + (idx - 1) * width for pos in x]
        ax.bar(offsets, values, width=width, label=INJECTOR_LABELS[injector])
    ax.set_xticks(list(x))
    ax.set_xticklabels([PARTNER_LABELS[p] for p in partner_order])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Study 1b summary figure.")
    parser.add_argument("--mutation-table", required=True)
    parser.add_argument("--advheur-table", required=True)
    parser.add_argument("--searchmut-table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.concat(
        [
            load_with_injector(Path(args.mutation_table), "mutation"),
            load_with_injector(Path(args.advheur_table), "advheur"),
            load_with_injector(Path(args.searchmut_table), "searchmut"),
        ],
        ignore_index=True,
    )

    collapse_frame = collapse_reduction_table(frame)
    welfare_frame = welfare_diff_table(frame)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_grouped_bars(
        axes[0],
        collapse_frame,
        "collapse_reduction",
        "Collapse reduction from monitoring with sanctions",
        "Mean held-out collapse reduction",
    )
    plot_grouped_bars(
        axes[1],
        welfare_frame,
        "adaptive_minus_closure_welfare",
        "Welfare advantage of adaptive quotas",
        "Mean welfare difference",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Fishery Commons: top-down governance summary", fontsize=14)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
