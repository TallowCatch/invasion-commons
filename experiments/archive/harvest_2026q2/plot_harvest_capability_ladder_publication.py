from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd


CELL_ORDER = [
    ("medium_h1", "balanced", 0.3, "Medium | balanced | 0.3"),
    ("medium_h1", "balanced", 0.5, "Medium | balanced | 0.5"),
    ("medium_h1", "adversarial_heavy", 0.3, "Medium | adversarial-heavy | 0.3"),
    ("medium_h1", "adversarial_heavy", 0.5, "Medium | adversarial-heavy | 0.5"),
    ("hard_h1", "balanced", 0.3, "Hard | balanced | 0.3"),
    ("hard_h1", "balanced", 0.5, "Hard | balanced | 0.5"),
    ("hard_h1", "adversarial_heavy", 0.3, "Hard | adversarial-heavy | 0.3"),
    ("hard_h1", "adversarial_heavy", 0.5, "Hard | adversarial-heavy | 0.5"),
]

CONTRAST_ORDER = [
    ("hybrid_minus_top_down_only", "H-TD"),
    ("hybrid_minus_bottom_up_only", "H-BU"),
    ("top_down_only_minus_bottom_up_only", "TD-BU"),
]

BREAK_COLUMNS = [
    ("first_ecological_break_injector", "Ecological break"),
    ("first_control_break_injector", "Control break"),
    ("first_costly_robustness_injector", "Costly robustness"),
]

RUNG_LABELS = {
    "none": "No break",
    "random": "Random",
    "mutation": "Mutation",
    "adversarial_heuristic": "Heuristic",
    "search_mutation": "Search",
}

RUNG_COLORS = {
    "none": "#E5E7EB",
    "random": "#D9F0D3",
    "mutation": "#A6DBA0",
    "adversarial_heuristic": "#5AAE61",
    "search_mutation": "#006D2C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a publication-style Harvest capability ladder figure.")
    parser.add_argument(
        "--ladder-csv",
        default="results/runs/showcase/curated/harvest_capability_ladder_stageB_capability_ladder.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="paper/paper_v4_institutional_commons/figures/harvest_capability_ladder_stageB_publication",
    )
    return parser.parse_args()


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot(df: pd.DataFrame, output_prefix: Path) -> None:
    _apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 6.1), sharey=True)
    text_color = {
        "none": "black",
        "random": "black",
        "mutation": "black",
        "adversarial_heuristic": "white",
        "search_mutation": "white",
    }

    for panel_idx, (ax, (break_col, title)) in enumerate(zip(axes, BREAK_COLUMNS)):
        ax.set_xlim(0, len(CONTRAST_ORDER))
        ax.set_ylim(0, len(CELL_ORDER))
        ax.invert_yaxis()
        ax.set_aspect("equal")

        for row_idx, (tier, partner_mix, pressure, row_label) in enumerate(CELL_ORDER):
            for col_idx, (contrast_name, contrast_label) in enumerate(CONTRAST_ORDER):
                row = df[
                    (df["tier"] == tier)
                    & (df["partner_mix"] == partner_mix)
                    & (df["adversarial_pressure"] == pressure)
                    & (df["contrast_name"] == contrast_name)
                ].iloc[0]
                rung = str(row[break_col])
                ax.add_patch(
                    Rectangle(
                        (col_idx, row_idx),
                        1,
                        1,
                        facecolor=RUNG_COLORS[rung],
                        edgecolor="white",
                        linewidth=1.6,
                    )
                )
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.52,
                    RUNG_LABELS[rung],
                    ha="center",
                    va="center",
                    fontsize=8.4,
                    color=text_color[rung],
                    fontweight="semibold" if rung in {"adversarial_heuristic", "search_mutation"} else "normal",
                )

        ax.set_xticks([idx + 0.5 for idx in range(len(CONTRAST_ORDER))])
        ax.set_xticklabels([label for _, label in CONTRAST_ORDER], fontsize=10.5)
        ax.xaxis.tick_top()
        ax.set_title(title, fontsize=12, pad=20)
        ax.tick_params(length=0, pad=8)

        if panel_idx == 0:
            ax.set_yticks([idx + 0.5 for idx in range(len(CELL_ORDER))])
            ax.set_yticklabels([label for _, _, _, label in CELL_ORDER], fontsize=9.5)
        else:
            ax.set_yticks([idx + 0.5 for idx in range(len(CELL_ORDER))])
            ax.tick_params(axis="y", left=False, labelleft=False)

        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=RUNG_COLORS[rung], edgecolor="none", label=label)
        for rung, label in [
            ("none", "No break"),
            ("random", "Random"),
            ("mutation", "Mutation"),
            ("adversarial_heuristic", "Heuristic"),
            ("search_mutation", "Search"),
        ]
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.24, right=0.99, top=0.86, bottom=0.13, wspace=0.52)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = _load(Path(args.ladder_csv))
    plot(df, Path(args.output_prefix))


if __name__ == "__main__":
    main()
