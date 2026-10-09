from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd


SCENARIO_ORDER = [
    ("community_irrigation", "Community irrigation"),
    ("regulated_fishery", "Regulated fishery"),
    ("forest_co_management", "Forest co-management"),
]

COLUMN_ORDER = [
    ("ideal", 0.3, "Ideal\n0.3"),
    ("ideal", 0.5, "Ideal\n0.5"),
    ("constrained", 0.3, "Constr.\n0.3"),
    ("constrained", 0.5, "Constr.\n0.5"),
]

WINNER_COLORS = {
    "hybrid": "#8DA0CB",
    "top_down_only": "#FC8D62",
    "bottom_up_only": "#66C2A5",
    "none": "#D9D9D9",
}

WINNER_LABELS = {
    "hybrid": "H",
    "top_down_only": "TD",
    "bottom_up_only": "BU",
    "none": "N",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a publication-style winner map for the institutional-friction matrix.")
    parser.add_argument("--showcase-dir", default="results/runs/showcase/curated")
    parser.add_argument(
        "--output-prefix",
        default="paper/paper_v4_institutional_commons/figures/institutional_friction_winner_map",
    )
    return parser.parse_args()


def _load_matrix(showcase_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scenario_key, scenario_label in SCENARIO_ORDER:
        ranking_path = showcase_dir / f"harvest_institutional_friction_{scenario_key}_ranking.csv"
        contrast_path = showcase_dir / f"harvest_institutional_friction_{scenario_key}_contrast_ci.csv"
        ranking = pd.read_csv(ranking_path)
        ranking = ranking[ranking["rank"] == 1].copy()
        contrast = pd.read_csv(contrast_path)
        contrast = contrast[contrast["contrast_name"] == "hybrid_minus_top_down_only"].copy()
        merged = ranking.merge(
            contrast[
                [
                    "governance_friction_regime",
                    "adversarial_pressure",
                    "delta__test_mean_patch_health_mean_mean",
                ]
            ],
            on=["governance_friction_regime", "adversarial_pressure"],
            how="left",
        )
        merged["scenario_label"] = scenario_label
        rows.append(
            merged[
                [
                    "scenario_label",
                    "governance_friction_regime",
                    "adversarial_pressure",
                    "condition",
                    "delta__test_mean_patch_health_mean_mean",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


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


def plot_winner_map(df: pd.DataFrame, output_prefix: Path) -> None:
    _apply_style()

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.set_xlim(0, len(COLUMN_ORDER))
    ax.set_ylim(0, len(SCENARIO_ORDER))
    ax.invert_yaxis()
    ax.set_aspect("equal")

    for row_idx, (_, scenario_label) in enumerate(SCENARIO_ORDER):
        for col_idx, (regime, pressure, _) in enumerate(COLUMN_ORDER):
            row = df[
                (df["scenario_label"] == scenario_label)
                & (df["governance_friction_regime"] == regime)
                & (df["adversarial_pressure"] == pressure)
            ].iloc[0]
            winner = row["condition"]
            delta = float(row["delta__test_mean_patch_health_mean_mean"])
            ax.add_patch(
                Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    facecolor=WINNER_COLORS[winner],
                    edgecolor="white",
                    linewidth=2.3,
                )
            )
            ax.text(
                col_idx + 0.5,
                row_idx + 0.40,
                WINNER_LABELS[winner],
                ha="center",
                va="center",
                fontsize=19,
                fontweight="bold",
                color="black",
            )
            ax.text(
                col_idx + 0.5,
                row_idx + 0.72,
                rf"$\Delta$PH {delta:+.2f}",
                ha="center",
                va="center",
                fontsize=10.5,
                color="black",
            )

    ax.set_xticks([idx + 0.5 for idx in range(len(COLUMN_ORDER))])
    ax.set_xticklabels([label for _, _, label in COLUMN_ORDER], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks([idx + 0.5 for idx in range(len(SCENARIO_ORDER))])
    ax.set_yticklabels([label for _, label in SCENARIO_ORDER], fontsize=11)
    ax.tick_params(length=0, pad=6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=WINNER_COLORS["hybrid"], edgecolor="none", label="Hybrid wins"),
        Patch(facecolor=WINNER_COLORS["top_down_only"], edgecolor="none", label="Top-down-only wins"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout(pad=0.25)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    showcase_dir = Path(args.showcase_dir)
    output_prefix = Path(args.output_prefix)
    df = _load_matrix(showcase_dir)
    plot_winner_map(df, output_prefix)


if __name__ == "__main__":
    main()
