import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAIN_METRICS = [
    ("hybrid_minus_top__test_garden_failure_mean", "Hybrid - top-down failure", "#264653"),
    ("hybrid_minus_top__test_mean_patch_health_mean", "Hybrid - top-down patch health", "#2a9d8f"),
    ("hybrid_minus_top__test_mean_welfare_mean", "Hybrid - top-down welfare", "#457b9d"),
]

MECH_METRICS = [
    ("hybrid_minus_top__test_mean_max_local_aggression_mean", "Hybrid - top-down local aggression", "#d1495b"),
    ("hybrid_minus_top__test_mean_neighborhood_overharvest_mean", "Hybrid - top-down neighborhood overharvest", "#e76f51"),
    ("hybrid_minus_top__test_mean_prevented_harvest_mean", "Hybrid - top-down prevented harvest", "#f4a261"),
    ("hybrid_minus_top__test_mean_credit_transferred_mean", "Hybrid - top-down credit transferred", "#8ab17d"),
]

TIER_LABELS = {"easy_h1": "Easy", "medium_h1": "Medium", "hard_h1": "Hard"}
MIX_LABELS = {"cooperative_heavy": "Coop", "balanced": "Bal", "adversarial_heavy": "Adv"}
INJECTOR_LABELS = {"random": "Rnd", "mutation": "Mut", "adversarial_heuristic": "AdvH"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Harvest invasion paired-delta confidence intervals.")
    parser.add_argument("--ci-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tiers", default="")
    parser.add_argument("--partner-mixes", default="")
    parser.add_argument("--injector-modes", default="")
    parser.add_argument("--pressures", default="")
    return parser.parse_args()


def _parse_csv_arg(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.tiers:
        out = out[out["tier"].isin(_parse_csv_arg(args.tiers))]
    if args.partner_mixes:
        out = out[out["partner_mix"].isin(_parse_csv_arg(args.partner_mixes))]
    if args.injector_modes:
        out = out[out["injector_mode_requested"].isin(_parse_csv_arg(args.injector_modes))]
    if args.pressures:
        out = out[out["adversarial_pressure"].isin([float(x) for x in _parse_csv_arg(args.pressures)])]
    return out.sort_values(["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure"]).reset_index(drop=True)


def _labels(df: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in df.iterrows():
        labels.append(
            "\n".join(
                [
                    TIER_LABELS.get(row["tier"], str(row["tier"])),
                    MIX_LABELS.get(row["partner_mix"], str(row["partner_mix"])),
                    INJECTOR_LABELS.get(row["injector_mode_requested"], str(row["injector_mode_requested"])),
                    f"p={row['adversarial_pressure']:.1f}",
                ]
            )
        )
    return labels


def _plot_grid(df: pd.DataFrame, metrics: list[tuple[str, str, str]], title: str, output: Path) -> None:
    n = len(metrics)
    rows = 2
    cols = 2 if n > 2 else n
    fig, axes = plt.subplots(rows, cols, figsize=(14, 9), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols).flatten()
    x = np.arange(len(df))
    labels = _labels(df)
    for ax, (metric, ylabel, color) in zip(axes, metrics):
        mean_col = f"{metric}_mean"
        lo_col = f"{metric}_ci95_low"
        hi_col = f"{metric}_ci95_high"
        means = df[mean_col].to_numpy(dtype=float)
        lowers = means - df[lo_col].to_numpy(dtype=float)
        uppers = df[hi_col].to_numpy(dtype=float) - means
        ax.bar(x, means, color=color, alpha=0.85)
        ax.errorbar(x, means, yerr=np.vstack([lowers, uppers]), fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.ci_csv)
    df = _filter(df, args)
    if df.empty:
        raise ValueError("No rows remain after applying plot filters.")
    prefix = Path(args.output_prefix)
    _plot_grid(
        df,
        MAIN_METRICS,
        "Harvest invasion: hybrid minus top-down paired deltas",
        prefix.with_name(prefix.name + "_main.png"),
    )
    _plot_grid(
        df,
        MECH_METRICS,
        "Harvest invasion: mechanism deltas",
        prefix.with_name(prefix.name + "_mechanisms.png"),
    )


if __name__ == "__main__":
    main()
