from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FISHERY_RUN_SPECS = [
    ("easy_v1", "mutation", "results/runs/ablation/curated/paper_v1_mutation_easy_v1_runs.csv"),
    ("easy_v1", "ollama_live", "results/runs/ablation/curated/paper_v1_llm_easy_v1_runs.csv"),
    ("medium_v1", "mutation", "results/runs/ablation/curated/paper_v1_mutation_medium_v1_runs.csv"),
    ("medium_v1", "ollama_live", "results/runs/ablation/curated/paper_v1_llm_medium_v1_runs.csv"),
]

FISHERY_TIER_LABELS = {
    "easy_v1": "Easy",
    "medium_v1": "Medium",
}

FISHERY_INJECTOR_LABELS = {
    "mutation": "Mutation",
    "ollama_live": "Live LLM",
}

FISHERY_INJECTOR_COLORS = {
    "mutation": "#ff8c42",
    "ollama_live": "#3d7fb1",
}

HARVEST_TIER_COLORS = {
    "hard_h1": "#355070",
    "medium_h1": "#2a9d8f",
}

HARVEST_TIER_LABELS = {
    "hard_h1": "Hard tier",
    "medium_h1": "Medium tier",
}

HARVEST_PARTNER_MARKERS = {
    "balanced": "o",
    "adversarial_heavy": "s",
}

HARVEST_PARTNER_LABELS = {
    "balanced": "Balanced",
    "adversarial_heavy": "Adversarial-heavy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate polished paper_v2 figures.")
    parser.add_argument(
        "--harvest-ci-csv",
        default="results/runs/showcase/curated/harvest_invasion_search_stageC_ci.csv",
    )
    parser.add_argument(
        "--fishery-output",
        default="paper/paper_v2/figures/governance_effect_ci.png",
    )
    parser.add_argument(
        "--harvest-output",
        default="paper/paper_v2/figures/harvest_tradeoff_scatter_talk.png",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def _bootstrap_ci(values: np.ndarray, *, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    point = float(values.mean())
    if values.size <= 1:
        return point, point, point
    samples = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        draw = values[rng.integers(0, values.size, size=values.size)]
        samples[idx] = float(draw.mean())
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return point, float(lo), float(hi)


def _load_fishery_baseline(n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, (tier, injector, path_text) in enumerate(FISHERY_RUN_SPECS):
        path = Path(path_text)
        df = pd.read_csv(path)
        none_df = df[df["condition"] == "none"].copy()
        ms_df = df[df["condition"] == "monitoring_sanctions"].copy()
        merged = ms_df.merge(
            none_df,
            on="run_id",
            suffixes=("_ms", "_none"),
            how="inner",
        )
        if merged.empty:
            raise ValueError(f"No matched runs found for {tier}/{injector} in {path}.")
        reduction = (
            merged["test_collapse_mean_none"].to_numpy(dtype=float)
            - merged["test_collapse_mean_ms"].to_numpy(dtype=float)
        )
        rng = np.random.default_rng(seed + idx)
        mean, lo, hi = _bootstrap_ci(reduction, n_boot=n_boot, rng=rng)
        rows.append(
            {
                "tier": tier,
                "injector": injector,
                "n_pairs": int(len(reduction)),
                "mean": mean,
                "low": lo,
                "high": hi,
                "label": f"{FISHERY_TIER_LABELS[tier]} / {FISHERY_INJECTOR_LABELS[injector]}",
            }
        )
    frame = pd.DataFrame(rows)
    order = pd.MultiIndex.from_tuples(
        [(tier, injector) for tier, injector, _ in FISHERY_RUN_SPECS],
        names=["tier", "injector"],
    )
    return (
        frame.set_index(["tier", "injector"])
        .loc[order]
        .reset_index()
    )


def _plot_fishery_baseline(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    y = np.arange(len(frame), dtype=float)
    labels = frame["label"].tolist()
    means = frame["mean"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    xerr = np.vstack([means - lows, highs - means])

    for band_start in (0, 2):
        ax.axhspan(band_start - 0.5, band_start + 1.5, color="#f5f5f5", zorder=0)
    ax.axhline(1.5, color="#d0d0d0", linewidth=1.0, zorder=1)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8, zorder=1)

    for idx, row in frame.iterrows():
        ax.errorbar(
            row["mean"],
            y[idx],
            xerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]],
            fmt="o",
            color=FISHERY_INJECTOR_COLORS[str(row["injector"])],
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            f"n={int(row['n_pairs'])}",
            xy=(row["high"], y[idx]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#555555",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Collapse reduction from monitoring with sanctions")
    ax.set_title("Fishery Commons matched baseline", fontsize=14, pad=10)
    ax.grid(axis="x", alpha=0.25)

    xmin = min(float(lows.min()), -0.02)
    xmax = max(float(highs.max()), 0.2)
    pad = 0.06 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad * 2.4)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7,
            label=FISHERY_INJECTOR_LABELS[injector],
        )
        for injector, color in FISHERY_INJECTOR_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load_harvest_ci(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
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
    df["plot_order"] = df.apply(
        lambda row: order[(row["tier"], row["partner_mix"], row["adversarial_pressure"])],
        axis=1,
    )
    return df.sort_values("plot_order").reset_index(drop=True)


def _plot_harvest_tradeoff(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.6), constrained_layout=True)

    x_col = "hybrid_minus_top__test_mean_welfare_mean_mean"
    y_col = "hybrid_minus_top__test_mean_patch_health_mean_mean"

    for (tier, partner_mix), gdf in df.groupby(["tier", "partner_mix"], sort=False):
        gdf = gdf.sort_values("adversarial_pressure")
        xs = gdf[x_col].to_numpy(dtype=float)
        ys = gdf[y_col].to_numpy(dtype=float)
        ax.plot(
            xs,
            ys,
            color=HARVEST_TIER_COLORS[str(tier)],
            linewidth=1.0,
            alpha=0.35,
            zorder=1,
        )

    x_range = float(df[x_col].max() - df[x_col].min())
    y_range = float(df[y_col].max() - df[y_col].min())
    x_offset = 0.014 * x_range if x_range > 0 else 0.01
    y_offset = 0.022 * y_range if y_range > 0 else 0.004

    for _, row in df.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        tier = str(row["tier"])
        partner_mix = str(row["partner_mix"])
        pressure = float(row["adversarial_pressure"])
        ax.scatter(
            x,
            y,
            s=90,
            color=HARVEST_TIER_COLORS[tier],
            marker=HARVEST_PARTNER_MARKERS[partner_mix],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
            zorder=3,
        )
        text_dx = x_offset if x <= 0 else -x_offset
        text_dy = y_offset if pressure <= 0.3 else -y_offset
        ax.text(
            x + text_dx,
            y + text_dy,
            f"p={pressure:.1f}",
            fontsize=8.5,
            color="#333333",
            ha="left" if text_dx > 0 else "right",
            va="center",
            zorder=4,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.75)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("Hybrid minus top-down welfare")
    ax.set_ylabel("Hybrid minus top-down patch health")
    ax.set_title("Harvest Commons: ecological gain versus welfare cost", fontsize=14, pad=10)
    ax.grid(alpha=0.25)

    tier_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=HARVEST_TIER_COLORS[tier],
            markeredgecolor="black",
            markersize=7,
            label=label,
        )
        for tier, label in HARVEST_TIER_LABELS.items()
    ]
    mix_handles = [
        Line2D(
            [0],
            [0],
            marker=HARVEST_PARTNER_MARKERS[mix],
            color="black",
            linestyle="None",
            markersize=7,
            label=label,
        )
        for mix, label in HARVEST_PARTNER_LABELS.items()
    ]
    legend_tier = ax.legend(
        handles=tier_handles,
        title="Tier",
        loc="upper left",
        bbox_to_anchor=(0.77, 0.98),
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
        bbox_to_anchor=(0.77, 0.80),
        frameon=False,
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=8,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    legend_mix._legend_box.align = "left"
    ax.text(
        0.77,
        0.67,
        "Pressure is shown\nby point labels.",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
        va="top",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fishery_frame = _load_fishery_baseline(
        n_boot=int(args.bootstrap_samples),
        seed=int(args.bootstrap_seed),
    )
    _plot_fishery_baseline(fishery_frame, Path(args.fishery_output))

    harvest_df = _load_harvest_ci(Path(args.harvest_ci_csv))
    _plot_harvest_tradeoff(harvest_df, Path(args.harvest_output))


if __name__ == "__main__":
    main()
