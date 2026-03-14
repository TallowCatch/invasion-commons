import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_SPECS = [
    ("hybrid_minus_top__test_mean_patch_health_mean", "Patch health", "#2a9d8f"),
    ("hybrid_minus_top__test_mean_welfare_mean", "Welfare", "#457b9d"),
    ("hybrid_minus_top__test_mean_max_local_aggression_mean", "Local aggression", "#d1495b"),
    ("hybrid_minus_top__test_mean_neighborhood_overharvest_mean", "Neighborhood overharvest", "#e76f51"),
]

GROUP_LABELS = {
    ("hard_h1", "adversarial_heavy"): "Hard / Adv",
    ("hard_h1", "balanced"): "Hard / Bal",
    ("medium_h1", "adversarial_heavy"): "Medium / Adv",
    ("medium_h1", "balanced"): "Medium / Bal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compact paper figures for Harvest invasion Stage C.")
    parser.add_argument("--delta-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_samples: int) -> tuple[float, float, float]:
    point = float(values.mean())
    if values.size <= 1:
        return point, point, point
    samples = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        draw = values[rng.integers(0, values.size, size=values.size)]
        samples[idx] = float(draw.mean())
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return point, float(lo), float(hi)


def _aggregate(delta_df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    order = [
        ("hard_h1", "adversarial_heavy"),
        ("hard_h1", "balanced"),
        ("medium_h1", "adversarial_heavy"),
        ("medium_h1", "balanced"),
    ]
    for tier, mix in order:
        gdf = delta_df[(delta_df["tier"] == tier) & (delta_df["partner_mix"] == mix)]
        if gdf.empty:
            continue
        row = {
            "tier": tier,
            "partner_mix": mix,
            "label": GROUP_LABELS[(tier, mix)],
            "n_pairs": int(len(gdf)),
        }
        for metric, _, _ in METRIC_SPECS:
            mean, lo, hi = _bootstrap_ci(gdf[metric].to_numpy(dtype=float), rng, n_samples)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, metrics: list[tuple[str, str, str]], title: str, output: Path) -> None:
    x = np.arange(len(df))
    labels = df["label"].tolist()
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (metric, ylabel, color) in zip(axes, metrics):
        mean_col = f"{metric}_mean"
        lo_col = f"{metric}_ci95_low"
        hi_col = f"{metric}_ci95_high"
        means = df[mean_col].to_numpy(dtype=float)
        lowers = means - df[lo_col].to_numpy(dtype=float)
        uppers = df[hi_col].to_numpy(dtype=float) - means
        ax.bar(x, means, color=color, alpha=0.88, width=0.65)
        ax.errorbar(x, means, yerr=np.vstack([lowers, uppers]), fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=9)
        ax.set_ylabel(f"Hybrid - top-down {ylabel}")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(title, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    delta_df = pd.read_csv(args.delta_csv)
    aggregated = _aggregate(delta_df, n_samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    prefix = Path(args.output_prefix)
    aggregated.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    _plot(
        aggregated,
        METRIC_SPECS[:2],
        "Harvest Commons: search-over-mutations outcomes",
        prefix.with_name(prefix.name + "_outcomes.png"),
    )
    _plot(
        aggregated,
        METRIC_SPECS[2:],
        "Harvest Commons: search-over-mutations mechanism shifts",
        prefix.with_name(prefix.name + "_mechanisms.png"),
    )


if __name__ == "__main__":
    main()
