import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd


CONDITION_COLORS = {
    "none": "#d9d9d9",
    "bottom_up_only": "#66c2a5",
    "top_down_only": "#fc8d62",
    "hybrid": "#8da0cb",
}

CONTRAST_COLORS = {
    "hybrid_minus_top_down_only": "#8da0cb",
    "hybrid_minus_bottom_up_only": "#1b9e77",
    "top_down_only_minus_bottom_up_only": "#d95f02",
}

INJECTOR_LABELS = {
    "random": "Random",
    "mutation": "Mutation",
    "adversarial_heuristic": "Heuristic",
    "search_mutation": "Search",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Harvest architecture follow-up outputs.")
    parser.add_argument("--ranking-csv", required=True)
    parser.add_argument("--contrast-ci-csv", required=True)
    parser.add_argument("--capability-ladder-csv", default=None)
    parser.add_argument("--aggression-summary-csv", default=None)
    parser.add_argument("--targeting-summary-csv", default=None)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def _load_optional_csv(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    path_obj = Path(path)
    if not path_obj.exists():
        return pd.DataFrame()
    return pd.read_csv(path_obj)


def _cell_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    tier_order = {"medium_h1": 0, "hard_h1": 1}
    mix_order = {"balanced": 0, "adversarial_heavy": 1}
    out = df.copy()
    out["_tier_order"] = out["tier"].map(tier_order).fillna(99)
    out["_mix_order"] = out["partner_mix"].map(mix_order).fillna(99)
    out["_pressure_order"] = out["adversarial_pressure"].astype(float)
    return out


def _cell_label(row: pd.Series) -> str:
    parts: list[str] = []
    if "scenario_preset" in row.index and pd.notna(row["scenario_preset"]) and str(row["scenario_preset"]).strip():
        parts.append(str(row["scenario_preset"]).replace("_", " "))
    if "governance_friction_regime" in row.index and pd.notna(row["governance_friction_regime"]) and str(row["governance_friction_regime"]).strip():
        parts.append(str(row["governance_friction_regime"]))
    tier = row["tier"].replace("_h1", "").replace("_", " ")
    mix = row["partner_mix"].replace("_", " ")
    parts.extend([tier, mix, f"p={row['adversarial_pressure']}"])
    return " | ".join(parts)


def _architecture_effect_plot(contrast_ci_df: pd.DataFrame, output: Path) -> None:
    if contrast_ci_df.empty:
        return
    focus = contrast_ci_df[contrast_ci_df["contrast_name"] == "hybrid_minus_top_down_only"].copy()
    if focus.empty:
        return
    focus = _cell_sort_columns(focus)
    focus = focus.sort_values(["_tier_order", "_mix_order", "_pressure_order"]).reset_index(drop=True)
    focus["cell"] = focus.apply(_cell_label, axis=1)
    y = np.arange(len(focus))[::-1]
    metrics = [
        (
            "delta__test_mean_patch_health_mean_mean",
            "delta__test_mean_patch_health_mean_ci95_low",
            "delta__test_mean_patch_health_mean_ci95_high",
            "Patch health",
            "#355C7D",
        ),
        (
            "delta__test_mean_neighborhood_overharvest_mean_mean",
            "delta__test_mean_neighborhood_overharvest_mean_ci95_low",
            "delta__test_mean_neighborhood_overharvest_mean_ci95_high",
            "Neighborhood overharvest",
            "#2A9D8F",
        ),
        (
            "delta__test_mean_welfare_mean_mean",
            "delta__test_mean_welfare_mean_ci95_low",
            "delta__test_mean_welfare_mean_ci95_high",
            "Welfare",
            "#C06C84",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.8), sharey=True, constrained_layout=True)
    for idx, (ax, (metric_col, low_col, high_col, title, color)) in enumerate(zip(axes, metrics)):
        lows = focus[low_col].to_numpy(dtype=float)
        highs = focus[high_col].to_numpy(dtype=float)
        means = focus[metric_col].to_numpy(dtype=float)
        ax.errorbar(
            means,
            y,
            xerr=np.vstack([means - lows, highs - means]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            markersize=5,
        )
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(title, fontsize=10)
        if idx == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(focus["cell"], fontsize=8)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    axes[0].set_ylabel("Decision cell")
    fig.suptitle("Stage C: hybrid relative to top-down-only", fontsize=12)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _contrast_bars(contrast_ci_df: pd.DataFrame, output: Path) -> None:
    if contrast_ci_df.empty:
        return
    focus = contrast_ci_df[
        contrast_ci_df["contrast_name"].isin(
            ["hybrid_minus_top_down_only", "hybrid_minus_bottom_up_only", "top_down_only_minus_bottom_up_only"]
        )
    ].copy()
    if focus.empty:
        return
    focus["cell"] = focus.apply(
        lambda row: f"{row['tier']} | {row['partner_mix']} | {row['injector_mode_requested']} | p={row['adversarial_pressure']}",
        axis=1,
    )
    metrics = [
        ("delta__test_mean_patch_health_mean_mean", "Patch health"),
        ("delta__test_mean_welfare_mean_mean", "Welfare"),
        ("delta__test_mean_neighborhood_overharvest_mean_mean", "Neighborhood overharvest"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 10), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (metric_col, title) in zip(axes, metrics):
        labels = np.arange(len(focus))
        colors = [CONTRAST_COLORS.get(name, "#457b9d") for name in focus["contrast_name"]]
        means = focus[metric_col].to_numpy(dtype=float)
        base = metric_col.removesuffix("_mean")
        low = means - focus[f"{base}_ci95_low"].to_numpy(dtype=float)
        high = focus[f"{base}_ci95_high"].to_numpy(dtype=float) - means
        ax.bar(labels, means, color=colors, alpha=0.85)
        ax.errorbar(labels, means, yerr=np.vstack([low, high]), fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xticks(labels)
        ax.set_xticklabels(focus["cell"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Harvest architecture pairwise deltas")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _capability_ladder_plot(ladder_df: pd.DataFrame, output: Path) -> None:
    if ladder_df.empty:
        return
    rung_order = ["none", "random", "mutation", "adversarial_heuristic", "search_mutation"]
    rung_map = {name: idx for idx, name in enumerate(rung_order)}
    rung_label = {
        "none": "No break",
        "random": "Random",
        "mutation": "Mutation",
        "adversarial_heuristic": "Heuristic",
        "search_mutation": "Search",
    }
    contrast_order = [
        "hybrid_minus_top_down_only",
        "hybrid_minus_bottom_up_only",
        "top_down_only_minus_bottom_up_only",
    ]
    contrast_labels = {
        "hybrid_minus_top_down_only": "H - TD",
        "hybrid_minus_bottom_up_only": "H - BU",
        "top_down_only_minus_bottom_up_only": "TD - BU",
    }
    break_cols = [
        ("first_ecological_break_injector", "Ecological break"),
        ("first_control_break_injector", "Control break"),
        ("first_costly_robustness_injector", "Costly robustness"),
    ]
    ladder_df = _cell_sort_columns(ladder_df)
    ladder_df = ladder_df.sort_values(["_tier_order", "_mix_order", "_pressure_order"]).copy()
    ladder_df["cell"] = ladder_df.apply(_cell_label, axis=1)
    cell_order = list(dict.fromkeys(ladder_df["cell"].tolist()))
    cmap = ListedColormap(["#E5E7EB", "#C7E9C0", "#7BC87C", "#41AB5D", "#006D2C"])
    norm = BoundaryNorm(np.arange(-0.5, len(rung_order) + 0.5, 1), cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.6), constrained_layout=True)
    for idx, (ax, (col, title)) in enumerate(zip(axes, break_cols)):
        pivot = (
            ladder_df.assign(value=ladder_df[col].map(rung_map))
            .pivot_table(index="cell", columns="contrast_name", values="value", aggfunc="first")
            .reindex(index=cell_order, columns=contrast_order)
        )
        data = pivot.to_numpy()
        ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(contrast_order)))
        ax.set_xticklabels([contrast_labels[c] for c in contrast_order], rotation=20, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(cell_order)))
        if idx == 0:
            ax.set_yticklabels(cell_order, fontsize=8)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = rung_order[int(data[i, j])]
                ax.text(j, i, rung_label[value], ha="center", va="center", fontsize=6, color="black")
    fig.suptitle("Harvest capability ladder", fontsize=12)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _incidence_plot(aggression_df: pd.DataFrame, targeting_df: pd.DataFrame, output: Path) -> None:
    if aggression_df.empty and targeting_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    if not aggression_df.empty:
        focus = aggression_df[aggression_df["contrast_name"] == "hybrid_minus_top_down_only"].copy()
        if focus.empty:
            focus = aggression_df.copy()
        order = ["high_aggression", "mid_aggression", "low_aggression"]
        labels = {
            "high_aggression": "High aggression",
            "mid_aggression": "Mid aggression",
            "low_aggression": "Low aggression",
        }
        x_positions = np.arange(len(order))
        rng = np.random.default_rng(7)
        for idx, group in enumerate(order):
            values = focus.loc[focus["aggression_group"] == group, "delta__mean_welfare_mean"].to_numpy(dtype=float)
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            axes[0].scatter(np.full_like(values, x_positions[idx]) + jitter, values, color="#8da0cb", alpha=0.35, s=24)
            axes[0].scatter(x_positions[idx], values.mean(), color="#355C7D", s=55, zorder=3)
        axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        axes[0].set_xticks(x_positions)
        axes[0].set_xticklabels([labels[g] for g in order], fontsize=9)
        axes[0].set_title("Aggression terciles", fontsize=10)
        axes[0].set_ylabel("Welfare delta")
        axes[0].grid(axis="y", alpha=0.25)
    else:
        axes[0].axis("off")
    if not targeting_df.empty:
        focus = targeting_df[targeting_df["contrast_name"] == "hybrid_minus_top_down_only"].copy()
        if focus.empty:
            focus = targeting_df.copy()
        metrics = [
            ("delta__mean_welfare_mean", "Targeted welfare", "#C06C84"),
            ("delta__mean_local_patch_health_mean", "Targeted patch health", "#2A9D8F"),
        ]
        y_positions = np.arange(len(metrics))[::-1]
        for idx, (metric, label, color) in enumerate(metrics):
            values = focus[metric].to_numpy(dtype=float)
            jitter = np.linspace(-0.08, 0.08, num=len(values))
            axes[1].scatter(values, np.full_like(values, y_positions[idx]) + jitter, color=color, alpha=0.35, s=24)
            axes[1].scatter(values.mean(), y_positions[idx], color=color, edgecolor="black", linewidth=0.4, s=55, zorder=3)
        axes[1].axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
        axes[1].set_yticks(y_positions)
        axes[1].set_yticklabels([m[1] for m in metrics], fontsize=9)
        axes[1].set_title("Targeted agents", fontsize=10)
        axes[1].set_xlabel("Delta")
        axes[1].grid(axis="x", alpha=0.25)
    else:
        axes[1].axis("off")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prefix = Path(args.output_prefix)
    ranking_df = pd.read_csv(args.ranking_csv)
    contrast_ci_df = pd.read_csv(args.contrast_ci_csv)
    ladder_df = _load_optional_csv(args.capability_ladder_csv)
    aggression_df = _load_optional_csv(args.aggression_summary_csv)
    targeting_df = _load_optional_csv(args.targeting_summary_csv)

    _architecture_effect_plot(contrast_ci_df, prefix.with_name(prefix.name + "_architecture.png"))
    _contrast_bars(contrast_ci_df, prefix.with_name(prefix.name + "_architecture_contrasts.png"))
    _capability_ladder_plot(ladder_df, prefix.with_name(prefix.name + "_capability_ladder.png"))
    _incidence_plot(aggression_df, targeting_df, prefix.with_name(prefix.name + "_welfare_incidence.png"))


if __name__ == "__main__":
    main()
