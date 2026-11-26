from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper_v2 contextual/mecanism artifacts from curated paper_v1 runs.")
    p.add_argument("--ablation-dir", default="results/runs/ablation/curated")
    p.add_argument("--showcase-dir", default="results/runs/showcase/curated")
    p.add_argument("--tag", default="paper_v1")
    return p.parse_args()


def _load_runs(ablation_dir: Path, injector: str, tier: str) -> pd.DataFrame:
    return pd.read_csv(ablation_dir / f"paper_v1_{injector}_{tier}_runs.csv")


def build_context_table(ablation_dir: Path, showcase_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for injector in ["mutation", "llm"]:
        runs = _load_runs(ablation_dir, injector, "medium_v1")
        none = runs[runs["condition"] == "none"].sort_values("run_id")
        ms = runs[runs["condition"] == "monitoring_sanctions"].sort_values("run_id")
        merged = ms.merge(none, on="run_id", suffixes=("_ms", "_none"), how="inner")
        if merged.empty:
            continue

        baseline_collapse = float(merged["test_collapse_mean_none"].mean())
        collapse_reduction = float((merged["test_collapse_mean_none"] - merged["test_collapse_mean_ms"]).mean())
        relative_collapse_reduction_pct = float(
            100.0
            * (
                (merged["test_collapse_mean_none"] - merged["test_collapse_mean_ms"])
                / merged["test_collapse_mean_none"].replace(0.0, np.nan)
            ).mean()
        )

        baseline_ttc = float(merged["time_to_collapse_none"].mean())
        ttc_gain = float((merged["time_to_collapse_ms"] - merged["time_to_collapse_none"]).mean())
        relative_ttc_gain_pct = float(
            100.0
            * (
                (merged["time_to_collapse_ms"] - merged["time_to_collapse_none"])
                / merged["time_to_collapse_none"].replace(0.0, np.nan)
            ).mean()
        )

        rows.append(
            {
                "injector": "mutation" if injector == "mutation" else "live_llm_json",
                "tier": "medium_v1",
                "n_pairs": int(len(merged)),
                "baseline_collapse_none": baseline_collapse,
                "collapse_reduction_abs": collapse_reduction,
                "collapse_reduction_rel_pct": relative_collapse_reduction_pct,
                "baseline_time_to_collapse_none": baseline_ttc,
                "time_to_collapse_gain_abs": ttc_gain,
                "time_to_collapse_gain_rel_pct": relative_ttc_gain_pct,
            }
        )

    out = pd.DataFrame(rows)
    out_path = showcase_dir / "paper_v2_table_context_effects_medium.csv"
    out.to_csv(out_path, index=False)
    return out


def build_mechanism_table(ablation_dir: Path, showcase_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for injector in ["mutation", "llm"]:
        for condition in ["none", "monitoring", "monitoring_sanctions"]:
            paths = sorted(ablation_dir.glob(f"paper_v1_{injector}_medium_v1_{condition}_run*_generations.csv"))
            if not paths:
                continue
            df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
            rows.append(
                {
                    "injector": "mutation" if injector == "mutation" else "live_llm_json",
                    "condition": condition,
                    "n_rows": int(len(df)),
                    "mean_high_harvest_frac": float(df["mean_high_harvest_frac"].mean()),
                    "mean_population_diversity": float(df["population_diversity"].mean()),
                    "mean_test_collapse_rate": float(df["test_collapse_rate"].mean()),
                    "mean_test_stock": float(df["test_mean_stock"].mean()),
                }
            )

    out = pd.DataFrame(rows)
    out_path = showcase_dir / "paper_v2_table_mechanism_proxies_medium.csv"
    out.to_csv(out_path, index=False)
    return out


def build_composite_sensitivity(ablation_dir: Path, showcase_dir: Path) -> pd.DataFrame:
    rank = pd.read_csv(ablation_dir / "paper_v1_table3_medium_effect_size_ranking.csv")
    m = rank.set_index("injector")
    vec = {
        inj: np.array(
            [
                m.loc[inj, "collapse_reduction_effect_size_dz"],
                m.loc[inj, "stock_gain_effect_size_dz"],
                m.loc[inj, "survival_gain_effect_size_dz"],
            ],
            dtype=float,
        )
        for inj in m.index
    }

    rng = np.random.default_rng(0)
    n_draws = 200000
    raw = rng.random((n_draws, 3))
    w = raw / raw.sum(axis=1, keepdims=True)
    s_llm = w @ vec["llm_json"]
    s_mut = w @ vec["mutation"]

    d = vec["llm_json"] - vec["mutation"]
    stock_threshold = float(-d[0] / (d[1] - d[0])) if abs(d[1] - d[0]) > 1e-12 else np.nan

    out = pd.DataFrame(
        [
            {
                "tier": "medium_v1",
                "n_weight_draws": n_draws,
                "llm_json_win_fraction": float((s_llm > s_mut).mean()),
                "mutation_win_fraction": float((s_mut > s_llm).mean()),
                "stock_weight_threshold_for_llm_json_win": stock_threshold,
                "delta_dz_collapse": float(d[0]),
                "delta_dz_stock": float(d[1]),
                "delta_dz_survival": float(d[2]),
            }
        ]
    )
    out_path = showcase_dir / "paper_v2_table_composite_sensitivity_medium.csv"
    out.to_csv(out_path, index=False)
    return out


def build_mechanism_figure(mech_df: pd.DataFrame, showcase_dir: Path) -> Path:
    df = mech_df.copy()
    cond_order = ["none", "monitoring", "monitoring_sanctions"]
    cond_label = {
        "none": "None",
        "monitoring": "Monitoring",
        "monitoring_sanctions": "Monitoring with sanctions",
    }
    inj_order = ["mutation", "live_llm_json"]
    inj_label = {
        "mutation": "Mutation",
        "live_llm_json": "Live LLM JSON",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for j, metric in enumerate(["mean_high_harvest_frac", "mean_test_stock"]):
        ax = axes[j]
        x = np.arange(len(cond_order))
        width = 0.34
        for i, inj in enumerate(inj_order):
            y = [
                float(
                    df[(df["injector"] == inj) & (df["condition"] == c)][metric].iloc[0]
                )
                for c in cond_order
            ]
            bars = ax.bar(x + (i - 0.5) * width, y, width, label=inj_label[inj], alpha=0.9)
            for b, val in zip(bars, y):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height(),
                    f"{val:.3f}" if metric == "mean_high_harvest_frac" else f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([cond_label[c] for c in cond_order])
        if metric == "mean_high_harvest_frac":
            ax.set_title("High-harvest pressure proxy")
            ax.set_ylabel("Mean high-harvest fraction")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        else:
            ax.set_title("Ecological state proxy")
            ax.set_ylabel("Mean held-out stock")
            none_mut = float(df[(df["injector"] == "mutation") & (df["condition"] == "none")][metric].iloc[0])
            none_llm = float(df[(df["injector"] == "live_llm_json") & (df["condition"] == "none")][metric].iloc[0])
            ax.axhline((none_mut + none_llm) / 2.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, ncols=2, loc="upper right")
    fig.suptitle("Medium-tier mechanism proxies under matched invasion settings", fontsize=12)

    out_path = showcase_dir / "paper_v2_figure_mechanism_proxies_medium.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    ablation_dir = Path(args.ablation_dir)
    showcase_dir = Path(args.showcase_dir)
    showcase_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context_table(ablation_dir, showcase_dir)
    mech = build_mechanism_table(ablation_dir, showcase_dir)
    sens = build_composite_sensitivity(ablation_dir, showcase_dir)
    fig_path = build_mechanism_figure(mech, showcase_dir)

    print(f"Wrote: {showcase_dir / 'paper_v2_table_context_effects_medium.csv'} ({len(ctx)} rows)")
    print(f"Wrote: {showcase_dir / 'paper_v2_table_mechanism_proxies_medium.csv'} ({len(mech)} rows)")
    print(f"Wrote: {showcase_dir / 'paper_v2_table_composite_sensitivity_medium.csv'} ({len(sens)} rows)")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
