from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "test_collapse_mean",
    "test_mean_stock_mean",
    "per_regime_survival_over_generations_mean",
    "time_to_collapse",
]

MATCH_KEYS = [
    "benchmark_pack",
    "generations",
    "population_size",
    "seeds_per_generation",
    "test_seeds_per_generation",
    "replacement_fraction",
    "adversarial_pressure",
    "collapse_penalty",
    "train_regen_rate",
    "train_obs_noise_std",
    "test_regen_rate",
    "test_obs_noise_std",
    "run_seed_stride",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-v1 matched matrix summarizer with CI gates and figure bundle."
    )
    parser.add_argument("--ablation-dir", default="results/runs/ablation")
    parser.add_argument("--showcase-dir", default="results/runs/showcase")
    parser.add_argument("--experiment-tag", default="paper_v1")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def _safe_float(v: Any) -> float:
    if v is None:
        return math.nan
    if isinstance(v, float):
        return v
    try:
        return float(v)
    except Exception:
        return math.nan


def _canon_injector(raw: str) -> str:
    low = raw.strip().lower()
    if low in {"llm", "llm_json", "llmjson"}:
        return "llm_json"
    if low in {"mutation"}:
        return "mutation"
    return low


def _parse_runs_stem(stem: str, experiment_tag: str) -> tuple[str, str] | None:
    pattern = rf"^{re.escape(experiment_tag)}_(?P<injector>[^_]+)_(?P<tier>.+)_runs$"
    m = re.match(pattern, stem)
    if not m:
        return None
    return _canon_injector(m.group("injector")), m.group("tier")


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _effect_size_dz(diffs: np.ndarray) -> float:
    arr = np.asarray(diffs, dtype=float)
    if arr.size <= 1:
        return math.nan
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return mean / sd


def _normalize_key(v: Any) -> Any:
    if isinstance(v, float) and math.isnan(v):
        return "__nan__"
    if pd.isna(v):
        return "__nan__"
    if isinstance(v, float):
        return round(v, 10)
    return v


def _assert_single_value(df: pd.DataFrame, key: str, context: str) -> Any:
    if key not in df.columns:
        return None
    vals = {_normalize_key(v) for v in df[key].tolist()}
    if len(vals) > 1:
        raise ValueError(f"Mismatched key '{key}' in {context}: {sorted(vals)}")
    return next(iter(vals))


def _assert_matched(a: pd.DataFrame, b: pd.DataFrame, context: str) -> None:
    for key in MATCH_KEYS:
        if key in a.columns and key in b.columns:
            va = _assert_single_value(a, key, context=f"{context}/left")
            vb = _assert_single_value(b, key, context=f"{context}/right")
            if va != vb:
                raise ValueError(f"Mismatched key '{key}' in {context}: left={va}, right={vb}")


def _load_runs(ablation_dir: Path, experiment_tag: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(ablation_dir.rglob(f"{experiment_tag}_*_runs.csv")):
        parsed = _parse_runs_stem(path.stem, experiment_tag=experiment_tag)
        if parsed is None:
            continue
        injector, tier = parsed
        if injector not in {"mutation", "llm_json"}:
            continue
        df = pd.read_csv(path)
        if "condition" not in df.columns or "run_id" not in df.columns:
            raise ValueError(f"Missing condition/run_id columns in {path}")
        df["injector"] = injector
        df["tier"] = tier
        df["source_runs_csv"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No runs files found matching '{experiment_tag}_*_runs.csv' under {ablation_dir}"
        )
    out = pd.concat(frames, ignore_index=True)
    return out


def _load_survival_curves(ablation_dir: Path, experiment_tag: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(ablation_dir.rglob(f"{experiment_tag}_*_survival_curves.csv")):
        stem = path.stem.replace("_survival_curves", "_runs")
        parsed = _parse_runs_stem(stem, experiment_tag=experiment_tag)
        if parsed is None:
            continue
        injector, tier = parsed
        if injector not in {"mutation", "llm_json"}:
            continue
        df = pd.read_csv(path)
        if "condition" not in df.columns or "generation" not in df.columns:
            continue
        df["injector"] = injector
        df["tier"] = tier
        df["source_survival_csv"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _within_governance_deltas(
    runs_df: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (injector, tier), gdf in runs_df.groupby(["injector", "tier"], sort=True):
        none_df = gdf[gdf["condition"] == "none"].copy()
        ms_df = gdf[gdf["condition"] == "monitoring_sanctions"].copy()
        if none_df.empty or ms_df.empty:
            continue
        _assert_matched(none_df, ms_df, context=f"within/{injector}/{tier}")
        merged = ms_df.merge(
            none_df,
            on="run_id",
            suffixes=("_ms", "_none"),
            how="inner",
        )
        if merged.empty:
            continue
        for metric in METRICS:
            col_ms = f"{metric}_ms"
            col_none = f"{metric}_none"
            if col_ms not in merged.columns or col_none not in merged.columns:
                raise ValueError(
                    f"Missing columns for metric '{metric}' in {injector}/{tier} merged deltas."
                )
            deltas = merged[col_ms].to_numpy(dtype=float) - merged[col_none].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_ci(deltas, n_boot=n_boot, rng=rng)
            row: dict[str, Any] = {
                "injector": injector,
                "tier": tier,
                "metric": metric,
                "n_pairs": int(len(merged)),
                "delta_ms_minus_none_mean": mean,
                "delta_ms_minus_none_ci95_low": lo,
                "delta_ms_minus_none_ci95_high": hi,
            }
            if metric == "test_collapse_mean":
                reduction = merged[col_none].to_numpy(dtype=float) - merged[col_ms].to_numpy(dtype=float)
                r_mean, r_lo, r_hi = _bootstrap_ci(reduction, n_boot=n_boot, rng=rng)
                row["collapse_reduction_none_minus_ms_mean"] = r_mean
                row["collapse_reduction_none_minus_ms_ci95_low"] = r_lo
                row["collapse_reduction_none_minus_ms_ci95_high"] = r_hi
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["tier", "injector", "metric"]).reset_index(drop=True)
    return out


def _cross_injector_deltas(
    runs_df: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (tier, condition), gdf in runs_df.groupby(["tier", "condition"], sort=True):
        mut_df = gdf[gdf["injector"] == "mutation"].copy()
        llm_df = gdf[gdf["injector"] == "llm_json"].copy()
        if mut_df.empty or llm_df.empty:
            continue
        _assert_matched(mut_df, llm_df, context=f"cross/{tier}/{condition}")
        merged = llm_df.merge(
            mut_df,
            on="run_id",
            suffixes=("_llm", "_mutation"),
            how="inner",
        )
        if merged.empty:
            continue
        for metric in METRICS:
            col_llm = f"{metric}_llm"
            col_mut = f"{metric}_mutation"
            if col_llm not in merged.columns or col_mut not in merged.columns:
                continue
            deltas = merged[col_llm].to_numpy(dtype=float) - merged[col_mut].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_ci(deltas, n_boot=n_boot, rng=rng)
            rows.append(
                {
                    "tier": tier,
                    "condition": condition,
                    "metric": metric,
                    "n_pairs": int(len(merged)),
                    "delta_llm_minus_mutation_mean": mean,
                    "delta_llm_minus_mutation_ci95_low": lo,
                    "delta_llm_minus_mutation_ci95_high": hi,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["tier", "condition", "metric"]).reset_index(drop=True)
    return out


def _governance_effect_ranking(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (injector, tier), gdf in runs_df.groupby(["injector", "tier"], sort=True):
        none_df = gdf[gdf["condition"] == "none"].copy()
        ms_df = gdf[gdf["condition"] == "monitoring_sanctions"].copy()
        if none_df.empty or ms_df.empty:
            continue
        _assert_matched(none_df, ms_df, context=f"ranking/{injector}/{tier}")
        merged = ms_df.merge(
            none_df,
            on="run_id",
            suffixes=("_ms", "_none"),
            how="inner",
        )
        if merged.empty:
            continue

        collapse_reduction = (
            merged["test_collapse_mean_none"].to_numpy(dtype=float)
            - merged["test_collapse_mean_ms"].to_numpy(dtype=float)
        )
        stock_gain = (
            merged["test_mean_stock_mean_ms"].to_numpy(dtype=float)
            - merged["test_mean_stock_mean_none"].to_numpy(dtype=float)
        )
        survival_gain = (
            merged["per_regime_survival_over_generations_mean_ms"].to_numpy(dtype=float)
            - merged["per_regime_survival_over_generations_mean_none"].to_numpy(dtype=float)
        )

        es_collapse = _effect_size_dz(collapse_reduction)
        es_stock = _effect_size_dz(stock_gain)
        es_survival = _effect_size_dz(survival_gain)
        pieces = np.array([es_collapse, es_stock, es_survival], dtype=float)
        pieces[~np.isfinite(pieces)] = np.nan
        composite = float(np.nanmean(pieces)) if np.isfinite(pieces).any() else math.nan

        rows.append(
            {
                "injector": injector,
                "tier": tier,
                "n_pairs": int(len(merged)),
                "collapse_reduction_mean": float(collapse_reduction.mean()),
                "stock_gain_mean": float(stock_gain.mean()),
                "survival_gain_mean": float(survival_gain.mean()),
                "collapse_reduction_effect_size_dz": es_collapse,
                "stock_gain_effect_size_dz": es_stock,
                "survival_gain_effect_size_dz": es_survival,
                "governance_score_composite": composite,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("governance_score_composite", ascending=False).reset_index(drop=True)
    out["rank_global"] = (
        out["governance_score_composite"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    out["rank_within_tier"] = (
        out.groupby("tier")["governance_score_composite"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    return out


def _llm_integrity(runs_df: pd.DataFrame) -> pd.DataFrame:
    llm_df = runs_df[runs_df["injector"] == "llm_json"].copy()
    if llm_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for tier, gdf in llm_df.groupby("tier", sort=True):
        fallback = float(gdf["llm_fallback_fraction"].mean()) if "llm_fallback_fraction" in gdf.columns else math.nan
        llm_json = float(gdf["llm_json_fraction"].mean()) if "llm_json_fraction" in gdf.columns else math.nan
        rows.append(
            {
                "tier": tier,
                "n_rows": int(len(gdf)),
                "llm_json_fraction_mean": llm_json,
                "llm_fallback_fraction_mean": fallback,
                "fallback_gt_0_10": bool(fallback > 0.10) if not math.isnan(fallback) else False,
                "llm_provider": str(gdf["llm_provider"].iloc[0]) if "llm_provider" in gdf.columns else "",
                "llm_model": str(gdf["llm_model"].iloc[0]) if "llm_model" in gdf.columns else "",
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("tier").reset_index(drop=True)


def _gate_report(within_df: pd.DataFrame, runs_df: pd.DataFrame) -> dict[str, Any]:
    medium = within_df[(within_df["tier"] == "medium_v1") & (within_df["metric"] == "test_collapse_mean")].copy()
    by_injector: dict[str, dict[str, float]] = {}
    for _, row in medium.iterrows():
        by_injector[str(row["injector"])] = {
            "collapse_reduction_mean": _safe_float(row.get("collapse_reduction_none_minus_ms_mean")),
            "collapse_reduction_ci95_low": _safe_float(row.get("collapse_reduction_none_minus_ms_ci95_low")),
            "collapse_reduction_ci95_high": _safe_float(row.get("collapse_reduction_none_minus_ms_ci95_high")),
        }

    primary_pass = (
        "mutation" in by_injector
        and "llm_json" in by_injector
        and by_injector["mutation"]["collapse_reduction_mean"] > 0
        and by_injector["llm_json"]["collapse_reduction_mean"] > 0
    )
    strong_pass = any(v["collapse_reduction_ci95_low"] > 0 for v in by_injector.values())

    medium_rows = runs_df[runs_df["tier"] == "medium_v1"]
    saturated = False
    if not medium_rows.empty:
        grouped = medium_rows.groupby("condition", sort=True)["test_collapse_mean"].mean()
        if not grouped.empty:
            saturated = bool((grouped > 0.95).all())

    return {
        "gate_primary_medium_reduction_both_injectors": bool(primary_pass),
        "gate_strong_ci_lb_gt_zero_at_least_one_injector": bool(strong_pass),
        "medium_saturation_all_conditions_gt_0_95": bool(saturated),
        "medium_by_injector": by_injector,
    }


def _save_table(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    lines = [f"# {title}", ""]
    if df.empty:
        lines.append("_No rows._")
    else:
        headers = list(df.columns)
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            vals: list[str] = []
            for h in headers:
                v = row[h]
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fig_a_governance_effect(within_df: pd.DataFrame, out_path: Path) -> None:
    focus = within_df[within_df["metric"] == "test_collapse_mean"].copy()
    if focus.empty:
        return
    focus["mean"] = focus["collapse_reduction_none_minus_ms_mean"]
    focus["low"] = focus["collapse_reduction_none_minus_ms_ci95_low"]
    focus["high"] = focus["collapse_reduction_none_minus_ms_ci95_high"]
    tiers = sorted(focus["tier"].unique().tolist())
    injectors = sorted(focus["injector"].unique().tolist())

    x = np.arange(len(tiers))
    width = 0.35 if len(injectors) == 2 else 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, inj in enumerate(injectors):
        sub = focus[focus["injector"] == inj].set_index("tier")
        means = [float(sub.loc[t, "mean"]) if t in sub.index else math.nan for t in tiers]
        lows = [float(sub.loc[t, "low"]) if t in sub.index else math.nan for t in tiers]
        highs = [float(sub.loc[t, "high"]) if t in sub.index else math.nan for t in tiers]
        err_lo = np.array(means) - np.array(lows)
        err_hi = np.array(highs) - np.array(means)
        ax.bar(
            x + (i - (len(injectors) - 1) / 2) * width,
            means,
            width=width,
            yerr=np.vstack([err_lo, err_hi]),
            capsize=4,
            label=inj,
            alpha=0.9,
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Collapse Reduction (none - monitoring+sanctions)")
    ax.set_title("Figure A: Governance Effect by Tier/Injector")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _fig_b_time_gain(within_df: pd.DataFrame, out_path: Path) -> None:
    focus = within_df[within_df["metric"] == "time_to_collapse"].copy()
    if focus.empty:
        return
    tiers = sorted(focus["tier"].unique().tolist())
    injectors = sorted(focus["injector"].unique().tolist())
    x = np.arange(len(tiers))
    width = 0.35 if len(injectors) == 2 else 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, inj in enumerate(injectors):
        sub = focus[focus["injector"] == inj].set_index("tier")
        means = [float(sub.loc[t, "delta_ms_minus_none_mean"]) if t in sub.index else math.nan for t in tiers]
        lows = [float(sub.loc[t, "delta_ms_minus_none_ci95_low"]) if t in sub.index else math.nan for t in tiers]
        highs = [float(sub.loc[t, "delta_ms_minus_none_ci95_high"]) if t in sub.index else math.nan for t in tiers]
        err_lo = np.array(means) - np.array(lows)
        err_hi = np.array(highs) - np.array(means)
        ax.bar(
            x + (i - (len(injectors) - 1) / 2) * width,
            means,
            width=width,
            yerr=np.vstack([err_lo, err_hi]),
            capsize=4,
            label=inj,
            alpha=0.9,
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Time-to-Collapse Gain (monitoring+sanctions - none)")
    ax.set_title("Figure B: Time-to-Collapse Gain Under Governance")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _fig_c_survival_trajectories(survival_df: pd.DataFrame, out_path: Path) -> None:
    med = survival_df[survival_df["tier"] == "medium_v1"].copy()
    if med.empty:
        return

    # Aggregate across named regimes (exclude synthetic overall_test).
    regime_filtered = med[med["regime"] != "overall_test"].copy()
    base = regime_filtered if not regime_filtered.empty else med.copy()
    agg = (
        base.groupby(["injector", "condition", "generation"], as_index=False)["survival_fraction_mean"]
        .mean()
    )
    injectors = sorted(agg["injector"].unique().tolist())
    fig, axes = plt.subplots(1, len(injectors), figsize=(6 * len(injectors), 4.5), sharey=True)
    if len(injectors) == 1:
        axes = [axes]
    for ax, inj in zip(axes, injectors):
        sub = agg[agg["injector"] == inj]
        for condition in ["none", "monitoring", "monitoring_sanctions"]:
            cdf = sub[sub["condition"] == condition].sort_values("generation")
            if cdf.empty:
                continue
            ax.plot(
                cdf["generation"].to_numpy(),
                cdf["survival_fraction_mean"].to_numpy(),
                label=condition,
                linewidth=2,
            )
        ax.set_title(f"{inj} | medium_v1")
        ax.set_xlabel("Generation")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Per-Regime Survival Fraction")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Figure C: Per-Regime Survival Trajectories (Medium Tier)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _fig_d_injector_compare(runs_df: pd.DataFrame, out_path: Path) -> None:
    focus = runs_df[runs_df["condition"].isin(["none", "monitoring_sanctions"])].copy()
    if focus.empty:
        return
    agg = (
        focus.groupby(["tier", "condition", "injector"], as_index=False)["test_collapse_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["ci"] = 1.96 * agg["se"].fillna(0.0)

    tiers = sorted(agg["tier"].unique().tolist())
    labels: list[str] = []
    for t in tiers:
        labels.append(f"{t}\nnone")
        labels.append(f"{t}\nmonitoring+sanc")
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, inj in enumerate(["mutation", "llm_json"]):
        means: list[float] = []
        cis: list[float] = []
        for t in tiers:
            for cond in ["none", "monitoring_sanctions"]:
                row = agg[(agg["tier"] == t) & (agg["condition"] == cond) & (agg["injector"] == inj)]
                if row.empty:
                    means.append(math.nan)
                    cis.append(0.0)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    cis.append(float(row["ci"].iloc[0]))
        ax.bar(x + (i - 0.5) * width, means, width=width, yerr=cis, capsize=4, label=inj, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Test Collapse Mean")
    ax.set_title("Figure D: Injector Comparison (Matched None vs Monitoring+Sanctions)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_methods(
    path: Path,
    experiment_tag: str,
    runs_df: pd.DataFrame,
    n_boot: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (injector, tier), gdf in runs_df.groupby(["injector", "tier"], sort=True):
        one = gdf.iloc[0]
        rows.append(
            {
                "injector": injector,
                "tier": tier,
                "n_rows": int(len(gdf)),
                "benchmark_pack": one.get("benchmark_pack", ""),
                "generations": int(one.get("generations", 0)),
                "seeds_per_generation": int(one.get("seeds_per_generation", 0)),
                "test_seeds_per_generation": int(one.get("test_seeds_per_generation", 0)),
                "replacement_fraction": float(one.get("replacement_fraction", math.nan)),
                "adversarial_pressure": float(one.get("adversarial_pressure", math.nan)),
                "train_regen_rate": float(one.get("train_regen_rate", math.nan)),
                "train_obs_noise_std": float(one.get("train_obs_noise_std", math.nan)),
                "llm_provider": one.get("llm_provider", ""),
                "llm_model": one.get("llm_model", ""),
            }
        )
    methods_df = pd.DataFrame(rows).sort_values(["tier", "injector"]).reset_index(drop=True)

    lines: list[str] = []
    lines.append("# paper_v1_methods")
    lines.append("")
    lines.append(f"- experiment_tag: `{experiment_tag}`")
    lines.append(f"- CI method: bootstrap mean CI (B={n_boot}, percentile 2.5/97.5)")
    lines.append("- analysis unit: per-run rows from `*_runs.csv`")
    lines.append("- governance contrast: monitoring_sanctions minus none")
    lines.append("- collapse effect report: both `ms-none` and `none-ms` (reduction) are reported")
    lines.append(
        "- composite governance score: mean of three paired effect sizes (dz): collapse reduction, stock gain, survival gain"
    )
    lines.append("")
    lines.append("## Cell Settings")
    lines.append("")
    if methods_df.empty:
        lines.append("_No rows loaded._")
    else:
        headers = list(methods_df.columns)
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in methods_df.iterrows():
            vals = []
            for h in headers:
                v = row[h]
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_main_results(
    path: Path,
    gate: dict[str, Any],
    within_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    within_path: Path,
    cross_path: Path,
    llm_path: Path,
    ranking_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# paper_v1_main_results")
    lines.append("")
    lines.append("## Gate Status")
    lines.append("")
    lines.append(
        f"- primary_gate_medium_reduction_both_injectors: `{gate['gate_primary_medium_reduction_both_injectors']}`"
    )
    lines.append(
        f"- strong_gate_ci_lb_gt_zero_at_least_one_injector: `{gate['gate_strong_ci_lb_gt_zero_at_least_one_injector']}`"
    )
    lines.append(
        f"- medium_saturation_all_conditions_gt_0_95: `{gate['medium_saturation_all_conditions_gt_0_95']}`"
    )
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    focus = within_df[within_df["metric"] == "test_collapse_mean"].copy()
    if focus.empty:
        lines.append("_No collapse-delta rows found._")
    else:
        for _, row in focus.sort_values(["tier", "injector"]).iterrows():
            reduction = float(row["collapse_reduction_none_minus_ms_mean"])
            lo = float(row["collapse_reduction_none_minus_ms_ci95_low"])
            hi = float(row["collapse_reduction_none_minus_ms_ci95_high"])
            lines.append(
                f"- `{row['tier']}` `{row['injector']}` collapse reduction (none - monitoring+sanc): "
                f"{reduction:.6f} [{lo:.6f}, {hi:.6f}]"
            )
    lines.append("")
    lines.append("## Medium Collapse Reduction by Injector")
    lines.append("")
    by_inj = gate.get("medium_by_injector", {})
    if not by_inj:
        lines.append("_No medium injector rows found._")
    else:
        for inj in sorted(by_inj.keys()):
            row = by_inj[inj]
            lines.append(
                f"- `{inj}`: mean={row['collapse_reduction_mean']:.6f}, "
                f"ci_low={row['collapse_reduction_ci95_low']:.6f}, "
                f"ci_high={row['collapse_reduction_ci95_high']:.6f}"
            )
    lines.append("")
    lines.append("## Composite Governance Ranking")
    lines.append("")
    if ranking_df.empty:
        lines.append("_No ranking rows found._")
    else:
        med = ranking_df[ranking_df["tier"] == "medium_v1"].sort_values(
            "governance_score_composite", ascending=False
        )
        target = med if not med.empty else ranking_df
        for _, row in target.head(4).iterrows():
            lines.append(
                f"- `{row['tier']}` `{row['injector']}` composite={float(row['governance_score_composite']):.6f} "
                f"(collapse_dz={float(row['collapse_reduction_effect_size_dz']):.6f}, "
                f"stock_dz={float(row['stock_gain_effect_size_dz']):.6f}, "
                f"survival_dz={float(row['survival_gain_effect_size_dz']):.6f})"
            )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- LLM cells are slower and can be sensitive to local model/runtime load.")
    pair_counts = sorted(
        {int(v) for v in within_df["n_pairs"].tolist() if pd.notna(v)}
    ) if not within_df.empty else []
    if pair_counts:
        lines.append(
            f"- CI uses bootstrap with finite paired-run counts {pair_counts}; intervals can remain wide."
        )
    else:
        lines.append("- CI uses bootstrap with finite runs; intervals can remain wide.")
    if not llm_df.empty:
        for _, row in llm_df.sort_values("tier").iterrows():
            lines.append(
                f"- `{row['tier']}` fallback fraction: {float(row['llm_fallback_fraction_mean']):.6f} "
                f"(flag>0.10: {bool(row['fallback_gt_0_10'])})."
            )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Table 1 (within-governance deltas): `{within_path}`")
    lines.append(f"- Table 1b (cross-injector deltas): `{cross_path}`")
    lines.append(f"- Table 2 (LLM integrity): `{llm_path}`")
    lines.append(f"- Table 3 (composite governance ranking): `{ranking_path}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ablation_dir = Path(args.ablation_dir)
    showcase_dir = Path(args.showcase_dir)
    out_ablation = ablation_dir / "curated"
    out_showcase = showcase_dir / "curated"
    out_ablation.mkdir(parents=True, exist_ok=True)
    out_showcase.mkdir(parents=True, exist_ok=True)

    runs_df = _load_runs(ablation_dir=ablation_dir, experiment_tag=args.experiment_tag)
    survival_df = _load_survival_curves(ablation_dir=ablation_dir, experiment_tag=args.experiment_tag)
    rng = np.random.default_rng(args.bootstrap_seed)

    within_df = _within_governance_deltas(
        runs_df=runs_df,
        n_boot=args.bootstrap_samples,
        rng=rng,
    )
    cross_df = _cross_injector_deltas(
        runs_df=runs_df,
        n_boot=args.bootstrap_samples,
        rng=rng,
    )
    llm_df = _llm_integrity(runs_df=runs_df)
    ranking_df = _governance_effect_ranking(runs_df=runs_df)
    gate = _gate_report(within_df=within_df, runs_df=runs_df)

    table1_csv = out_ablation / f"{args.experiment_tag}_table1_main_deltas.csv"
    table1_md = out_ablation / f"{args.experiment_tag}_table1_main_deltas.md"
    table1b_csv = out_ablation / f"{args.experiment_tag}_table1b_cross_injector.csv"
    table1b_md = out_ablation / f"{args.experiment_tag}_table1b_cross_injector.md"
    table2_csv = out_ablation / f"{args.experiment_tag}_table2_llm_integrity.csv"
    table2_md = out_ablation / f"{args.experiment_tag}_table2_llm_integrity.md"
    table3_csv = out_ablation / f"{args.experiment_tag}_table3_effect_size_ranking.csv"
    table3_md = out_ablation / f"{args.experiment_tag}_table3_effect_size_ranking.md"
    table3_med_csv = out_ablation / f"{args.experiment_tag}_table3_medium_effect_size_ranking.csv"
    table3_med_md = out_ablation / f"{args.experiment_tag}_table3_medium_effect_size_ranking.md"

    _save_table(within_df, table1_csv, table1_md, title="Table 1: Main Deltas With CI")
    _save_table(cross_df, table1b_csv, table1b_md, title="Table 1b: Cross Injector Deltas With CI")
    _save_table(llm_df, table2_csv, table2_md, title="Table 2: LLM Integrity")
    _save_table(ranking_df, table3_csv, table3_md, title="Table 3: Composite Governance Ranking")
    _save_table(
        ranking_df[ranking_df["tier"] == "medium_v1"].copy(),
        table3_med_csv,
        table3_med_md,
        title="Table 3 (Medium): Composite Governance Ranking",
    )

    gate_json = out_showcase / f"{args.experiment_tag}_gates.json"
    gate_json.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    methods_md = out_showcase / f"{args.experiment_tag}_methods.md"
    main_md = out_showcase / f"{args.experiment_tag}_main_results.md"
    _write_methods(
        path=methods_md,
        experiment_tag=args.experiment_tag,
        runs_df=runs_df,
        n_boot=args.bootstrap_samples,
    )
    _write_main_results(
        path=main_md,
        gate=gate,
        within_df=within_df,
        llm_df=llm_df,
        ranking_df=ranking_df,
        within_path=table1_csv,
        cross_path=table1b_csv,
        llm_path=table2_csv,
        ranking_path=table3_csv,
    )

    if not args.skip_figures:
        _fig_a_governance_effect(
            within_df=within_df,
            out_path=out_showcase / f"{args.experiment_tag}_figureA_governance_effect.png",
        )
        _fig_b_time_gain(
            within_df=within_df,
            out_path=out_showcase / f"{args.experiment_tag}_figureB_time_to_collapse_gain.png",
        )
        _fig_c_survival_trajectories(
            survival_df=survival_df,
            out_path=out_showcase / f"{args.experiment_tag}_figureC_medium_survival_trajectories.png",
        )
        _fig_d_injector_compare(
            runs_df=runs_df,
            out_path=out_showcase / f"{args.experiment_tag}_figureD_injector_comparison.png",
        )

    print(f"Saved: {table1_csv}")
    print(f"Saved: {table1_md}")
    print(f"Saved: {table1b_csv}")
    print(f"Saved: {table1b_md}")
    print(f"Saved: {table2_csv}")
    print(f"Saved: {table2_md}")
    print(f"Saved: {table3_csv}")
    print(f"Saved: {table3_md}")
    print(f"Saved: {table3_med_csv}")
    print(f"Saved: {table3_med_md}")
    print(f"Saved: {gate_json}")
    print(f"Saved: {methods_md}")
    print(f"Saved: {main_md}")
    if not args.skip_figures:
        print("Saved: Figure A/B/C/D under showcase curated folder")


if __name__ == "__main__":
    main()
