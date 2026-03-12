import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_METRICS = [
    "train_garden_failure_mean",
    "test_garden_failure_mean",
    "train_garden_failure_last",
    "test_garden_failure_last",
    "test_mean_patch_health_mean",
    "test_mean_welfare_mean",
    "test_mean_credit_transferred_mean",
    "test_mean_aggressive_request_fraction_mean",
    "test_mean_max_local_aggression_mean",
    "test_mean_neighborhood_overharvest_mean",
    "test_mean_capped_action_fraction_mean",
    "test_mean_targeted_agent_fraction_mean",
    "test_mean_prevented_harvest_mean",
    "test_mean_patch_variance_mean",
    "test_mean_requested_harvest_mean",
    "test_mean_realized_harvest_mean",
    "time_to_garden_failure",
    "first_generation_test_failure_ge_0_8",
    "per_regime_health_survival_over_generations_mean",
    "population_diversity_mean",
]

PAIR_METRICS = [
    "test_garden_failure_mean",
    "test_mean_patch_health_mean",
    "test_mean_welfare_mean",
    "test_mean_credit_transferred_mean",
    "test_mean_max_local_aggression_mean",
    "test_mean_neighborhood_overharvest_mean",
    "test_mean_capped_action_fraction_mean",
    "test_mean_targeted_agent_fraction_mean",
    "test_mean_prevented_harvest_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Harvest invasion matrix outputs.")
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def _ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    margin = 1.96 * se
    return (mean, mean - margin, mean + margin)


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def _aggregate_with_ci(per_run_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure", "condition"]
    rows = []
    regime_cols = sorted([c for c in per_run_df.columns if c.startswith("per_regime_health_survival_over_generations__")])
    metric_keys = SUMMARY_METRICS + regime_cols
    for keys, gdf in per_run_df.groupby(group_cols, sort=True):
        row = {
            "tier": keys[0],
            "partner_mix": keys[1],
            "injector_mode_requested": keys[2],
            "adversarial_pressure": keys[3],
            "condition": keys[4],
            "n_runs": int(len(gdf)),
        }
        for key in metric_keys:
            mean, lo, hi = _ci95(gdf[key].tolist())
            row[f"{key}_mean"] = round(mean, 6)
            row[f"{key}_ci95_low"] = round(lo, 6)
            row[f"{key}_ci95_high"] = round(hi, 6)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure", "test_garden_failure_mean_mean", "test_mean_patch_health_mean_mean", "test_mean_welfare_mean_mean"],
            ascending=[True, True, True, True, True, False, False],
        ).reset_index(drop=True)
    return out


def _build_ranking_table(table_df: pd.DataFrame) -> pd.DataFrame:
    if table_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure"]
    for keys, gdf in table_df.groupby(group_cols, sort=True):
        ranked = gdf.sort_values(
            ["test_garden_failure_mean_mean", "test_mean_patch_health_mean_mean", "test_mean_welfare_mean_mean"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "tier": keys[0],
                    "partner_mix": keys[1],
                    "injector_mode_requested": keys[2],
                    "adversarial_pressure": keys[3],
                    "condition": row["condition"],
                    "rank": rank,
                    "test_garden_failure_mean_mean": row["test_garden_failure_mean_mean"],
                    "test_mean_patch_health_mean_mean": row["test_mean_patch_health_mean_mean"],
                    "test_mean_welfare_mean_mean": row["test_mean_welfare_mean_mean"],
                }
            )
    return pd.DataFrame(rows)


def _build_paired_delta_df(per_run_df: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure", "run_id"]
    top = per_run_df[per_run_df["condition"] == "top_down_only"].copy()
    hybrid = per_run_df[per_run_df["condition"] == "hybrid"].copy()
    merged = hybrid.merge(top, on=join_cols, suffixes=("_hybrid", "_top"))
    rows = []
    for _, row in merged.iterrows():
        out = {key: row[key] for key in join_cols}
        for metric in PAIR_METRICS:
            out[f"hybrid_minus_top__{metric}"] = float(row[f"{metric}_hybrid"] - row[f"{metric}_top"])
        rows.append(out)
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_samples: int) -> tuple[float, float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(values.mean())
    if values.size == 1:
        return (point, point, point)
    samples = []
    for _ in range(n_samples):
        idx = rng.integers(0, values.size, size=values.size)
        samples.append(float(values[idx].mean()))
    lo, hi = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975])
    return (point, float(lo), float(hi))


def _build_paired_ci_df(delta_df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    group_cols = ["tier", "partner_mix", "injector_mode_requested", "adversarial_pressure"]
    for keys, gdf in delta_df.groupby(group_cols, sort=True):
        row = {
            "tier": keys[0],
            "partner_mix": keys[1],
            "injector_mode_requested": keys[2],
            "adversarial_pressure": keys[3],
            "n_pairs": int(len(gdf)),
        }
        for metric in [c for c in delta_df.columns if c.startswith("hybrid_minus_top__")]:
            mean, lo, hi = _bootstrap_ci(gdf[metric].to_numpy(dtype=float), rng=rng, n_samples=n_samples)
            row[f"{metric}_mean"] = round(mean, 6)
            row[f"{metric}_ci95_low"] = round(lo, 6)
            row[f"{metric}_ci95_high"] = round(hi, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    per_run_df = pd.read_csv(args.runs_csv)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    table_df = _aggregate_with_ci(per_run_df)
    ranking_df = _build_ranking_table(table_df)
    delta_df = _build_paired_delta_df(per_run_df)
    ci_df = _build_paired_ci_df(delta_df, n_samples=args.bootstrap_samples, seed=args.bootstrap_seed)

    best_counts = pd.Series(dtype=int)
    if not ranking_df.empty:
        best_counts = ranking_df[ranking_df["rank"] == 1]["condition"].value_counts().sort_index()

    lines = [
        "# Harvest invasion summary",
        "",
        "## Best condition frequency",
        "",
    ]
    for condition, count in best_counts.items():
        lines.append(f"- `{condition}` wins {int(count)} cell(s).")

    if not delta_df.empty:
        lines.extend(
            [
                "",
                "## Mean paired hybrid minus top-down deltas",
                "",
                f"- Mean failure delta: `{delta_df['hybrid_minus_top__test_garden_failure_mean'].mean():.4f}`",
                f"- Mean patch-health delta: `{delta_df['hybrid_minus_top__test_mean_patch_health_mean'].mean():.4f}`",
                f"- Mean welfare delta: `{delta_df['hybrid_minus_top__test_mean_welfare_mean'].mean():.4f}`",
                f"- Mean local-aggression delta: `{delta_df['hybrid_minus_top__test_mean_max_local_aggression_mean'].mean():.4f}`",
                f"- Mean neighborhood-overharvest delta: `{delta_df['hybrid_minus_top__test_mean_neighborhood_overharvest_mean'].mean():.4f}`",
            ]
        )
    lines.extend([
        "",
        "## Table preview",
        "",
        _markdown_table(table_df.head(24)) if not table_df.empty else "No rows.",
        "",
    ])

    table_csv = str(output_prefix.with_name(output_prefix.name + "_table.csv"))
    ranking_csv = str(output_prefix.with_name(output_prefix.name + "_ranking.csv"))
    delta_csv = str(output_prefix.with_name(output_prefix.name + "_delta.csv"))
    ci_csv = str(output_prefix.with_name(output_prefix.name + "_ci.csv"))
    summary_md = str(output_prefix.with_name(output_prefix.name + "_summary.md"))

    table_df.to_csv(table_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)
    ci_df.to_csv(ci_csv, index=False)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"Saved: {table_csv}")
    print(f"Saved: {ranking_csv}")
    print(f"Saved: {delta_csv}")
    print(f"Saved: {ci_csv}")
    print(f"Saved: {summary_md}")


if __name__ == "__main__":
    main()
