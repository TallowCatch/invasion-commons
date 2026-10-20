from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    ("mean_patch_health", "patch_health", "higher_is_better"),
    ("mean_welfare", "welfare", "higher_is_better"),
    ("mean_max_local_aggression", "max_local_aggression", "lower_is_better"),
    ("mean_neighborhood_overharvest", "neighborhood_overharvest", "lower_is_better"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Harvest Commons high-power matched deltas with bootstrap confidence intervals."
    )
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ci-csv", required=True)
    parser.add_argument("--pairs-csv", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    return parser.parse_args()


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


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs_df = pd.read_csv(args.runs_csv)
    rng = np.random.default_rng(args.bootstrap_seed)

    hybrid_df = runs_df[runs_df["condition"] == "hybrid"].copy()
    top_df = runs_df[runs_df["condition"] == "top_down_only"].copy()
    if hybrid_df.empty or top_df.empty:
        raise ValueError("Runs CSV must include both 'hybrid' and 'top_down_only' conditions.")

    merge_cols = ["tier", "social_mix", "run_id"]
    pairs = hybrid_df.merge(top_df, on=merge_cols, suffixes=("_hybrid", "_top"))
    if pairs.empty:
        raise ValueError("No matched hybrid/top_down_only pairs found.")

    ci_rows: list[dict[str, object]] = []
    for (tier, social_mix), gdf in pairs.groupby(["tier", "social_mix"], sort=True):
        for metric_col, metric_name, preference in METRICS:
            diffs = gdf[f"{metric_col}_hybrid"].to_numpy(dtype=float) - gdf[f"{metric_col}_top"].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_ci(diffs, n_boot=args.bootstrap_samples, rng=rng)
            ci_rows.append(
                {
                    "tier": tier,
                    "social_mix": social_mix,
                    "metric": metric_name,
                    "preference": preference,
                    "n_pairs": int(len(diffs)),
                    "delta_mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(args.ci_csv, index=False)

    if args.pairs_csv:
        pair_cols = list(merge_cols)
        for metric_col, metric_name, _ in METRICS:
            pair_cols.append(f"{metric_col}_hybrid")
            pair_cols.append(f"{metric_col}_top")
            pairs[f"{metric_name}_delta"] = pairs[f"{metric_col}_hybrid"] - pairs[f"{metric_col}_top"]
            pair_cols.append(f"{metric_name}_delta")
        pairs[pair_cols].to_csv(args.pairs_csv, index=False)

    summary_rows = []
    for (tier, social_mix), gdf in ci_df.groupby(["tier", "social_mix"], sort=True):
        row = {"tier": tier, "social_mix": social_mix}
        for metric_name in [m[1] for m in METRICS]:
            mdf = gdf[gdf["metric"] == metric_name].iloc[0]
            row[f"{metric_name}_delta_mean"] = round(float(mdf["delta_mean"]), 4)
            row[f"{metric_name}_ci95"] = f"[{float(mdf['ci95_low']):.4f}, {float(mdf['ci95_high']):.4f}]"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    lines = [
        "# Harvest Commons high-power uncertainty summary",
        "",
        "Paired deltas are computed as `hybrid - top_down_only` at matched `run_id` within each `(tier, social_mix)` cell.",
        "",
        "## Mean deltas with 95% bootstrap confidence intervals",
        "",
        _markdown_table(summary_df),
        "",
    ]
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"Saved: {args.output}")
    print(f"Saved: {args.ci_csv}")
    if args.pairs_csv:
        print(f"Saved: {args.pairs_csv}")


if __name__ == "__main__":
    main()
