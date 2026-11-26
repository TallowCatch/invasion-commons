import argparse
import glob
import math
import os

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Fishery RL evaluation summaries.")
    parser.add_argument(
        "--summary-glob",
        default="results/runs/rl_fishery/**/*_summary.csv",
        help="Glob that resolves to per-run Fishery RL summary CSVs.",
    )
    parser.add_argument("--output-prefix", default="results/runs/rl_fishery/curated/fishery_rl")
    return parser.parse_args()


def _ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean()) if arr.size else 0.0
    if arr.size <= 1:
        return mean, mean, mean
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    delta = 1.96 * se
    return mean, mean - delta, mean + delta


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.summary_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No summary files matched: {args.summary_glob}")

    frames = [pd.read_csv(path) for path in paths]
    per_run_df = pd.concat(frames, ignore_index=True)

    metric_keys = [
        "train_collapse_mean",
        "train_mean_welfare_mean",
        "test_collapse_mean",
        "test_mean_stock_mean",
        "test_mean_welfare_mean",
        "per_regime_survival_over_generations_mean",
    ]
    metric_keys.extend(sorted(c for c in per_run_df.columns if c.startswith("per_regime_survival_over_generations__")))

    rows: list[dict[str, object]] = []
    for (condition, benchmark_pack), gdf in per_run_df.groupby(["condition", "benchmark_pack"], dropna=False, sort=True):
        row: dict[str, object] = {
            "condition": condition,
            "benchmark_pack": benchmark_pack,
            "n_runs": int(len(gdf)),
        }
        for key in metric_keys:
            if key not in gdf.columns:
                continue
            mean, lo, hi = _ci95(gdf[key].tolist())
            row[f"{key}_mean"] = round(mean, 4)
            row[f"{key}_ci95_low"] = round(lo, 4)
            row[f"{key}_ci95_high"] = round(hi, 4)
        rows.append(row)

    table_df = pd.DataFrame(rows)
    if not table_df.empty and "test_collapse_mean_mean" in table_df.columns:
        table_df = table_df.sort_values("test_collapse_mean_mean", ascending=True).reset_index(drop=True)

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    runs_path = f"{args.output_prefix}_runs.csv"
    table_path = f"{args.output_prefix}_table.csv"
    md_path = f"{args.output_prefix}_table.md"
    per_run_df.to_csv(runs_path, index=False)
    table_df.to_csv(table_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown_table(table_df))
        f.write("\n")
    print(f"Saved: {runs_path}")
    print(f"Saved: {table_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
