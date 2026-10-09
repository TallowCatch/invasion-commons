import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize tiered governance-ablation table outputs into one comparison."
    )
    parser.add_argument("--ablation-dir", default="results/runs/ablation")
    parser.add_argument("--output-prefix", default="results/runs/ablation/tiered_ablation_summary")
    return parser.parse_args()


def _parse_meta(path: Path) -> tuple[str, str]:
    # Expected examples:
    # tiered_mutation_easy_v1_table.csv
    # tiered_llm_medium_v1_table.csv
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 5 or parts[0] != "tiered":
        return "unknown", "unknown"
    injector = parts[1]
    tier = "_".join(parts[2:-1])  # strip trailing 'table'
    return injector, tier


def _read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    injector, tier = _parse_meta(path)
    df["injector"] = injector
    df["tier"] = tier
    return df


def _metric(row: pd.Series, key: str) -> float:
    return float(row[f"{key}_mean"])


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (injector, tier), gdf in df.groupby(["injector", "tier"], sort=True):
        by_cond = {str(r["condition"]): r for _, r in gdf.iterrows()}
        if "none" not in by_cond or "monitoring_sanctions" not in by_cond:
            continue
        none = by_cond["none"]
        ms = by_cond["monitoring_sanctions"]
        rows.append(
            {
                "injector": injector,
                "tier": tier,
                "none_test_collapse_mean": _metric(none, "test_collapse_mean"),
                "ms_test_collapse_mean": _metric(ms, "test_collapse_mean"),
                "collapse_reduction_ms_vs_none": _metric(none, "test_collapse_mean")
                - _metric(ms, "test_collapse_mean"),
                "none_test_mean_stock": _metric(none, "test_mean_stock_mean"),
                "ms_test_mean_stock": _metric(ms, "test_mean_stock_mean"),
                "stock_gain_ms_vs_none": _metric(ms, "test_mean_stock_mean")
                - _metric(none, "test_mean_stock_mean"),
                "none_survival_mean": _metric(none, "per_regime_survival_over_generations_mean"),
                "ms_survival_mean": _metric(ms, "per_regime_survival_over_generations_mean"),
                "survival_gain_ms_vs_none": _metric(ms, "per_regime_survival_over_generations_mean")
                - _metric(none, "per_regime_survival_over_generations_mean"),
                "none_time_to_collapse": _metric(none, "time_to_collapse"),
                "ms_time_to_collapse": _metric(ms, "time_to_collapse"),
                "time_to_collapse_gain": _metric(ms, "time_to_collapse")
                - _metric(none, "time_to_collapse"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["injector", "tier"]).reset_index(drop=True)
    return out


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows found._"
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ablation_dir = Path(args.ablation_dir)
    paths = sorted(ablation_dir.rglob("tiered_*_table.csv"))
    if not paths:
        raise FileNotFoundError(f"No tiered table files found under {ablation_dir}")

    frames = [_read_table(p) for p in paths]
    all_tables = pd.concat(frames, ignore_index=True)
    summary = _build_summary(all_tables)

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = out_prefix.with_suffix(".csv")
    summary_md = out_prefix.with_suffix(".md")

    summary.to_csv(summary_csv, index=False)
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Tiered Ablation Summary\n\n")
        f.write(_markdown_table(summary))
        f.write("\n")

    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_md}")


if __name__ == "__main__":
    main()
