import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Study 1b matrix outputs.")
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _best_condition(group: pd.DataFrame) -> tuple[str, float, float, float]:
    ranked = group.sort_values(
        ["test_collapse_mean", "test_mean_welfare_mean", "test_mean_stock_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    return (
        str(best["condition"]),
        float(best["test_collapse_mean"]),
        float(best["test_mean_welfare_mean"]),
        float(best["test_mean_stock_mean"]),
    )


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
    df = pd.read_csv(args.runs_csv)

    scenario_cols = ["benchmark_pack", "partner_mix", "adversarial_pressure"]
    scenario_rows = []
    for keys, gdf in df.groupby(scenario_cols, sort=True):
        best_condition, best_collapse, best_welfare, best_stock = _best_condition(gdf)
        none_row = gdf[gdf["condition"] == "none"].iloc[0]
        sanc_row = gdf[gdf["condition"] == "monitoring_sanctions"].iloc[0]
        adaptive_row = gdf[gdf["condition"] == "adaptive_quota"].iloc[0] if "adaptive_quota" in set(gdf["condition"]) else None
        closure_row = gdf[gdf["condition"] == "temporary_closure"].iloc[0] if "temporary_closure" in set(gdf["condition"]) else None
        adaptive_better_than_closure = None
        if adaptive_row is not None and closure_row is not None:
            adaptive_better_than_closure = float(adaptive_row["test_mean_welfare_mean"] - closure_row["test_mean_welfare_mean"])
        scenario_rows.append(
            {
                "benchmark_pack": keys[0],
                "partner_mix": keys[1],
                "adversarial_pressure": keys[2],
                "best_condition": best_condition,
                "best_collapse": best_collapse,
                "best_mean_welfare": best_welfare,
                "best_mean_stock": best_stock,
                "sanctions_minus_none": float(none_row["test_collapse_mean"] - sanc_row["test_collapse_mean"]),
                "adaptive_minus_none": float(none_row["test_collapse_mean"] - adaptive_row["test_collapse_mean"]) if adaptive_row is not None else 0.0,
                "closure_minus_none": float(none_row["test_collapse_mean"] - closure_row["test_collapse_mean"]) if closure_row is not None else 0.0,
                "adaptive_minus_closure_welfare": adaptive_better_than_closure if adaptive_better_than_closure is not None else 0.0,
                "adaptive_mean_welfare": float(adaptive_row["test_mean_welfare_mean"]) if adaptive_row is not None else 0.0,
                "closure_mean_welfare": float(closure_row["test_mean_welfare_mean"]) if closure_row is not None else 0.0,
                "sanctions_quota_clipped": float(sanc_row["test_mean_quota_clipped_total"]),
                "sanctions_repeat_offender_rate": float(sanc_row["test_mean_repeat_offender_rate"]),
                "sanctions_recovery_lag": float(sanc_row["test_mean_stock_recovery_lag"]),
            }
        )

    scenario_df = pd.DataFrame(scenario_rows)
    best_counts = scenario_df["best_condition"].value_counts().sort_index()

    lines = [
        "# study1b follow-up summary",
        "",
        "## Best governance condition frequency",
        "",
    ]
    for condition, count in best_counts.items():
        lines.append(f"- `{condition}` wins {int(count)} scenario(s).")

    lines.extend(
        [
            "",
            "## Mechanism readout",
            "",
            f"- Mean `monitoring_sanctions - none` collapse reduction: `{scenario_df['sanctions_minus_none'].mean():.4f}`.",
            f"- Mean `adaptive_quota - none` collapse reduction: `{scenario_df['adaptive_minus_none'].mean():.4f}`.",
            f"- Mean `temporary_closure - none` collapse reduction: `{scenario_df['closure_minus_none'].mean():.4f}`.",
            f"- Mean `adaptive_quota - temporary_closure` welfare difference: `{scenario_df['adaptive_minus_closure_welfare'].mean():.4f}`.",
            f"- Under `monitoring_sanctions`, mean clipped harvest is `{scenario_df['sanctions_quota_clipped'].mean():.4f}` and repeat-offender rate is `{scenario_df['sanctions_repeat_offender_rate'].mean():.4f}`.",
            "",
            "## Scenario table",
            "",
            _markdown_table(scenario_df),
            "",
        ]
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
