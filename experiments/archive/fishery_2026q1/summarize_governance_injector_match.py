import argparse
import os
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize matched governance comparison across mutation and live LLM injectors."
    )
    parser.add_argument(
        "--mutation-table",
        default="results/runs/ablation/governance_match_step4_mutation_table.csv",
    )
    parser.add_argument(
        "--llm-table",
        default="results/runs/ablation/governance_match_step4_ollama_live_table.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/runs/ablation/governance_match_step4_injector_split",
    )
    return parser.parse_args()


def _read_table(path: str, injector: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing table: {path}")
    df = pd.read_csv(path)
    df["injector"] = injector
    return df


def _round_value(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    return v


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
    mut = _read_table(args.mutation_table, injector="mutation")
    llm = _read_table(args.llm_table, injector="llm_json_ollama_live")
    combined = pd.concat([mut, llm], ignore_index=True)

    focus_conditions = ["none", "monitoring_sanctions"]
    focus = combined[combined["condition"].isin(focus_conditions)].copy()
    if focus.empty:
        raise RuntimeError("No rows found for conditions: none, monitoring_sanctions")

    key_cols = [
        "injector",
        "condition",
        "n_runs",
        "test_collapse_mean_mean",
        "test_collapse_mean_ci95_low",
        "test_collapse_mean_ci95_high",
        "test_collapse_last_mean",
        "test_mean_stock_mean_mean",
        "time_to_collapse_mean",
        "first_generation_test_collapse_ge_0_8_mean",
        "per_regime_survival_over_generations_mean_mean",
    ]
    key = focus[key_cols].copy()

    benefit_rows = []
    for injector, idf in focus.groupby("injector", sort=False):
        by_cond = {row["condition"]: row for _, row in idf.iterrows()}
        if "none" not in by_cond or "monitoring_sanctions" not in by_cond:
            continue
        none = by_cond["none"]
        ms = by_cond["monitoring_sanctions"]
        benefit_rows.append(
            {
                "injector": injector,
                "collapse_reduction_mean": float(none["test_collapse_mean_mean"] - ms["test_collapse_mean_mean"]),
                "stock_gain_mean": float(ms["test_mean_stock_mean_mean"] - none["test_mean_stock_mean_mean"]),
                "survival_gain_mean": float(
                    ms["per_regime_survival_over_generations_mean_mean"]
                    - none["per_regime_survival_over_generations_mean_mean"]
                ),
                "time_to_collapse_gain": float(ms["time_to_collapse_mean"] - none["time_to_collapse_mean"]),
                "first_ge_0_8_delay": float(
                    ms["first_generation_test_collapse_ge_0_8_mean"]
                    - none["first_generation_test_collapse_ge_0_8_mean"]
                ),
            }
        )
    benefit = pd.DataFrame(benefit_rows)

    cross_rows = []
    for cond in focus_conditions:
        cdf = focus[focus["condition"] == cond]
        if len(cdf) < 2:
            continue
        by_inj = {row["injector"]: row for _, row in cdf.iterrows()}
        if "mutation" not in by_inj or "llm_json_ollama_live" not in by_inj:
            continue
        mut_row = by_inj["mutation"]
        llm_row = by_inj["llm_json_ollama_live"]
        cross_rows.append(
            {
                "condition": cond,
                "delta_test_collapse_mean_llm_minus_mutation": float(
                    llm_row["test_collapse_mean_mean"] - mut_row["test_collapse_mean_mean"]
                ),
                "delta_test_mean_stock_llm_minus_mutation": float(
                    llm_row["test_mean_stock_mean_mean"] - mut_row["test_mean_stock_mean_mean"]
                ),
                "delta_survival_llm_minus_mutation": float(
                    llm_row["per_regime_survival_over_generations_mean_mean"]
                    - mut_row["per_regime_survival_over_generations_mean_mean"]
                ),
            }
        )
    cross = pd.DataFrame(cross_rows)

    key = key.map(_round_value)
    benefit = benefit.map(_round_value) if not benefit.empty else benefit
    cross = cross.map(_round_value) if not cross.empty else cross

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    key_csv = f"{args.output_prefix}_key.csv"
    benefit_csv = f"{args.output_prefix}_benefit.csv"
    cross_csv = f"{args.output_prefix}_cross.csv"
    md_path = f"{args.output_prefix}.md"

    key.to_csv(key_csv, index=False)
    benefit.to_csv(benefit_csv, index=False)
    cross.to_csv(cross_csv, index=False)

    lines = []
    lines.append("# Governance-Conditioned Matched Comparison")
    lines.append("")
    lines.append("Same run settings; only governance and injector dimensions vary.")
    lines.append("")
    lines.append("## Key Rows")
    lines.append("")
    lines.append(_markdown_table(key))
    lines.append("")
    lines.append("## Monitoring+Sanctions Benefit vs None")
    lines.append("")
    lines.append(_markdown_table(benefit) if not benefit.empty else "_No benefit rows._")
    lines.append("")
    lines.append("## Live LLM vs Mutation Delta (Within Condition)")
    lines.append("")
    lines.append(_markdown_table(cross) if not cross.empty else "_No cross rows._")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {key_csv}")
    print(f"Saved: {benefit_csv}")
    print(f"Saved: {cross_csv}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
