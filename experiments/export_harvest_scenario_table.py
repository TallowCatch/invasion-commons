from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fishery_sim.harvest_benchmarks import HARVEST_SCENARIO_PRESETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export literature-backed Harvest scenario metadata.")
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = []
    for scenario_name, preset in HARVEST_SCENARIO_PRESETS.items():
        rows.append(
            {
                "scenario_preset": scenario_name,
                "institutional_archetype": preset["institutional_archetype"],
                "tier": preset["tier"],
                "partner_mix": preset["partner_mix"],
                "default_friction_regime": preset["default_friction_regime"],
                "preferred_governance_comparison": " vs ".join(preset["preferred_governance_comparison"]),
                "citations": "; ".join(preset["citations"]),
                "notes": preset["notes"],
            }
        )
    table_df = pd.DataFrame(rows)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = str(output_prefix.with_name(output_prefix.name + "_scenario_table.csv"))
    md_path = str(output_prefix.with_name(output_prefix.name + "_scenario_table.md"))
    table_df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Harvest scenario mapping\n\n")
        handle.write(_markdown_table(table_df))
        handle.write("\n")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
