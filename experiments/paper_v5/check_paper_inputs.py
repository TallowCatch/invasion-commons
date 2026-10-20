from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = {
    "paper source": [
        "paper/paper_v5_scalable_oversight_commons/main.tex",
        "paper/paper_v5_scalable_oversight_commons/refs.bib",
    ],
    "scripts": [
        "experiments/paper_v5/analyze_harvest_oversight_stageA.py",
        "experiments/paper_v5/plot_scalable_oversight_paper_v5.py",
        "experiments/paper_v5/analyze_llm_bridge_uncertainty.py",
        "experiments/paper_v5/analyze_stagea_stress_regimes.py",
        "experiments/paper_v5/run_targeted_threshold_replay.py",
        "experiments/paper_v5/analyze_threshold_replay_grid.py",
        "experiments/paper_v5/audit_threshold_sweep_completeness.py",
        "experiments/paper_v5/run_overseer_limit_ablation.py",
        "experiments/paper_v5/analyze_overseer_limit_ablation.py",
        "experiments/paper_v5/audit_overseer_limit_ablation.py",
    ],
    "figures": [
        "paper/paper_v5_scalable_oversight_commons/figures/fig01_evidence_chain.pdf",
        "paper/paper_v5_scalable_oversight_commons/figures/fig02_method_schematic.pdf",
        "paper/paper_v5_scalable_oversight_commons/figures/fig03_capability_gap.pdf",
        "paper/paper_v5_scalable_oversight_commons/figures/fig04_winner_map.pdf",
        "paper/paper_v5_scalable_oversight_commons/figures/fig05_case_trace.pdf",
        "paper/paper_v5_scalable_oversight_commons/figures/fig07_threshold_robustness.pdf",
        "paper/paper_v5_scalable_oversight_commons/figures/fig06_llm_bridge.pdf",
    ],
    "tables": [
        "paper/paper_v5_scalable_oversight_commons/tables/table_stress_settings.tex",
        "paper/paper_v5_scalable_oversight_commons/tables/table_stagea_condition_means.tex",
        "paper/paper_v5_scalable_oversight_commons/tables/table_actor_capability_validation.tex",
        "paper/paper_v5_scalable_oversight_commons/tables/table_threshold_sensitivity.tex",
        "paper/paper_v5_scalable_oversight_commons/tables/table_threshold_robustness_sweep.tex",
        "paper/paper_v5_scalable_oversight_commons/tables/table_llm_bank_validity.tex",
        "paper/paper_v5_scalable_oversight_commons/tables/table_llm_bridge_outcomes.tex",
    ],
    "stage A data": [
        "results/runs/showcase/curated/harvest_oversight_gap_stageA_table.csv",
        "results/runs/showcase/curated/harvest_oversight_gap_stageA_ranking.csv",
        "results/runs/showcase/curated/harvest_oversight_gap_stageA_contrast_delta.csv",
        "results/runs/showcase/curated/harvest_oversight_gap_stageA_contrast_ci.csv",
        "results/runs/showcase/curated/harvest_oversight_gap_stageA_oversight_case_trace.csv",
        "results/runs/showcase/curated/harvest_oversight_gap_stageA_threshold_sensitivity.csv",
        "results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered.csv",
        "results/runs/threshold_replay/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered_runs.csv",
        "results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_runs.csv",
        "results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_generation_history.csv",
        "results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_strategy_history.csv",
    ],
    "LLM bridge data": [
        "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_bank.csv",
        "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_summary.csv",
        "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_map_samples.csv",
        "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_constrained_map_samples.csv",
        "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_bank.csv",
        "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_summary.csv",
        "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_map_samples.csv",
        "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_constrained_map_samples.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check paper reproducibility inputs.")
    parser.add_argument("--output-md", default="notes/PAPER_INPUT_CHECK.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    ok = True
    for group, paths in REQUIRED.items():
        for text_path in paths:
            path = Path(text_path)
            exists = path.exists()
            ok = ok and exists
            rows.append((group, text_path, exists))

    lines = [
        "# Paper Input Check",
        "",
        "| Group | Path | Status |",
        "| --- | --- | --- |",
    ]
    for group, path, exists in rows:
        lines.append(f"| {group} | `{path}` | {'present' if exists else 'missing'} |")
    missing = [path for _, path, exists in rows if not exists]
    lines.extend(["", f"Overall status: {'pass' if ok else 'blocked'}", ""])
    if missing:
        lines.append("## Missing Inputs")
        lines.append("")
        for path in missing:
            lines.append(f"- `{path}`")
        lines.append("")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")
    print(f"Status: {'pass' if ok else 'blocked'}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
