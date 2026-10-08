"""Package the bounded source data needed to audit the current paper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "paper/paper_v5_scalable_oversight_commons/data/sources"
PRIMARY = ROOT / "results/runs/budgeted_reviewer_confirmation_v1"
SUPPLEMENTARY = (
    "results/runs/showcase/curated/harvest_oversight_gap_stageA_table.csv",
    "results/runs/showcase/curated/harvest_oversight_gap_stageA_ranking.csv",
    "results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered.csv",
    "results/runs/showcase/curated/harvest_oversight_gap_stageA_oversight_case_trace.csv",
    "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_bank.csv",
    "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_map_samples.csv",
    "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_map_summary.csv",
    "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_bank.csv",
    "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_map_samples.csv",
    "results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_map_summary.csv",
)


def describe(path: Path) -> dict:
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def package() -> dict:
    required = [PRIMARY / "completion.json", PRIMARY / "manifest.json",
                PRIMARY / "frozen_cases.json.gz", PRIMARY / "labels.json.gz",
                PRIMARY / "decisions.json.gz", *(ROOT / name for name in SUPPLEMENTARY)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed paper inputs: {missing}")
    if len(list((PRIMARY / "blocks").glob("*.json.gz"))) != 1920:
        raise ValueError("Expected 1,920 complete confirmation blocks")

    DEST.mkdir(parents=True, exist_ok=True)
    primary_bundle = DEST / "budgeted_reviewer_confirmation_v1.tar.gz"
    with tarfile.open(primary_bundle, "w:gz") as archive:
        archive.add(PRIMARY, arcname=PRIMARY.name)

    supplemental = DEST / "supplementary"
    supplemental.mkdir(exist_ok=True)
    record = {"primary_bundle": describe(primary_bundle), "supplementary": {}}
    for relative in SUPPLEMENTARY:
        source = ROOT / relative
        target = supplemental / source.name
        shutil.copy2(source, target)
        record["supplementary"][source.name] = describe(target)
    (DEST / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    record = package()
    print(json.dumps({"primary_bundle_bytes": record["primary_bundle"]["bytes"],
                      "supplementary_files": len(record["supplementary"])}, indent=2))
