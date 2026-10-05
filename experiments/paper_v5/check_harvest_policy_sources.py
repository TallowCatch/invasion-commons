"""Compare saved model policies with their supplied numeric anchors; no inference."""
from dataclasses import asdict
from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd

from experiments.common.extract_harvest_oversight_case import _strategy_from_row
from experiments.common.validate_harvest_mechanisms import (ROOT, SCENARIOS, pin_manifest, read_json,
                                                    serial_episode, write_json)
from experiments.paper_v5.analyze_harvest_validation import interval
from fishery_sim.harvest import HarvestStrategySpec
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_scenario
from fishery_sim.harvest_llm_population import _harvest_bank_variation_anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/runs/validation_v1/policy_sources")
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "notes/research_review/completed_checks")
    args = parser.parse_args()
    banks = {model: ROOT / f"results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_{model}_bank.csv"
             for model in ["qwen2_5_3b", "llama3_2_3b"]}
    pin_manifest(args.output_dir, {"kind": "anchor_control", "models": list(banks), "shares": [0, .5, 1],
        "population_samples": 8, "seeds_per_population": 4, "mechanisms": ["none", "uniform_off"],
        "population_seed_base": 93_000_000, "weather_seed_base": 94_000_000},
        [*banks.values(), Path(__file__), ROOT / "fishery_sim/harvest_llm_population.py"])
    started = time.monotonic()
    for model, path in banks.items():
        bank = pd.read_csv(path).fillna("")
        if set(bank.bank_prompt_version) != {"harvest_bank_v3_attitude_anchors"}:
            raise ValueError("Anchor control requires known v3 prompts")
        pools = {attitude: group.sort_values("prompt_nonce").reset_index(drop=True)
                 for attitude, group in bank.groupby("bank_attitude")}
        for scenario in SCENARIOS:
            for share in [0, .5, 1]:
                for mechanism in ["none", "uniform_off"]:
                    output = args.output_dir / f"{model}__{scenario}__{share}__{mechanism}.json.gz"
                    if output.exists():
                        continue
                    rows = []
                    for sample in range(8):
                        rng = np.random.default_rng(93_000_000 + sample)
                        attitudes = ["exploitative"] * int(6 * share) + ["cooperative"] * (6-int(6*share))
                        rng.shuffle(attitudes)
                        selected = [pools[att].iloc[int(rng.integers(len(pools[att])))] for att in attitudes]
                        for source in ["model", "anchors"]:
                            policies = []
                            for index, row in enumerate(selected):
                                spec = _strategy_from_row(row)
                                if source == "anchors":
                                    anchors = _harvest_bank_variation_anchors(row.bank_attitude, 20., int(row.prompt_nonce))
                                    spec = HarvestStrategySpec(strategy_id=spec.strategy_id, **anchors)
                                spec.strategy_id = f"agent_{index}"
                                policies.append(spec)
                            for offset in range(4):
                                seed = 94_000_000 + sample*100 + offset
                                cfg = make_harvest_cfg_for_scenario(scenario, seed=seed)
                                out = serial_episode(cfg, policies, mechanism, trace=False)
                                rows.append({"model": model, "source": source, "scenario": scenario,
                                    "share": share, "mechanism": mechanism, "sample": sample, "seed": seed,
                                    "policies": [asdict(p) for p in policies], **out["metrics"]})
                    write_json(output, rows, compressed=True)
            print(f"Policy-source control complete: {model} / {scenario}", flush=True)
    rows = []
    for path in sorted(args.output_dir.glob("*.json.gz")):
        rows.extend(read_json(path, True))
    df = pd.DataFrame(rows).drop(columns="policies")
    if len(df) != 1536:
        raise ValueError("Incomplete policy-source control")
    keys = ["model", "source", "scenario", "share", "mechanism", "sample", "seed"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate source-control keys")
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.summary_dir / "policy_source_episodes.csv", index=False)
    metrics = ["global_unsafe_rate", "garden_failure_event", "mean_patch_health", "total_welfare"]
    samples = df.groupby(keys[:-1])[metrics].mean().reset_index()
    samples.groupby(keys[:5])[metrics].mean().to_csv(args.summary_dir / "policy_source_summary.csv")
    differences = []
    for key, group in samples.groupby(["model", "scenario", "share", "mechanism"]):
        a = group[group.source.eq("model")].set_index("sample")
        b = group[group.source.eq("anchors")].set_index("sample")
        for metric in metrics:
            mean, half = interval(a[metric] - b[metric])
            differences.append({**dict(zip(["model", "scenario", "share", "mechanism"], key)),
                                "metric": metric, "model_minus_anchors": mean, "ci_half": half, "n_pairs": len(a)})
    pd.DataFrame(differences).to_csv(args.summary_dir / "policy_source_differences.csv", index=False)
    write_json(args.output_dir / "completion.json", {"episodes": len(df), "seconds": time.monotonic()-started})


if __name__ == "__main__":
    main()
