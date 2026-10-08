from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one grouped batch of threshold-replay shards.")
    parser.add_argument("--shards-json", required=True, help="Compact JSON array of shard specifications.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def _build_command(shard: dict, output_prefix: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.run_harvest_invasion_matrix",
        "--scenario-presets",
        str(shard["scenario_preset"]),
        "--conditions",
        str(shard["conditions"]),
        "--actor-capability-levels",
        str(shard["actor_capability_level"]),
        "--overseer-capability-levels",
        str(shard["overseer_capability_level"]),
        "--n-runs",
        str(shard["n_runs"]),
        "--generations",
        str(shard["generations"]),
        "--population-size",
        str(shard["population_size"]),
        "--seeds-per-generation",
        str(shard["seeds_per_generation"]),
        "--test-seeds-per-generation",
        str(shard["test_seeds_per_generation"]),
        "--replacement-fraction",
        str(shard["replacement_fraction"]),
        "--local-safety-margin",
        str(shard["local_safety_margin"]),
        "--global-min-mean-patch-health",
        str(shard["global_min_mean_patch_health"]),
        "--max-workers",
        str(shard["max_workers"]),
        "--output-prefix",
        str(output_prefix),
        "--experiment-tag",
        str(shard["experiment_tag"]),
        "--no-progress",
    ]


def main() -> None:
    args = parse_args()
    shards = json.loads(args.shards_json)
    if not isinstance(shards, list) or not shards:
        raise ValueError("shards-json must decode to a non-empty list")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Shard group size: {len(shards)}")
    for index, shard in enumerate(shards, start=1):
        slug = str(shard["slug"])
        output_prefix = output_root / slug
        runs_csv = output_prefix.with_name(output_prefix.name + "_runs.csv")
        if args.resume and runs_csv.exists():
            print(f"[{index}/{len(shards)}] Skipping existing shard: {slug}")
            continue

        command = _build_command(shard, output_prefix)
        print(f"[{index}/{len(shards)}] Running {slug}")
        print(" ".join(shlex.quote(part) for part in command))
        subprocess.run(command, check=True)

    print("Completed shard group.")


if __name__ == "__main__":
    main()
