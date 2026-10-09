from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path


FULL_STAGEA_THRESHOLD_CONFIG = {
    "scenarios": ["community_irrigation", "forest_co_management"],
    "conditions": "none,bottom_up_only,top_down_only,hybrid",
    "actor_capability_levels": ["low_actor", "medium_actor", "high_actor"],
    "overseer_capability_levels": ["strong_overseer", "limited_overseer", "weak_overseer"],
    "local_margins": [0.00, 0.025, 0.05, 0.075, 0.10],
    "global_thresholds": [9.0, 9.5, 10.0, 10.5, 11.0],
    "n_runs": 5,
    "generations": 15,
    "population_size": 6,
    "seeds_per_generation": 32,
    "test_seeds_per_generation": 32,
    "replacement_fraction": 0.2,
    "max_workers": 1,
    "run_name": "harvest_oversight_gap_threshold_replay_full_grid",
    "experiment_tag": "harvest_threshold_replay_sharded",
    "output_root": "results/runs/threshold_replay/gh/harvest_oversight_gap_threshold_replay_full_grid",
    "merged_prefix": "results/runs/threshold_replay/curated/harvest_oversight_gap_threshold_replay_full_grid",
    "summary_csv": "results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid.csv",
    "analysis_prefix": "results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid",
}


THRESHOLD_OVERRIDE_KEYS = {
    "n_runs",
    "generations",
    "population_size",
    "seeds_per_generation",
    "test_seeds_per_generation",
    "replacement_fraction",
    "max_workers",
}


def _slug_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def threshold_replay_config(overrides: Mapping[str, object] | None = None) -> dict:
    cfg = dict(FULL_STAGEA_THRESHOLD_CONFIG)
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in cfg:
            raise KeyError(f"Unknown threshold replay config key: {key}")
        cfg[key] = value
    return cfg


def threshold_replay_cells(overrides: Mapping[str, object] | None = None) -> list[dict]:
    cfg = threshold_replay_config(overrides)
    cells: list[dict] = []
    for scenario in cfg["scenarios"]:
        for local_margin in cfg["local_margins"]:
            for global_threshold in cfg["global_thresholds"]:
                for actor_level in cfg["actor_capability_levels"]:
                    for overseer_level in cfg["overseer_capability_levels"]:
                        slug = (
                            f"{scenario}__lm{_slug_float(local_margin)}__gh{_slug_float(global_threshold)}"
                            f"__{actor_level}__{overseer_level}"
                        )
                        cells.append(
                            {
                                "slug": slug,
                                "scenario_preset": scenario,
                                "conditions": cfg["conditions"],
                                "actor_capability_level": actor_level,
                                "overseer_capability_level": overseer_level,
                                "local_safety_margin": local_margin,
                                "global_min_mean_patch_health": global_threshold,
                                "n_runs": cfg["n_runs"],
                                "generations": cfg["generations"],
                                "population_size": cfg["population_size"],
                                "seeds_per_generation": cfg["seeds_per_generation"],
                                "test_seeds_per_generation": cfg["test_seeds_per_generation"],
                                "replacement_fraction": cfg["replacement_fraction"],
                                "max_workers": cfg["max_workers"],
                                "experiment_tag": cfg["experiment_tag"],
                            }
                        )
    return cells


def grouped_threshold_replay_jobs(
    group_size: int = 6,
    max_groups: int | None = None,
    overrides: Mapping[str, object] | None = None,
) -> list[dict]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    cells = threshold_replay_cells(overrides)
    jobs = []
    for idx in range(0, len(cells), group_size):
        chunk = cells[idx : idx + group_size]
        group_index = idx // group_size
        jobs.append(
            {
                "group_index": group_index,
                "group_slug": f"group_{group_index:03d}",
                "shard_count": len(chunk),
                "shards_json": json.dumps(chunk, separators=(",", ":")),
            }
        )
    if max_groups is not None:
        jobs = jobs[:max_groups]
    return jobs


def github_threshold_matrix_payload(
    group_size: int = 6,
    max_groups: int | None = None,
    overrides: Mapping[str, object] | None = None,
    run_name_suffix: str = "",
) -> dict:
    cfg = threshold_replay_config(overrides)
    suffix = run_name_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    run_name = f"{cfg['run_name']}{suffix}"
    output_root = f"results/runs/threshold_replay/gh/{run_name}"
    merged_prefix = f"results/runs/threshold_replay/curated/{run_name}"
    summary_csv = f"results/runs/showcase/curated/{run_name}.csv"
    analysis_prefix = f"results/runs/showcase/curated/{run_name}"

    jobs = grouped_threshold_replay_jobs(group_size=group_size, max_groups=max_groups, overrides=overrides)
    return {
        "matrix": {"include": jobs},
        "run_name": run_name,
        "output_root": output_root,
        "merged_prefix": merged_prefix,
        "summary_csv": summary_csv,
        "analysis_prefix": analysis_prefix,
        "group_size": group_size,
        "total_cells": len(threshold_replay_cells(overrides)),
        "total_groups": len(jobs),
    }


def _read_request_file(path: str | None) -> dict:
    if not path:
        return {}
    request_path = Path(path)
    if not request_path.exists():
        return {}
    return json.loads(request_path.read_text())


def _coerce_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or emit sharded threshold-replay jobs.")
    parser.add_argument("--request-file", default=None)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--seeds-per-generation", type=int, default=None)
    parser.add_argument("--test-seeds-per-generation", type=int, default=None)
    parser.add_argument("--replacement-fraction", type=float, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--run-name-suffix", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = _read_request_file(args.request_file)
    overrides = {
        key: request[key]
        for key in THRESHOLD_OVERRIDE_KEYS
        if key in request and request[key] is not None
    }
    cli_overrides = {
        "n_runs": args.n_runs,
        "generations": args.generations,
        "population_size": args.population_size,
        "seeds_per_generation": args.seeds_per_generation,
        "test_seeds_per_generation": args.test_seeds_per_generation,
        "replacement_fraction": args.replacement_fraction,
        "max_workers": args.max_workers,
    }
    overrides.update({key: value for key, value in cli_overrides.items() if value is not None})
    group_size = int(request.get("group_size", args.group_size))
    max_groups = _coerce_optional_int(request.get("max_groups", args.max_groups))
    run_name_suffix = str(request.get("run_name_suffix", args.run_name_suffix))

    payload = github_threshold_matrix_payload(
        group_size=group_size,
        max_groups=max_groups,
        overrides=overrides,
        run_name_suffix=run_name_suffix,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(f"Run name: {payload['run_name']}")
    print(f"Total logical shards: {payload['total_cells']}")
    print(f"Group size: {payload['group_size']}")
    print(f"Total workflow jobs: {payload['total_groups']}")
    print(f"Summary CSV: {payload['summary_csv']}")
    jobs = payload["matrix"]["include"]
    preview = jobs[: min(5, len(jobs))]
    for job in preview:
        print(f"{job['group_slug']}: {job['shard_count']} shard(s)")


if __name__ == "__main__":
    main()
