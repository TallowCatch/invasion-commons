import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local resumable Harvest invasion shard sweeps, then merge and summarize."
    )
    parser.add_argument(
        "--stage",
        choices=["stage_b_llm", "stage_c_llm"],
        default="stage_b_llm",
        help="Preset sweep to execute.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Merged output prefix. If omitted, uses the stage default.",
    )
    parser.add_argument(
        "--shard-dir",
        default=None,
        help="Directory for per-cell shard CSVs. If omitted, uses the stage default.",
    )
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:3b-instruct")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout-s", type=float, default=120.0)
    parser.add_argument("--llm-temperature", type=float, default=0.8)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def stage_config(stage: str) -> dict:
    if stage == "stage_b_llm":
        return {
            "tiers": ["medium_h1", "hard_h1"],
            "partner_mixes": ["balanced", "adversarial_heavy"],
            "conditions": ["top_down_only", "hybrid"],
            "injector_modes": ["mutation", "llm_json"],
            "pressures": ["0.3", "0.5"],
            "n_runs": "3",
            "generations": "12",
            "population_size": "6",
            "seeds_per_generation": "24",
            "test_seeds_per_generation": "24",
            "replacement_fraction": "0.2",
            "run_name": "harvest_invasion_llm_stageB_local",
            "experiment_tag": "harvest_invasion_llm_stageB_local",
        }
    if stage == "stage_c_llm":
        return {
            "tiers": ["medium_h1", "hard_h1"],
            "partner_mixes": ["balanced", "adversarial_heavy"],
            "conditions": ["top_down_only", "hybrid"],
            "injector_modes": ["llm_json"],
            "pressures": ["0.3"],
            "n_runs": "5",
            "generations": "15",
            "population_size": "6",
            "seeds_per_generation": "32",
            "test_seeds_per_generation": "32",
            "replacement_fraction": "0.2",
            "run_name": "harvest_invasion_llm_stageC_local",
            "experiment_tag": "harvest_invasion_llm_stageC_local",
        }
    raise ValueError(f"Unknown stage: {stage}")


def shard_slug(tier: str, partner_mix: str, condition: str, injector_mode: str, pressure: str) -> str:
    return f"{tier}__{partner_mix}__{condition}__{injector_mode}__p{pressure}".replace(".", "p")


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    args = parse_args()
    cfg = stage_config(args.stage)

    default_output_prefix = f"results/runs/harvest_invasion/curated/{cfg['run_name']}"
    default_shard_dir = f"results/runs/harvest_invasion/local_shards/{cfg['run_name']}"
    output_prefix = Path(args.output_prefix or default_output_prefix)
    shard_dir = Path(args.shard_dir or default_shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_prefix = Path(f"results/runs/showcase/curated/{cfg['run_name']}")
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, list[str]]] = []
    for tier in cfg["tiers"]:
        for partner_mix in cfg["partner_mixes"]:
            for condition in cfg["conditions"]:
                for injector_mode in cfg["injector_modes"]:
                    for pressure in cfg["pressures"]:
                        slug = shard_slug(tier, partner_mix, condition, injector_mode, pressure)
                        shard_prefix = shard_dir / slug
                        runs_csv = shard_prefix.with_name(shard_prefix.name + "_runs.csv")
                        if args.resume and runs_csv.exists():
                            continue
                        cmd = [
                            sys.executable,
                            "-m",
                            "experiments.run_harvest_invasion_matrix",
                            "--tiers",
                            tier,
                            "--partner-mixes",
                            partner_mix,
                            "--conditions",
                            condition,
                            "--injector-modes",
                            injector_mode,
                            "--adversarial-pressures",
                            pressure,
                            "--n-runs",
                            cfg["n_runs"],
                            "--generations",
                            cfg["generations"],
                            "--population-size",
                            cfg["population_size"],
                            "--seeds-per-generation",
                            cfg["seeds_per_generation"],
                            "--test-seeds-per-generation",
                            cfg["test_seeds_per_generation"],
                            "--replacement-fraction",
                            cfg["replacement_fraction"],
                            "--max-workers",
                            "1",
                            "--output-prefix",
                            str(shard_prefix),
                            "--experiment-tag",
                            cfg["experiment_tag"],
                            "--llm-provider",
                            args.llm_provider,
                            "--llm-model",
                            args.llm_model,
                            "--llm-api-key-env",
                            args.llm_api_key_env,
                            "--llm-timeout-s",
                            str(args.llm_timeout_s),
                            "--llm-temperature",
                            str(args.llm_temperature),
                        ]
                        if args.llm_base_url:
                            cmd.extend(["--llm-base-url", args.llm_base_url])
                        jobs.append((slug, cmd))

    total_jobs = len(
        cfg["tiers"]
    ) * len(cfg["partner_mixes"]) * len(cfg["conditions"]) * len(cfg["injector_modes"]) * len(cfg["pressures"])
    print(f"Stage: {args.stage}")
    print(f"Shard dir: {shard_dir}")
    print(f"Output prefix: {output_prefix}")
    print(f"Pending shards: {len(jobs)} / {total_jobs}")

    env = os.environ.copy()
    for idx, (slug, cmd) in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] Running {slug}")
        run_cmd(cmd, env=env)

    if args.skip_merge:
        return

    run_cmd(
        [
            sys.executable,
            "-m",
            "experiments.merge_harvest_invasion_outputs",
            "--input-dir",
            str(shard_dir),
            "--output-prefix",
            str(output_prefix),
        ]
    )

    if not args.skip_summary:
        run_cmd(
            [
                sys.executable,
                "-m",
                "experiments.summarize_harvest_invasion",
                "--runs-csv",
                str(output_prefix.with_name(output_prefix.name + "_runs.csv")),
                "--output-prefix",
                str(summary_prefix),
            ]
        )

    if not args.skip_plot:
        env_plot = os.environ.copy()
        env_plot.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        env_plot.setdefault("XDG_CACHE_HOME", "/tmp")
        run_cmd(
            [
                sys.executable,
                "-m",
                "experiments.plot_harvest_invasion",
                "--ci-csv",
                str(summary_prefix.with_name(summary_prefix.name + "_ci.csv")),
                "--output-prefix",
                str(summary_prefix),
            ],
            env=env_plot,
        )

    print("Completed local shard sweep, merge, summary, and plots.")


if __name__ == "__main__":
    main()
