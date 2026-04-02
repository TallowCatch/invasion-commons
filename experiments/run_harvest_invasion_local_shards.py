import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from experiments.harvest_invasion_presets import shard_slug
    from experiments.harvest_invasion_presets import stage_cells
    from experiments.harvest_invasion_presets import stage_config
    from experiments.harvest_invasion_presets import stage_names
except ModuleNotFoundError:  # pragma: no cover
    from harvest_invasion_presets import shard_slug
    from harvest_invasion_presets import stage_cells
    from harvest_invasion_presets import stage_config
    from harvest_invasion_presets import stage_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local resumable Harvest invasion shard sweeps, then merge and summarize."
    )
    parser.add_argument(
        "--stage",
        choices=stage_names(),
        default="stage_b_llm_narrow",
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
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


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

    cells = stage_cells(cfg)
    jobs: list[tuple[str, list[str]]] = []
    for cell in cells:
        tier = cell["tier"]
        partner_mix = cell["partner_mix"]
        condition = cell["condition"]
        injector_mode = cell["injector_mode"]
        pressure = cell["pressure"]
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
            "--government-trigger",
            str(cfg["government_trigger"]),
            "--strict-cap-frac",
            str(cfg["strict_cap_frac"]),
            "--relaxed-cap-frac",
            str(cfg["relaxed_cap_frac"]),
            "--soft-trigger",
            str(cfg["soft_trigger"]),
            "--deterioration-threshold",
            str(cfg["deterioration_threshold"]),
            "--activation-warmup",
            str(cfg["activation_warmup"]),
            "--aggressive-request-threshold",
            str(cfg["aggressive_request_threshold"]),
            "--aggressive-agent-fraction-trigger",
            str(cfg["aggressive_agent_fraction_trigger"]),
            "--local-neighborhood-trigger",
            str(cfg["local_neighborhood_trigger"]),
        ]
        if args.llm_base_url:
            cmd.extend(["--llm-base-url", args.llm_base_url])
        jobs.append((slug, cmd))

    total_jobs = len(cells)
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
                "--agent-history-csv",
                str(output_prefix.with_name(output_prefix.name + "_agent_history.csv")),
            ]
        )

    if not args.skip_plot:
        env_plot = os.environ.copy()
        env_plot.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        env_plot.setdefault("XDG_CACHE_HOME", "/tmp")
        if cfg.get("plot_mode") == "institutional":
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "experiments.plot_harvest_architecture_followup",
                    "--ranking-csv",
                    str(summary_prefix.with_name(summary_prefix.name + "_ranking.csv")),
                    "--contrast-ci-csv",
                    str(summary_prefix.with_name(summary_prefix.name + "_contrast_ci.csv")),
                    "--capability-ladder-csv",
                    str(summary_prefix.with_name(summary_prefix.name + "_capability_ladder.csv")),
                    "--aggression-summary-csv",
                    str(summary_prefix.with_name(summary_prefix.name + "_aggression_summary.csv")),
                    "--targeting-summary-csv",
                    str(summary_prefix.with_name(summary_prefix.name + "_targeting_summary.csv")),
                    "--output-prefix",
                    str(summary_prefix),
                ],
                env=env_plot,
            )
        else:
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
