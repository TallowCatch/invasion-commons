import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize result artifacts into curated/exploratory/scratch folders."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Subdirectory inside results-dir that contains invasion/ablation/showcase/baselines.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually move files (default is dry-run).")
    return parser.parse_args()


SCRATCH_MARKERS = (
    "tmp",
    "smoke",
    "check",
    "bench_smoke",
)


def _is_scratch(name: str) -> bool:
    lower = name.lower()
    return any(m in lower for m in SCRATCH_MARKERS)


def _classify_invasion(name: str) -> str:
    lower = name.lower()
    if _is_scratch(name):
        return "scratch"
    curated_prefixes = (
        "invasion_step4_match_",
        "invasion_visual_none",
        "invasion_visual_monitoring_sanctions",
        "invasion_visual_benchmark_v2",
        "invasion_regime_split",
        "invasion_baseline_tuned",
        "invasion_governance_tuned",
    )
    if lower.startswith(curated_prefixes):
        return "curated"
    return "exploratory"


def _classify_ablation(name: str) -> str:
    lower = name.lower()
    if _is_scratch(name) or "step3_smoke" in lower:
        return "scratch"
    curated_prefixes = (
        "paper_v1_",
        "governance_match_step4_",
        "governance_ablation_highpower_combined_summary",
        "governance_ablation_highpower_heldout_v1_table",
        "governance_ablation_highpower_mixed_v1_table",
        "tiered_",
    )
    if lower.startswith(curated_prefixes):
        return "curated"
    return "exploratory"


def _classify_showcase(name: str) -> str:
    lower = name.lower()
    curated_names = {
        "episode_stock_harvest_dynamics.gif",
        "invasion_train_vs_heldout_dynamics.gif",
        "invasion_none_vs_monitoring_sanctions.gif",
        "showcase_report.md",
    }
    if lower.startswith("paper_v1_"):
        return "curated"
    if lower in curated_names:
        return "curated"
    if _is_scratch(name):
        return "scratch"
    return "exploratory"


def _classify_baselines(name: str) -> str:
    lower = name.lower()
    if _is_scratch(name):
        return "scratch"
    return "curated"


def _classify_bucket(run_type: str, name: str) -> str:
    if run_type == "invasion":
        return _classify_invasion(name)
    if run_type == "ablation":
        return _classify_ablation(name)
    if run_type == "showcase":
        return _classify_showcase(name)
    if run_type == "baselines":
        return _classify_baselines(name)
    return "exploratory"


def _safe_move(src: Path, dst: Path, apply: bool) -> tuple[str, Path]:
    final_dst = dst
    if final_dst.exists():
        base = final_dst.with_suffix("")
        ext = final_dst.suffix
        i = 1
        while (base.parent / f"{base.name}_{i}{ext}").exists():
            i += 1
        final_dst = base.parent / f"{base.name}_{i}{ext}"
    action = "MOVE" if apply else "PLAN"
    print(f"{action}: {src} -> {final_dst}")
    if apply:
        final_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(final_dst))
    return action, final_dst


def _ensure_layout(results_dir: Path, runs_dir: Path, archive_tag: str) -> None:
    base_dirs = [
        results_dir / "archive" / archive_tag,
        results_dir / "archive" / "runtime_cache",
    ]
    for run_type in ("invasion", "ablation", "showcase", "baselines"):
        for bucket in ("curated", "exploratory", "scratch"):
            base_dirs.append(runs_dir / run_type / bucket)
    for d in base_dirs:
        d.mkdir(parents=True, exist_ok=True)


def _clean_runtime_noise(results_dir: Path, apply: bool) -> int:
    moved = 0
    cache_dir = results_dir / ".mplconfig"
    if cache_dir.exists():
        dst = results_dir / "archive" / "runtime_cache" / ".mplconfig"
        _safe_move(cache_dir, dst, apply=apply)
        moved += 1
    for root, _, files in os.walk(results_dir):
        for name in files:
            if name != ".DS_Store":
                continue
            src = Path(root) / name
            if apply:
                print(f"REMOVE: {src}")
                src.unlink(missing_ok=True)
            else:
                print(f"PLAN_REMOVE: {src}")
            moved += 1
    return moved


def _move_legacy_top_level(results_dir: Path, archive_dir: Path, apply: bool) -> int:
    moved = 0
    for src in sorted(results_dir.iterdir()):
        if src.is_dir():
            continue
        if src.suffix.lower() not in {".csv", ".md", ".gif", ".png", ".json"}:
            continue
        dst = archive_dir / src.name
        _safe_move(src, dst, apply=apply)
        moved += 1
    return moved


def _organize_run_type(runs_dir: Path, run_type: str, apply: bool) -> int:
    moved = 0
    root = runs_dir / run_type
    if not root.exists():
        return moved
    for src in sorted(root.rglob("*")):
        if not src.is_file():
            continue
        if src.name.startswith("."):
            continue
        bucket = _classify_bucket(run_type, src.name)
        dst = root / bucket / src.name
        if src == dst:
            continue
        _safe_move(src, dst, apply=apply)
        moved += 1
    return moved


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    runs_dir = results_dir / args.runs_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    archive_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = results_dir / "archive" / archive_tag
    _ensure_layout(results_dir=results_dir, runs_dir=runs_dir, archive_tag=archive_tag)

    moved = 0
    moved += _clean_runtime_noise(results_dir=results_dir, apply=args.apply)
    moved += _move_legacy_top_level(results_dir=results_dir, archive_dir=archive_dir, apply=args.apply)
    for run_type in ("invasion", "ablation", "showcase", "baselines"):
        moved += _organize_run_type(runs_dir=runs_dir, run_type=run_type, apply=args.apply)

    print(
        f"{'Moved' if args.apply else 'Planned'} {moved} file(s). "
        f"Structured outputs should now live under {runs_dir}/<type>/(curated|exploratory|scratch)."
    )


if __name__ == "__main__":
    main()
