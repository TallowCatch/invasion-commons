import argparse
import os
import shutil
from datetime import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize result CSV/MD artifacts into structured folders."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--apply", action="store_true", help="Actually move files (default is dry-run).")
    return parser.parse_args()


def _target_subdir(name: str) -> str:
    lower = name.lower()
    if "ablation" in lower:
        return "runs/ablation"
    if "showcase" in lower:
        return "runs/showcase"
    if "invasion" in lower:
        return "runs/invasion"
    if "sweep" in lower:
        return "runs/baselines"
    return ""


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    archive_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(results_dir, "archive", archive_tag)
    base_dirs = [
        os.path.join(results_dir, "runs", "invasion"),
        os.path.join(results_dir, "runs", "ablation"),
        os.path.join(results_dir, "runs", "showcase"),
        os.path.join(results_dir, "runs", "baselines"),
        archive_dir,
    ]
    for d in base_dirs:
        os.makedirs(d, exist_ok=True)

    moved = 0
    for name in sorted(os.listdir(results_dir)):
        src = os.path.join(results_dir, name)
        if os.path.isdir(src):
            continue
        if not (name.endswith(".csv") or name.endswith(".md")):
            continue

        subdir = _target_subdir(name)
        if subdir:
            dst = os.path.join(results_dir, subdir, name)
        else:
            dst = os.path.join(archive_dir, name)

        print(f"{'MOVE' if args.apply else 'PLAN'}: {src} -> {dst}")
        if args.apply:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                base, ext = os.path.splitext(dst)
                i = 1
                while os.path.exists(f"{base}_{i}{ext}"):
                    i += 1
                dst = f"{base}_{i}{ext}"
            shutil.move(src, dst)
        moved += 1

    print(
        f"{'Moved' if args.apply else 'Planned'} {moved} file(s). "
        f"Structured outputs should now live under {results_dir}/runs/."
    )


if __name__ == "__main__":
    main()

