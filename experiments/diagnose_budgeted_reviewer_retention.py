"""Post-hoc diagnostic: how much of resolved-safe original requests was retained."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.run_matched_oversight import read_json


def diagnose(directory: Path) -> pd.DataFrame:
    directory = Path(directory)
    labels = pd.DataFrame(read_json(directory / "labels.json.gz"))
    decisions = pd.DataFrame(read_json(directory / "decisions.json.gz"))
    safe = labels[(labels.pre_global_safe.eq(1)) & labels.reference_label.eq("safe")]
    joined = decisions.merge(safe[["case_id", "game", "context"]],
                             on="case_id", validate="many_to_one")
    if joined.empty:
        raise ValueError("No resolved-safe original proposals")
    if joined.scale.lt(0).any() or joined.scale.gt(1).any():
        raise ValueError("Invalid retained fraction")
    rows = joined.groupby(["game", "inspection_budget", "method"], as_index=False).agg(
        n_safe=("case_id", "size"), mean_retained_fraction=("scale", "mean"),
        completely_stopped=("scale", lambda values: int(values.eq(0).sum())),
        partially_restricted=("scale", lambda values: int(values.between(0, 1, inclusive="neither").sum())),
        unchanged=("scale", lambda values: int(values.eq(1).sum())))
    if not (rows.n_safe == rows.completely_stopped + rows.partially_restricted + rows.unchanged).all():
        raise ValueError("Restriction accounting mismatch")
    output = directory / "posthoc"
    output.mkdir(exist_ok=True)
    rows.to_csv(output / "resolved_safe_retention.csv", index=False)
    (output / "README.md").write_text(
        "# Post-hoc retained-activity diagnostic\n\n"
        "Computed after reading the primary confirmation result. This is not a "
        "predeclared primary endpoint or part of the completion manifest. "
        "It distinguishes partial scaling from stopping a resolved-safe "
        "original request. The mean is case-weighted and has no uncertainty "
        "interval here. Closed-loop extraction and welfare are separate outcomes.\n",
        encoding="utf8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    print(diagnose(parser.parse_args().input_dir).to_string(index=False))
