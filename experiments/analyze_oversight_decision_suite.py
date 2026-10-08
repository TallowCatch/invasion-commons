"""Finite-suite diagnostics. No confidence intervals treating cases as IID."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_matched_oversight import read_json, digest, FISHERY_METHODS
from experiments.analyze_matched_oversight import LABELS
from fishery_sim.oversight_protocol import METHODS

GROUP = ["game", "regime", "source", "pre_global_safe"]


DELIVERABLES = {"analysis/case_labels.csv", "analysis/coverage.csv", "analysis/decisions.csv",
                "analysis/method_summary.csv", "analysis/experiment.md",
                "analysis/decision_coverage.pdf", "analysis/decision_coverage.svg",
                "analysis/decision_coverage.png"}


def verify_artifacts(directory, *, require_complete=True):
    directory = Path(directory)
    for file, expected in read_json(directory / "checksums.json").items():
        if digest(directory / file) != expected:
            raise ValueError(f"Artifact hash mismatch: {file}")
    freeze = read_json(directory / "freeze.json")
    for part in ("cases", "labels"):
        if digest(directory / f"{part}.json.gz") != freeze[f"{part}_sha256"]:
            raise ValueError(f"Frozen {part} changed")
    cases, labels, decisions = [read_json(directory / (name + ".json.gz"))
                                for name in ("cases", "labels", "decisions")]
    pending = read_json(directory / "pending_analysis.json")
    ids = [c["case_id"] for c in cases]
    if (len(set(ids)) != len(ids) or [l["case_id"] for l in labels] != ids
            or len(cases) != pending["cases"] or len(decisions) != pending["decisions"]):
        raise ValueError("Case/label inventory mismatch")
    frame = pd.DataFrame(decisions)
    by_case = {key: values for key, values in frame.groupby("case_id")}
    if set(by_case) != set(ids):
        raise ValueError("Decision inventory mismatch")
    for case, label in zip(cases, labels, strict=True):
        rows = by_case[case["case_id"]]
        methods = METHODS if case["game"] == "harvest" else FISHERY_METHODS
        if len(rows) != len(methods) or set(rows.method) != set(methods):
            raise ValueError("Missing or duplicate method decision")
        if (not rows.reference_label.eq(label["reference_label"]).all()
                or not rows.original_risk.eq(label["risk"]).all()
                or not rows.content_sha256.eq(case["content_sha256"]).all()):
            raise ValueError("Monitors were not scored on identical frozen evidence")
    if require_complete:
        completion = read_json(directory / "completion.json")
        if completion["cases"] != len(cases) or completion["decisions"] != len(decisions):
            raise ValueError("Completion counts differ from raw evidence")
        checksums_path = directory / "deliverable_checksums.json"
        if digest(checksums_path) != completion["deliverable_checksums_sha256"]:
            raise ValueError("Deliverable checksum inventory changed")
        deliverables = read_json(checksums_path)
        if not DELIVERABLES.issubset(deliverables):
            raise ValueError("Missing required analysis deliverables")
        for path, expected in deliverables.items():
            if digest(directory / path) != expected:
                raise ValueError(f"Deliverable changed: {path}")
        manifest = read_json(directory / "manifest.json")
        if not all(deliverables.get("source/" + path) == expected
                   for path, expected in manifest["source_sha256"].items()):
            raise ValueError("Source snapshots differ from recorded code hashes")
    return cases, labels, frame


def tables(cases, labels, decisions):
    metadata = pd.DataFrame([{k: c[k] for k in ("case_id", "game", "regime", "source", "content_sha256")}
                            | {k: label[k] for k in ("pre_global_safe", "reference_label", "risk", "risk_lower", "risk_upper")}
                            for c, label in zip(cases, labels, strict=True)])
    coverage = []
    for keys, frame in metadata.groupby(GROUP):
        counts = frame.reference_label.value_counts()
        row = dict(zip(GROUP, keys), n_cases=len(frame), unique_content=frame.content_sha256.nunique())
        row.update({"n_"+label: int(counts.get(label, 0)) for label in ("safe", "risky", "unresolved")})
        row["coverage_gate"] = ("pass" if row["n_safe"] >= 10 and row["n_risky"] >= 10 else "incomplete") if (
            row["source"] == "structural" and row["pre_global_safe"] == 1) else "not_applicable"
        coverage.append(row)
    summaries = []
    for keys, frame in decisions.groupby(GROUP + ["method"]):
        risky, safe = frame.reference_label.eq("risky"), frame.reference_label.eq("safe")
        row = dict(zip(GROUP + ["method"], keys), n_cases=len(frame),
            n_risky=int(risky.sum()), n_safe=int(safe.sum()), n_unresolved=int(frame.unresolved.sum()),
            risky_accepted=int(frame.harmful_accepted.sum()), safe_rejected=int(frame.safe_rejected.sum()),
            risky_accept_rate=float(frame.loc[risky, "harmful_accepted"].mean()),
            safe_reject_rate=float(frame.loc[safe, "safe_rejected"].mean()),
            safe_retained_fraction=float(frame.loc[safe, "retained_fraction"].mean()),
            safe_activity_denominator=int(frame.loc[safe, "retained_fraction"].notna().sum()),
            mean_original_extraction=float(frame.original_extraction.mean()),
            mean_retained_extraction=float(frame.retained_extraction.mean()),
            executed_risky=int(frame.executed_label.eq("risky").sum()),
            executed_unresolved=int(frame.executed_label.eq("unresolved").sum()),
            mean_executed_risk=float(frame.executed_risk.mean()),
            monitor_infeasible=int(frame.status.eq("infeasible").sum()),
            budget_exhausted=int(frame.status.eq("budget_exhausted").sum()),
            approved=int(frame.approved.sum()), abstained=int(frame.abstained.sum()))
        for cost in ("candidate_evaluations", "component_evaluations", "request_inspections", "transmitted_scalars"):
            row["mean_"+cost] = float(frame[cost].mean())
        summaries.append(row)
    return metadata, pd.DataFrame(coverage), pd.DataFrame(summaries)


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    data = summary[summary.source.eq("structural") & summary.pre_global_safe.eq(1)]
    panels = list(data.groupby(["game", "regime"]))
    fig, axes = plt.subplots(len(panels), 2, figsize=(9, 2.7*len(panels)), squeeze=False)
    for row, ((game, regime), frame) in enumerate(panels):
        frame = frame.set_index("method").reindex([m for m in METHODS if m in set(frame.method)]).reset_index()
        for col, (metric, name) in enumerate((("risky_accept_rate", "Risky proposals accepted"),
                                              ("safe_reject_rate", "Safe proposals rejected"))):
            ax = axes[row, col]
            for pos, record in enumerate(frame.itertuples()):
                value = getattr(record, metric)
                color = "#176B70" if record.method.startswith("joint") else "#98582D" if record.method.startswith("local") else "#555B60"
                if np.isfinite(value):
                    ax.plot(100*value, pos, "D" if record.method.startswith("joint") else "o", color=color, ms=4.5)
                else:
                    ax.text(50, pos, "unavailable", va="center", fontsize=7)
            ax.set_xlim(-3, 103)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_yticks(range(len(frame)), [LABELS[m] for m in frame.method] if col == 0 else [])
            ax.invert_yaxis()
            ax.set_title(f"{game.title()} / {regime.replace('_', ' ')}: {name}", loc="left", fontsize=9)
            ax.set_xlabel("Percent of corresponding resolved proposals")
            ax.grid(axis="x", alpha=.15, lw=.5)
    fig.text(.02, .012, "Fixed structural challenges, initially safe states only. No deployment-prevalence interpretation.\n"
             "All methods share cases, target and intervention menu; reference-uncertain cases remain in the tables.", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"decision_coverage.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze(directory):
    directory = Path(directory)
    if (directory / "completion.json").exists():
        # The primary report is immutable after successful completion.
        verify_artifacts(directory)
        return None
    cases, labels, decisions = verify_artifacts(directory, require_complete=False)
    metadata, coverage, summary = tables(cases, labels, decisions)
    output = directory / "analysis"
    output.mkdir(exist_ok=False)
    metadata.to_csv(output / "case_labels.csv", index=False)
    coverage.to_csv(output / "coverage.csv", index=False)
    decisions.to_csv(output / "decisions.csv", index=False)
    summary.to_csv(output / "method_summary.csv", index=False)
    plot(summary, output)
    lines = ["# Decision-Case Coverage Repair", "", "## Question",
        "Can the oversight comparison measure both risky approvals and unnecessary restrictions?", "",
        "## Data and Coverage", f"{len(cases)} cases; {len(decisions)} matched method judgments.",
        "Recorded rows use original no-intervention proposals, with occurrence frequency preserved. "
        "Structural cases are a deduplicated, fixed grid, not policy samples.", "",
        "| Game / regime | Source | Initially safe | Safe | Risky | Unresolved | Gate |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for row in coverage.itertuples():
        lines.append(f"| {row.game} / {row.regime} | {row.source} | {bool(row.pre_global_safe)} | {row.n_safe} | {row.n_risky} | {row.n_unresolved} | {row.coverage_gate} |")
    lines += ["", "## Results", "Full method results, denominators, executed-action risk, retained extraction and "
        "checking/communication proxies are in method_summary.csv. Every original label is frozen before monitor comparison.", "",
        "## Limits", "These are one-step judgments on previously seen trajectories and designed states. "
        "No new policy population, held-out confirmation, training, long-run welfare estimate or capability ordering was produced.",
        "Wilson intervals estimate reference risk within each case. They are pointwise, not simultaneous "
        "certificates across the suite. Identical content shares reference draws; repeated natural occurrences are dependent.",
        "Harvest native replay resets episode bookkeeping. Its parity check covers physical one-step health and extraction, "
        "not historical collapse streaks, agent memory or multistep recovery.",
        "Monitor status 'infeasible' means its model failed to certify the menu; it does not establish true infeasibility.",
        "A safe original proposal can have zero extraction, so its retained-fraction denominator is unavailable. "
        "Retained one-step extraction is not long-run welfare.", "",
        "## Next Decision", "Review the coverage gates and strong-local controls before a new study. "
        "No adaptive filling, additional game sweep or paper claim is authorized automatically by this report.", "",
        "## Provenance", "manifest.json and source/ capture code/protocol/runtime. freeze.json records the cases and labels "
        "before monitor evaluation; checksums.json protects completed artifacts. Offline analysis never reruns episodes.", ""]
    (output / "experiment.md").write_text("\n".join(lines))
    return coverage, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    analyze(parser.parse_args().input_dir)
