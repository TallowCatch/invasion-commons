"""Analyze held-out matched oversight at policy-context level."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.run_matched_oversight import digest, read_json

KEY = ["game", "policy_group", "regime"]
METRICS = ["global_unsafe_rate", "onset_count", "total_welfare", "mean_patch_health",
           "mean_realized_harvest", "checks_per_step", "transmitted_scalars_per_step"]


def analyze(directory, jobs=None):
    directory = Path(directory)
    from experiments.common.run_heldout_oversight import (
        FISHERY_METHODS, HARVEST_METHODS, job_name, jobs_for, verify_blocks,
    )
    cfg = read_json(directory / "manifest.json")["protocol"]
    jobs = jobs if jobs is not None else jobs_for(cfg)
    verify_blocks(directory, jobs)
    blocks = [read_json(directory / "blocks" / f"{job_name(job)}.json.gz") for job in jobs]
    episode_rows = []
    for job, block in zip(jobs, blocks, strict=True):
        row = dict(zip(["game", "policy_group", "regime", "context", "weather", "method"], job))
        row.update(block["metrics"])
        row["checks_per_step"] = row["component_evaluations"] / row["t_end"]
        row["transmitted_scalars_per_step"] = sum(
            item["transmitted_scalars"] for item in block["trace"]) / row["t_end"]
        episode_rows.append(row)
    episodes = pd.DataFrame(episode_rows)
    if episodes.duplicated(KEY + ["context", "weather", "method"]).any():
        raise ValueError("Duplicate episode")

    cases = read_json(directory / "frozen_cases.json.gz")
    labels = pd.DataFrame(read_json(directory / "labels.json.gz"))
    decisions = pd.DataFrame(read_json(directory / "decisions.json.gz"))
    case_ids = {case["case_id"] for case in cases}
    if (len(case_ids) != len(cases) or set(labels.case_id) != case_ids
            or labels.case_id.duplicated().any()):
        raise ValueError("Frozen case and label inventory mismatch")
    expected = {(case["case_id"], method) for case in cases
                for method in (HARVEST_METHODS if case["game"] == "harvest" else FISHERY_METHODS)}
    actual = list(zip(decisions.case_id, decisions.method))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing, duplicated, or unexpected monitor decisions")
    if decisions.candidate_evaluations.gt(cfg["settings"]["candidate_budget"]).any():
        raise ValueError("Monitor exceeded candidate budget")
    annotated = decisions.merge(labels, on="case_id", validate="many_to_one")
    for _, frame in annotated.groupby("case_id"):
        if len(frame) != (len(HARVEST_METHODS) if frame.iloc[0].game == "harvest" else len(FISHERY_METHODS)):
            raise ValueError("Different monitors did not judge the same case")

    grouped = episodes.groupby(KEY + ["context", "method"], as_index=False)[METRICS].mean()
    summary = grouped.groupby(KEY + ["method"], as_index=False)[METRICS].mean()
    summary["n_contexts"] = cfg["contexts"]
    contrast_rows = []
    for keys, frame in grouped.groupby(KEY):
        joint = "joint_uncertain" if keys[0] == "harvest" else "joint_nominal"
        locals_ = ("local_uncertain", "local_conservative_uncertain") if keys[0] == "harvest" else (
            "local_nominal", "local_conservative")
        reference = frame[frame.method.eq(joint)].set_index("context")
        for local in locals_:
            comparison = frame[frame.method.eq(local)].set_index("context")
            if set(reference.index) != set(comparison.index):
                raise ValueError("Unpaired population contexts")
            for context in sorted(reference.index):
                for metric in METRICS:
                    contrast_rows.append(dict(zip(KEY, keys), context=int(context),
                        left=joint, right=local, metric=metric,
                        difference=float(reference.loc[context, metric] - comparison.loc[context, metric])))
    contrasts = pd.DataFrame(contrast_rows)
    contrast_summary = contrasts.groupby(KEY + ["left", "right", "metric"], as_index=False).agg(
        mean_difference=("difference", "mean"), min_context_difference=("difference", "min"),
        max_context_difference=("difference", "max"), n_contexts=("context", "nunique"))

    coverage = labels.groupby(KEY + ["pre_global_safe", "reference_label"], as_index=False).size()
    coverage = coverage.rename(columns={"size": "n_original_proposals"})
    decision_rows = []
    for keys, frame in annotated.groupby(KEY + ["pre_global_safe", "method"]):
        risky = int(frame.reference_label.eq("risky").sum())
        safe = int(frame.reference_label.eq("safe").sum())
        decision_rows.append(dict(zip(KEY + ["pre_global_safe", "method"], keys),
            n_cases=len(frame), risky_resolved=risky, safe_resolved=safe,
            unresolved=int(frame.reference_label.eq("unresolved").sum()),
            harmful_accepted=int(frame.harmful_accepted.sum()), safe_rejected=int(frame.safe_rejected.sum()),
            harmful_accept_rate=float(frame.harmful_accepted.sum()/risky) if risky else np.nan,
            safe_reject_rate=float(frame.safe_rejected.sum()/safe) if safe else np.nan,
            approval_rate=float(frame.approved.mean()), abstention_rate=float(frame.abstained.mean()),
            mean_candidate_evaluations=float(frame.candidate_evaluations.mean()),
            mean_component_evaluations=float(frame.component_evaluations.mean()),
            mean_request_inspections=float(frame.request_inspections.mean()),
            mean_transmitted_scalars=float(frame.transmitted_scalars.mean())))
    decision_quality = pd.DataFrame(decision_rows)

    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("episodes", episodes), ("context_means", grouped),
                        ("outcomes", summary), ("paired_context_differences", contrasts),
                        ("paired_contrasts", contrast_summary), ("coverage", coverage),
                        ("decision_quality", decision_quality)):
        frame.to_csv(output / f"{name}.csv", index=False)

    mixed = cfg["profile"] == "mixed_pilot"
    contract = "MIXED_POLICY_REPAIR_PROTOCOL.md" if mixed else "HELDOUT_POLICY_PILOT_PROTOCOL.md"
    lines = ["# Mixed-policy coverage repair" if mixed else "# Held-out policy oversight development pilot", "",
        f"Frozen contract: `notes/research_review/{contract}`.",
        f"Completed {len(episodes)} episodes and judged {len(labels)} original proposals with "
        f"{len(decisions)} matched monitor decisions. No training or search occurred.", "",
        "## Coverage among initially safe no-intervention states", "",
        "| Game / stratum / regime | Safe | Risky | Unresolved |", "| --- | ---: | ---: | ---: |"]
    for keys, frame in labels[labels.pre_global_safe.eq(1)].groupby(KEY):
        counts = frame.reference_label.value_counts()
        lines.append(f"| {' / '.join(map(str, keys))} | {counts.get('safe', 0)} | "
                     f"{counts.get('risky', 0)} | {counts.get('unresolved', 0)} |")
    lines += ["", "## Interpretation boundary", "",
        "These are development observations from four independent policy contexts per stratum "
        "(one for smoke); weather streams and timesteps are nested, not independent replicates.",
        "The aggressive stratum tests behavioral pressure, not a validated general capability ordering.",
        "Harvest local reports are pooled into a global predicate. All methods share the same "
        "target and intervention menu, but component checks and communication are distinct cost proxies.",
        "The one-step reference labels use Wilson intervals; unresolved cases are not treated as safe.",
        "Full-episode effects include policy responses after trajectories diverge and are not equal "
        "to the frozen one-step judgment effects. Already-unsafe states are reported separately.",
        "This pilot does not validate Cleanup, reviewer-resource scaling, or a scalar capability gap.", "",
        "## Next gate", "",
        "Inspect decision coverage and context-level trade-offs, then either repair the sampling "
        "or freeze a new confirmatory protocol with explicit precision targets. Do not tune to a winner.", ""]
    (output / "experiment.md").write_text("\n".join(lines), encoding="utf8")
    return dict(episodes=len(episodes), cases=len(labels), decisions=len(decisions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    print(analyze(parser.parse_args().input_dir))
