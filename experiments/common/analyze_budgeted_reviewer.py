"""Analysis for the predeclared reviewer-inspection development pilot."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.run_matched_oversight import read_json

CELL = ["game", "policy_group", "regime"]
OUTCOMES = ["unsafe_fixed_horizon", "global_unsafe_rate", "onset_count",
            "garden_failure_event", "total_welfare", "mean_patch_health",
            "mean_realized_harvest", "component_checks_per_step",
            "requests_inspected_per_step", "transmitted_scalars_per_step"]


def analyze(directory, jobs=None):
    directory = Path(directory)
    from experiments.common.run_budgeted_reviewer import MODES, check_blocks, jobs_for, name_for
    cfg = read_json(directory / "manifest.json")["protocol"]
    jobs = jobs if jobs is not None else jobs_for(cfg)
    check_blocks(directory, jobs)
    episode_rows = []
    for job in jobs:
        block = read_json(directory / "blocks" / f"{name_for(job)}.json.gz")
        metrics = block["metrics"]
        horizon, length = cfg["horizon"], metrics["t_end"]
        if not 1 <= length <= horizon:
            raise ValueError("Invalid episode length")
        terminal = int(metrics["garden_failure_event"])
        if terminal and not block["trace"][-1]["global_unsafe"]:
            raise ValueError("Terminal failure must end unsafe")
        if terminal == 0 and length != horizon:
            raise ValueError("Truncated episode without terminal failure")
        row = dict(zip(["game", "policy_group", "regime", "context", "weather",
                        "mode", "inspection_budget"], job))
        row.update(metrics)
        observed_unsafe = sum(int(step["global_unsafe"]) for step in block["trace"])
        row["unsafe_fixed_horizon"] = (observed_unsafe + terminal * (horizon - length)) / horizon
        row["component_checks_per_step"] = metrics["component_evaluations"] / length
        row["requests_inspected_per_step"] = metrics.get("request_inspections", 0) / length
        row["transmitted_scalars_per_step"] = metrics.get("transmitted_scalars", 0) / length
        episode_rows.append(row)
    episodes = pd.DataFrame(episode_rows)
    if episodes.duplicated(CELL + ["context", "weather", "mode", "inspection_budget"]).any():
        raise ValueError("Duplicate episode")
    contexts = episodes.groupby(CELL + ["context", "mode", "inspection_budget"],
                                as_index=False)[OUTCOMES].mean()
    outcomes = contexts.groupby(CELL + ["mode", "inspection_budget"], as_index=False)[OUTCOMES].mean()
    outcomes["n_contexts"] = cfg["contexts"]

    cases = read_json(directory / "frozen_cases.json.gz")
    labels = pd.DataFrame(read_json(directory / "labels.json.gz"))
    decisions = pd.DataFrame(read_json(directory / "decisions.json.gz"))
    ids = {case["case_id"] for case in cases}
    if len(ids) != len(cases) or set(labels.case_id) != ids or labels.case_id.duplicated().any():
        raise ValueError("Frozen case/label mismatch")
    expected = {(case_id, k, mode) for case_id in ids for k in cfg["inspection_budgets"] for mode in MODES}
    actual = list(zip(decisions.case_id, decisions.inspection_budget, decisions.method))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing, duplicate or unexpected decision")
    if (decisions.request_inspections != decisions.inspection_budget).any():
        raise ValueError("Decision used the wrong request-inspection budget")
    if (decisions.candidate_evaluations > cfg["settings"]["candidate_budget"]).any():
        raise ValueError("Candidate limit exceeded")
    joined = decisions.merge(labels, on="case_id", validate="many_to_one")
    coverage = labels.groupby(CELL + ["pre_global_safe", "reference_label"], as_index=False).size()
    coverage = coverage.rename(columns={"size": "n_original_proposals"})
    quality = []
    for keys, frame in joined.groupby(CELL + ["pre_global_safe", "inspection_budget", "method"]):
        risky = int(frame.reference_label.eq("risky").sum())
        safe = int(frame.reference_label.eq("safe").sum())
        quality.append(dict(zip(CELL + ["pre_global_safe", "inspection_budget", "mode"], keys),
            n_cases=len(frame), risky_resolved=risky, safe_resolved=safe,
            unresolved=int(frame.reference_label.eq("unresolved").sum()),
            harmful_accepted=int(frame.harmful_accepted.sum()), safe_rejected=int(frame.safe_rejected.sum()),
            harmful_accept_rate=float(frame.harmful_accepted.sum()/risky) if risky else np.nan,
            safe_reject_rate=float(frame.safe_rejected.sum()/safe) if safe else np.nan,
            approval_rate=float(frame.approved.mean()), abstention_rate=float(frame.abstained.mean()),
            mean_component_evaluations=float(frame.component_evaluations.mean()),
            mean_request_inspections=float(frame.request_inspections.mean()),
            mean_transmitted_scalars=float(frame.transmitted_scalars.mean())))
    quality = pd.DataFrame(quality)

    contrasts = []
    for keys, frame in contexts.groupby(CELL + ["inspection_budget"]):
        joint = frame[frame["mode"].eq("joint")].set_index("context")
        for local_mode in ("local_optimistic", "local_bounded"):
            local = frame[frame["mode"].eq(local_mode)].set_index("context")
            if set(joint.index) != set(local.index):
                raise ValueError("Context pairing differs between monitors")
            for context in sorted(joint.index):
                for metric in OUTCOMES:
                    contrasts.append(dict(zip(CELL + ["inspection_budget"], keys),
                        context=int(context), left="joint", right=local_mode, metric=metric,
                        difference=float(joint.loc[context, metric] - local.loc[context, metric])))
    contrasts = pd.DataFrame(contrasts)
    contrast_summary = contrasts.groupby(CELL + ["inspection_budget", "left", "right", "metric"],
        as_index=False).agg(mean_difference=("difference", "mean"),
                            min_context_difference=("difference", "min"),
                            max_context_difference=("difference", "max"),
                            n_contexts=("context", "nunique"))

    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("episodes", episodes), ("context_means", contexts),
                        ("outcomes", outcomes), ("coverage", coverage),
                        ("decision_quality", quality), ("paired_context_differences", contrasts),
                        ("paired_contrasts", contrast_summary)):
        frame.to_csv(output / f"{name}.csv", index=False)
    lines = ["# Budgeted reviewer development pilot", "",
        "Frozen contract: `notes/research_review/BUDGETED_REVIEWER_PILOT_PROTOCOL.md`.",
        f"Completed {len(episodes)} episodes, {len(cases)} original proposals and "
        f"{len(decisions)} matched reviewer decisions. Tracking: local files and source hashes.", "",
        "## Coverage among initially safe states", "",
        "| Game / policy mixture / setting | Safe | Risky | Unresolved |",
        "| --- | ---: | ---: | ---: |"]
    for keys, frame in labels[labels.pre_global_safe.eq(1)].groupby(CELL):
        counts = frame.reference_label.value_counts()
        lines.append(f"| {' / '.join(map(str, keys))} | {counts.get('safe', 0)} | "
                     f"{counts.get('risky', 0)} | {counts.get('unresolved', 0)} |")
    lines += ["", "## Boundaries", "",
        "`inspection_budget` counts current requests revealed to the monitor, not model intelligence.",
        "Uninspected requests use the declared upper bound. Local Harvest reports are pooled.",
        "Four population contexts per mixture (one in smoke) support a development comparison only.",
        "Weather streams, timesteps and repeated physical cases are not independent contexts.",
        "The fixed-horizon unsafe rate treats terminal failure as absorbing; observed-step rate is also saved.",
        "One-step reference labels do not certify long-run safety. Unresolved labels remain separate.",
        "A later confirmatory study needs new seeds and a precision target. Cleanup policy admission is separate.", ""]
    (output / "experiment.md").write_text("\n".join(lines), encoding="utf8")
    return dict(episodes=len(episodes), cases=len(cases), decisions=len(decisions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    print(analyze(parser.parse_args().input_dir))
