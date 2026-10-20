"""Context-level, descriptive analysis of nested actor-search pressure."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from experiments.common.run_matched_oversight import read_json
from fishery_sim.budgeted_oversight import MODES


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def analyze(output: Path, cfg: dict) -> dict:
    labels = read_json(output / "labels.json.gz")
    decisions = read_json(output / "decisions.json.gz")
    label_by_id = {row["case_id"]: row for row in labels}
    if len(label_by_id) != len(labels):
        raise ValueError("Duplicate reference labels")
    by_case = defaultdict(list)
    for row in decisions:
        if row["case_id"] not in label_by_id:
            raise ValueError("Decision without reference label")
        by_case[row["case_id"]].append(row)
    expected_per_case = len(cfg["inspection_budgets"]) * len(MODES)
    if any(len(rows) != expected_per_case for rows in by_case.values()):
        raise ValueError("Incomplete or duplicate reviewer decisions")

    selection_rows = []
    summaries = []
    for context in range(cfg["contexts"]):
        selection = read_json(output / "selection" / f"context_{context:03d}.json.gz")
        for actor_budget in cfg["actor_budgets"]:
            index = int(selection["selected"][str(actor_budget)])
            if index < 0 or index >= actor_budget:
                raise ValueError("Selected candidate outside nested search budget")
            train_score = float(selection["train_scores"][index])
            selection_rows.append(dict(context=context, actor_budget=actor_budget,
                                       selected_index=index, train_score=train_score,
                                       candidate_evaluations=actor_budget * cfg["train_weather_streams"]))
            group_labels = [row for row in labels if row["context"] == context
                            and row["actor_budget"] == actor_budget]
            counts = {label: sum(row["reference_label"] == label for row in group_labels)
                      for label in ("safe", "risky", "unresolved")}
            base_blocks = [read_json(output / "blocks" /
                           f"c{context:03d}__a{actor_budget:02d}__w{weather}__none__k0.json.gz")
                           for weather in range(cfg["weather_streams"])]
            heldout_payoff = mean(block["metrics"]["entrant_payoff"] for block in base_blocks)
            heldout_unsafe = mean(block["metrics"]["fixed_horizon_unsafe_rate"] for block in base_blocks)
            total_steps = sum(int(block["metrics"]["t_end"]) for block in base_blocks)
            common = dict(context=context, actor_budget=actor_budget, train_score=train_score,
                          selected_index=index, heldout_entrant_payoff=heldout_payoff,
                          no_review_unsafe_rate=heldout_unsafe,
                          resolved_risky=counts["risky"], resolved_safe=counts["safe"],
                          unresolved=counts["unresolved"],
                          excluded_unsafe_prestates=total_steps - len(group_labels))
            arms = [("none", 0)] + [(mode, budget) for budget in cfg["inspection_budgets"]
                                     for mode in cfg["modes"]]
            for mode, inspection_budget in arms:
                if mode == "none":
                    block_metrics = [block["metrics"] for block in base_blocks]
                    harmful_accepted, safe_restricted = counts["risky"], 0
                    retained = 1.0 if counts["safe"] else None
                else:
                    block_metrics = [read_json(output / "blocks" /
                        f"c{context:03d}__a{actor_budget:02d}__w{weather}__{mode}__k{inspection_budget}.json.gz")
                        ["metrics"] for weather in range(cfg["weather_streams"])]
                    selected = [row for row in decisions if row["context"] == context
                                and row["actor_budget"] == actor_budget
                                and row["mode"] == mode and row["inspection_budget"] == inspection_budget]
                    if len(selected) != len(group_labels):
                        raise ValueError("Decision count does not match frozen proposal count")
                    harmful_accepted = sum(row["harmful_accepted"] for row in selected)
                    safe_restricted = sum(row["safe_rejected"] for row in selected)
                    safe_scales = [row["scale"] for row in selected if row["safe_resolved"]]
                    retained = mean(safe_scales) if safe_scales else None
                summaries.append(dict(**common, mode=mode, inspection_budget=inspection_budget,
                    harmful_accepted=harmful_accepted, safe_restricted=safe_restricted,
                    harmful_approval_rate=_rate(harmful_accepted, counts["risky"]),
                    safe_restriction_rate=_rate(safe_restricted, counts["safe"]),
                    retained_safe_fraction=retained,
                    global_unsafe_rate=mean(row["fixed_horizon_unsafe_rate"] for row in block_metrics),
                    mean_patch_health=mean(row["mean_patch_health"] for row in block_metrics),
                    total_welfare=mean(row["total_welfare"] for row in block_metrics),
                    request_inspections=mean(row["request_inspections"] for row in block_metrics),
                    candidate_evaluations=mean(row["candidate_evaluations"] for row in block_metrics),
                    component_evaluations=mean(row["component_evaluations"] for row in block_metrics),
                    transmitted_scalars=mean(row["transmitted_scalars"] for row in block_metrics)))

    _write_csv(output / "analysis" / "candidate_summary.csv", selection_rows)
    _write_csv(output / "analysis" / "context_summary.csv", summaries)
    return dict(context_rows=len(summaries), selection_rows=len(selection_rows),
                resolved_safe=sum(row["reference_label"] == "safe" for row in labels),
                resolved_risky=sum(row["reference_label"] == "risky" for row in labels),
                unresolved=sum(row["reference_label"] == "unresolved" for row in labels))
