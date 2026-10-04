"""Context-clustered analysis of fresh budgeted reviewer decisions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_budgeted_reviewer import analyze as analyze_base
from experiments.run_matched_oversight import read_json


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _counts(frame: pd.DataFrame) -> dict:
    return {name: int(frame[name].sum()) for name in (
        "safe_resolved", "risky_resolved", "unresolved", "safe_rejected", "harmful_accepted")}


def _difference(table: pd.DataFrame, comparator: str) -> tuple[float, float]:
    counts = table.groupby("method", as_index=True)[
        ["safe_resolved", "risky_resolved", "safe_rejected", "harmful_accepted"]].sum()
    left, right = counts.loc["joint"], counts.loc[comparator]
    return (_rate(left.harmful_accepted, left.risky_resolved)
            - _rate(right.harmful_accepted, right.risky_resolved),
            _rate(left.safe_rejected, left.safe_resolved)
            - _rate(right.safe_rejected, right.safe_resolved))


def _cluster_interval(table: pd.DataFrame, comparator: str, rng: np.random.Generator,
                      replicates: int = 4000) -> tuple[float, float, float, float]:
    contexts = sorted(table.context.unique())
    risk, safe = [], []
    blocks = {context: table[table.context.eq(context)] for context in contexts}
    for _ in range(replicates):
        sample = rng.choice(contexts, size=len(contexts), replace=True)
        resampled = pd.concat([blocks[int(context)] for context in sample], ignore_index=True)
        risky_diff, safe_diff = _difference(resampled, comparator)
        risk.append(risky_diff)
        safe.append(safe_diff)

    def interval(values):
        valid = np.asarray(values, dtype=float)
        valid = valid[np.isfinite(valid)]
        if len(valid) != replicates:
            return float("nan"), float("nan")
        return tuple(map(float, np.quantile(valid, [0.025, 0.975])))

    return (*interval(risk), *interval(safe))


def analyze(directory: Path, jobs=None) -> dict:
    directory = Path(directory)
    from experiments.run_budgeted_reviewer_confirmation import jobs_for
    cfg = read_json(directory / "manifest.json")["protocol"]
    jobs = jobs if jobs is not None else jobs_for(cfg)
    base = analyze_base(directory, jobs)
    labels = pd.DataFrame(read_json(directory / "labels.json.gz"))
    decisions = pd.DataFrame(read_json(directory / "decisions.json.gz"))
    if labels.case_id.duplicated().any():
        raise ValueError("Duplicate case ID")
    joined = decisions.merge(labels[["case_id", "game", "context", "pre_global_safe",
                                     "reference_label"]], on="case_id", validate="many_to_one")
    joined = joined[joined.pre_global_safe.eq(1)].copy()
    if not len(joined):
        raise ValueError("No initially safe cases")
    table = joined.groupby(["game", "context", "inspection_budget", "method"],
                           as_index=False)[["safe_resolved", "risky_resolved", "unresolved",
                                            "safe_rejected", "harmful_accepted"]].sum()
    expected = {(cell["game"], context, budget, mode)
                for cell in cfg["cells"] for context in range(cfg["contexts"])
                for budget in cfg["inspection_budgets"] for mode in cfg["modes"]}
    actual = set(map(tuple, table[["game", "context", "inspection_budget", "method"]].itertuples(
        index=False, name=None)))
    if actual != expected:
        raise ValueError("Missing context/mode/budget decision coverage")

    primary = []
    for (game, budget, mode), frame in table.groupby(["game", "inspection_budget", "method"]):
        counts = _counts(frame)
        primary.append(dict(game=game, inspection_budget=budget, method=mode,
            n_contexts=cfg["contexts"],
            contexts_with_safe=int(frame.safe_resolved.gt(0).sum()),
            contexts_with_risky=int(frame.risky_resolved.gt(0).sum()),
            **counts, harmful_accept_rate=_rate(counts["harmful_accepted"], counts["risky_resolved"]),
            safe_reject_rate=_rate(counts["safe_rejected"], counts["safe_resolved"])))
    primary_frame = pd.DataFrame(primary).sort_values(["game", "inspection_budget", "method"])

    contrasts = []
    rng = np.random.default_rng(20260923)
    if cfg["profile"] == "confirm":
        for game in sorted(table.game.unique()):
            current = table[table.game.eq(game) & table.inspection_budget.eq(6)]
            for comparator in ("local_bounded", "local_optimistic"):
                risky_diff, safe_diff = _difference(current, comparator)
                risk_lo, risk_hi, safe_lo, safe_hi = _cluster_interval(current, comparator, rng)
                coverage = primary_frame[(primary_frame.game.eq(game)) &
                                         (primary_frame.inspection_budget.eq(6)) &
                                         (primary_frame.method.eq("joint"))].iloc[0]
                contrasts.append(dict(game=game, inspection_budget=6, left="joint", right=comparator,
                    n_contexts=cfg["contexts"],
                    contexts_with_safe=int(coverage.contexts_with_safe),
                    contexts_with_risky=int(coverage.contexts_with_risky),
                    harmful_accept_difference=risky_diff, harmful_accept_ci_low=risk_lo,
                    harmful_accept_ci_high=risk_hi, safe_reject_difference=safe_diff,
                    safe_reject_ci_low=safe_lo, safe_reject_ci_high=safe_hi,
                    cluster_bootstrap_replicates=4000))
    contrasts_frame = pd.DataFrame(contrasts)

    out = directory / "analysis"
    table.to_csv(out / "context_decision_counts.csv", index=False)
    primary_frame.to_csv(out / "primary_decision_quality.csv", index=False)
    contrasts_frame.to_csv(out / "primary_paired_contrasts.csv", index=False)
    lines = ["# Fresh-seed budgeted reviewer confirmation", "",
        "Protocol: `notes/research_review/BUDGETED_REVIEWER_CONFIRMATION_PROTOCOL.md`.",
        f"Profile: `{cfg['profile']}`. {base['episodes']} episodes, {base['cases']} frozen original "
        f"proposals, {base['decisions']} matched reviewer decisions.",
        "Tracking: local files, source snapshots and checksums.", "",
        "## Primary decision coverage", "",
        "Rates use only proposals from initially safe states. Unresolved labels are excluded from "
        "the safe/risky denominators and retained in the table.", "",
        "| Game | Safe | Risky | Unresolved | Contexts with safe | Contexts with risky |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for game in sorted(primary_frame.game.unique()):
        row = primary_frame[(primary_frame.game.eq(game)) &
                            (primary_frame.inspection_budget.eq(cfg["inspection_budgets"][-1])) &
                            (primary_frame.method.eq("joint"))].iloc[0]
        lines.append(f"| {game} | {row.safe_resolved} | {row.risky_resolved} | {row.unresolved} "
                     f"| {row.contexts_with_safe} | {row.contexts_with_risky} |")
    lines += ["", "## Interpretation boundary", "",
        "The two strata were selected using a prior development pilot. The current population "
        "seeds are fresh; no claim transfers to other mixtures or tasks.",
        "Contexts are the independent units. Timesteps, weather streams and paired methods are not.",
        "The 95% context-bootstrap intervals estimate sampling variability over these policy "
        "generators; they do not cover model, threshold or game-family uncertainty.",
        "A missing safe or risky denominator makes that rate unavailable. No p-values or "
        "equivalence claims are made.",
        "Closed-loop outcomes are in `outcomes.csv` and `paired_contrasts.csv`; they are secondary "
        "and can rank methods differently from the immediate decisions.",
        "The smoke profile is an engineering check only; it has no uncertainty intervals.", ""]
    (out / "experiment.md").write_text("\n".join(lines), encoding="utf8")
    return dict(**base, primary_rows=len(primary_frame), contrast_rows=len(contrasts_frame))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    print(analyze(parser.parse_args().input_dir))
