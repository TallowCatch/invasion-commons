from __future__ import annotations

import pandas as pd
import pytest

from experiments.paper_v5.analyze_reviewer_longrun import analyze


def _episodes():
    rows = []
    for context in range(64):
        for mode, welfare in (("joint", 2.0), ("local_bounded", 1.0),
                              ("local_optimistic", 3.0)):
            rows.append(dict(game="fishery", policy_group="mix", regime="pilot",
                             context=context, weather=0, mode=mode,
                             inspection_budget=6, total_welfare=welfare,
                             mean_patch_health=welfare + 10,
                             unsafe_fixed_horizon=0.0))
    return pd.DataFrame(rows)


def test_paired_longrun_uses_contexts(tmp_path):
    source = tmp_path / "episodes.csv"
    _episodes().to_csv(source, index=False)
    summary = analyze(source, tmp_path / "out", draws=100)
    bounded = summary.loc[
        (summary.comparison == "joint_minus_local_bounded")
        & (summary.metric == "total_welfare")
    ].iloc[0]
    assert bounded.independent_contexts == 64
    assert bounded.mean_difference == 1.0
    assert bounded.ci_95_low == bounded.ci_95_high == 1.0
    assert len(pd.read_csv(tmp_path / "out" / "longrun_context_differences.csv")) == 384


def test_paired_longrun_rejects_missing_mode(tmp_path):
    source = tmp_path / "episodes.csv"
    _episodes().query("not (context == 0 and mode == 'joint')").to_csv(source, index=False)
    with pytest.raises(ValueError, match="Missing paired mode"):
        analyze(source, tmp_path / "out", draws=100)
