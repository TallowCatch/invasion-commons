import pandas as pd
import pytest

from experiments.paper_v5.audit_research_evidence import (
    CONTEXT,
    METRICS,
    local_global_table,
    paired_contrasts,
    trace_events,
    within_group_ranges,
)


def sample_row(condition="hybrid", run_id=1):
    return {**dict(zip(CONTEXT, ["example", "low_actor", "weak_overseer"])),
            "condition": condition, "run_id": run_id,
            **dict.fromkeys(METRICS, 0.2),
            "test_all_local_safe_step_fraction_mean": 0.5,
            "test_local_pass_global_fail_rate_mean": 0.1}


def test_joint_rates_include_local_failure_global_safety():
    rates = local_global_table(pd.DataFrame([sample_row()])).iloc[0]
    assert rates.all_pass_global_safe == pytest.approx(0.4)
    assert rates.all_pass_global_unsafe == pytest.approx(0.1)
    assert rates.some_fail_global_safe == pytest.approx(0.4)
    assert rates.some_fail_global_unsafe == pytest.approx(0.1)


def test_joint_rates_reject_inconsistent_inputs():
    row = sample_row()
    row["test_local_pass_global_fail_rate_mean"] = 0.8
    with pytest.raises(ValueError, match="inconsistent"):
        local_global_table(pd.DataFrame([row]))


def test_trace_separates_onset_persistence_and_unknown_initial_state():
    trace = pd.DataFrame({
        "global_unsafe": [1, 0, 1, 1],
        "all_local_safe": [1, 0, 1, 1],
        "local_pass_global_fail": [1, 0, 1, 1],
        "mean_patch_health_before": [9, 9, 11, 9],
    })
    result = trace_events(trace)
    assert result["lpgf_steps"] == 3
    assert result["lpgf_safe_to_unsafe_known_previous_state"] == 1
    assert result["lpgf_already_unsafe_known_previous_state"] == 1
    assert result["lpgf_previous_state_unavailable"] == 1
    assert result["some_local_fail_global_safe_steps"] == 1


def test_invariance_is_checked_within_underlying_run():
    df = pd.DataFrame({"run": [1, 1, 2, 2], "metric": [1, 1, 4, 4]})
    assert within_group_ranges(df, ["run"], ["metric"]) == {"metric": 0}


def test_architecture_contrasts_require_matching_run_ids():
    df = pd.DataFrame([sample_row("hybrid", 1), sample_row("top_down_only", 2),
                       sample_row("bottom_up_only", 1)])
    with pytest.raises(ValueError, match="Unpaired"):
        paired_contrasts(df)


def test_architecture_contrasts_preserve_pairing():
    rows = [sample_row(condition, run) for condition in
            ["hybrid", "top_down_only", "bottom_up_only"] for run in [1, 2]]
    rows[0][METRICS[0]] = 0.3
    rows[1][METRICS[0]] = 0.5
    contrasts = paired_contrasts(pd.DataFrame(rows))
    selected = contrasts[(contrasts.contrast == "hybrid_minus_top_down_only")
                         & (contrasts.metric == METRICS[0])].iloc[0]
    assert selected.n_pairs == 2
    assert selected.mean_difference == pytest.approx(0.2)
    assert selected.min_run_difference == pytest.approx(0.1)
    assert selected.max_run_difference == pytest.approx(0.3)
