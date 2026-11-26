import numpy as np
import pandas as pd

from experiments.paper_v5.plot_reviewer_decisions import estimate
from experiments.paper_v5.replay_coupled_local import replay_case
from fishery_sim.budgeted_oversight import decide_budgeted_fishery, mask_requests
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_oversight import FisherySnapshot
from fishery_sim.oversight_protocol import MonitorSettings


def test_replay_uses_only_the_declared_inspected_subset():
    cfg = FisheryConfig(n_agents=6, obs_noise_std=0, regen_rate=.7)
    proposals = [.2, .4, .1, .8, .6, .3]
    case = dict(case_id="known-case", game="fishery", config=vars(cfg),
                state=dict(stock=20, below_count=0, collapsed=False), proposals=proposals)
    visible = mask_requests(proposals, 3, "known-case")
    replayed = replay_case(case, 3, MonitorSettings())
    direct = decide_budgeted_fishery(cfg, FisherySnapshot(20), visible,
                                    "local_coupled", MonitorSettings())
    assert replayed == direct
    changed = np.asarray(proposals)
    changed[np.isnan(visible)] = 1 - changed[np.isnan(visible)]
    assert replay_case({**case, "proposals": changed.tolist()}, 3, MonitorSettings()) == replayed


def test_plot_estimate_resamples_contexts_and_preserves_case_weighting():
    frame = pd.DataFrame(dict(context=[0, 1], safe_rejected=[1, 0],
                              safe_resolved=[1, 9]))
    estimate_value, low, high = estimate(frame, replicates=100, seed=1)
    assert estimate_value == .1
    assert 0 <= low <= estimate_value <= high <= 1
