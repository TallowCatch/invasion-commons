"""Fixed structural challenges and native one-step replay, not policy samples."""
from dataclasses import replace
import hashlib
import json

import numpy as np

from .harvest import BaseHarvestAgent, HarvestAction, run_harvest_episode
from .oversight_protocol import harvest_nominal_next


def harvest_challenge_cases(cfg):
    if cfg.n_agents != 6 or cfg.patch_max != 20:
        raise ValueError("This declared grid requires six patches of capacity 20")
    cases = {}
    for mean in (8., 10., 11., 14., 18.):
        profiles = {
            "uniform": np.full(6, mean),
            "alternating": mean + np.array([-2., 2.] * 3),
            "clustered": mean + np.array([-2.] * 3 + [2.] * 3),
            "half_depleted": np.minimum([0.] * 3 + [2 * mean] * 3, cfg.patch_max),
        }
        for profile, health in profiles.items():
            for demand in (0., .1, .25, .5, .75, 1.):
                concentrated = np.clip(6 * demand - np.arange(6), 0, 1)
                layouts = {"uniform": np.full(6, demand), "concentrated": concentrated,
                           "reverse_concentrated": concentrated[::-1]}
                for layout, actions in layouts.items():
                    key = json.dumps([health.tolist(), actions.tolist()], separators=(",", ":"))
                    alias = dict(health_parameter=mean, health_profile=profile,
                                 demand_mean=demand, allocation=layout)
                    if key in cases:
                        cases[key]["design"]["aliases"].append(alias)
                    else:
                        cases[key] = dict(case_id="harvest-" + hashlib.sha256(key.encode()).hexdigest()[:20],
                            state=health.tolist(), proposals=actions.tolist(),
                            design=dict(aliases=[alias], actual_mean_health=float(health.mean()),
                                failed_patch_fraction=float(np.mean(health < cfg.local_patch_failure_threshold))))
    return list(cases.values())


class FixedRequestAgent(BaseHarvestAgent):
    def __init__(self, request):
        self.request = request

    def act(self, observation, inbox, t):
        return HarvestAction(harvest_frac=self.request)


def validate_native_replay(cfg, health, requests, seed):
    """Check the reference formula on the original simulator's physical step."""
    replay_cfg = replace(cfg, horizon=1, seed=seed, communication_enabled=False,
                         side_payments_enabled=False)
    agents = [FixedRequestAgent(float(a)) for a in requests]
    result = run_harvest_episode(replay_cfg, agents, record_trace=True,
                                 initial_patch_health=np.asarray(health))
    native = result["episode_trace_rows"][0]
    weather = np.random.default_rng(seed).normal(0, cfg.weather_noise_std, cfg.n_agents)
    predicted = np.clip(harvest_nominal_next(cfg, health, requests) + weather, 0, cfg.patch_max)
    np.testing.assert_allclose(predicted, json.loads(native["patch_health_after_json"]), atol=1e-12, rtol=0)
    extracted = float(np.minimum(np.asarray(requests) * cfg.max_harvest_per_agent, health).sum())
    np.testing.assert_allclose(extracted, native["mean_realized_harvest"], atol=1e-12, rtol=0)
    return dict(seed=seed, next_health=predicted.tolist(), extracted=extracted, native_parity=True)
