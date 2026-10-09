"""Post hoc checks on L2 and L3 prompted by a review of the draft (decided 9 Oct 2026, after all results were seen).

1. One-round gain, two ways. Proposition 1(i) uses the largest excess available to an agent in a round (6 t minus its
   allowance). The pre-registered measure is the excess the model actually took. Both are averaged over over-take steps
   in E0 (no fine), with 95% bootstrap intervals over the 10 populations (ratio of sums, resampling populations).
2. Stability of the stopping fine. Resample the 10 populations (paired across fines) and recompute the stopping fine
   with the frozen rule: at most 5% of LLM agent-rounds over-taken at that fine and every larger one.

Run:  PYTHONPATH=. python -m experiments.oversight.l2_review_checks --store STORE --out DIR
STORE holds claude_l2_v1/ and claude_l3_v1/ (branch l2-results).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from experiments.oversight import run_l2_llm_agents as l2
from fishery_sim import llm_actor as L

MODELS = {"gpt-oss_120b-cloud": "gpt-oss-120b", "nemotron-3-super_cloud": "Nemotron 3 Super"}
B, BOOT_SEED, STOP = 4000, 20261019, 0.05
THR = l2.OVER_T / L.MAX_CATCH


def episodes(store, model):
    for run in ("claude_l2_v1", "claude_l3_v1"):
        for p in sorted((store / run / model / "episodes").glob("E*__*.json")):
            if not p.name.startswith("EM__"):
                yield int(re.match(r"E(\d+)__", p.name).group(1)), json.loads(p.read_text())


def ratio_ci(num, den, rng):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), size=(B, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return [float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def stopping_fine(rates, fines):
    for F in fines:
        if all(rates[G] <= STOP for G in fines if G >= F):
            return F
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    store, out, res = Path(a.store), Path(a.out), {}
    for m, label in MODELS.items():
        rng = np.random.default_rng(BOOT_SEED)
        taken, avail, n_over = collections.defaultdict(float), collections.defaultdict(float), collections.defaultdict(int)
        rate = collections.defaultdict(dict)
        for F, e in episodes(store, m):
            n = k = 0
            for s in e["steps"]:
                for i in e["llm_agents"]:
                    n += 1
                    if s["taken"][i] > s["allowance"][i] + THR:
                        k += 1
                        if F == 0:
                            taken[e["context"]] += (s["taken"][i] - s["allowance"][i]) * L.MAX_CATCH
                            avail[e["context"]] += (1.0 - s["allowance"][i]) * L.MAX_CATCH
                            n_over[e["context"]] += 1
            rate[F][e["context"]] = k / n
        ctx = sorted(rate[0])
        fines = sorted(rate)
        observed = stopping_fine({F: np.mean([rate[F][c] for c in ctx]) for F in fines}, fines)
        boot = collections.Counter()
        for _ in range(B):
            idx = rng.integers(0, len(ctx), len(ctx))
            boot[stopping_fine({F: np.mean([rate[F][ctx[j]] for j in idx]) for F in fines}, fines)] += 1
        res[label] = dict(
            observed_excess_t=ratio_ci([taken[c] for c in ctx], [n_over[c] for c in ctx], rng),
            largest_available_excess_t=ratio_ci([avail[c] for c in ctx], [n_over[c] for c in ctx], rng),
            fines_t=fines, observed_stopping_fine_t=observed,
            bootstrap_stopping_fine_share={str(k): v / B for k, v in sorted(boot.items())},
            overtake_rate_by_fine={str(F): float(np.mean([rate[F][c] for c in ctx])) for F in fines})
    out.mkdir(parents=True, exist_ok=True)
    (out / "l2_review_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("observed_excess_t", "largest_available_excess_t",
                                               "observed_stopping_fine_t", "bootstrap_stopping_fine_share")}
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
