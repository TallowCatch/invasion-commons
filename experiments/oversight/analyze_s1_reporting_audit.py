"""Analysis for experiment S1 (Claude audit, October 2026)."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

B, SEED = 4000, 20261006


def boot(x, y, rng, ratio=None):
    """Paired context bootstrap of mean difference (or ratio-of-sums difference)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if ratio is None:
        d = x - y
        draws = [d[rng.integers(0, n, n)].mean() for _ in range(B)]
        est = d.mean()
    else:
        dx, dy = (np.asarray(v, float) for v in ratio)
        f = lambda i: x[i].sum() / max(dx[i].sum(), 1e-12) - y[i].sum() / max(dy[i].sum(), 1e-12)
        est = f(np.arange(n))
        draws = [f(rng.integers(0, n, n)) for _ in range(B)]
    return dict(estimate=float(est), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    E = pd.DataFrame([json.loads(l) for l in gzip.open(run / "episodes.jsonl.gz", "rt")])
    E["steps"] = E["t_end"]
    keys = ["game", "protocol", "actor", "d"]
    T = E.groupby(keys).agg(contexts=("context", "nunique"), scored=("scored_steps", "sum"),
                            exec_risky=("exec_risky", "sum"), exec_unresolved=("exec_unresolved", "sum"),
                            req_safe=("req_safe", "sum"), useful_loss=("useful_loss", "sum"),
                            total_harvest=("total_harvest", "mean"), honest_harvest=("honest_harvest", "mean"),
                            misreporter_payoff=("misreporter_payoff", "mean"), mean_health=("mean_health", "mean"),
                            unsafe_fixed=("unsafe_fixed", "mean"), audits=("audits", "sum"),
                            messages=("messages", "sum"), steps=("steps", "sum"), catches=("catches", "mean"),
                            caught_honest=("caught_honest", "sum")).reset_index()
    T["unsafe_approval_rate"] = T.exec_risky / T.scored
    T["usefulness_loss_rate"] = T.useful_loss / T.req_safe.replace(0, np.nan)
    T["audits_per_step"] = T.audits / T.steps
    T["messages_per_step"] = T.messages / T.steps
    T.to_csv(out / "condition_table.csv", index=False)

    def cond(game, protocol, d=None, actor="fixed"):
        m = (E.game == game) & (E.protocol == protocol) & (E.actor == actor)
        if d is not None and protocol not in ("none", "full"):
            m &= np.isclose(E.d, d)
        return E[m].sort_values("context").reset_index(drop=True)

    S = {}
    # H1 engineering gate: d=0 report-based protocols identical to full
    mism = 0
    for game in ("fishery", "harvest"):
        f = cond(game, "full")
        for p in E[(E.game == game)].protocol.unique():
            if p in ("full", "none"):
                continue
            c = cond(game, p, 0.0)
            for k in ("total_harvest", "exec_risky", "useful_loss", "mean_health"):
                mism += int(not np.allclose(c[k].values, f[k].values))
    S["H1_mismatches"] = mism
    for game in ("fishery", "harvest"):
        protos = [p for p in ("report", "rand1", "rand2", "targ1", "targ2", "peer", "peer_collude")
                  if (E[(E.game == game)].protocol == p).any()]
        for d in (0.25, 0.5):
            full = cond(game, "full")
            base = cond(game, "report", d)
            G = {}
            G["report_minus_full_unsafe"] = boot(base.exec_risky, full.exec_risky, rng, ratio=(base.scored_steps, full.scored_steps))
            for p in protos[1:]:
                c = cond(game, p, d)
                G[f"{p}_minus_report_unsafe"] = boot(c.exec_risky, base.exec_risky, rng, ratio=(c.scored_steps, base.scored_steps))
                G[f"{p}_minus_full_total_harvest"] = boot(c.total_harvest, full.total_harvest, rng)
            if game == "harvest":
                for k in (1, 2):
                    t, r = cond(game, f"targ{k}", d), cond(game, f"rand{k}", d)
                    G[f"targ{k}_minus_rand{k}_unsafe"] = boot(t.exec_risky, r.exec_risky, rng, ratio=(t.scored_steps, r.scored_steps))
                    G[f"targ{k}_minus_rand{k}_misreporter_payoff"] = boot(t.misreporter_payoff, r.misreporter_payoff, rng)
            pc, pp = cond(game, "peer_collude", d), cond(game, "peer", d)
            G["peer_collude_minus_peer_unsafe"] = boot(pc.exec_risky, pp.exec_risky, rng, ratio=(pc.scored_steps, pp.scored_steps))
            G["report_minus_full_misreporter_payoff"] = boot(base.misreporter_payoff, full.misreporter_payoff, rng)
            G["report_minus_full_total_harvest"] = boot(base.total_harvest, full.total_harvest, rng)
            S[f"{game}_d{d}"] = G
        # does misreporting pay? payoff(d) - payoff(0) per protocol on held-out contexts
        pay = {}
        for p in protos:
            z = cond(game, p, 0.0)
            pay[p] = {str(d): boot(cond(game, p, d).misreporter_payoff, z.misreporter_payoff, rng) for d in (0.25, 0.5)}
        S[f"{game}_misreporting_gain"] = pay
    S["adaptive_search"] = json.loads((run / "adaptive_search.json").read_text())
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print("H1 mismatches:", mism)


if __name__ == "__main__":
    main()
