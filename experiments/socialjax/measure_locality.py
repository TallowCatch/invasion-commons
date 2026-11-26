"""Task 1 of notes/socialjax/locality/protocol.md: measure locality (chi, zeta, Lambda_w, xi_loc) per testbed.

Scripted bots (CPU):
  .venv-socialjax/bin/python -m experiments.socialjax.measure_locality --out results/locality/measure.json
  (--testbeds harvest,rooms ... ; --seeds N to override)
Trained IPPO agents (CREATE; scripts/create/measure_locality_ippo.sbatch):
  python -m experiments.socialjax.measure_locality --params-root /scratch/users/$USER/ippo_overseer \
      --runs A_none_seed1,B_random_k2_c1.0_seed1 --episodes 32 --out DIR/locality_ippo.json
  Each run's params.msgpack is loaded with the trainer's own network (train_ippo_overseer.socialjax_actor_critic)
  and played under the run's own overseer settings (its config.json), sampling actions as the trainer's eval does.

Everything is measured on the same rollouts for every region layout of a testbed. During the bot rollouts each
layout's own overseers check K agents per step at random (K = 2; 4 with 14 agents; 1 in Coins), with no penalty
and no freeze, so only the sneaky and migrator bots react to them.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from overseer.socialjax import locality as L
from overseer.socialjax.decentralised import DecConfig
from overseer.socialjax.game_bots import kind_codes, make_policy
from overseer.socialjax.games import make_rules
from overseer.socialjax.local_records import LocalRecordOverseer
from overseer.socialjax.regions import region_grid
from overseer.socialjax.rooms import make_rooms_rules, rooms_policy

STEPS = 999
WINDOWS = (1, 10, 50)


def _pops(n):
    return {f"{n} honest": ("honest",) * n,
            f"1 greedy + {n - 1} honest": ("greedy",) + ("honest",) * (n - 1),
            f"{n} sneaky": ("sneaky",) * n,
            f"{n} migrator": ("migrator",) * n}


def testbeds():
    """name -> dict(rules factory, policy factory, layouts [(label, (rows, cols) -> grid)], K, seeds, zeta kind)."""
    def strips(R, layout="strips"):
        return lambda rules: region_grid(rules.shape, R, layout)

    def rooms(rules):
        return rules.room_grid

    harvest_layouts = [("strips R=2", strips(2)), ("strips R=4", strips(4)), ("bands R=2", strips(2, "rows")),
                       ("grid R=4", strips(4, "grid"))]
    return {
        "harvest": dict(rules=lambda: make_rules("harvest", num_inner_steps=STEPS + 1), policy=make_policy,
                        layouts=harvest_layouts, K=2, seeds=10, zeta="harvest", pops=_pops(7)),
        "harvest_large": dict(rules=lambda: make_rules("harvest_large", num_inner_steps=STEPS + 1),
                              policy=make_policy, layouts=[("strips R=2", strips(2)), ("strips R=4", strips(4)),
                                                           ("grid R=4", strips(4, "grid"))],
                              K=4, seeds=5, zeta="harvest", pops=_pops(14)),
        "rooms": dict(rules=lambda: make_rooms_rules("rooms", num_inner_steps=STEPS + 1), policy=rooms_policy,
                      layouts=[("rooms (R=2)", rooms), ("strips R=4", strips(4))], K=2, seeds=10, zeta="harvest",
                      pops=_pops(7)),
        "rooms_door": dict(rules=lambda: make_rooms_rules("rooms_door", num_inner_steps=STEPS + 1),
                           policy=rooms_policy, layouts=[("rooms (R=2)", rooms), ("strips R=4", strips(4))], K=2,
                           seeds=10, zeta="harvest", pops=_pops(7)),
        "rooms_large": dict(rules=lambda: make_rooms_rules("rooms_large", num_inner_steps=STEPS + 1),
                            policy=rooms_policy, layouts=[("rooms (R=4)", rooms), ("strips R=2", strips(2))], K=4,
                            seeds=5, zeta="harvest", pops=_pops(14)),
        "coins": dict(rules=lambda: make_rules("coins", num_inner_steps=STEPS + 1), policy=make_policy,
                      layouts=[("bands R=2", strips(2, "rows")), ("strips R=2", strips(2))], K=1, seeds=10,
                      zeta="coins", pops={"2 honest": ("honest",) * 2, "1 greedy + 1 honest": ("greedy", "honest"),
                                          "2 sneaky": ("sneaky",) * 2, "2 migrator": ("migrator",) * 2}),
        "cleanup": dict(rules=lambda: make_rules("cleanup", num_inner_steps=STEPS + 1), policy=make_policy,
                        layouts=[("strips R=2", strips(2)), ("strips R=4", strips(4)), ("bands R=2", strips(2, "rows"))],
                        K=2, seeds=10, zeta="cleanup", pops=_pops(7)),
    }


# ------------------------------------------------------------------------------------------------ one trace
def measure_trace(trace, rules, rgrid, kind, key, cf_events=40):
    """Raw per-run measures for one region grid: chi, per-violation stays and zetas (lists, pooled later)."""
    reg = L.regions_of(trace["pos"], rgrid)
    ev = L.violation_events(trace)
    out = dict(chi=L.crossing_rate(reg), agent_steps=int(reg[1:].size), crossings=int((reg[1:] != reg[:-1]).sum()),
               violations=int(len(ev)),
               stays={str(w): L.stays(reg, ev, w).tolist() for w in WINDOWS})
    if kind == "harvest":
        spawn = np.zeros(rules.shape, bool)
        sp = np.asarray(rules.env.SPAWNS_APPLE)
        spawn[sp[:, 0], sp[:, 1]] = True
        dyn, sta, size = L.harvest_footprint(trace, ev, rgrid, spawn)
        inside, total = L.harvest_counterfactual(trace, ev, rgrid, spawn, key, horizon=100, max_events=cf_events)
        out.update(zeta=dyn.tolist(), zeta_static=sta.tolist(), footprint=size.tolist(),
                   cf_inside=inside.tolist(), cf_total=total.tolist())
    elif kind == "cleanup":
        orchard = np.asarray(rules.env.POTENTIAL_APPLE)
        out["zeta"] = L.cleanup_zeta(trace, ev, rgrid, orchard).tolist()
    elif kind == "coins":
        out["zeta"] = L.coins_zeta(trace, ev, rgrid).tolist()
    return out


def pool(runs):
    """Pool per-run measures of one (testbed, layout, population) into the reported numbers."""
    chi = sum(r["crossings"] for r in runs) / max(1, sum(r["agent_steps"] for r in runs))
    zeta_v = np.concatenate([np.asarray(r["zeta"], float) for r in runs]) if runs else np.zeros(0)
    nv = int(sum(r["violations"] for r in runs))
    zeta = float(np.nanmean(zeta_v)) if np.isfinite(zeta_v).any() else float("nan")
    res = dict(chi=chi, violations=nv, zeta=zeta, runs=len(runs))
    if "zeta_static" in runs[0]:
        zs = np.concatenate([np.asarray(r["zeta_static"], float) for r in runs])
        res["zeta_static"] = float(np.nanmean(zs)) if np.isfinite(zs).any() else float("nan")
        ins = sum(sum(r["cf_inside"]) for r in runs)
        tot = sum(sum(r["cf_total"]) for r in runs)
        res["zeta_cf"] = ins / tot if tot > 0 else float("nan")
        res["cf_events"] = int(sum(len(r["cf_total"]) for r in runs))
        res["cf_lost_per_violation"] = tot / max(1, res["cf_events"])
        # bootstrap over violations for the counterfactual ratio
        a = np.concatenate([np.asarray(r["cf_inside"]) for r in runs])
        b = np.concatenate([np.asarray(r["cf_total"]) for r in runs])
        if len(b) > 5 and b.sum() > 0:
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(500):
                i = rng.integers(0, len(b), len(b))
                if b[i].sum() > 0:
                    bs.append(a[i].sum() / b[i].sum())
            res["zeta_cf_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    zeta_for_lambda = zeta if np.isfinite(zeta) else 1.0
    for w in WINDOWS:
        st = np.concatenate([np.asarray(r["stays"][str(w)], float) for r in runs])
        zv = np.where(np.isnan(zeta_v), zeta_for_lambda, zeta_v) if len(zeta_v) else zeta_v
        res[f"S_{w}"] = float(np.nanmean(st)) if np.isfinite(st).any() else float("nan")
        res[f"S_{w}_formula"] = (1 - chi) ** (w - 1)
        res[f"Lambda_{w}_formula"] = L.lambda_formula(chi, zeta, w) if np.isfinite(zeta) else float("nan")
        res[f"Lambda_{w}_direct"] = L.lambda_direct(st, zv) if len(st) else float("nan")
        res[f"xi_{w}_formula"] = 1 - res[f"Lambda_{w}_formula"]
        res[f"xi_{w}_direct"] = 1 - res[f"Lambda_{w}_direct"]
    return res


# ------------------------------------------------------------------------------------------------ bots
def run_bots(names, out_path, seeds_override=None):
    tb = testbeds()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(out_path.read_text()) if out_path.exists() else dict(rows=[], done=[])
    for name in names:
        g = tb[name]
        rules = g["rules"]()
        policy = g["policy"](rules)
        seeds = range(seeds_override or g["seeds"])
        for label, make_grid in g["layouts"]:
            tag = f"{name} | {label}"
            if tag in d["done"]:
                continue
            t0 = time.time()
            rgrid = np.asarray(make_grid(rules))
            cfg = DecConfig(k=g["K"], chooser="random", window=1)
            ov = LocalRecordOverseer(rules, cfg, records="local", region_grid=rgrid)
            for pop, kinds in g["pops"].items():
                act = L.bot_act(policy, kind_codes(kinds), ov.region_grid)
                runs = []
                for s in seeds:
                    tr = L.collect_trace(ov, act, jax.random.PRNGKey(s), STEPS, keep_grid=g["zeta"] != "coins")
                    runs.append(measure_trace(tr, rules, rgrid, g["zeta"], jax.random.PRNGKey(1000 + s)))
                row = dict(testbed=name, layout=label, regions=int(rgrid.max()) + 1, population=pop,
                           n_agents=rules.env.num_agents, K=g["K"], **pool(runs))
                row["apples_taken"] = None
                d["rows"].append(row)
                print(f"{tag} | {pop}: chi {row['chi']:.4f} zeta {row['zeta']:.3f} "
                      f"xi(10) {row['xi_10_direct']:.3f} ({row['violations']} violations)", flush=True)
            jax.clear_caches()
            d["done"].append(tag)
            out_path.write_text(json.dumps(d, indent=1, default=float))
            print(f"  {tag}: {time.time() - t0:.0f} s", flush=True)
    return d


# ------------------------------------------------------------------------------------------------ IPPO
def run_ippo(params_root, runs, episodes, out_path):
    """chi, S_w, zeta and Lambda_w for trained IPPO agents, every layout of the run's game."""
    import flax.serialization

    from experiments.socialjax.train_ippo_overseer import socialjax_actor_critic

    tb = testbeds()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(out_path.read_text()) if out_path.exists() else dict(rows=[], done=[])
    for run in runs:
        if run in d["done"]:
            continue
        rdir = Path(params_root) / run
        cfgj = json.loads((rdir / "config.json").read_text())
        env_name = cfgj.get("env", "harvest")
        fields = DecConfig.__dataclass_fields__
        ov_kw = {k: v for k, v in cfgj["overseer"].items() if k in fields}  # stage 1 configs have extra fields
        ov_kw["region_k"] = tuple(ov_kw.get("region_k") or ())
        g = tb[env_name]
        rules = g["rules"]()
        network = socialjax_actor_critic()(rules.env.action_space().n, activation="relu")
        obs_shape = rules.env.observation_space()[0].shape
        params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, *obs_shape)))
        params = flax.serialization.from_bytes(params, (rdir / "params.msgpack").read_bytes())

        def act(key, obs, state, mem, params=params):
            pi, _ = network.apply(params, obs)
            return pi.sample(seed=key).astype(jnp.int32), mem

        # play under the run's own overseer (regions as trained); measure every layout on the same rollouts
        cfg = DecConfig(**ov_kw)
        ov = LocalRecordOverseer(rules, cfg, records="handover")
        t0 = time.time()
        traces = [L.collect_trace(ov, act, jax.random.PRNGKey(20_000 + e), STEPS, keep_grid=g["zeta"] != "coins")
                  for e in range(episodes)]
        for label, make_grid in g["layouts"]:
            rgrid = np.asarray(make_grid(rules))
            per = [measure_trace(tr, rules, rgrid, g["zeta"], jax.random.PRNGKey(3000 + e))
                   for e, tr in enumerate(traces)]
            row = dict(run=run, testbed=env_name, layout=label, regions=int(rgrid.max()) + 1,
                       population="IPPO " + run, n_agents=rules.env.num_agents, overseer=asdict(cfg), **pool(per))
            row["apples_taken"] = float(np.mean([tr["taken"].sum() for tr in traces]))
            d["rows"].append(row)
            print(f"{run} | {label}: chi {row['chi']:.4f} zeta {row['zeta']:.3f} xi(10) {row['xi_10_direct']:.3f}"
                  f" ({row['violations']} violations, {row['apples_taken']:.0f} apples/episode)", flush=True)
        d["done"].append(run)
        out_path.write_text(json.dumps(d, indent=1, default=float))
        print(f"  {run}: {time.time() - t0:.0f} s", flush=True)
        jax.clear_caches()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/locality/measure.json")
    ap.add_argument("--testbeds", default=",".join(testbeds()))
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--params-root", default=None, help="directory holding <run>/params.msgpack and config.json")
    ap.add_argument("--runs", default="", help="comma-separated run names under --params-root")
    ap.add_argument("--episodes", type=int, default=32)
    a = ap.parse_args()
    if a.params_root:
        run_ippo(a.params_root, [r for r in a.runs.split(",") if r], a.episodes, a.out)
    else:
        run_bots([t for t in a.testbeds.split(",") if t], a.out, a.seeds)


if __name__ == "__main__":
    main()
