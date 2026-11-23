"""H5: how much commons harm passes the individual rule? (local pass, global fail; written 23 Nov 2026)

Protocol and definitions: notes/formulation/h5_lpgf/protocol.md. Write-up: notes/formulation/h5_lpgf/results.md.

Harvest measures, all counted as creation events between t - 1 and t (so steps after a collapse add nothing):
  H_ext   cells of patches that lose their last apple (permanent: such a patch can never regrow);
  H_dead  apple cells that become locally dead (no apple on the cell or within two squares; reversible);
  June    downward crossings of g_t = 1[apples_t >= tau * apples_0], tau in TAUS.
LPGF: (a) share of harm steps / crossings where every agent passed the rule; (b) share of harm created in such
steps; (c) per-event attribution of each dead cell or extinct patch to the eats that removed its last apples:
legal solo, legal joint, or violation involved.
Clean Up (once its params exist locally): crossings of pollution >= 0.4 (regrowth stops) and of the apple predicate,
step-level (a) and window-level (a_W, no violation in the rule's W-step window).

Usage (venv .venv-socialjax):
  python -m experiments.socialjax.measure_lpgf --params-root results/create_pull \
      --runs A_none_seed1,B_random_k2_c1.0_seed1 --bots --episodes 32
A run directory needs params.msgpack; its overseer comes from config.json if present (needed for Clean Up runs),
else from the stage 1 run name (<stage>_<chooser>_k<k>_c<penalty apples>_seed<s>, or <stage>_none_seed<s>).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from overseer.socialjax.harvest_overseer import APPLE, NEIGHBOURHOOD

STEPS = 999
TAUS = (0.1, 0.25, 0.5)
CLOSED = ((0, 0),) + NEIGHBOURHOOD  # C(c): the cell and its 12 neighbours
CU_DEPLETION = 0.4  # clean_up.py thresholdDepletion: no regrowth at or above this pollution
BOT_POPS = {"7 honest": ("honest",) * 7, "7 greedy": ("greedy",) * 7, "1 greedy + 6 honest": ("greedy",) + ("honest",) * 6}


# ------------------------------------------------------------------------------------------------ map helpers
def shift_sum(x, offsets):
    """Sum of x shifted by each (dr, dc) in offsets, zero outside the map: out[..., r, c] = sum x[..., r+dr, c+dc].
    Works on (..., rows, cols)."""
    x = np.asarray(x).astype(np.int32)
    rows, cols = x.shape[-2:]
    pad = [(0, 0)] * (x.ndim - 2) + [(2, 2), (2, 2)]
    p = np.pad(x, pad)
    return sum(p[..., 2 + dr:2 + dr + rows, 2 + dc:2 + dc + cols] for dr, dc in offsets)


def patches(spawn_mask):
    """Label apple cells joined when one is within two squares of the other (the regrowth neighbourhood).
    Returns (labels (rows, cols) with -1 off the apple cells, number of patches)."""
    spawn_mask = np.asarray(spawn_mask, bool)
    labels = np.full(spawn_mask.shape, -1, int)
    k = 0
    for r, c in zip(*np.nonzero(spawn_mask)):
        if labels[r, c] >= 0:
            continue
        stack, labels[r, c] = [(r, c)], k
        while stack:
            a, b = stack.pop()
            for dr, dc in NEIGHBOURHOOD:
                y, x = a + dr, b + dc
                if 0 <= y < spawn_mask.shape[0] and 0 <= x < spawn_mask.shape[1] and spawn_mask[y, x] \
                        and labels[y, x] < 0:
                    labels[y, x] = k
                    stack.append((y, x))
        k += 1
    return labels, k


def dead_cells(apples, spawn_mask):
    """Locally dead: an apple cell with no apple on it and none within two squares (cannot regrow next step)."""
    return np.asarray(spawn_mask, bool) & ~apples & (shift_sum(apples, NEIGHBOURHOOD) == 0)


def eat_maps(eat_pos, eaten, violation, shape):
    """Per step, maps of legal and violating eats. eat_pos (T, N, 2) cells after the step, eaten/violation (T, N)."""
    T = eaten.shape[0]
    legal = np.zeros((T,) + tuple(shape), np.int32)
    viol = np.zeros_like(legal)
    t_idx, i_idx = np.nonzero(eaten)
    r, c = eat_pos[t_idx, i_idx, 0], eat_pos[t_idx, i_idx, 1]
    v = violation[t_idx, i_idx].astype(bool)
    np.add.at(viol, (t_idx[v], r[v], c[v]), 1)
    np.add.at(legal, (t_idx[~v], r[~v], c[~v]), 1)
    return legal, viol


def _classify(n_legal, n_viol):
    """Event classes from the responsible eats: 0 legal solo, 1 legal joint, 2 violation involved, 3 no eat."""
    return np.where(n_viol > 0, 2, np.where(n_legal == 1, 0, np.where(n_legal >= 2, 1, 3)))


CLASSES = ("legal_solo", "legal_joint", "violation", "no_eat")


# ------------------------------------------------------------------------------------------------ Harvest
def harvest_measures(apples, eat_pos, eaten, violation, spawn_mask, taus=TAUS):
    """Per-episode sums for one Harvest episode.
    apples (T+1, rows, cols) bool: the map at reset, then after each step; eat_pos (T, N, 2) agents' cells after
    each step; eaten, violation (T, N) bool."""
    apples = np.asarray(apples, bool)
    eaten = np.asarray(eaten, bool)
    violation = np.asarray(violation, bool) & eaten
    spawn_mask = np.asarray(spawn_mask, bool)
    T = eaten.shape[0]
    labels, n_patch = patches(spawn_mask)
    all_pass = ~violation.any(1)  # (T,) every agent passed the rule at step t
    legal, viol = eat_maps(np.asarray(eat_pos), eaten, violation, spawn_mask.shape)
    out = dict(steps=int(T), eats=int(eaten.sum()), violations=int(violation.sum()),
               all_pass_steps=int(all_pass.sum()))

    # --- H_dead: newly locally dead cells, attributed to the eats inside C(c) at that step
    dead = dead_cells(apples, spawn_mask)  # (T+1, rows, cols)
    new_dead = dead[1:] & ~dead[:-1]
    revived = dead[:-1] & ~dead[1:]
    nl, nv = shift_sum(legal, CLOSED), shift_sum(viol, CLOSED)
    t_d, r_d, c_d = np.nonzero(new_dead)
    cls = _classify(nl[t_d, r_d, c_d], nv[t_d, r_d, c_d])
    h_dead = new_dead.sum((1, 2))
    # lasting: the cell stays dead from then to the end of the episode (its last death)
    last_alive = T - np.argmax(~dead[::-1], axis=0)  # (rows, cols) last index at which the cell was not dead
    lasting = t_d == last_alive[r_d, c_d]  # alive at dead[t_d], dead at t_d + 1 and never revived
    out.update({f"dead_lasting_{k}": int(((cls == j) & lasting).sum()) for j, k in enumerate(CLASSES)})
    out.update(dead_new=int(new_dead.sum()), dead_revived=int(revived.sum()), dead_end=int(dead[-1].sum()),
               dead_new_allpass=int(h_dead[all_pass].sum()), dead_steps=int((h_dead > 0).sum()),
               dead_steps_allpass=int(((h_dead > 0) & all_pass).sum()),
               **{f"dead_{k}": int((cls == j).sum()) for j, k in enumerate(CLASSES)})
    # per eat: did it help kill at least one cell (a new dead cell within its closed neighbourhood)?
    kills_near = shift_sum(new_dead, CLOSED) > 0  # (T, rows, cols)
    t_e, i_e = np.nonzero(eaten)
    k_e = kills_near[t_e, eat_pos[t_e, i_e, 0], eat_pos[t_e, i_e, 1]]
    v_e = violation[t_e, i_e]
    out.update(legal_eats_killing=int((k_e & ~v_e).sum()), violating_eats_killing=int((k_e & v_e).sum()))

    # --- H_ext: patches that lose their last apple, attributed to the eats inside the patch at that step
    idx = labels[spawn_mask]
    sizes = np.bincount(idx, minlength=n_patch)
    per_patch = np.stack([np.bincount(idx, weights=a[spawn_mask], minlength=n_patch) for a in apples])  # (T+1, P)
    ext = (per_patch[:-1] > 0) & (per_patch[1:] == 0)  # (T, P)
    leg_p = np.stack([np.bincount(idx, weights=m[spawn_mask], minlength=n_patch) for m in legal])
    vio_p = np.stack([np.bincount(idx, weights=m[spawn_mask], minlength=n_patch) for m in viol])
    t_x, p_x = np.nonzero(ext)
    cls_x = _classify(leg_p[t_x, p_x], vio_p[t_x, p_x])
    h_ext = (ext * sizes).sum(1)
    out.update(ext_patches=int(ext.sum()), ext_cells=int(h_ext.sum()), ext_cells_allpass=int(h_ext[all_pass].sum()),
               ext_steps=int((h_ext > 0).sum()), ext_steps_allpass=int(((h_ext > 0) & all_pass).sum()),
               patches_alive_end=int((per_patch[-1] > 0).sum()),
               **{f"ext_cells_{k}": int(sizes[p_x[cls_x == j]].sum()) for j, k in enumerate(CLASSES)})

    # --- June predicate: downward crossings of apples >= tau * apples_0
    count = apples.reshape(T + 1, -1).sum(1)
    a0 = count[0]
    for tau in taus:
        g = count >= tau * a0
        down = g[:-1] & ~g[1:]
        first = np.zeros(T, bool)
        if down.any():
            first[np.argmax(down)] = True
        out.update({f"tau{tau}_cross": int(down.sum()), f"tau{tau}_cross_allpass": int((down & all_pass).sum()),
                    f"tau{tau}_first": int(first.sum()), f"tau{tau}_first_allpass": int((first & all_pass).sum()),
                    f"tau{tau}_june_steps": int((all_pass & ~g[1:]).sum())})  # June's raw form, all steps
    out.update(apples0=int(a0), apples_end=int(count[-1]), apples_mean=float(count[1:].mean()),
               alive_end=int(count[-1] > 0), ever_empty=int((count[1:] == 0).any()))
    return out


def harvest_from_trace(trace, spawn_mask):
    apples = np.concatenate([trace["grid0"][None], trace["grid"]], 0) == APPLE
    return harvest_measures(apples, trace["pos"][1:], trace["taken"], trace["violation"], spawn_mask)


# ------------------------------------------------------------------------------------------------ Clean Up
def cleanup_measures(pollution, apples_count, apples0, violation, window, taus=TAUS, depletion=CU_DEPLETION):
    """Per-episode sums for one Clean Up episode. pollution, apples_count (T,) after each step; violation (T, N).
    Crossings are counted from step 1 on (the pollution at reset is not in the trace)."""
    pollution, apples_count = np.asarray(pollution, float), np.asarray(apples_count, float)
    violation = np.asarray(violation, bool)
    T = violation.shape[0]
    any_v = violation.any(1)
    all_pass = ~any_v
    csum = np.concatenate([[0], np.cumsum(any_v)])
    lo = np.maximum(np.arange(T) - window + 1, 0)
    window_pass = (csum[np.arange(T) + 1] - csum[lo]) == 0  # nobody violated in [t - W + 1, t]
    out = dict(steps=int(T), eats=None, violations=int(violation.sum()), all_pass_steps=int(all_pass.sum()))

    def crossings(g, name):
        down = np.zeros(T, bool)
        down[1:] = g[:-1] & ~g[1:]
        out.update({f"{name}_cross": int(down.sum()), f"{name}_cross_allpass": int((down & all_pass).sum()),
                    f"{name}_cross_windowpass": int((down & window_pass).sum())})

    crossings(pollution < depletion, "pollution")
    for tau in taus:
        crossings(apples_count >= tau * apples0, f"tau{tau}")
    out.update(apples0=int(apples0), apples_end=int(apples_count[-1]), apples_mean=float(apples_count.mean()),
               alive_end=int(apples_count[-1] > 0), ever_empty=int((apples_count == 0).any()),
               polluted_share=float((pollution >= depletion).mean()), pollution_mean=float(pollution.mean()))
    return out


def cleanup_from_trace(trace, rules):
    from overseer.socialjax.games import CU_APPLE
    a0 = int((trace["grid0"] == CU_APPLE).sum())
    return cleanup_measures(trace["pollution"], trace["resource"], a0, trace["violation"], rules.lookback)


# ------------------------------------------------------------------------------------------------ pooling
def ratio(rows, num, den):
    n, d = sum(r[num] for r in rows), sum(r[den] for r in rows)
    return n / d if d else float("nan")


def bootstrap(rows, stat, B=2000, seed=0):
    """Point value and 95% percentile interval of stat(rows), resampling episodes."""
    rng = np.random.default_rng(seed)
    point = stat(rows)
    vals = np.array([stat([rows[j] for j in rng.integers(0, len(rows), len(rows))]) for _ in range(B)], float)
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return [point, float("nan"), float("nan")]
    return [point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mean_of(key):
    return lambda rows: float(np.mean([r[key] for r in rows]))


def share(num_keys, den_key):
    num_keys = (num_keys,) if isinstance(num_keys, str) else num_keys
    return lambda rows: (sum(r[k] for r in rows for k in num_keys) / d
                         if (d := sum(r[den_key] for r in rows)) else float("nan"))


def summarise_harvest(rows):
    s = dict(n=len(rows))
    for key in ("eats", "violations", "apples_end", "alive_end", "ever_empty", "ext_cells", "ext_patches",
                "dead_new", "dead_revived", "dead_end"):
        s[key] = bootstrap(rows, mean_of(key))
    s["violation_share_of_eats"] = bootstrap(rows, share("violations", "eats"))
    legal_keys = ("ext_cells_legal_solo", "ext_cells_legal_joint")
    s["lpgf_c_ext"] = bootstrap(rows, share(legal_keys, "ext_cells"))
    s["lpgf_b_ext"] = bootstrap(rows, share("ext_cells_allpass", "ext_cells"))
    s["lpgf_c_dead"] = bootstrap(rows, share(("dead_legal_solo", "dead_legal_joint"), "dead_new"))
    s["lpgf_c_dead_joint"] = bootstrap(rows, share("dead_legal_joint", "dead_new"))
    s["lpgf_b_dead"] = bootstrap(rows, share("dead_new_allpass", "dead_new"))
    s["lpgf_a_dead_steps"] = bootstrap(rows, share("dead_steps_allpass", "dead_steps"))
    s["legal_dead_per_episode"] = bootstrap(rows, lambda rr: float(np.mean(
        [r["dead_legal_solo"] + r["dead_legal_joint"] for r in rr])))
    s["revived_share"] = bootstrap(rows, share("dead_revived", "dead_new"))
    lasting = tuple(f"dead_lasting_{k}" for k in CLASSES)
    for r in rows:
        r["dead_lasting"] = sum(r[k] for k in lasting)
    s["dead_lasting"] = bootstrap(rows, mean_of("dead_lasting"))
    s["lpgf_c_dead_lasting"] = bootstrap(rows, share(("dead_lasting_legal_solo", "dead_lasting_legal_joint"),
                                                     "dead_lasting"))
    s["dead_lasting_counts"] = {k: int(sum(r[f"dead_lasting_{k}"] for r in rows)) for k in CLASSES}
    s["legal_eat_kill_rate"] = bootstrap(rows, lambda rr: ratio(
        [dict(a=r["legal_eats_killing"], b=r["eats"] - r["violations"]) for r in rr], "a", "b"))
    s["violating_eat_kill_rate"] = bootstrap(rows, share("violating_eats_killing", "violations"))
    s["no_eat_events"] = int(sum(r["dead_no_eat"] + (r["ext_cells_no_eat"] > 0) for r in rows))
    s["ext_counts"] = {k: int(sum(r[f"ext_cells_{k}"] for r in rows)) for k in CLASSES}
    s["dead_counts"] = {k: int(sum(r[f"dead_{k}"] for r in rows)) for k in CLASSES}
    for tau in TAUS:
        s[f"tau{tau}_cross"] = bootstrap(rows, mean_of(f"tau{tau}_cross"))
        s[f"tau{tau}_lpgf_a"] = bootstrap(rows, share(f"tau{tau}_cross_allpass", f"tau{tau}_cross"))
        s[f"tau{tau}_lpgf_a_first"] = bootstrap(rows, share(f"tau{tau}_first_allpass", f"tau{tau}_first"))
        s[f"tau{tau}_june_rate"] = bootstrap(rows, share(f"tau{tau}_june_steps", "steps"))
    return s


def summarise_cleanup(rows):
    s = dict(n=len(rows))
    for key in ("violations", "apples_end", "alive_end", "polluted_share", "pollution_cross"):
        s[key] = bootstrap(rows, mean_of(key))
    for name in ("pollution",) + tuple(f"tau{t}" for t in TAUS):
        s[f"{name}_lpgf_a"] = bootstrap(rows, share(f"{name}_cross_allpass", f"{name}_cross"))
        s[f"{name}_lpgf_aW"] = bootstrap(rows, share(f"{name}_cross_windowpass", f"{name}_cross"))
    return s


# ------------------------------------------------------------------------------------------------ runner
RUN_NAME = re.compile(r"^[A-Za-z]+_(?P<chooser>[a-z]+)(?:_k(?P<k>\d+))?(?:_c(?P<c>[\d.]+))?_seed\d+$")


def overseer_settings(rdir):
    """(env name, rules kwargs, DecConfig kwargs) from config.json, else from a stage 1 run name."""
    from overseer.socialjax.decentralised import DecConfig
    from overseer.socialjax.games import make_rules
    cfg_path = Path(rdir) / "config.json"
    fields = DecConfig.__dataclass_fields__
    if cfg_path.exists():
        cfgj = json.loads(cfg_path.read_text())
        ov = {k: v for k, v in cfgj["overseer"].items() if k in fields}
        ov["region_k"] = tuple(ov.get("region_k") or ())
        if ov.get("inspector_table") is not None:
            ov.pop("inspector_table")  # a situation table changes only who is caught, not the game's dynamics
        return cfgj.get("env", "harvest"), cfgj.get("rules", {}), ov
    m = RUN_NAME.match(Path(rdir).name)
    if not m:
        raise ValueError(f"no config.json and cannot parse run name {Path(rdir).name!r}")
    k = int(m["k"] or 0)
    chooser = m["chooser"] if k > 0 else "none"
    unit = make_rules("harvest", num_inner_steps=10).reward_unit
    return "harvest", {}, dict(k=k, chooser=chooser, window=1, penalty=float(m["c"] or 0) * unit)


def play(rules, cfg_kw, act, episodes, seed0, measure):
    import jax

    from overseer.socialjax import locality as L
    from overseer.socialjax.decentralised import DecConfig
    from overseer.socialjax.local_records import LocalRecordOverseer
    ov = LocalRecordOverseer(rules, DecConfig(**cfg_kw), records="handover")  # the central overseer, step for step
    rows = []
    for e in range(episodes):
        tr = L.collect_trace(ov, act(ov), jax.random.PRNGKey(seed0 + e), STEPS, keep_grid=True)
        rows.append(measure(tr))
        del tr
    return rows


def measurer(env, rules):
    if env == "cleanup":
        return lambda tr: cleanup_from_trace(tr, rules)
    spawn = np.zeros(rules.shape, bool)
    sp = np.asarray(rules.env.SPAWNS_APPLE)
    spawn[sp[:, 0], sp[:, 1]] = True
    return lambda tr: harvest_from_trace(tr, spawn)


def run_ippo(rdir, episodes):
    import flax.serialization
    import jax
    import jax.numpy as jnp

    from experiments.socialjax.train_ippo_overseer import socialjax_actor_critic
    from overseer.socialjax.games import make_rules
    env, rules_kw, ov_kw = overseer_settings(rdir)
    rules = make_rules(env, num_inner_steps=STEPS + 1, **rules_kw)
    network = socialjax_actor_critic()(rules.env.action_space().n, activation="relu")
    obs_shape = rules.env.observation_space()[0].shape
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, *obs_shape)))
    params = flax.serialization.from_bytes(params, (Path(rdir) / "params.msgpack").read_bytes())

    def act_fn(key, obs, state, mem):
        pi, _ = network.apply(params, obs)
        return pi.sample(seed=key).astype(jnp.int32), mem

    rows = play(rules, ov_kw, lambda ov: act_fn, episodes, 20_000, measurer(env, rules))
    return env, ov_kw, rows


def run_bots(env, kinds, episodes):
    from overseer.socialjax import locality as L
    from overseer.socialjax.game_bots import kind_codes, make_policy
    from overseer.socialjax.games import make_rules
    rules = make_rules(env, num_inner_steps=STEPS + 1)
    policy = make_policy(rules)
    rows = play(rules, dict(k=0, chooser="none", window=1),
                lambda ov: L.bot_act(policy, kind_codes(kinds), ov.region_grid), episodes, 30_000,
                measurer(env, rules))
    return rows


def fmt(ci, pct=False, nd=1):
    p, lo, hi = ci
    if p != p:  # NaN
        return "n/a"
    f = (lambda x: f"{100 * x:.{nd}f}") if pct else (lambda x: f"{x:.{nd}f}")
    return f"{f(p)} [{f(lo)}, {f(hi)}]" + ("%" if pct else "")


def print_tables(d):
    hv = [(name, r) for name, r in d["policies"].items() if r["env"] == "harvest"]
    print("\n| policy | n | eats | violations | alive at end | patch cells lost | LPGF_c ext | dead cells new | "
          "LPGF_c dead | LPGF_b dead | revived |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in hv:
        s = r["summary"]
        print(f"| {name} | {s['n']} | {fmt(s['eats'])} | {fmt(s['violations'])} | {fmt(s['alive_end'], True, 0)} | "
              f"{fmt(s['ext_cells'])} | {fmt(s['lpgf_c_ext'], True)} | {fmt(s['dead_new'])} | "
              f"{fmt(s['lpgf_c_dead'], True)} | {fmt(s['lpgf_b_dead'], True)} | {fmt(s['revived_share'], True, 0)} |")
    print("\n| policy | " + " | ".join(f"tau {t}: crossings, LPGF_a" for t in TAUS) + " |")
    print("|---|" + "---|" * len(TAUS))
    for name, r in hv:
        s = r["summary"]
        print(f"| {name} | " + " | ".join(f"{fmt(s[f'tau{t}_cross'])}, {fmt(s[f'tau{t}_lpgf_a'], True, 0)}"
                                          for t in TAUS) + " |")
    for name, r in d["policies"].items():
        if r["env"] == "cleanup":
            print(name, json.dumps(r["summary"], default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params-root", default="results/create_pull")
    ap.add_argument("--runs", default="", help="comma-separated run directories under --params-root")
    ap.add_argument("--bots", action="store_true", help="also the scripted reference populations")
    ap.add_argument("--bot-env", default="harvest", choices=("harvest", "cleanup"))
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--out", default="results/h5_lpgf/lpgf.json")
    ap.add_argument("--tables", action="store_true", help="only print the tables from --out")
    a = ap.parse_args()
    out = Path(a.out)
    d = json.loads(out.read_text()) if out.exists() else dict(policies={})
    if a.tables:
        print_tables(d)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [("ippo", r) for r in a.runs.split(",") if r]
    if a.bots:
        jobs += [("bots", p) for p in BOT_POPS]
    for kind, name in jobs:
        label = name if kind == "ippo" else f"bots: {name}" + ("" if a.bot_env == "harvest" else f" ({a.bot_env})")
        if label in d["policies"]:
            continue
        t0 = time.time()
        if kind == "ippo":
            rdir = Path(a.params_root) / name
            if not (rdir / "params.msgpack").exists():
                print(f"skip {name}: no params.msgpack", flush=True)
                continue
            env, ov_kw, rows = run_ippo(rdir, a.episodes)
        else:
            env, ov_kw, rows = a.bot_env, dict(k=0, chooser="none"), run_bots(a.bot_env, BOT_POPS[name], a.episodes)
        summ = summarise_cleanup(rows) if env == "cleanup" else summarise_harvest(rows)
        d["policies"][label] = dict(env=env, overseer={k: v for k, v in ov_kw.items() if k in
                                                       ("k", "chooser", "penalty", "window")},
                                    episodes=rows, summary=summ)
        out.write_text(json.dumps(d, indent=1, default=float))
        print(f"{label}: {len(rows)} episodes, {time.time() - t0:.0f} s", flush=True)
        import jax
        jax.clear_caches()
    print_tables(d)


if __name__ == "__main__":
    main()
