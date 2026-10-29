"""Study R1a: do random audits and fines stop Selfish strategies taking over in Willis et al.'s commons?

Protocol (frozen, with amendments): notes/claude_audit_20261005/studies/R1a_willis_cpr_audits/protocol.md
Game, evolution and strategies: Willis, Zhao, Du and Leibo's `emergent_llm` at external/emergent_llm (commit be78ff9,
unmodified). The overseer: overseer/ (UniformRandom audits, Perfect detection of D, Fine), through
overseer/adapters/emergent_llm.py. No LLM calls: the strategies are their committed, pre-written Python classes.

Modes (run from the repo root):
  PYTHONPATH=.:external/emergent_llm/src python3 -m experiments.oversight.run_r1a_willis_audits prediction
  PYTHONPATH=.:external/emergent_llm/src python3 -m experiments.oversight.run_r1a_willis_audits pilot
  PYTHONPATH=.:external/emergent_llm/src python3 -m experiments.oversight.run_r1a_willis_audits timing
  PYTHONPATH=.:external/emergent_llm/src python3 -m experiments.oversight.run_r1a_willis_audits grid --R 20 \
      [--conditions NAME ...] [--processes 7] [--max-hours 5.5]
  PYTHONPATH=.:external/emergent_llm/src python3 -m experiments.oversight.run_r1a_willis_audits list

- prediction: e_pred = dPi / dD per n from 2,000 fine-free games at the starting mix (seed R1A_BASE + 900); written
  with a UTC timestamp to notes/.../runs/claude_r1a_v1/prediction.json (and a copy in results/runs/claude_r1a_v1/)
  before any fined run. It refuses to overwrite an existing prediction.
- pilot: 2 runs x 200 generations, n = 64, e = 0 (protocol "Unit, runs, intervals"); timing only, not analysed.
- timing: short probes (50 generations) of the slower condition types, to cost the grid; not analysed.
- grid: the conditions (protocol "Conditions", minus anything dropped by amendment), R runs each, in priority order
  (H0 baselines, then H1/H2 at q = 0.1 for n = 64 and n = 4, then q = 0.25, then the secondary arm). Each run is saved
  as soon as it finishes; finished runs are skipped on a restart. --max-hours stops *starting* runs after that wall time.

Seeds: run i seeds Python's and numpy's generators with R1A_BASE + i (so run i starts from the same population in every
condition); audits use stable_seed(R1A_BASE, "audit", run, generation, game, round), game = repetition * (512 / n) + match.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from overseer.adapters import emergent_llm as E

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "runs" / "claude_r1a_v1"
SUMMARY = ROOT / "notes" / "claude_audit_20261005" / "runs" / "claude_r1a_v1"
PREDICTION = SUMMARY / "prediction.json"

R1A_BASE = E.R1A_BASE
PRED_SEED = R1A_BASE + 900
PRED_GAMES = 2000
NS = (64, 4)
QS = (0.1, 0.25)
MULTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SECONDARY_MULTS = (1.25, 2.0)
GENERATIONS = 2000
WINDOW = 100
POP, G, BETA, MU = 512, 4, 1.0, 0.0025
# Willis rundown s4.5 (my re-implementation of their rule, others all-C, one deviation mid-game)
FORWARD_GAIN = {4: 1.30, 64: 1.97}

# Conditions dropped by the protocol's R rule (Amendment 3, after the pilot). --include-dropped runs them anyway.
DROPPED_Q: tuple = (0.25,)            # Amendment 3
DROPPED_MULTS: tuple = (0.75, 1.5)    # Amendment 3


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path: Path, obj, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if gz:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"))
    else:
        tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def load_json(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text())


# ------------------------------------------------------------------ prediction
def predict_one(n, registry, games=PRED_GAMES, seed=PRED_SEED):
    """Fine-free games at the evolution's starting mix: every seat draws a (model, attitude) gene uniformly from the 24
    (so the six base genotypes are uniform) and then one of its 128 strategies, as `sample_spec` does."""
    E.seed_globals(seed)
    desc = E.AuditedCommonPoolDescription.standard(n)
    view = desc.strategy_view()
    genes = sorted(registry.available_genes, key=str)
    pay, dn, sel, game_id = [], [], [], []
    for k in range(games):
        specs = [registry.sample_spec(random.choice(genes)) for _ in range(n)]
        players = [s.create_player(f"p{i}", view) for i, s in enumerate(specs)]
        res = E.AuditedCommonPoolGame(players, desc, key=("prediction", n, k)).play_game()
        pay += res.total_payoffs
        dn += res.defections_nonfinal
        sel += [E.base_attitude(s.gene) == E.Attitude.SELFISH for s in specs]
        game_id += [k] * n
        E._purge_player_loggers()
    pay, dn, sel, game_id = map(np.asarray, (pay, dn, sel, game_id))
    d_pi = float(pay[sel].mean() - pay[~sel].mean())
    d_d = float(dn[sel].mean() - dn[~sel].mean())
    # descriptive interval: bootstrap over games (not in the protocol)
    rng = np.random.default_rng(26101441)
    by_game = np.arange(games)
    boot = []
    idx_of = [np.flatnonzero(game_id == g) for g in by_game]
    for _ in range(1000):
        pick = np.concatenate([idx_of[g] for g in rng.integers(0, games, games)])
        p, d, s = pay[pick], dn[pick], sel[pick]
        boot.append((p[s].mean() - p[~s].mean()) / (d[s].mean() - d[~s].mean()))
    return dict(
        n=n, games=games, agents=int(len(pay)), selfish_agents=int(sel.sum()),
        mean_payoff_selfish=float(pay[sel].mean()), mean_payoff_collective=float(pay[~sel].mean()),
        mean_nonfinal_D_selfish=float(dn[sel].mean()), mean_nonfinal_D_collective=float(dn[~sel].mean()),
        delta_payoff=d_pi, delta_D=d_d, e_pred=d_pi / d_d,
        e_pred_boot95_descriptive=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        welfare_efficiency_at_start_mix=float((pay.mean() - desc.min_player_welfare())
                                              / (desc.max_player_welfare() - desc.min_player_welfare())),
        myopic_gain_full_stock=2.0, myopic_F_star={str(q): 2.0 / q for q in QS},
        forward_looking_gain=FORWARD_GAIN[n],
    )


def mode_prediction(args):
    for p in (PREDICTION, RAW / "prediction.json"):
        if p.exists() and not args.force:
            raise SystemExit(f"{p} exists; the prediction is computed once, before any fined run")
    t0 = time.time()
    registry = E.load_registry()
    out = dict(study="R1a", computed_utc=None, seed=PRED_SEED, games_per_n=PRED_GAMES,
               definition="e_pred = (mean payoff Selfish - mean payoff Collective) / (mean non-final-round D Selfish - "
                          "mean non-final-round D Collective), per agent per game, fine-free games at the starting mix",
               models=list(E.MODELS), per_n={})
    for n in NS:
        out["per_n"][str(n)] = predict_one(n, registry)
        print(n, {k: out["per_n"][str(n)][k] for k in ("delta_payoff", "delta_D", "e_pred")}, flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)
    out["computed_utc"] = utc_now()
    save_json(PREDICTION, out)
    save_json(RAW / "prediction.json", out)
    print("wrote", PREDICTION, out["computed_utc"])


def read_prediction():
    if not PREDICTION.exists():
        raise SystemExit(f"no prediction at {PREDICTION}: run `prediction` first")
    return load_json(PREDICTION)


# ------------------------------------------------------------------ conditions
def scale_for(n, e_pred):
    """The e scale used at group size n. Amendment 2: e_pred(4) < 0, so n = 4 uses e_pred(64)."""
    return e_pred[n] if e_pred[n] > 0 else e_pred[64]


def conditions(e_pred: dict, include_dropped=False):
    """Priority-ordered conditions. e_pred: {n: value}. Each: name, n, q, mult, e, F, exempt, arm, priority."""
    out = []

    def add(n, q, m, exempt, arm, prio):
        e = m * scale_for(n, e_pred) if m else 0.0
        name = f"n{n}_base" if m == 0 else f"n{n}_q{q}_e{m}" + ("" if exempt else "_noexempt")
        out.append(dict(name=name, n=n, q=q if m else 0.0, mult=m, e=e, F=(e / q if m else 0.0), exempt=exempt,
                        arm=arm, priority=prio))

    for n in NS:
        add(n, 0.0, 0.0, True, "baseline", 1)
    order = (1.25, 0.5, 0.75, 1.0, 1.5, 2.0)   # H2's level first
    for q in QS:
        for n in NS:
            for m in order:
                dropped = q in DROPPED_Q or m in DROPPED_MULTS
                if dropped and not include_dropped:
                    continue
                add(n, q, m, True, "primary" + ("_dropped" if dropped else ""), 2 if q == 0.1 else 3)
    for m in SECONDARY_MULTS:
        add(64, 0.1, m, False, "secondary", 4)
    out.sort(key=lambda c: c["priority"])
    return out


def condition_by_name(name, e_pred):
    for c in conditions(e_pred, include_dropped=True):
        if c["name"] == name:
            return c
    raise KeyError(name)


# ------------------------------------------------------------------ one evolution run
_REGISTRY = None


def _registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = E.load_registry()
    return _REGISTRY


def window_summary(stats, desc, window=WINDOW):
    w = stats[-window:]
    lo, hi = desc.min_player_welfare(), desc.max_player_welfare()
    mean = lambda k: float(np.mean([s[k] for s in w]))
    wt, wl = mean("welfare_transfer"), mean("welfare_lost")
    models = sorted(w[0]["model_shares"])
    switch = next((s["generation"] for s in stats if s["collective_frequency"] > 0.5), None)
    return dict(
        collective_frequency=mean("collective_frequency"),
        welfare_transfer=wt, welfare_lost=wl,
        efficiency_transfer=(wt - lo) / (hi - lo), efficiency_lost=(wl - lo) / (hi - lo),
        fines_per_game=mean("fines_per_game"), audits_per_game=mean("audits_per_game"),
        caught_per_game=mean("caught_per_game"), cooperation_rate=mean("cooperation_rate"),
        defections_nonfinal_per_agent_game=mean("defections_nonfinal_per_agent_game"),
        final_stock_mean=mean("final_stock_mean"),
        model_shares={m: float(np.mean([s["model_shares"].get(m, 0.0) for s in w])) for m in models},
        first_generation_collective_above_half=switch,
    )


def run_one(cond: dict, run: int, generations=GENERATIONS, out_dir: Path | None = None, deadline=None):
    """One evolution run; returns (status, seconds). Saves results/runs/claude_r1a_v1/runs/<cond>/run_XXX.json.gz."""
    out_dir = Path(out_dir or RAW / "runs")
    path = out_dir / cond["name"] / f"run_{run:03d}.json.gz"
    if path.exists():
        return "exists", 0.0
    if deadline is not None and time.time() > deadline:
        return "skipped_deadline", 0.0
    logging.disable(logging.WARNING)
    t0 = time.time()
    registry = _registry()
    desc = E.AuditedCommonPoolDescription.standard(cond["n"], cond["q"], cond["F"], cond["exempt"])
    E.seed_globals(R1A_BASE + run)
    ev = E.AuditedCulturalEvolution(E.evolution_config(desc, generations, population_size=POP, beta=BETA,
                                                       mutation_rate=MU, final_window=WINDOW, games_per_agent=G),
                                    registry, run=run)
    stats = ev.run_tournament()
    genes = sorted({str(g) for f in ev.gene_frequencies for g in f})
    counts = [[int(round(f.get(g, 0.0) * POP)) for g in genes]
              for f in ({str(k): v for k, v in fr.items()} for fr in ev.gene_frequencies)]
    secs = time.time() - t0
    rec = dict(study="R1a", condition=cond, run=run, seed=R1A_BASE + run, generations=generations,
               population=POP, games_per_agent=G, beta=BETA, mutation_rate=MU, window=WINDOW,
               models=list(E.MODELS), genes=genes, gene_counts=counts, generation_stats=stats,
               window_summary=window_summary(stats, desc, min(WINDOW, generations)),
               runtime_s=round(secs, 2), finished_utc=utc_now(), host_cpus=os.cpu_count())
    save_json(path, rec, gz=True)
    return "done", secs


def _task(args):
    cond, run, generations, out_dir, deadline = args
    t0 = time.time()
    try:
        status, secs = run_one(cond, run, generations, out_dir, deadline)
    except Exception as exc:  # report, keep the pool alive
        return dict(condition=cond["name"], run=run, status=f"error: {exc!r}", seconds=time.time() - t0)
    return dict(condition=cond["name"], run=run, status=status, seconds=round(secs, 2), pid=os.getpid(),
                utc=utc_now())


def run_pool(tasks, processes, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")   # fresh interpreters: no inherited generator state
    done = []
    with ctx.Pool(processes, maxtasksperchild=None) as pool, open(log_path, "a") as log:
        for r in pool.imap(_task, tasks, chunksize=1):
            log.write(json.dumps(r) + "\n")
            log.flush()
            done.append(r)
            print(r, flush=True)
    return done


# ------------------------------------------------------------------ modes
def mode_pilot(args):
    cond = dict(name="pilot_n64_base", n=64, q=0.0, mult=0.0, e=0.0, F=0.0, exempt=True, arm="pilot", priority=0)
    out = RAW / "pilot"
    t0 = time.time()
    res = run_pool([(cond, 9000 + i, 200, out, None) for i in range(2)], 2, out / "pilot_log.jsonl")
    wall = time.time() - t0
    per_run = [r["seconds"] for r in res]
    rec = dict(computed_utc=utc_now(), runs=2, generations=200, n=64, e=0, processes=2, wall_s=round(wall, 1),
               per_run_s=per_run, per_generation_s=float(np.mean(per_run)) / 200, host_cpus=os.cpu_count(),
               note="pilot runs use run indices 9000-9001 (seeds R1A_BASE + 9000..9001); not analysed")
    save_json(out / "pilot_timing.json", rec)
    print(json.dumps(rec, indent=2))


def mode_timing(args):
    """Short probes of each condition type (50 generations), run in parallel at the grid's process count."""
    pred = read_prediction()
    e_pred = {int(k): v["e_pred"] for k, v in pred["per_n"].items()}
    probes = ["n4_base", "n4_q0.1_e1.25", "n4_q0.25_e1.25", "n64_q0.1_e1.25", "n64_q0.25_e1.25", "n64_q0.1_e2.0"]
    out = RAW / "timing_probe"
    gens = args.generations or 50
    tasks = [(condition_by_name(p, e_pred), 9100 + i, gens, out, None) for p in probes for i in range(1)]
    procs = args.processes or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    res = run_pool(tasks, procs, out / "timing_log.jsonl")
    rec = dict(computed_utc=utc_now(), generations=gens, processes=procs, wall_s=round(time.time() - t0, 1),
               per_generation_s={r["condition"]: r["seconds"] / gens for r in res},
               note="timing probes, run indices 9100+, not analysed")
    save_json(out / "timing_probe.json", rec)
    print(json.dumps(rec, indent=2))


def mode_grid(args):
    pred = read_prediction()
    e_pred = {int(k): v["e_pred"] for k, v in pred["per_n"].items()}
    conds = conditions(e_pred, include_dropped=args.include_dropped)
    if args.conditions:
        wanted = set(args.conditions)
        conds = [c for c in conditions(e_pred, include_dropped=True) if c["name"] in wanted]
        missing = wanted - {c["name"] for c in conds}
        if missing:
            raise SystemExit(f"unknown conditions: {sorted(missing)}")
    runs = range(args.run_start, args.run_start + args.R)
    deadline = None if args.max_hours is None else time.time() + 3600 * args.max_hours
    tasks = [(c, r, args.generations or GENERATIONS, RAW / "runs", deadline) for c in conds for r in runs]
    procs = args.processes or max(1, (os.cpu_count() or 2) - 1)
    save_json(RAW / "grid_conditions.json", dict(written_utc=utc_now(), e_pred=e_pred, R=args.R,
                                                 conditions=conditions(e_pred, include_dropped=True)))
    print(f"{len(tasks)} runs over {len(conds)} conditions, {procs} processes, deadline {args.max_hours} h", flush=True)
    t0 = time.time()
    res = run_pool(tasks, procs, RAW / "grid_log.jsonl")
    print(f"grid wall {time.time() - t0:.0f} s;", {s: sum(r['status'] == s for r in res)
                                                   for s in sorted({r['status'] for r in res})})


def mode_list(args):
    pred = read_prediction()
    e_pred = {int(k): v["e_pred"] for k, v in pred["per_n"].items()}
    for c in conditions(e_pred, include_dropped=True):
        print(f"{c['priority']} {c['arm']:16s} {c['name']:22s} n={c['n']:3d} q={c['q']:<5} e={c['e']:.4f} "
              f"F={c['F']:.4f} exempt={c['exempt']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prediction", "pilot", "timing", "grid", "list"])
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--run-start", type=int, default=0)
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--include-dropped", action="store_true", help="also run conditions dropped by Amendment 2")
    ap.add_argument("--processes", type=int)
    ap.add_argument("--generations", type=int)
    ap.add_argument("--max-hours", type=float)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    {"prediction": mode_prediction, "pilot": mode_pilot, "timing": mode_timing, "grid": mode_grid,
     "list": mode_list}[args.mode](args)


if __name__ == "__main__":
    main()
