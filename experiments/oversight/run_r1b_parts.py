"""Study R1b, Parts 1 and 2: audit-aware (Pool A) and blind (Pool B) gpt-oss-120b strategies in Willis et al.'s CPR.

Protocol (frozen, with amendments): notes/claude_audit_20261005/studies/R1b_audit_aware_strategies/protocol.md
Strategies: written by run_r1b_generate.py into <root>/strategies/pool_{A,B}/common_pool/gpt-oss-120b.py.
Game, evolution and overseer: R1a's (overseer/adapters/emergent_llm.py): UniformRandom(q) audits, Perfect detection
of D, Fine(F), final round exempt. Pool A plays the audit-aware game (strategies read q, F and their own audits and
fines); Pool B plays R1a's game (strategies see none of it). No LLM calls.

Modes (run from the repo root):
  PYTHONPATH=. python3 -m experiments.oversight.run_r1b_parts part1 [--root DIR] [--processes 4]
  PYTHONPATH=. python3 -m experiments.oversight.run_r1b_parts prediction [--root DIR] [--summary-dir DIR]
  PYTHONPATH=. python3 -m experiments.oversight.run_r1b_parts grid --R 20 [--conditions NAME ...] [--run-start 0]
  PYTHONPATH=. python3 -m experiments.oversight.run_r1b_parts list [--root DIR]

- part1: every strategy of each pool plays 20 games as one of n players, the other n - 1 seats drawn uniformly (with
  replacement) from the same pool, at n = 64 and n = 4, q = 0.1, e = qF in {0, 0.5, 1, 2, 3, 4}. Common random
  numbers: a (strategy, game) has the same opponents, the same seeds for Python's and numpy's generators and the
  same audit draws at every e, so differences across e come from the fine only. One file per (pool, n, e):
  <out>/part1/pool_<P>_n<n>_e<e>.json.gz; existing files are skipped.
- prediction: e_pred(B) = dPi / dD at n = 64 from 2,000 fine-free games among Pool B strategies at the evolution's
  starting mix (seed R1B_BASE + 900), exactly R1a's computation (run_r1a_willis_audits.predict_one). Written with a
  UTC timestamp to <out>/prediction.json (and --summary-dir/prediction.json) before any evolution run; refuses to
  overwrite.
- grid: Willis et al.'s cultural evolution as in R1a (512 agents, G = 4, Fermi beta = 1, mutation 0.25%, 2,000
  generations, final window 100) at n = 64, q = 0.1, e in {0, 0.5, 1.0, 1.25, 2.0} x e_pred(B), each pool on its own
  (genotype = attitude; one model). Run i seeds Python's and numpy's generators with R1B_BASE + i; audits use
  stable_seed(R1B_BASE, "audit", run, generation, game, round). One file per run:
  <out>/runs/<pool>_e<mult>/run_XXX.json.gz; existing runs are skipped. Needs the prediction.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from experiments.oversight import run_r1a_willis_audits as R
from overseer.adapters import emergent_llm as E
from overseer.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "runs" / "claude_r1b_v1"
SUMMARY = ROOT / "notes" / "claude_audit_20261005" / "runs" / "claude_r1b_v1"
R1B_BASE = E.R1B_BASE
PRED_SEED = R1B_BASE + 900
PRED_GAMES = 2000
Q = 0.1
P1_NS = (64, 4)
P1_E = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
P1_GAMES = 20
N_EVO = 64
MULTS = (0.0, 0.5, 1.0, 1.25, 2.0)
MULT_ORDER = (1.25, 0.0, 2.0, 1.0, 0.5)    # H4's level first, then the baseline
GENERATIONS, WINDOW = R.GENERATIONS, R.WINDOW
POP, G, BETA, MU = R.POP, R.G, R.BETA, R.MU

utc_now, save_json, load_json = R.utc_now, R.save_json, R.load_json


# ------------------------------------------------------------------ pools
_REG = {}


def registry(strategies_dir, pool):
    key = (str(strategies_dir), pool)
    if key not in _REG:
        _REG[key] = E.load_pool(strategies_dir, pool)
    return _REG[key]


def spec_key(spec):
    name = spec.strategy_class.__name__
    return (E.base_attitude(spec.gene).value, spec.gene.attitude.name, int(name.rsplit("_", 1)[1]))


def pool_specs(reg):
    """Every strategy of a pool, in a fixed order (base attitude, synonym, index)."""
    return sorted((s for g in reg.available_genes for s in reg.get_all_specs(g)), key=spec_key)


# ------------------------------------------------------------------ Part 1
def part1_path(out, pool, n, e):
    """e = None is Amendment 3's no-audit level (q = 0, F = 0); it is a separate file the H1-H3 code does not read."""
    if e is None:
        return Path(out) / "part1" / f"pool_{pool}_n{n}_none.json.gz"
    return Path(out) / "part1" / f"pool_{pool}_n{n}_e{e:g}.json.gz"


def part1_unit(pool, n, e, strategies_dir, out, games=P1_GAMES, q=Q):
    path = part1_path(out, pool, n, e)
    if path.exists():
        return "exists", 0.0
    logging.disable(logging.WARNING)
    t0 = time.time()
    specs = pool_specs(registry(strategies_dir, pool))
    m = len(specs)
    desc = E.r1b_description(pool, n, 0.0, 0.0) if e is None else E.r1b_description(pool, n, q, e / q)
    view = desc.strategy_view()
    game_cls = desc.game_type()
    r = desc.n_rounds
    shape = (m, games)
    focal_d, focal_f, focal_net, focal_ext, group_d, stock, audits = (np.zeros(shape) for _ in range(7))
    for j, spec in enumerate(specs):
        name = spec.strategy_class.__name__
        for g in range(games):
            key = ("part1", pool, n, name, g)               # no e: the same draws at every e
            rng = np.random.default_rng(stable_seed(R1B_BASE, *key, "opponents"))
            opp = rng.integers(0, m, n - 1)
            players = [spec.create_player("focal", view)] + [specs[o].create_player(f"p{i}", view)
                                                              for i, o in enumerate(opp)]
            E.seed_globals(stable_seed(R1B_BASE, *key, "globals"))
            res = game_cls(players, desc, key=key, seed=R1B_BASE).play_game()
            focal_d[j, g] = res.defections_nonfinal[0] / (r - 1)
            focal_f[j, g] = res.fine_totals[0]
            focal_net[j, g] = res.total_payoffs[0]
            focal_ext[j, g] = res.extraction_totals[0]
            audits[j, g] = res.audits_per_player[0]
            group_d[j, g] = float(np.mean(res.defections_nonfinal)) / (r - 1)
            stock[j, g] = res.final_stock
        E._purge_player_loggers()
    secs = time.time() - t0
    rec = dict(study="R1b", part=1, pool=pool, n=n, q=0.0 if e is None else q, e="none" if e is None else e,
               F=0.0 if e is None else e / q, games=games, n_rounds=r, seed_base=R1B_BASE,
               strategies=[dict(cls=s.strategy_class.__name__, attitude=s.gene.attitude.value,
                                base=E.base_attitude(s.gene).value) for s in specs],
               focal_defect_rate_nonfinal=focal_d.tolist(), focal_fines=focal_f.tolist(),
               focal_net_payoff=focal_net.tolist(), focal_extraction=focal_ext.tolist(),
               focal_audits=audits.tolist(), group_defect_rate_nonfinal=group_d.tolist(),
               final_stock=stock.tolist(), runtime_s=round(secs, 2), finished_utc=utc_now())
    save_json(path, rec, gz=True)
    return "done", secs


def _part1_task(a):
    t0 = time.time()
    try:
        status, secs = part1_unit(*a)
    except Exception as exc:
        return dict(unit=f"{a[0]} n{a[1]} e{a[2]}", status=f"error: {exc!r}", seconds=round(time.time() - t0, 1))
    return dict(unit=f"{a[0]} n{a[1]} e{a[2]}", status=status, seconds=round(secs, 1), utc=utc_now())


def mode_part1(args):
    out = args.out or args.root
    tasks = [(p, n, e, args.strategies or Path(args.root) / "strategies", out, args.games or P1_GAMES)
             for n in args.ns for p in args.pools for e in (*P1_E, None)]  # None: Amendment 3 no-audit level
    res = run_pool(_part1_task, tasks, args.processes, Path(out) / "part1" / "part1_log.jsonl")
    print({s: sum(r["status"] == s for r in res) for s in sorted({r["status"] for r in res})})


# ------------------------------------------------------------------ Part 2: prediction
def prediction_path(root):
    return Path(root) / "prediction.json"


def mode_prediction(args):
    out = Path(args.out or args.root)
    targets = [prediction_path(out)] + ([Path(args.summary_dir) / "prediction.json"] if args.summary_dir else [])
    for p in targets:
        if p.exists() and not args.force:
            raise SystemExit(f"{p} exists; the prediction is computed once, before any evolution run")
    t0 = time.time()
    reg = registry(args.strategies or Path(args.root) / "strategies", "B")
    per = R.predict_one(N_EVO, reg, games=args.pred_games or PRED_GAMES, seed=PRED_SEED)
    per.pop("myopic_F_star", None)      # R1a's q grid; R1b uses q = 0.1 only
    per["myopic_F_star_q0.1"] = 2.0 / Q
    rec = dict(study="R1b", pool="B", computed_utc=None, seed=PRED_SEED, games=per["games"], n=N_EVO,
               definition="e_pred(B) = (mean payoff Selfish - mean payoff Collective) / (mean non-final-round D Selfish"
                          " - mean non-final-round D Collective), per agent per game, fine-free games among Pool B "
                          "strategies at the starting mix (uniform over the 8 attitude genes, then a strategy)",
               strategies_per_gene={str(g): reg.count_strategies(g) for g in sorted(reg.available_genes, key=str)},
               per_n={str(N_EVO): per}, e_pred_B=per["e_pred"])
    rec["runtime_s"] = round(time.time() - t0, 1)
    rec["computed_utc"] = utc_now()
    for p in targets:
        save_json(p, rec)
    print(json.dumps({k: per[k] for k in ("delta_payoff", "delta_D", "e_pred")}), "->", ", ".join(map(str, targets)))


def read_e_pred(root, path=None):
    p = Path(path) if path else prediction_path(root)
    if not p.exists():
        raise SystemExit(f"no prediction at {p}: run `prediction` first (protocol: before any evolution run)")
    rec = load_json(p)
    return float(rec["e_pred_B"]), rec


# ------------------------------------------------------------------ Part 2: conditions and runs
def conditions(e_pred_b):
    """Priority-ordered conditions: name, pool, n, q, mult, e, F, exempt, priority."""
    if not e_pred_b > 0:
        raise SystemExit(f"e_pred(B) = {e_pred_b:.4f} is not positive: the frozen fine grid is undefined; amend first")
    # Amendment 3: a true no-audit baseline per pool (q = 0, so Pool A strategies read audit_prob = 0), run right
    # after the H4 cells
    out = [dict(name=f"{pool}_none", pool=pool, n=N_EVO, q=0.0, mult=0.0, e=0.0, F=0.0, exempt=True, priority=0.5)
           for pool in E.POOLS]
    for prio, m in enumerate(MULT_ORDER):
        for pool in E.POOLS:
            e = m * e_pred_b
            out.append(dict(name=f"{pool}_e{m:g}", pool=pool, n=N_EVO, q=Q, mult=m, e=e, F=e / Q, exempt=True,
                            priority=prio))
    out.sort(key=lambda c: c["priority"])
    return out


def run_path(out, cond, run):
    return Path(out) / "runs" / cond["name"] / f"run_{run:03d}.json.gz"


def run_one(cond, run, strategies_dir, out, generations=GENERATIONS, deadline=None, pred_utc=None):
    path = run_path(out, cond, run)
    if path.exists():
        return "exists", 0.0
    if deadline is not None and time.time() > deadline:
        return "skipped_deadline", 0.0
    logging.disable(logging.WARNING)
    t0 = time.time()
    reg = registry(strategies_dir, cond["pool"])
    desc = E.r1b_description(cond["pool"], cond["n"], cond["q"], cond["F"], cond["exempt"])
    E.seed_globals(R1B_BASE + run)
    ev = E.AuditedCulturalEvolution(E.evolution_config(desc, generations, population_size=POP, beta=BETA,
                                                       mutation_rate=MU, final_window=WINDOW, games_per_agent=G),
                                    reg, run=run, audit_seed=R1B_BASE)
    stats = ev.run_tournament()
    genes = sorted({str(g) for f in ev.gene_frequencies for g in f})
    counts = [[int(round(f.get(g, 0.0) * POP)) for g in genes]
              for f in ({str(k): v for k, v in fr.items()} for fr in ev.gene_frequencies)]
    secs = time.time() - t0
    rec = dict(study="R1b", part=2, condition=cond, run=run, seed=R1B_BASE + run, audit_seed=R1B_BASE,
               generations=generations, population=POP, games_per_agent=G, beta=BETA, mutation_rate=MU,
               window=WINDOW, model=E.R1B_MODEL_KEY, prediction_utc=pred_utc,
               strategies_per_gene={str(g): reg.count_strategies(g) for g in sorted(reg.available_genes, key=str)},
               genes=genes, gene_counts=counts, generation_stats=stats,
               window_summary=R.window_summary(stats, desc, min(WINDOW, generations)),
               runtime_s=round(secs, 2), finished_utc=utc_now(), host_cpus=os.cpu_count())
    save_json(path, rec, gz=True)
    return "done", secs


def _grid_task(a):
    cond, run = a[0], a[1]
    t0 = time.time()
    try:
        status, secs = run_one(*a)
    except Exception as exc:
        return dict(condition=cond["name"], run=run, status=f"error: {exc!r}", seconds=round(time.time() - t0, 1))
    return dict(condition=cond["name"], run=run, status=status, seconds=round(secs, 2), pid=os.getpid(),
                utc=utc_now())


def run_pool(fn, tasks, processes, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    procs = processes or max(1, (os.cpu_count() or 2) - 1)
    done = []
    with open(log_path, "a") as log:
        if procs == 1:
            it = map(fn, tasks)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(procs)   # fresh interpreters: no inherited generator state
            it = pool.imap(fn, tasks, chunksize=1)
        try:
            for r in it:
                log.write(json.dumps(r) + "\n")
                log.flush()
                done.append(r)
                print(r, flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    return done


def selected_conditions(args):
    e_pred_b, rec = read_e_pred(args.root, args.prediction)
    conds = conditions(e_pred_b)
    if args.conditions:
        wanted = set(args.conditions)
        missing = wanted - {c["name"] for c in conds}
        if missing:
            raise SystemExit(f"unknown conditions: {sorted(missing)}")
        conds = [c for c in conds if c["name"] in wanted]
    return conds, rec


def mode_grid(args):
    conds, pred = selected_conditions(args)
    out = Path(args.out or args.root)
    runs = range(args.run_start, args.run_start + args.R)
    deadline = None if args.max_hours is None else time.time() + 3600 * args.max_hours
    strategies = args.strategies or Path(args.root) / "strategies"
    tasks = [(c, r, strategies, out, args.generations or GENERATIONS, deadline, pred.get("computed_utc"))
             for c in conds for r in runs]
    save_json(out / "grid_conditions.json", dict(written_utc=utc_now(), e_pred_B=pred["e_pred_B"],
                                                 prediction_utc=pred.get("computed_utc"),
                                                 conditions=conditions(pred["e_pred_B"])))
    print(f"{len(tasks)} runs over {len(conds)} conditions", flush=True)
    t0 = time.time()
    res = run_pool(_grid_task, tasks, args.processes, out / "grid_log.jsonl")
    print(f"grid wall {time.time() - t0:.0f} s;", {s: sum(r['status'] == s for r in res)
                                                   for s in sorted({r['status'] for r in res})})


def mode_list(args):
    conds, _ = selected_conditions(args)
    for c in conds:
        print(f"{c['priority']} {c['name']:10s} pool={c['pool']} n={c['n']} q={c['q']} mult={c['mult']:<5} "
              f"e={c['e']:.4f} F={c['F']:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["part1", "prediction", "grid", "list"])
    ap.add_argument("--root", type=Path, default=RAW,
                    help="inputs: <root>/strategies and <root>/prediction.json (default %(default)s)")
    ap.add_argument("--strategies", type=Path, help="strategy store (default <root>/strategies)")
    ap.add_argument("--prediction", type=Path, help="prediction file to read (default <root>/prediction.json)")
    ap.add_argument("--out", type=Path, help="where part1/, runs/ and prediction.json are written (default --root)")
    ap.add_argument("--summary-dir", type=Path, help="prediction: also write a copy here (e.g. notes/.../claude_r1b_v1)")
    ap.add_argument("--pools", nargs="*", default=list(E.POOLS))
    ap.add_argument("--ns", nargs="*", type=int, default=list(P1_NS))
    ap.add_argument("--games", type=int, help=f"part1: games per strategy (default {P1_GAMES})")
    ap.add_argument("--pred-games", type=int, help=f"prediction: games (default {PRED_GAMES})")
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--run-start", type=int, default=0)
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--processes", type=int)
    ap.add_argument("--generations", type=int)
    ap.add_argument("--max-hours", type=float)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    {"part1": mode_part1, "prediction": mode_prediction, "grid": mode_grid, "list": mode_list}[args.mode](args)


if __name__ == "__main__":
    main()
