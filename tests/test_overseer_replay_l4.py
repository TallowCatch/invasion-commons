"""Admission test A for the overseer (overseer/DESIGN.md, ADMISSION.md): exact replay of stored LLM Fishery games.

For every stored episode, the logged requests and catches are fed to an overseer built from the package parts and set
up like that episode's cell: SharedCut (MSY target), UniformRandom(q) with the runner's audit seeds, a Perfect detector
with L2's 0.06 t tolerance, and Fine(F), Memory(rule) or nothing. The stock is advanced by the adapter from the logged
catches. Allowances, scale, audited, caught, fines, payoffs and the stock path must equal the log.

The main set is L4 (results/runs/l2_results_branch/claude_l4_v1, the l2-results branch extract). L2, L2's EM re-run,
L3 and L5 are replayed as well: they use the same game loop, and L2's EM cells are the only stored games with memory.
Each set is skipped if its folder is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.oversight import run_l2_llm_agents as l2
from experiments.oversight import run_l4_llm_identification as l4
from experiments.oversight.claude_oversight_common import fishery_setup
from fishery_sim import llm_actor as L
from overseer import Ledger, Overseer
from overseer.adapters.fishery_text import FisheryText
from overseer.allowance import SharedCut
from overseer.audit import UniformRandom
from overseer.detect import Perfect
from overseer.sanction import Fine, Memory

ROOT = Path(__file__).resolve().parents[1] / "results" / "runs" / "l2_results_branch"
# (folder, seeds the runner used). L3 ran through L2's runner with --cells; L5 through L4's runner.
SETS = {"L4": ("claude_l4_v1", l4.SEEDS), "L2": ("claude_l2_v1", l2.SEEDS), "L2_EM": ("claude_l2_em_v2", l2.SEEDS),
        "L3": ("claude_l3_v1", l2.SEEDS), "L5": ("claude_l5_v1", l4.SEEDS)}
TOL = l2.OVER_T / L.MAX_CATCH
ATOL = 1e-12


def build(e, seeds):
    """The overseer for one stored episode's cell, and the game adapter."""
    cfg, _, _ = fishery_setup(e["context"], seeds["pop"], l2.HORIZON)
    game = FisheryText(cfg, l2.TARGET)
    q = e.get("q", l2.Q)  # L2/L3 episodes predate the q field; they all used L2's Q
    if e["consequence"] == "fine":
        sanctions = (Fine(e["fine"]),)
    elif e["consequence"] == "memory":
        # claude_l2_v1's EM games used the 'share' rule (the field was added with Amendment 5's 'tonnes' rule)
        sanctions = (Memory(e.get("memory_rule") or "share", cfg.n_agents),)
    else:
        sanctions = ()
    ov = Overseer(cfg.n_agents, game.resource, allowance=SharedCut(),
                  audit=UniformRandom(q, seeds["audit"], prefix=(e["context"],)),
                  detector=Perfect(tol=TOL), sanctions=sanctions)
    return ov, game


def replay(e, seeds):
    """Returns (problems, bit_exact): a list of mismatch descriptions (empty = match), and whether every float
    compared was bit-for-bit equal."""
    ov, game = build(e, seeds)
    state, pay, problems, exact = game.initial_state(), np.zeros(ov.n_agents), [], True

    def same(name, t, got, want):
        nonlocal exact
        got, want = np.asarray(got, float), np.asarray(want, float)
        exact &= bool(np.array_equal(got, want))
        if not np.allclose(got, want, rtol=0, atol=ATOL):
            problems.append(f"t={t} {name}: overseer {got.tolist()} log {want.tolist()}")

    for s in e["steps"]:
        t = s["t"]
        same("stock", t, state.stock, s["stock"])
        obs = game.observe(t, state, s["requests"])
        d = ov.begin_window(obs)
        same("scale", t, d.scale, s["scale"])
        same("allowance", t, d.caps, s["allowance"])
        out = ov.end_window(obs, Ledger(np.asarray(s["taken"], float), d.caps, obs.requests))
        for name in ("audited", "caught"):
            if getattr(out, name).tolist() != s[name]:
                problems.append(f"t={t} {name}: overseer {getattr(out, name).tolist()} log {s[name]}")
        same("msy_break", t, float(game.resource.harm(state, s["taken"])), float(s["msy_break"]))
        state, payoffs = game.step(state, np.asarray(s["taken"], float))
        pay += payoffs
    same("fines", "end", ov.fines, e["fines"])
    same("payoff", "end", pay, e["payoff"])
    same("final_stock", "end", state.stock, e["final_stock"])
    if bool(state.collapsed) != e["collapsed"]:
        problems.append(f"collapsed: overseer {state.collapsed} log {e['collapsed']}")
    return problems, exact


def episodes(folder):
    return sorted((ROOT / folder).glob("*/episodes/*.json"))


def run_set(key):
    folder, seeds = SETS[key]
    files = episodes(folder)
    if not files:
        pytest.skip(f"{ROOT / folder} absent (extract the l2-results branch)")
    bad, exact = {}, 0
    for f in files:
        problems, bit = replay(json.loads(f.read_text()), seeds)
        exact += bit
        if problems:
            bad[str(f.relative_to(ROOT))] = problems[:3]
    return len(files), exact, bad


@pytest.mark.parametrize("key", list(SETS))
def test_replay_matches_log(key):
    n, exact, bad = run_set(key)
    print(f"{key}: {n - len(bad)}/{n} episodes match ({exact} bit-for-bit)")
    assert not bad, json.dumps(bad, indent=1)


def test_l4_has_every_cell_type():
    """The L4 set covers fines (several F and q) and the no-audit cell; L2's EM sets cover both memory rules."""
    files = episodes(SETS["L4"][0])
    if not files:
        pytest.skip("L4 data absent")
    kinds = {(json.loads(f.read_text())["consequence"]) for f in files}
    assert kinds == {"fine", "none"}


if __name__ == "__main__":  # PYTHONPATH=. python3 tests/test_overseer_replay_l4.py
    for k in SETS:
        n, exact, bad = run_set(k)
        print(f"{k}: {n - len(bad)}/{n} match, {exact} bit-for-bit")
        for f, p in list(bad.items())[:5]:
            print("  ", f, p)
