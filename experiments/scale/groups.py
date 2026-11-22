"""M overseer groups in Willis et al.'s Common Pool Resource game (written 22 Nov 2026; notes/socialjax/locality/).

The game and the strategies are Willis, Zhao, Du & Leibo's (2026), through overseer/adapters/emergent_llm.py: N
players, 20 rounds, capacity 4N; C takes S/(2N), D takes S/N; logistic regrowth. Strategies are the LLM-written
Python strategies of their committed pool (no LLM calls).

**Why audits can be applied after the game.** The strategies read only their own extraction, never audits or fines
(R1a Amendment 1), and fines never change the stock (AuditedCommonPoolGame). So a game's defection matrix D
(rounds x N) is the same whatever the overseer does, and every overseer design below is applied to the same D.
This is exact for these fixed strategies; it says nothing about strategies that react to audits.

Overseers. Each round t = 0 .. T - 2 (the final round is exempt, as in R1a) the overseers audit K agents in total.
An audited D is caught (perfect detection).
- central: one overseer, K audits among all N;
- groups: the N agents are split into M fixed groups of N/M (randomly, or by the strategies' base attitude, so that
  groups differ in how often they defect); overseer m audits k_m among its own group only. k_m = K/M, with the
  remainder handed to random groups each round (as DecConfig does). Agents never change group: chi = 0;
- pooled groups: before each round each overseer sends one number to a coordinator (how many agents in its group it
  has caught before) and gets back its budget (one number): the K audits are split in proportion to those counts
  when they add up to K or more, else each group gets its count and the rest is split equally.
Choosers: random; history-first (agents caught before, most catches first, then the longest unaudited, ties random:
the same order as the SocialJax "history" chooser without decay).
Communication between overseers per round: central collects N records (w = 1); independent groups send none;
pooled groups send M counts up and M budgets down, 2M numbers.
"""
from __future__ import annotations

import random

import numpy as np

import overseer.adapters.emergent_llm as E


def play_game(registry, n, seed, genes=None):
    """One game of n players drawn uniformly over the pool's genes. Returns (D (rounds, n) bool, base attitudes
    (n,) str, stock path)."""
    E.seed_globals(seed)
    genes = genes or sorted(registry.available_genes, key=str)
    desc = E.AuditedCommonPoolDescription.standard(n, q=0.0)
    view = desc.strategy_view()
    specs = [registry.sample_spec(random.choice(genes)) for _ in range(n)]
    players = [s.create_player(f"p{i}", view) for i, s in enumerate(specs)]
    game = E.AuditedCommonPoolGame(players, desc, key=(seed, 0, 0))
    game.play_game()
    D = ~np.asarray(game.history.actions[:desc.n_rounds], bool)
    att = np.array([E.base_attitude(s.gene).value for s in specs])
    return D, att, game.stock_path.copy()


def assign_groups(n, M, rng, attitudes=None):
    """(n,) group ids, N/M per group: a random split, or (attitudes given) agents sorted by attitude then split."""
    if attitudes is None:
        order = rng.permutation(n)
    else:
        order = np.lexsort((rng.random(n), attitudes))
    g = np.empty(n, int)
    g[order] = np.repeat(np.arange(M), n // M)
    return g


def budgets(K, M, rng, weights=None):
    """Per-group audits this round: K/M each with the remainder to random groups, or split by `weights`."""
    if weights is None:
        base = np.full(M, K // M)
        base[rng.permutation(M)[:K % M]] += 1
        return base
    w = np.asarray(weights, float)
    if w.sum() >= K:
        raw = K * w / w.sum()
        b = np.floor(raw).astype(int)
        rest = K - b.sum()
        order = np.argsort(-(raw - b) + 1e-9 * rng.random(M))
        b[order[:rest]] += 1
        return b
    b = w.astype(int)
    return b + budgets(K - b.sum(), M, rng)


def audit(D, groups, K, chooser, rng, pooled=False):
    """Apply one overseer design to a defection matrix D (rounds, n). Returns counts and the caught share."""
    T, n = D.shape
    M = int(groups.max()) + 1
    members = [np.flatnonzero(groups == m) for m in range(M)]
    catches = np.zeros(n)
    since = np.zeros(n)
    caught = defections = audits = 0
    for t in range(T - 1):  # final round exempt
        if chooser == "random":
            score = rng.random(n)
        else:
            score = catches + 1e-3 * since + 1e-6 * rng.random(n)
        if M == 1:
            b = np.array([K])
        elif pooled:
            b = budgets(K, M, rng, [(catches[idx] > 0).sum() for idx in members])
        else:
            b = budgets(K, M, rng)
        chosen = np.zeros(n, bool)
        for m, idx in enumerate(members):
            k = min(int(b[m]), len(idx))
            if k:
                chosen[idx[np.argsort(-score[idx])[:k]]] = True
        hit = chosen & D[t]
        caught += int(hit.sum())
        defections += int(D[t].sum())
        audits += int(chosen.sum())
        catches += hit
        since = np.where(chosen, 0, since + 1)
    return dict(caught=caught, defections=defections, audits=audits,
                caught_share=caught / defections if defections else float("nan"))


def comm_share(n, M, pooled):
    """Numbers sent between overseers per round, as a share of the N records a central overseer collects."""
    if M == 1:
        return 1.0
    return 2 * M / n if pooled else 0.0


# ------------------------------------------------------------------ version (b): each overseer for its own group
def cpr_welfare(n, defect_mask, rounds=20):
    """Per-player extraction over a game where players in defect_mask always play D and the rest always C
    (Willis's round: C takes S/(2n), D takes S/n; stock -= extraction; S += 2S(1 - S/k); capped at k = 4n)."""
    S, k = 4.0 * n, 4.0 * n
    total = np.zeros(n)
    for _ in range(rounds):
        share = 0.5 * S / n
        ext = share * (1 + defect_mask)
        total += ext
        S = max(S - ext.sum(), 0.0)
        S = min(S + 2 * S * (1 - S / k), k)
    return total


def own_group_view(n, M, rounds=20):
    """What deterring one group is worth to the whole commons and to that group, when every other group complies.
    Returns dict: total welfare gain, the deterred group's own gain (negative = it gains by defecting), and the
    share of the harm of its defection that lands on itself."""
    size = n // M
    g = np.zeros(n, bool)
    g[:size] = True
    all_c = cpr_welfare(n, np.zeros(n, bool), rounds)
    g_d = cpr_welfare(n, g, rounds)
    total_gain = all_c.sum() - g_d.sum()        # what deterring group 0 is worth to everyone
    own_gain = all_c[g].sum() - g_d[g].sum()     # ... to group 0 itself
    harm_others = all_c[~g].sum() - g_d[~g].sum()
    return dict(n=n, M=M, group_size=size, total_gain=float(total_gain), own_gain=float(own_gain),
                harm_to_others=float(harm_others))
