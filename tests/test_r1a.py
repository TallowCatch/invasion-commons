"""Tests for R1a: random audits and fines in Willis et al.'s CPR (overseer/adapters/emergent_llm.py).

Protocol: notes/claude_audit_20261005/studies/R1a_willis_cpr_audits/protocol.md
Run: PYTHONPATH=.:external/emergent_llm/src python3 -m pytest -q tests/test_r1a.py
"""
import random

import numpy as np
import pytest

emergent = pytest.importorskip("overseer.adapters.emergent_llm")

from emergent_llm.common import Action, Attitude  # noqa: E402
from emergent_llm.games.common_pool import CommonPoolDescription, CommonPoolGame  # noqa: E402
from emergent_llm.players import SimplePlayer  # noqa: E402
from emergent_llm.tournament.cultural_evolution import CulturalEvolution  # noqa: E402

from overseer.seeding import stable_seed  # noqa: E402

E = emergent
ADesc = E.AuditedCommonPoolDescription
AGame = E.AuditedCommonPoolGame


@pytest.fixture(scope="module")
def registry():
    return E.load_registry()


def _pattern_players(n, rule):
    """Programmed players: rule(i, round) -> True for C."""
    return [SimplePlayer(f"p{i}", (lambda h, i=i: Action.C if rule(i, h.round_number) else Action.D))
            for i in range(n)]


def _mixed(i, t):
    return (i + t) % 3 != 0


def _llm_players(registry, n, seed, view):
    E.seed_globals(seed)
    genes = sorted(registry.available_genes, key=str)
    return [registry.sample_spec(random.choice(genes)).create_player(f"p{i}", view) for i in range(n)]


# ------------------------------------------------------------------ the pool
def test_pool_is_the_papers_three_models(registry):
    assert registry.available_models == set(E.MODELS)
    assert "claude-fable-5" not in registry.available_models
    assert len(registry.available_genes) == 24
    assert all(registry.count_strategies(g) == 128 for g in registry.available_genes)


def test_bounds_unchanged():
    for n in (4, 64):
        a, b = ADesc.standard(n, q=0.25, fine=9.0), CommonPoolDescription(n_players=n, n_rounds=20, capacity=4 * n)
        assert a.max_payoff() == b.max_payoff() and a.min_payoff() == b.min_payoff()
        assert a.max_player_welfare() == b.max_player_welfare()
        assert a.min_player_welfare() == b.min_player_welfare()
        assert a.strategy_view() == b and type(a.strategy_view()) is CommonPoolDescription


# ------------------------------------------------------------------ fines never change the stock
def test_stock_follows_extraction_by_hand():
    n = 4
    d = ADesc.standard(n, q=1.0, fine=50.0, exempt_final_round=False)
    g = AGame(_pattern_players(n, _mixed), d, key=(0, 0, 0))
    g.play_game()
    S, k = 4.0 * n, 4.0 * n
    for t in range(20):
        acts = np.array([_mixed(i, t) for i in range(n)])
        share = 0.5 * S / n
        ext = share + share * (~acts)
        assert np.array_equal(g.extraction[t], ext)
        S = max(S - ext.sum(), 0)
        S = min(S + 2 * S * (1 - S / k), k)
        assert g.stock_path[t + 1] == S
    assert g.fines.sum() > 0


@pytest.mark.parametrize("n", [4, 64])
def test_fines_never_change_stock_or_actions(registry, n):
    view = CommonPoolDescription(n_players=n, n_rounds=20, capacity=4 * n)
    out = []
    for F in (0.0, 7.5, 1e6):
        players = _llm_players(registry, n, 11, view)
        g = AGame(players, ADesc.standard(n, q=0.5, fine=F), key=(3, 1, 2))
        E.seed_globals(99)
        res = g.play_game()
        out.append((g.stock_path.copy(), g.history.actions.copy(), g.extraction.copy(), g.audited.copy(), res))
    for stock, acts, ext, aud, res in out[1:]:
        assert np.array_equal(stock, out[0][0])
        assert np.array_equal(acts, out[0][1])
        assert np.array_equal(ext, out[0][2])
        assert np.array_equal(aud, out[0][3])
        assert res.extraction_totals == out[0][4].extraction_totals


def test_strategies_see_extraction_not_fines():
    n = 4
    g = AGame(_pattern_players(n, lambda i, t: False), ADesc.standard(n, q=1.0, fine=3.0), key=(0, 0, 0))
    g.play_game()
    assert np.array_equal(g.history.payoffs, g.extraction)
    assert (g.fines[:-1] == 3.0).all()


# ------------------------------------------------------------------ final-round exemption
def test_final_round_exemption():
    n = 4
    res = {}
    for exempt in (True, False):
        g = AGame(_pattern_players(n, lambda i, t: False), ADesc.standard(n, q=1.0, fine=2.0, exempt_final_round=exempt),
                  key=(1, 2, 3))
        r = g.play_game()
        res[exempt] = (g, r)
    g, r = res[True]
    assert not g.audited[-1].any() and (g.fines[-1] == 0).all()
    assert g.audited[:-1].all() and (g.fines[:-1] == 2.0).all()
    assert r.fine_totals == [2.0 * 19] * n
    g2, r2 = res[False]
    assert g2.audited.all() and r2.fine_totals == [2.0 * 20] * n
    assert np.array_equal(g.audited[:-1], g2.audited[:-1])


def test_final_round_exemption_with_random_audits():
    n = 64
    a = AGame(_pattern_players(n, _mixed), ADesc.standard(n, q=0.25, fine=1.0, exempt_final_round=True), key=(5, 6, 7))
    b = AGame(_pattern_players(n, _mixed), ADesc.standard(n, q=0.25, fine=1.0, exempt_final_round=False), key=(5, 6, 7))
    a.play_game(), b.play_game()
    assert np.array_equal(a.audited[:-1], b.audited[:-1]) and np.array_equal(a.fines[:-1], b.fines[:-1])
    assert not a.audited[-1].any() and b.audited[-1].any()


# ------------------------------------------------------------------ audit determinism and nesting
def test_audits_match_protocol_seed_key():
    n, q = 64, 0.25
    key = (7, 123, 4)
    g = AGame(_pattern_players(n, _mixed), ADesc.standard(n, q=q, fine=1.0, exempt_final_round=False), key=key)
    g.play_game()
    for t in range(20):
        u = np.random.default_rng(stable_seed(E.R1A_BASE, "audit", *key, t)).random(n)
        assert np.array_equal(g.audited[t], u < q)


def test_audits_deterministic_and_nested():
    n = 64
    aud = {}
    for q in (0.1, 0.25):
        runs = []
        for _ in range(2):
            g = AGame(_pattern_players(n, _mixed), ADesc.standard(n, q=q, fine=1.0), key=(2, 9, 1))
            g.play_game()
            runs.append(g.audited.copy())
        assert np.array_equal(runs[0], runs[1])
        aud[q] = runs[0]
    assert (aud[0.1] <= aud[0.25]).all() and aud[0.1].sum() < aud[0.25].sum()
    other = AGame(_pattern_players(n, _mixed), ADesc.standard(n, q=0.25, fine=1.0), key=(2, 10, 1))
    other.play_game()
    assert not np.array_equal(other.audited, aud[0.25])


def test_audits_do_not_touch_global_generators():
    n = 4
    E.seed_globals(5)
    g = AGame(_pattern_players(n, _mixed), ADesc.standard(n, q=0.5, fine=1.0), key=(0, 0, 0))
    g.play_game()
    a = (random.random(), np.random.random())
    E.seed_globals(5)
    a0 = (random.random(), np.random.random())
    assert a == a0


# ------------------------------------------------------------------ e = 0 is their game
@pytest.mark.parametrize("n,seed", [(4, 1), (4, 2), (64, 3)])
def test_zero_fine_game_identical_to_theirs(registry, n, seed):
    view = CommonPoolDescription(n_players=n, n_rounds=20, capacity=4 * n)
    E.seed_globals(seed)
    theirs = CommonPoolGame(_llm_players(registry, n, seed, view), view)
    E.seed_globals(seed + 1000)
    t_res = theirs.play_game()
    for q in (0.0, 0.25):
        ours = AGame(_llm_players(registry, n, seed, view), ADesc.standard(n, q=q, fine=0.0), key=(0, 0, 0))
        E.seed_globals(seed + 1000)
        o_res = ours.play_game()
        assert np.array_equal(theirs.history.actions, ours.history.actions)
        assert np.array_equal(theirs.history.payoffs, ours.history.payoffs)
        assert t_res.total_payoffs == o_res.total_payoffs
        assert theirs.stock == ours.stock


@pytest.mark.parametrize("n,pop,gens,q", [(4, 32, 6, 0.25), (64, 128, 3, 0.1)])
def test_zero_fine_evolution_identical_to_theirs(registry, n, pop, gens, q):
    base = CommonPoolDescription(n_players=n, n_rounds=20, capacity=4 * n)
    E.seed_globals(77)
    theirs = CulturalEvolution(E.evolution_config(base, gens, population_size=pop), registry)
    t = theirs.run_tournament()
    E.seed_globals(77)
    ours = E.AuditedCulturalEvolution(E.evolution_config(ADesc.standard(n, q=q, fine=0.0), gens, population_size=pop),
                                      registry, run=0)
    stats = ours.run_tournament()
    assert theirs.gene_frequencies == ours.gene_frequencies
    assert [str(s) for s in theirs.population] == [str(s) for s in ours.population]
    for g, rec in enumerate(stats):
        their_mean = t.generation_results[g].summary.results_df["mean_payoff"].mean()
        assert rec["welfare_lost"] == pytest.approx(their_mean, rel=0, abs=1e-12)
        assert rec["fines_per_game"] == 0.0


def test_evolution_streams_end_in_same_state(registry):
    n, pop, gens = 4, 16, 3
    base = CommonPoolDescription(n_players=n, n_rounds=20, capacity=4 * n)
    E.seed_globals(8)
    CulturalEvolution(E.evolution_config(base, gens, population_size=pop), registry).run_tournament()
    after_theirs = (random.random(), float(np.random.random()))
    E.seed_globals(8)
    E.AuditedCulturalEvolution(E.evolution_config(ADesc.standard(n, q=0.5, fine=0.0), gens, population_size=pop),
                               registry).run_tournament()
    assert (random.random(), float(np.random.random())) == after_theirs


# ------------------------------------------------------------------ payoff bookkeeping
def test_payoff_bookkeeping_game():
    n, F = 64, 4.0
    g = AGame(_pattern_players(n, _mixed), ADesc.standard(n, q=0.25, fine=F), key=(1, 1, 1))
    r = g.play_game()
    D = ~g.history.actions
    assert not (g.caught & ~D).any()                     # a C is never caught
    assert np.array_equal(g.caught, g.audited & D)       # every audited D is caught
    assert np.array_equal(g.fines, F * g.caught)
    assert np.allclose(np.array(r.total_payoffs), np.array(r.extraction_totals) - np.array(r.fine_totals),
                       rtol=0, atol=1e-12)
    assert r.fine_totals == (F * g.caught.sum(axis=0)).tolist()
    assert r.defections_nonfinal == D[:-1].sum(axis=0).tolist()
    assert r.audits_per_player == g.audited.sum(axis=0).tolist()
    assert g.overseer.fines.tolist() == r.fine_totals    # the overseer's own per-player record agrees


def test_payoff_bookkeeping_generation(registry):
    n, pop = 4, 32
    E.seed_globals(4)
    ev = E.AuditedCulturalEvolution(E.evolution_config(ADesc.standard(n, q=0.5, fine=3.0), 2, population_size=pop),
                                    registry, run=1)
    stats = ev.run_tournament()
    G = 4
    for rec in stats:
        n_games = G * pop // n
        fines_per_player_game = rec["fines_per_game"] * n_games / (pop * G)
        assert rec["welfare_transfer"] - rec["welfare_lost"] == pytest.approx(fines_per_player_game, abs=1e-9)
        assert rec["fines_per_game"] == pytest.approx(3.0 * rec["caught_per_game"], abs=1e-9)
        fb = rec["fines_by_base"]
        assert fb["collective"] + fb["selfish"] == pytest.approx(rec["fines_per_game"] * n_games, abs=1e-9)
        assert sum(rec["model_shares"].values()) == pytest.approx(1.0)


# ------------------------------------------------------------------ their imitation uses fitness differences only
def test_imitation_depends_on_differences_only(registry):
    n, pop = 4, 32
    E.seed_globals(3)
    ev = CulturalEvolution(E.evolution_config(CommonPoolDescription(n_players=n, n_rounds=20, capacity=16), 1,
                                              population_size=pop), registry)
    fit = np.linspace(-0.3, 1.2, pop)
    np.random.seed(10)
    a = ev._imitate(fit)
    np.random.seed(10)
    b = ev._imitate(fit + 5.0)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


# ------------------------------------------------------------------ the grid runner's job list (for CI lanes)
def _grid_args(**kw):
    import argparse
    base = dict(runs="0-19", conditions=None, n=None, q=None, e_mult=None, exempt="any", include_dropped=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_job_list_fixed_and_shards_partition_it():
    from experiments.oversight import run_r1a_willis_audits as R
    e_pred = {64: 0.9619609053802177, 4: -0.5859053635835585}
    jobs = R.job_list(e_pred, _grid_args())
    assert len(jobs) == 12 * 20
    assert [c["name"] for c, _ in jobs[:2]] == ["n64_base", "n64_base"]
    # n = 4 uses the n = 64 scale (Amendment 2), so no fine is negative
    assert all(c["F"] >= 0 for c, _ in jobs)
    # the dropped conditions come after the analysed grid, so including them leaves earlier indices unchanged
    more = R.job_list(e_pred, _grid_args(include_dropped=True))
    assert [(c["name"], r) for c, r in more[:240]] == [(c["name"], r) for c, r in jobs]
    N = 7
    shards = [[i for i in range(len(jobs)) if i % N == k] for k in range(N)]
    assert sorted(i for s in shards for i in s) == list(range(len(jobs)))
    sub = R.job_list(e_pred, _grid_args(n=[64], q=[0.1], e_mult=[1.25], exempt="yes", runs="0-24"))
    assert [(c["name"], r) for c, r in sub] == [("n64_q0.1_e1.25", r) for r in range(25)]
    assert R.parse_runs("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]


def test_existing_result_is_never_overwritten(tmp_path):
    from experiments.oversight import run_r1a_willis_audits as R
    cond = dict(name="n4_base", n=4, q=0.0, mult=0.0, e=0.0, F=0.0, exempt=True, arm="baseline", priority=1)
    p = tmp_path / "n4_base" / "run_000.json.gz"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"sentinel")
    assert R.run_one(cond, 0, generations=2, out_dir=tmp_path) == ("exists", 0.0)
    assert p.read_bytes() == b"sentinel"
    assert R.run_one(cond, 1, generations=2, out_dir=tmp_path, deadline=0.0) == ("skipped_deadline", 0.0)
    status, _ = R.run_one(cond, 2, generations=2, out_dir=tmp_path)
    assert status == "done" and (tmp_path / "n4_base" / "run_002.json.gz").exists()
