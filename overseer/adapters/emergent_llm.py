"""Adapter for Willis, Zhao, Du and Leibo's Common Pool Resource game (`emergent_llm`, commit be78ff9, MIT).

Study R1a (protocol: notes/claude_audit_20261005/studies/R1a_willis_cpr_audits/protocol.md).

Their code is vendored read-only at external/emergent_llm and is never edited. Everything here subclasses it:

- `AuditedCommonPoolDescription(CommonPoolDescription)`: adds audit_prob q, fine F and the final-round switch. The
  fitness-normalisation bounds (`max_payoff`, `min_payoff`) and the welfare bounds (`max_player_welfare`,
  `min_player_welfare`) are inherited unchanged, as the protocol requires.
- `AuditedCommonPoolGame(CommonPoolGame)`: the same round as theirs (C takes S/(2n), D takes S/n, logistic regrowth at
  rate 2, capped at k). The stock is reduced by the **extraction**, never by the net payoff. After the round, this
  project's `Overseer` audits and fines:
    * audits  `UniformRandom(q, R1A_BASE, prefix=("audit", run, generation, game))`, window = round index, so the
      draw is numpy.random.default_rng(stable_seed(R1A_BASE, "audit", run, generation, game, round)).random(n) and
      agent i (seat i in the game) is audited if u_i < q. The draw does not depend on q, F or n's condition, so the
      agents audited at q = 0.1 are a subset of those audited at q = 0.25 in the same (run, generation, game, round).
      The audit generator is separate from Python's and numpy's global generators, which the strategies and their
      evolution code use, so auditing never shifts their random streams. `CachedUniformRandom` is UniformRandom with
      the round's draw kept (same numbers); a round in which nobody is audited skips the rest of the overseer cycle,
      since nothing can be caught or fined (speed only).
    * detection `Perfect()`. The ledger is in units of the C share: taken = 1 for C and 2 for D, allowance = 1. So an
      audited D is always caught and a C never is, whatever the stock (even at stock 0).
    * sanction `Fine(F)`: a flat fine per catch, subtracted from the payoff only.
    * final round: with `exempt_final_round=True` (primary) the overseer is not called in the last round: no draw, no
      audit, no fine. Draws in earlier rounds are the same in both arms.
- What the strategies see. The history the strategies read (`PlayerHistory.my_payoffs`) holds the **extraction**, not
  the net payoff, so the existing strategies are exactly blind to audits and fines (R1a Amendment 1). Fines are kept
  per player and per round in `self.fines`; `play_game` returns net totals (extraction - fines), which their cultural
  evolution uses as fitness.

`AuditedCulturalEvolution(CulturalEvolution)` runs their Fermi-imitation evolution unchanged except that each game is
an `AuditedCommonPoolGame` keyed by (run, generation, game). It consumes Python's and numpy's global generators in
exactly the order theirs does (shuffle per repetition, games in order, then `_imitate`, `_mutate` and the strategy
refresh, all their own methods), so at F = 0 a run is bit-for-bit their run on the same seed (tests/test_r1a.py).
It keeps compact per-generation statistics instead of their per-generation tournament objects.

Import note. `emergent_llm.generation` imports the LLM client libraries (anthropic, openai, ollama, google-genai) and
`emergent_llm.tournament` imports `pathos`, none of which are installed here and none of which R1a uses (no LLM calls,
no mixture tournaments). `_stub_unused_imports()` puts empty placeholder modules in `sys.modules` for any of them
that cannot be imported, before their package is imported. A placeholder raises if anything is called on it. Their
package also imports matplotlib at import time; if matplotlib is missing (for example on a bare CI image) a permissive
placeholder is used, since nothing here plots with their code. So running R1a needs only numpy and pandas (their
tournament code builds pandas tables) besides this repository and the vendored package.
"""
from __future__ import annotations

import importlib
import logging
import random
import sys
import types
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from overseer.audit import UniformRandom
from overseer.core import Ledger, Overseer, ResourceModel, WindowObs
from overseer.detect import Perfect
from overseer.sanction import Fine

R1A_BASE = 26_101_440
REPO = Path(__file__).resolve().parents[2]
EMERGENT_ROOT = REPO / "external" / "emergent_llm"
STRATEGIES_DIR = EMERGENT_ROOT / "strategies"
MODELS = ("claude-haiku-4-5", "gemini-3.1-flash-lite-preview", "gpt-5.4-mini")  # claude-fable-5 excluded


# ------------------------------------------------------------------ importing their package
class _Placeholder(types.ModuleType):
    """An empty stand-in for a library their package imports but R1a never uses."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        class _Unavailable(Exception):
            def __init__(self, *a, **k):
                raise RuntimeError(f"{self.__class__.__module__}.{name} is a placeholder: not installed, not used by R1a")

        _Unavailable.__name__ = name
        _Unavailable.__module__ = self.__name__
        setattr(self, name, _Unavailable)
        return _Unavailable


class _Anything:
    """Accepts any attribute access, call, item or update; used only if matplotlib is missing (see below)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __call__(self, *a, **k):
        return self

    def __getitem__(self, k):
        return self

    def __setitem__(self, k, v):
        pass

    def __iter__(self):
        return iter(())


class _LoosePlaceholder(types.ModuleType):
    """matplotlib stand-in. Their package imports matplotlib at import time (common/plotting.py sets rcParams;
    tournament/results.py imports pyplot and ticker), but R1a's runner never plots with it."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything()


STUBBED: list[str] = []


def _stub_unused_imports():
    groups = {"anthropic": ["anthropic"], "openai": ["openai"], "ollama": ["ollama"],
              "google.genai": ["google", "google.genai", "google.genai.errors"],
              "pathos": ["pathos", "pathos.multiprocessing"]}
    for probe, names in groups.items():
        try:
            importlib.import_module(probe)
            continue
        except Exception:
            pass
        for name in names:
            if name not in sys.modules or isinstance(sys.modules[name], _Placeholder):
                sys.modules[name] = _Placeholder(name)
                STUBBED.append(name)
        for name in names:
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(sys.modules[parent], child, sys.modules[name])
    try:
        importlib.import_module("matplotlib.pyplot")
    except Exception:
        for name in ("matplotlib", "matplotlib.pyplot", "matplotlib.ticker"):
            sys.modules[name] = _LoosePlaceholder(name)
            STUBBED.append(name)
        sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
        sys.modules["matplotlib"].ticker = sys.modules["matplotlib.ticker"]


_stub_unused_imports()
if str(EMERGENT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EMERGENT_ROOT / "src"))

from emergent_llm.common import Attitude, Gene  # noqa: E402
from emergent_llm.games.base_game import GameResult  # noqa: E402
from emergent_llm.games.common_pool import CommonPoolDescription, CommonPoolGame  # noqa: E402
from emergent_llm.generation.strategy_registry import StrategyRegistry  # noqa: E402
from emergent_llm.tournament.configs import CulturalEvolutionConfig  # noqa: E402
from emergent_llm.tournament.cultural_evolution import CulturalEvolution  # noqa: E402


def load_registry(models=MODELS) -> StrategyRegistry:
    """Their committed CPR pool for the paper's three models (3 x 8 attitudes x 128 = 3,072 strategies)."""
    return StrategyRegistry(strategies_dir=STRATEGIES_DIR, game_name="common_pool", models=list(models))


def base_attitude(gene: Gene) -> Attitude:
    return gene.attitude.to_base_attitude()


# ------------------------------------------------------------------ the game
@dataclass(frozen=True)
class AuditedCommonPoolDescription(CommonPoolDescription):
    """Their CPR description plus the overseer's settings. Bounds are inherited unchanged."""
    audit_prob: float = 0.0
    fine: float = 0.0
    exempt_final_round: bool = True

    @classmethod
    def game_type(cls):
        return AuditedCommonPoolGame

    def strategy_view(self) -> CommonPoolDescription:
        """The description the strategies are given: their own class, with no audit fields."""
        return CommonPoolDescription(n_players=self.n_players, n_rounds=self.n_rounds, capacity=self.capacity)

    @classmethod
    def standard(cls, n, q=0.0, fine=0.0, exempt_final_round=True, n_rounds=20):
        """Their `common_pool_default` (capacity 4n, 20 rounds) with an overseer."""
        return cls(n_players=n, n_rounds=n_rounds, capacity=4 * n, audit_prob=float(q), fine=float(fine),
                   exempt_final_round=bool(exempt_final_round))


class CachedUniformRandom(UniformRandom):
    """overseer.audit.UniformRandom with the round's draw kept, so asking twice in one round builds one generator.
    Same numbers as UniformRandom (default_rng(stable_seed(seed, *prefix, window, *suffix)).random(n) < q)."""

    def __init__(self, q, seed, prefix=(), suffix=()):
        super().__init__(q, seed, prefix, suffix)
        self._window, self._audited = None, None

    def draw(self, window, n):
        if window != self._window:
            self._window, self._audited = window, self.rng(window).random(n) < self.q   # = UniformRandom.choose
        return self._audited.copy()

    def choose(self, obs, ledger, overseer):
        return self.draw(obs.window, overseer.n_agents)


class CprResource(ResourceModel):
    """The CPR as the overseer sees it, in units of the C share S/(2n). The allowance is the C take (1 share)."""

    def __init__(self, n):
        self.n = n

    def safe_total(self, state):
        return float(self.n)          # everyone takes the C share: the MSY harvest, stock stays at k

    def harm(self, state, takes):
        return bool(np.asarray(takes, float).sum() > self.n)

    def first_best_per_window(self, state=None):
        return float(self.n)


@dataclass
class AuditedGameResult(GameResult):
    extraction_totals: list = field(default_factory=list)   # per player, summed over rounds
    fine_totals: list = field(default_factory=list)         # per player
    audits_per_player: list = field(default_factory=list)
    caught_per_player: list = field(default_factory=list)
    defections_nonfinal: list = field(default_factory=list)  # per player, rounds 1..r-1
    defections_final: list = field(default_factory=list)     # per player, round r
    final_stock: float = 0.0


class AuditedCommonPoolGame(CommonPoolGame):
    """Their CPR round, then audits and fines from `overseer`. `key` = (run, generation, game) for the audit seed."""

    def __init__(self, players, description: AuditedCommonPoolDescription, key=(0, 0, 0), seed=R1A_BASE):
        super().__init__(players, description)
        self.key = tuple(key)
        self.seed = int(seed)
        self._new_overseer()

    def _new_overseer(self):
        d = self.description
        n, r = d.n_players, d.n_rounds
        q = float(getattr(d, "audit_prob", 0.0))
        F = float(getattr(d, "fine", 0.0))
        self.active = q > 0.0
        self.overseer = Overseer(
            n_agents=n, resource=CprResource(n),
            audit=CachedUniformRandom(q, self.seed, prefix=("audit", *self.key)) if self.active else None,
            detector=Perfect(), sanctions=(Fine(F),))
        self.extraction = np.zeros((r, n))
        self.fines = np.zeros((r, n))
        self.audited = np.zeros((r, n), bool)
        self.caught = np.zeros((r, n), bool)
        self.stock_path = np.zeros(r + 1)
        self.stock_path[0] = self.stock

    def _calculate_payoffs(self, actions):
        t = self.history.round_number
        n = self.description.n_players
        stock_before = self.stock
        # --- their round, line for line (common_pool.py), except that the stock falls by the extraction
        share = 0.5 * self.stock / n
        extraction = share + share * (~actions).astype(np.float64)
        self.stock -= np.sum(extraction)
        self.stock = max(self.stock, 0)
        self.stock += 2 * self.stock * (1 - self.stock / self.description.capacity)
        self.stock = min(self.stock, self.description.capacity)
        # --- the overseer (payoffs only; the stock above is already final for this round)
        self.extraction[t] = extraction
        self.stock_path[t + 1] = self.stock
        last = t == self.description.n_rounds - 1
        if self.active and not (last and self.description.exempt_final_round):
            # Speed only: if the (deterministic) draw audits nobody this round, nothing can be caught or fined, so the
            # overseer's full cycle is skipped. When someone is audited, end_window reuses the same draw.
            if self.overseer.audit.draw(t, n).any():
                obs = WindowObs(window=t, state=stock_before)
                ledger = Ledger(taken=1.0 + (~actions).astype(np.float64), allowance=np.ones(n))
                outcome = self.overseer.end_window(obs, ledger)
                self.audited[t] = outcome.audited
                self.caught[t] = outcome.caught
                self.fines[t] = outcome.sanctions.fines
        return extraction            # recorded in the history: the strategies see what they took, not the fines

    def reset(self):
        super().reset()
        self._new_overseer()

    def play_game(self) -> AuditedGameResult:
        res = super().play_game()
        r = self.description.n_rounds
        D = ~self.history.actions[:r]
        extraction = self.extraction.sum(axis=0)
        fines = self.fines.sum(axis=0)
        net = (extraction - fines).tolist()
        return AuditedGameResult(
            player_names=res.player_names, total_payoffs=net, total_cooperations=res.total_cooperations,
            cooperations_by_round=res.cooperations_by_round, history=res.history, description=res.description,
            extraction_totals=extraction.tolist(), fine_totals=fines.tolist(),
            audits_per_player=self.audited.sum(axis=0).tolist(), caught_per_player=self.caught.sum(axis=0).tolist(),
            defections_nonfinal=D[:r - 1].sum(axis=0).tolist(), defections_final=D[r - 1].astype(int).tolist(),
            final_stock=float(self.stock))


# ------------------------------------------------------------------ cultural evolution
def seed_globals(seed: int):
    """Seed Python's and numpy's global generators (their code and the strategies use both)."""
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)


def _purge_player_loggers():
    """LLMPlayer makes one named logger per player object; drop them so 2,000 generations do not hold ~10^5 loggers.
    No random draw is involved."""
    d = logging.Logger.manager.loggerDict
    for name in [k for k in d if k.startswith("LLMPlayer(")]:
        del d[name]


class AuditedCulturalEvolution(CulturalEvolution):
    """Their cultural evolution with audited games. `config.game_description` is an AuditedCommonPoolDescription.

    Generation g plays games indexed 0..G*pop/n - 1 in their order (repetition-major), keyed (run, g, game).
    `generation_stats` keeps one dict per generation; `gene_frequencies` is their list (fine-grained genes).
    """

    def __init__(self, config: CulturalEvolutionConfig, registry: StrategyRegistry, run: int = 0):
        self.run = int(run)
        self.generation_stats: list[dict] = []
        super().__init__(config, registry)

    def reset(self):
        super().reset()
        self.generation_stats = []

    def _play_generation(self, players):
        """FairTournament._run_repetition, repeated G times, with audited games; same random draws as theirs."""
        desc = self.config.game_description
        n = desc.n_players
        per_rep = len(players) // n
        results = []  # (players in seat order, AuditedGameResult)
        for rep in range(self.config.games_per_agent):
            shuffled = players.copy()
            random.shuffle(shuffled)
            for m in range(per_rep):
                seat = shuffled[m * n:(m + 1) * n]
                game = AuditedCommonPoolGame(seat, desc, key=(self.run, self.generation, rep * per_rep + m))
                results.append((seat, game.play_game()))
        return results

    def _run_generation(self):
        desc = self.config.game_description
        view = desc.strategy_view() if hasattr(desc, "strategy_view") else desc
        players = [spec.create_player(f"player_{i}", view) for i, spec in enumerate(self.population)]
        index = {id(p): i for i, p in enumerate(players)}
        games = self._play_generation(players)

        N = len(players)
        net = np.zeros(N)
        ext = np.zeros(N)
        fin = np.zeros(N)
        dn = np.zeros(N)
        coop = np.zeros(N)
        n_games = len(games)
        game_fines = np.zeros(n_games)
        game_audits = np.zeros(n_games)
        game_caught = np.zeros(n_games)
        game_stock = np.zeros(n_games)
        for k, (seat, res) in enumerate(games):
            ids = [index[id(p)] for p in seat]
            net[ids] += res.total_payoffs
            ext[ids] += res.extraction_totals
            fin[ids] += res.fine_totals
            dn[ids] += res.defections_nonfinal
            coop[ids] += res.total_cooperations
            game_fines[k] = sum(res.fine_totals)
            game_audits[k] = sum(res.audits_per_player)
            game_caught[k] = sum(res.caught_per_player)
            game_stock[k] = res.final_stock

        G = self.config.games_per_agent
        genes = [spec.gene for spec in self.population]
        is_c = np.array([base_attitude(g) == Attitude.COLLECTIVE for g in genes])
        model_of = np.array([g.model for g in genes])
        rec = dict(
            generation=self.generation,
            collective_frequency=float(is_c.mean()),
            welfare_transfer=float(ext.mean() / G),       # mean extraction per player per game
            welfare_lost=float(net.mean() / G),           # mean net payoff per player per game
            payoff_collective=float(net[is_c].mean() / G) if is_c.any() else None,
            payoff_selfish=float(net[~is_c].mean() / G) if (~is_c).any() else None,
            fines_per_game=float(game_fines.mean()),
            audits_per_game=float(game_audits.mean()),
            caught_per_game=float(game_caught.mean()),
            defections_nonfinal_per_agent_game=float(dn.mean() / G),
            cooperation_rate=float(coop.mean() / (G * desc.n_rounds)),
            final_stock_mean=float(game_stock.mean()),
            model_shares={m: float((model_of == m).mean()) for m in sorted(set(model_of))},
            fines_by_base={"collective": float(fin[is_c].sum()), "selfish": float(fin[~is_c].sum())},
        )
        self.generation_stats.append(rec)

        # --- from here on, exactly their _run_generation (cultural_evolution.py L161-188)
        fitnesses = net
        pi_max = desc.max_payoff() * G
        pi_min = desc.min_payoff() * G
        fitnesses_normalised = (fitnesses - pi_min) / (pi_max - pi_min)
        new_genes, imitated = self._imitate(fitnesses_normalised)
        new_genes, mutated = self._mutate(new_genes)
        changed = imitated | mutated
        self.population = [
            self.registry.sample_spec(new_genes[i]) if changed[i] else spec
            for i, spec in enumerate(self.population)
        ]
        self._last_fitnesses = fitnesses
        del players, games
        _purge_player_loggers()

    def run_tournament(self):
        """Their loop (frequencies before each generation, then one more after the last), without their summary."""
        for _ in range(self.config.n_generations):
            self.gene_frequencies.append(self._calculate_gene_frequencies())
            self._run_generation()
            self.generation += 1
        self.gene_frequencies.append(self._calculate_gene_frequencies())
        return self.generation_stats


def evolution_config(desc, n_generations, population_size=512, beta=1.0, mutation_rate=0.0025, final_window=100,
                     games_per_agent=4) -> CulturalEvolutionConfig:
    return CulturalEvolutionConfig(game_description=desc, population_size=population_size, beta=beta,
                                   mutation_rate=mutation_rate, n_generations=n_generations,
                                   final_window=min(final_window, n_generations), games_per_agent=games_per_agent)


def gene_counts(freqs: dict, population_size: int) -> dict:
    """Fine-grained gene frequencies -> {"model[attitude]": count}."""
    return {str(g): int(round(f * population_size)) for g, f in freqs.items()}


def collapse_counts(counts: dict) -> Counter:
    """{"model[sub-attitude]": count} -> {"model[base attitude]": count}."""
    out = Counter()
    for key, c in counts.items():
        model, att = key[:-1].split("[")
        out[f"{model}[{Attitude(att).to_base_attitude().value}]"] += c
    return out
