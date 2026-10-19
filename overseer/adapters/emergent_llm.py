"""Adapter for Willis, Zhao, Du and Leibo's Common Pool Resource game (`emergent_llm`, commit be78ff9, MIT).

Study R1a (protocol: notes/claude_audit_20261005/studies/R1a_willis_cpr_audits/protocol.md).
Study R1b (protocol: notes/claude_audit_20261005/studies/R1b_audit_aware_strategies/protocol.md) adds, at the end of
this file, the audit-aware game for strategies that are told the rule (Pool A), its prompt text and its test harness.
R1a's classes behave exactly as before (the evolution now builds games with `desc.game_type()`, which is
AuditedCommonPoolGame for R1a's description, and takes an `audit_seed` that defaults to R1A_BASE).

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

    def __init__(self, config: CulturalEvolutionConfig, registry: StrategyRegistry, run: int = 0,
                 audit_seed: int = R1A_BASE):
        self.run = int(run)
        self.audit_seed = int(audit_seed)   # R1a: R1A_BASE (unchanged); R1b passes R1B_BASE
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
        game_cls = desc.game_type()   # AuditedCommonPoolGame (R1a); AuditAwareCommonPoolGame for R1b's Pool A
        for rep in range(self.config.games_per_agent):
            shuffled = players.copy()
            random.shuffle(shuffled)
            for m in range(per_rep):
                seat = shuffled[m * n:(m + 1) * n]
                game = game_cls(seat, desc, key=(self.run, self.generation, rep * per_rep + m), seed=self.audit_seed)
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


# ================================================================== R1b: audit-aware strategies
# Protocol: notes/claude_audit_20261005/studies/R1b_audit_aware_strategies/protocol.md (Amendment 1 holds the exact
# prompt additions below). Everything above is R1a's and behaves exactly as before (tests/test_r1a.py).
#
# Pool B (blind) strategies play AuditedCommonPoolGame with AuditedCommonPoolDescription: they get their own
# CommonPoolDescription (no audit fields) and their own PlayerHistory, with the extraction as my_payoffs, so they
# cannot see q, F, audits or fines.
# Pool A (audit-aware) strategies play AuditAwareCommonPoolGame with AuditAwareCommonPoolDescription. Same round, same
# stock, same audit draws, same fines; the only differences are what the strategies are given:
#   * game_description = CommonPoolAuditView: their CommonPoolDescription plus audit_prob (q) and fine (F);
#   * history = AuditedPlayerHistory: their PlayerHistory plus my_audited and my_fines per past round, and my_payoffs
#     net of the fine (the prompt's payoff formula subtracts the fine).
import time as _time  # noqa: E402

from emergent_llm.common import Action, COLLECTIVE, PlayerHistory  # noqa: E402
from emergent_llm.generation import prompts as _willis_prompts  # noqa: E402

R1B_BASE = 26_101_450
R1B_MODEL_KEY = "gpt-oss-120b"     # file stem in the strategy registry (make_safe of the model's name)
POOLS = ("A", "B")


@dataclass(frozen=True)
class CommonPoolAuditView(CommonPoolDescription):
    """What a Pool A strategy is given as `game_description`: their CPR description plus q and F."""
    audit_prob: float = 0.0
    fine: float = 0.0


@dataclass
class AuditedPlayerHistory(PlayerHistory):
    """Their PlayerHistory plus this player's own audits and fines, one entry per past round. READ-ONLY arrays."""
    my_audited: np.ndarray
    my_fines: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        self.my_audited.flags.writeable = False
        self.my_fines.flags.writeable = False


@dataclass(frozen=True)
class AuditAwareCommonPoolDescription(AuditedCommonPoolDescription):
    """AuditedCommonPoolDescription whose strategies are told q and F and see their own audits and fines."""

    @classmethod
    def game_type(cls):
        return AuditAwareCommonPoolGame

    def strategy_view(self) -> CommonPoolAuditView:
        return CommonPoolAuditView(n_players=self.n_players, n_rounds=self.n_rounds, capacity=self.capacity,
                                   audit_prob=float(self.audit_prob), fine=float(self.fine))


class AuditAwareCommonPoolGame(AuditedCommonPoolGame):
    """AuditedCommonPoolGame (same stock, extraction, audits and fines) whose players get AuditedPlayerHistory."""

    def player_history(self, i) -> AuditedPlayerHistory:
        base = self.history.for_player(i)
        r = self.history.round_number
        return AuditedPlayerHistory(my_actions=base.my_actions, my_payoffs=base.my_payoffs,
                                    opponent_cooperators=base.opponent_cooperators,
                                    my_audited=self.audited[:r, i].copy(), my_fines=self.fines[:r, i].copy())

    def _play_round(self, players):
        """BaseGame._play_round, line for line, with the audit-aware history."""
        stock = self.get_current_stock()
        action_enums = [player(history=self.player_history(i), current_stock=stock) for i, player in enumerate(players)]
        actions = Action.to_bool_array(action_enums)
        payoffs = self._calculate_payoffs(actions)
        self.history.record(actions, payoffs)

    def _calculate_payoffs(self, actions):
        t = self.history.round_number
        extraction = super()._calculate_payoffs(actions)     # stock, audits and fines exactly as in R1a
        return extraction - self.fines[t]                     # my_payoffs: net of the fine (as the prompt states)


def r1b_description(pool, n, q=0.0, fine=0.0, exempt_final_round=True, n_rounds=20):
    cls = AuditAwareCommonPoolDescription if pool == "A" else AuditedCommonPoolDescription
    return cls.standard(n, q, fine, exempt_final_round, n_rounds)


# ------------------------------------------------------------------ R1b prompts (Willis et al.'s, plus one section)
# Their system prompts are local strings inside generate_strategy_description / generate_strategy_code, so they are
# copied here verbatim; tests/test_r1b.py checks that both still appear in their source unchanged.
DESCRIPTION_SYSTEM_PROMPT = """You are an AI assistant with expertise in strategic thinking.

Output only the strategy description itself — no preamble, no meta-commentary, no conclusion.

Do not:
- Restate the game rules, parameters, or payoff structure
- Discuss why the strategy is good or weigh its tradeoffs
- Add framing like "Here is...", "I'll design...", or "In summary..."
- Hedge with caveats about tournament conditions or opponent uncertainty

Use natural language with optional pseudocode. Be precise about decision rules and edge cases."""
CODE_SYSTEM_PROMPT = "You are an expert Python programmer implementing game theory strategies."

# The one added section (protocol "Setting", Pool A). Inserted right after their CPR game description, in both the
# strategy prompt and the code prompt. Recorded verbatim in the protocol's Amendment 1.
AUDIT_SECTION = """

AUDITS AND FINES:
- audit_prob: audit probability q (float, 0 ≤ q ≤ 1), readable as game_description.audit_prob
- fine: fine F per audited defection (float, F ≥ 0), readable as game_description.fine
- Each round except the last, each player is audited independently with probability q.
- The last round (t = r-1) is never audited.
- An audited player who played D pays the fine F, deducted from its payoff for that round. An audited player who played C pays nothing.
- Fines do not change the stock: the stock falls by what the players consume, exactly as above.
- After each round, each player observes whether it was audited and the fine it paid (its own audits and fines only, in its history).

PAYOFF WITH FINES:
For player i in round t with current stock S:
π_i,t = S/(2n) if player i plays C
π_i,t = S/n - F × a_i,t if player i plays D
where a_i,t = 1 if player i is audited in round t, a_i,t = 0 otherwise.

EXPECTED FINE EXAMPLE (n=4, capacity=100, stock=100, q=0.1, F=20):
- Playing D instead of C consumes 100/4 - 100/8 = 12.5 more this round.
- Expected fine for playing D: q × F = 0.1 × 20 = 2."""

_DEF_OLD = "class CommonPoolDescription:\n    n_players: int\n    n_rounds: int\n    capacity: int\n"
_DEF_NEW = _DEF_OLD + "    audit_prob: float\n    fine: float\n"
_HIST_OLD = "    opponent_cooperators: NDArray[np.int_]  # [round] — count of cooperating opponents\n"
_HIST_NEW = (_HIST_OLD
             + "    my_audited: NDArray[np.bool_]           # [round] — whether this player was audited\n"
             + "    my_fines: NDArray[np.float64]           # [round] — fine this player paid (0.0 if none)\n")
_NOTE_OLD = "# opponent_cooperators[r] is in [0, n_opponents].\n"
_NOTE_NEW = _NOTE_OLD + "# my_payoffs[r] is after the fine: my_fines[r] has already been deducted.\n"


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old[:60]!r} in Willis et al.'s prompt")
    return text.replace(old, new)


def willis_game_text() -> str:
    return _willis_prompts.format_game_description("common_pool")


def pool_a_interface_text() -> str:
    """Their get_interface_description(CommonPoolDescription) with the two description fields and the two history
    fields added (their function hard-codes the PlayerHistory block, so the additions are made here)."""
    text = _willis_prompts.get_interface_description(CommonPoolDescription)
    text = _replace_once(text, _DEF_OLD, _DEF_NEW)
    text = _replace_once(text, _HIST_OLD, _HIST_NEW)
    return _replace_once(text, _NOTE_OLD, _NOTE_NEW)


def strategy_prompt(pool, attitude: Attitude) -> str:
    """Step 1 (natural-language strategy). Pool B: their prompt unchanged. Pool A: plus AUDIT_SECTION."""
    text = _willis_prompts.create_strategy_user_prompt(attitude, "common_pool")
    if pool == "B":
        return text
    game = willis_game_text()
    return _replace_once(text, game, game + AUDIT_SECTION)


def code_prompt(pool, strategy_description: str) -> str:
    """Step 2 (code). Pool B: their prompt unchanged. Pool A: plus AUDIT_SECTION and the Pool A interface."""
    if pool == "B":
        return _willis_prompts.create_code_user_prompt(strategy_description, "common_pool")
    # Built around a placeholder, so text inside the model's description can never be mistaken for the game text.
    text = _willis_prompts.create_code_user_prompt(_PLACEHOLDER, "common_pool")
    game = willis_game_text()
    text = _replace_once(text, game, game + AUDIT_SECTION)
    text = _replace_once(text, _willis_prompts.get_interface_description(CommonPoolDescription),
                         pool_a_interface_text())
    return _replace_once(text, _PLACEHOLDER, strategy_description)


_PLACEHOLDER = "\x00R1B_STRATEGY_DESCRIPTION\x00"


# ------------------------------------------------------------------ R1b test harness for Pool A
def test_aware_strategy_class(strategy_class, allowed_time: float | None = 2.0, sweep=(0.5, 2.0),
                              large=((0.1, 20.0), (0.1, 0.0))) -> float:
    """Willis et al.'s test_strategy_class (generation/test_strategies.py), run in the audit-aware game. The same
    games: an exhaustive sweep over opponent cooperator counts at n = 4, r = 5 (here with q, F = `sweep`, so the
    audited branches run), then their eight n = 256 opponent mixtures, the first four at large[0] and the last four
    at large[1] (F = 0 catches division by the fine). Same time limit as theirs (2 s)."""
    from emergent_llm.generation.test_strategies import make_fixed_opponents
    from emergent_llm.players import (Cooperator, Defector, GradualDefector, LLMPlayer, PeriodicDefector, Random,
                                      RandomCooperator, RandomDefector, SimplePlayer)
    start = _time.time()
    desc = AuditAwareCommonPoolDescription.standard(4, sweep[0], sweep[1], n_rounds=5)
    for _, opponents in make_fixed_opponents(3, 5):
        player = LLMPlayer("test", Gene("dummy", COLLECTIVE), desc.strategy_view(), strategy_class, max_errors=0)
        AuditAwareCommonPoolGame([player] + opponents, desc).play_game()
    m = 256
    mixtures = [
        [SimplePlayer(f"c_{i}", Cooperator) for i in range(m - 1)],
        [SimplePlayer(f"d_{i}", Defector) for i in range(m - 1)],
        [SimplePlayer(f"r_{i}", Random) for i in range(m - 1)],
        [SimplePlayer(f"r_{i}", RandomCooperator) for i in range(m - 1)],
        [SimplePlayer(f"r_{i}", RandomDefector) for i in range(m - 1)],
        [SimplePlayer(f"alternating_{i}", PeriodicDefector(2)) for i in range(m - 1)],
        [SimplePlayer(f"grad_{i}", GradualDefector(10 + i)) for i in range(m - 1)],
        [SimplePlayer(f"period_{i}", PeriodicDefector(2 + i)) for i in range(m - 1)]]
    for k, opps in enumerate(mixtures):
        q, F = large[0] if k < 4 else large[1]
        d = AuditAwareCommonPoolDescription.standard(m, q, F)
        player = LLMPlayer("test", Gene("dummy", COLLECTIVE), d.strategy_view(), strategy_class, max_errors=0)
        AuditAwareCommonPoolGame([player] + opps, d).play_game()
    total = _time.time() - start
    if allowed_time and total > allowed_time:
        raise RuntimeError(f"Strategy took {total:.1f}s")
    return total


test_aware_strategy_class.__test__ = False   # not a pytest test


def load_pool(strategies_dir, pool) -> StrategyRegistry:
    """One R1b pool as their StrategyRegistry: <strategies_dir>/pool_<pool>/common_pool/gpt-oss-120b.py."""
    return StrategyRegistry(strategies_dir=Path(strategies_dir) / f"pool_{pool}", game_name="common_pool",
                            models=[R1B_MODEL_KEY])
