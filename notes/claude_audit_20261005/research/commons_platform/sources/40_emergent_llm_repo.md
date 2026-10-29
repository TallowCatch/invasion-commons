# [40] emergent_llm: code for "Evaluating Collective Behaviour of Hundreds of LLM Agents"

- **Title:** emergent_llm (GitHub repository)
- **Author:** Richard Willis (`Copyright (c) 2026 Richard Willis`, LICENSE)
- **Year:** 2026
- **Identifier:** https://github.com/willis-richard/emergent_llm, commit `be78ff97959cf11c631fdeb4a1708a2eeb3eb747` (2 Oct 2026, "Move tests into test folder, split runs.sh into components")
- **Licence:** MIT
- **Source type:** source code (cloned `--depth 1` to the scratchpad; read only, nothing installed or executed)
- **Local clone:** `/private/tmp/claude-501/-Users-ameerfiras-invasion-commons-invasion-commons/cfe06c49-9d3e-4d81-b792-bdc11cc119da/scratchpad/repos/emergent_llm`
- **Date accessed:** 14 Oct 2026
- **GitHub description:** "Code for 'Evaluating Collective Behaviour of Hundreds of LLM Agents' (AAMAS SE 2026)"

All paths below are relative to the repo root. Line numbers refer to the commit above.

## 1. Actions are binary

`src/emergent_llm/common/actions.py` L8–14
```python
class Action(Enum):
    """Core actions in social dilemma games.
    There are only two possible actions, namely Cooperate or Defect,
    which are called C and D for convenience.
    """
    D = 0  # Defect - represented as False/0 in boolean arrays
    C = 1  # Cooperate - represented as True/1 in boolean arrays
```

## 2. What a strategy sees (anonymous, aggregate history)

`src/emergent_llm/common/history.py` L10–33
```python
@dataclass
class PlayerHistory:
    """
    Anonymous, aggregated history from one player's perspective.

    Opponents are anonymous: only the *count* of cooperating opponents is
    visible per round, not which specific opponents cooperated.

    On round 0, all arrays have length 0 (not None).
    """
    my_actions: NDArray[np.bool_]           # [round] — this player's actions
    my_payoffs: NDArray[np.float64]         # [round] — this player's payoffs
    opponent_cooperators: NDArray[np.int_]  # [round] — count of cooperating opponents
```
`GameHistory.record(actions_row, payoffs_row)` (L58–67) stores only actions and payoffs; `for_player` (L69–81) computes `opponent_cooperators = (self._round_coop_counts[:r] - my_actions.astype(np.int_))`.

`src/emergent_llm/players/base_player.py` L14–25 (strategy interface)
```python
    @abstractmethod
    def __call__(self, history: PlayerHistory, current_stock=None) -> Action:
        """
        ...
        For common_pool, current_stock is a float and strategies should be
        written as `def __call__(self, history, current_stock) -> Action`.
        """
```

`src/emergent_llm/common/game_description.py` L44–53: `has_stock()` returns False by default; "Games that override this to return True must also supply a float via BaseGame.get_current_stock(), and their strategies receive it as a second argument to __call__."

## 3. Round loop

`src/emergent_llm/games/base_game.py` L50–61
```python
    def _play_round(self, players: Sequence[BasePlayer]) -> None:
        """Play a single round, recording into self.history."""
        stock = self.get_current_stock()

        action_enums = [
            player(history=self.history.for_player(i), current_stock=stock)
            for i, player in enumerate(players)
        ]

        actions = Action.to_bool_array(action_enums)
        payoffs = self._calculate_payoffs(actions)
        self.history.record(actions, payoffs)
```

## 4. Common Pool Resource game

`src/emergent_llm/games/common_pool.py` L116–146
```python
class CommonPoolGame(BaseGame):
    """
    Common pool resource with growth rate of 2 * x * (1 - x / capacity).
    - Defect consumes stock/n
    - Cooperate consumes stock/2n
    - Remaining stock grows by growth rate
    """
    ...
    def _calculate_payoffs(self,
                           actions: NDArray[np.bool_]) -> NDArray[np.float64]:
        share = 0.5 * self.stock / self.description.n_players
        payoffs = share + share * (~actions).astype(np.float64)

        self.stock -= np.sum(payoffs)
        self.stock = max(self.stock, 0)
        self.stock += 2 * self.stock * (1 - self.stock / self.description.capacity)
        self.stock = min(self.stock, self.description.capacity)

        return payoffs
```
Note: payoff equals extraction; the stock is reduced by `np.sum(payoffs)`. A fine subtracted from `payoffs` before this line would wrongly feed back into the stock.

Description and bounds, L77–113:
```python
@dataclass(frozen=True)
class CommonPoolDescription(GameDescription):
    """Description for Common Pool Game."""
    capacity: int
    ...
        if not (self.capacity >= 2 * self.n_players):
    ...
    def max_player_welfare(self) -> float:
        """All-C every round, except All-D in the last"""
        return 0.5 * (self.capacity / self.n_players) * (self.n_rounds + 1)

    def min_player_welfare(self) -> float:
        """All-D in the first round"""
        return self.capacity / self.n_players
```
`_compute_payoff_bounds` (L16–74): min individual payoff = focal All-C vs others All-D; max = focal C-then-D vs All-C, "(Optimum within this strategy class; not provably the global optimum.)"

## 5. Default parameters (the only place the CPR settings are fixed)

`src/emergent_llm/games/helper.py` L31–44
```python
STANDARD_GENERATORS: dict[str, Callable[..., GameDescription]] = {
    'public_goods_default':
        lambda n_players, n_rounds=20: PublicGoodsDescription(
            n_players=n_players, n_rounds=n_rounds, k=2.0),
    'collective_risk_default':
        lambda n_players, n_rounds=20: CollectiveRiskDescription(
            n_players=n_players,
            n_rounds=n_rounds,
            m=max(2, n_players // 2),
            k=2.0),
    'common_pool_default':
        lambda n_players, n_rounds=20: CommonPoolDescription(
            n_players=n_players, n_rounds=n_rounds, capacity=n_players * 4),
}
```
(PGG: payoff `n_c*k/n + 1[D]`, `public_goods.py` L58–64. CRD: `k*1[n_c>=m] + 1[D]`, `collective_risk.py` L62–71.)

## 6. Attitudes (8 synonyms, 2 families)

`src/emergent_llm/common/attitudes.py` L5–30
```python
class Attitude(StrEnum):
    """Player attitudes for strategy generation."""
    COLLECTIVE = "collective"
    PROSOCIAL = "prosocial"
    ALTRUISTIC = "altruistic"
    BENEVOLENT = "benevolent"

    SELFISH = "selfish"
    SELFINTERESTED = "self-interested"
    OPPORTUNISTIC = "opportunistic"
    INDIVIDUALISTIC = "individualistic"
```
`Gene` = `(model: str, attitude: Attitude)` (`common/gene.py` L8–14).

## 7. Generation prompts

`src/emergent_llm/generation/prompts.py` L95–153 (CPR game description, verbatim)
```
GAME SPECIFICATION: Common Pool Resource Game
PARAMETERS:
- n: number of players (integer, n ≥ 2)
- r: number of rounds (integer, r > 1)
- capacity: maximum sustainable stock level (float, capacity)

STATE:
- stock: current stock level (float, 0 ≤ stock ≤ capacity)

GAME STRUCTURE:
- Type: Simultaneous-move, repeated game
- Rounds: r identical rounds, indexed t ∈ {0, 1, ..., r-1}
- Players: n players, indexed i ∈ {1, 2, ..., n}
- State: Common pool stock (initially = capacity)

ACTIONS PER ROUND:
Each player i has action set A_i = {C, D} where:
- C (Cooperate): Consume stock/(2n) from common pool
- D (Defect): Consume stock/n from common pool

PAYOFF CALCULATION:
For player i in a single round with current stock S:
π_i = S/(2n) if player i plays C
π_i = S/n if player i plays D

STOCK DYNAMICS:
After all players consume:
1. Total consumption = Σ(i=1 to n) consumption_i
2. Stock after consumption = S - total_consumption
3. Stock growth = 2 × S_remaining × (1 - S_remaining/capacity)
4. New stock = min(S_remaining + growth, capacity)

EXAMPLE CALCULATIONS (n=4, capacity=100, initial stock=100):
1. All players play C:
   - Each gets: 100/8 = 12.5
   - Total consumption: 4 × 12.5 = 50
   - Stock after consumption: 100 - 50 = 50
   - Growth: 2 × 50 × (1 - 50/100) = 50
   - New stock: min(50 + 50, 100) = 100

2. All players play D:
   - Each gets: 100/4 = 25
   - Total consumption: 4 × 25 = 100
   - Stock after consumption: 100 - 100 = 0
   - Growth: 2 × 0 × (1 - 0/100) = 0
   - New stock: min(0 + 0, 100) = 0

3. 2 players play C, 2 play D:
   - Cooperators get: 100/8 = 12.5 each
   - Defectors get: 100/4 = 25 each
   - Total consumption: 2×12.5 + 2×25 = 75
   - Stock after consumption: 100 - 75 = 25
   - Growth: 2 × 25 × (1 - 25/100) = 37.5
   - New stock: min(25 + 37.5, 100) = 62.5

TOTAL GAME PAYOFF:
Total payoff for player i over r rounds = Σ(t=0 to r-1) π_i,t
Note: Payoffs depend on both current actions and accumulated stock depletion from previous rounds.
```
Note the prompt example uses capacity 100 with n=4, while the experiments use capacity 4n (16 at n=4).

Task prompt with the attitude wording, `prompts.py` L158–175 (verbatim)
```python
def create_strategy_user_prompt(attitude: Attitude, game_name: str) -> str:
    """Create user prompt for strategy description generation."""

    state = ", history and state" if game_name == "common_pool" else " and history"

    return f"""{format_game_description(game_name)}

Assumptions:
- Common knowledge: all players know the game rules, parameters and payoff structure.
- Aggregate observation: the players are anonymous. After each round, you observe how many of your opponents cooperated, but not which ones.
- No communication: players cannot communicate or otherwise share information.
- Opponents are independent strategies from other AI systems — no shared norms or predetermined coordination.

Design a {attitude} strategy for this game that only depends on the game parameters{state}. Your strategy should be adaptive and robust to a wide range of opponent behaviours.

1. Specify decision rules - When exactly do you cooperate vs defect?
2. Handle edge cases - What do you do in the first round, last round, etc.?
3. Be {attitude} - Clearly align with the {attitude} mindset."""
```

Description system prompt, `generation/create_strategies.py` L227–237:
```
You are an AI assistant with expertise in strategic thinking.

Output only the strategy description itself — no preamble, no meta-commentary, no conclusion.
...
Use natural language with optional pseudocode. Be precise about decision rules and edge cases.
```
Code system prompt, L258: `"You are an expert Python programmer implementing game theory strategies."`

Code prompt, `prompts.py` L189–247: requires "a single class inheriting from BaseStrategy", `__init__(self, game_description)` and `__call__(self, history{stock_arg})`, forbids imports, globals, `del`, file operations, system calls, raising exceptions, `with`, async. The `PlayerHistory` shown to the model is hard-coded in `get_interface_description` (L249–280).

## 8. Validation of generated code

`create_strategies.py` L320–428 `validate_strategy_code`: AST whitelist (`allowed_types` L327–343), banned nodes `(ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Delete, ast.With, ast.AsyncWith, ast.Raise)`, banned calls `{'eval', 'exec', 'compile', 'open', '__import__', 'globals', 'locals', 'vars', 'dir'}`; exactly one class; `__call__` signature must be `['self', 'history', 'current_stock']` when `has_stock()`.

`create_strategies.py` L831–862: up to `max_retries` coding attempts per description (`generate.sh` passes `--max_retries 5`); a refusal is not retried.

`src/emergent_llm/generation/test_strategies.py` L90–102
```python
def test_strategy_class(strategy_class: type[BaseStrategy],
                        game_name: str,
                        allowed_time: float | None) -> float:
    start = time.time()
    _run_exhaustive_sweep(strategy_class, game_name,
                         n_players=4,
                         n_rounds=5,
                         n_games=1)
    _run_fixed_opponents(strategy_class, game_name, n_players=256)
    total = time.time() - start
    if allowed_time and total > allowed_time:
        raise RuntimeError(f"Strategy took {total:.1f}s")
    return total
```
(called with `allowed_time=2`, `create_strategies.py` L465). The exhaustive sweep plays every opponent-cooperator-count history (`make_fixed_opponents`, L35–47); the n=256 check uses 8 fixed opponent mixtures (All-C, All-D, Random, 90% C, 90% D, alternating, gradual defectors, periodic defectors).

Runtime fallback, `players/players.py` L59–70: on an exception the player returns C if its base attitude is Collective, else D, and raises after `max_errors=2`.

## 9. LLM providers (Ollama supported)

`create_strategies.py` L873–875: `choices=["openai", "anthropic", "ollama", "google", "openrouter"]`; L957–958: `client = ollama.Client(host=os.environ["OLLAMA_HOST"])`; L609–610: "NOTE: Ollama has no effort equivalent; reasoning_effort is ignored here".

## 10. Experiment settings

`scripts/config.sh` L7–16
```bash
BETA=1
GAMES_PER_AGENT=4

GAMES=("public_goods" "collective_risk" "common_pool")
PROVIDER_MODELS=(
    "openai gpt-5.4-mini"
    "google gemini-3.1-flash-lite-preview"
    "anthropic claude-haiku-4-5"
)
EVOLUTION_PLAYERS=(4 64)
```
`scripts/generate.sh` L16–18: `--full_attitudes descriptions --n 128` (128 per synonym, 8 synonyms → 1024 per model per game).

`scripts/analyse.sh`: fingerprinting `--n_rounds 7 --n_games 30` (L6–12); self-play `--matches 200 --group-sizes 4 16 64 256` (L24–31); cultural evolution (L55–69):
```bash
            python scripts/run_cultural_evolution.py \
                   --game ${game} \
                   --n_players $n_players \
                   --n_rounds 20 \
                   --population_size 512 \
                   --beta $beta \
                   --mutation_rate 0.0025 \
                   --n_generations 1000 \
                   --final_window 100 \
                   --games_per_agent $g \
                   --n_runs 100 \
```
plus sensitivity configs `"1 16 evolution_G16"`, `"0.25 4 evolution_beta0.25"`, `"4 4 evolution_beta4"` (L44–49). Note: the paper states 2000 generations; the script says 1000. No `--models` filter is passed, so the registry would also load `strategies/*/claude-fable-5.py`.

Deposited strategies (counted with grep): `strategies/common_pool/{claude-haiku-4-5,gemini-3.1-flash-lite-preview,gpt-5.4-mini}.py` 1024 classes each (128 per synonym); `claude-fable-5.py` 64 (8 per synonym), not used in the paper.

## 11. Cultural evolution

`src/emergent_llm/tournament/cultural_evolution.py` L142–163 (play, fitness normalised to single-agent bounds)
```python
        pi_max = self.config.game_description.max_payoff() * self.config.games_per_agent
        pi_min = self.config.game_description.min_payoff() * self.config.games_per_agent
        fitnesses_normalised = (fitnesses - pi_min) / (pi_max - pi_min)
```
L189–208 (Fermi imitation)
```python
    def _imitate(self, fitnesses: np.ndarray) -> tuple[list[Gene], np.ndarray]:
        """Fermi pairwise-comparison rule (Traulsen, Nowak, Pacheco, 2006)."""
        n = len(fitnesses)
        partners = (np.arange(n) + np.random.randint(1, n, size=n)) % n
        deltas = fitnesses[partners] - fitnesses
        probs = 1.0 / (1.0 + np.exp(-self.config.beta * deltas))
        copy_decision = np.random.random(n) < probs
```
L210–236: model and attitude mutate independently with probability `mu`; attitude mutation flips the base family and picks one of its four synonyms. L171–176: agents whose gene changed draw a fresh strategy (`registry.sample_spec`); unchanged agents keep their strategy.

`tournament/fair_tournament.py` L54–73: each repetition shuffles the population and splits it into consecutive groups of `n_players`; repetitions = `games_per_agent`.

No random seed is set anywhere in `src/` (grep for `seed` returns nothing).

## 12. Behavioural fingerprinting

`scripts/diversity.py` L183–195 (`compute_features`): for every opponent-count trajectory, play `n_games` games and record the focal's mean cooperation for each history prefix; PCA (`sklearn.decomposition.PCA`, L700–709); `compute_delta` (L340–356) "Standardised centroid distance"; `find_centroid_strategies` (L927–) picks the strategies stored in `representative_strategies/`.

## 12b. The deposited strategies predate the current prompt text

`src/emergent_llm/generation/replication_prompts.py` L10–16 (docstring):
```
The conditions are held as close as possible: identical game spec, parameters,
history rendering, field names and derived fields. Three things necessarily
differ — the round base above, the extra INTERFACE block in the code condition,
and the closing question. The round base is a genuine confound and cannot be
removed: the descriptions were generated against 1-based rounds and the code is
written against 0-based ones, so neither condition can be relabelled without
misrepresenting what its author saw.
```
The current `prompts.py` says "indexed t ∈ {0, 1, ..., r-1}". The paper's Appendix A.1 prompt says "Rounds: r identical rounds" and sums "t=1 to r". So the deposited descriptions came from an earlier 1-based version of the prompt.

## 13. Results metrics

`src/emergent_llm/tournament/results.py` L70–93 `plot_welfare_curves`: `efficiency = (group_data['mean_player_welfare'] - min_w) / w_range`; L1267–1281 "Welfare efficiency against collective ratio"; L1300–1347 relative Schelling diagram.

**Relevance.** This is the exact code an overseer extension would modify. The CPR game is 30 lines; the strategy interface is fixed by a hard-coded `PlayerHistory` in the prompt and in `common/history.py`; Ollama is already a supported provider, so audited-game strategies can be regenerated without new client code.
