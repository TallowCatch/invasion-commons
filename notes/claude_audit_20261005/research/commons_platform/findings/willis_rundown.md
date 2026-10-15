# Richard Willis's research and code: a rundown, and how an overseer study can build on it

Sub-agent 3 of the commons-platform deep-research run. Written 14 Oct 2026. Claims are tagged with source numbers from `../sources/` ([40]–[47]). Statements marked **(my analysis)** are my own reasoning or arithmetic, not claims from the sources.

---

## 0. Summary

- Willis's current paper [41] has LLMs (Claude Haiku 4.5, Gemini 3.1 Flash Lite, GPT-5.4 Mini) write a strategy once, in words and then in Python. The strategy is a fixed program that maps the history (and, in the commons, the stock) to C or D. He then studies 9,216 such programs in three n-player games with three analyses: an exhaustive "fingerprint", self-play mixtures of Collective and Selfish strategies, and cultural evolution by payoff-biased (Fermi) imitation [40][41].
- The commons (CPR) game is about 30 lines of code. C takes S/(2n) and D takes S/n. Regrowth is logistic with rate 2 and capacity k = 4n. There are 20 rounds and no enforcer of any kind [40][41]. Its main result: in groups of 64, Selfish strategies take over and CPR welfare efficiency falls to 15%. In groups of 4 it stays at 91% [41].
- The paper asks for "cooperation-supporting institutions" and leaves them untested [41]. Willis's other work studies one institution, binding reward transfer, which he deliberately sets against punishment because shirking is hard to detect [43][44]. **No paper of his tests an external audit-and-fine overseer.** That gap is open, and his code makes it cheap to fill.
- Adding an overseer whose fines the strategies cannot see takes about 4–6 files and 150–250 lines. Making the strategies audit-aware, and regenerating them through Ollama (already a supported provider), takes about 8–10 files and 300–450 lines plus the prompt text **(my analysis)**.
- Becker threshold under their payoff rule: D's one-round gain over C is S_t/(2n). At full stock that is 2 units, so **F\* = S_t/(2nq) = 2/q** at full stock, the same for every n **(my analysis)**. One design trap: in their game, defecting in the final round *raises* welfare, and their welfare maximum assumes everyone does it [41]. An overseer that fines every D therefore also deters a welfare-raising act in round 20.

---

## 1. Architecture of `emergent_llm` (repo commit `be78ff9`, MIT) [40]

### 1.1 Games
All three games share `BaseGame` (`src/emergent_llm/games/base_game.py`). Each round, every player is called with its own anonymous history and the current stock. The actions become a boolean array, `_calculate_payoffs` returns payoffs, and `GameHistory.record` stores actions and payoffs. Nothing else is stored [40].

| Game | File | Round payoff | Default parameters (`games/helper.py`) |
|---|---|---|---|
| Public Goods | `games/public_goods.py` | `n_c·k/n + 1[D]` | k = 2 |
| Collective Risk | `games/collective_risk.py` | `k·1[n_c ≥ m] + 1[D]` | m = max(2, n//2), k = 2 |
| Common Pool Resource | `games/common_pool.py` | C: S/(2n); D: S/n; the remaining stock regrows by `2·S'·(1−S'/capacity)`, capped at capacity | capacity = 4n, start stock = capacity |

All games use 20 rounds by default [40]. In the CPR code the payoff *is* the extraction: the stock is reduced by `np.sum(payoffs)` (`common_pool.py` L129–139) [40]. This matters when adding fines (section 4).

### 1.2 What a strategy sees
A strategy is a class with `__init__(self, game_description)` and `__call__(self, history, current_stock)` (CPR) or `__call__(self, history)` (PGG, CRD) [40]. `PlayerHistory` holds just three read-only arrays: `my_actions`, `my_payoffs` and `opponent_cooperators` (a count per round, with no identities), plus `round_number` [40]. The game description gives `n_players`, `n_rounds` and `capacity`. The paper describes this as anonymity: "each player observes only the aggregate cooperator count from the previous round" [41]. In fact the code gives the whole count history, not just the last round [40].

### 1.3 How strategies are generated
There are two LLM calls per strategy [40][41]:
1. **Description.** System prompt: "You are an AI assistant with expertise in strategic thinking. Output only the strategy description itself …". The user prompt is the game specification plus the task prompt [40].
2. **Code.** System prompt: "You are an expert Python programmer implementing game theory strategies." The user prompt has the specification, the description, a hard-coded interface block (`Action`, the description dataclass, `PlayerHistory`) and a template class [40].

**CPR game text** (`generation/prompts.py` L95–153), key lines quoted exactly [40]:
> "C (Cooperate): Consume stock/(2n) from common pool / D (Defect): Consume stock/n from common pool"
> "3. Stock growth = 2 × S_remaining × (1 - S_remaining/capacity) / 4. New stock = min(S_remaining + growth, capacity)"

It also includes three worked examples at n = 4, capacity 100: all C, all D, and 2C/2D. The full text is in [40].

**Attitude wording** (`prompts.py` L171–175), verbatim [40]:
> "Design a {attitude} strategy for this game that only depends on the game parameters{state}. Your strategy should be adaptive and robust to a wide range of opponent behaviours.
> 1. Specify decision rules - When exactly do you cooperate vs defect?
> 2. Handle edge cases - What do you do in the first round, last round, etc.?
> 3. Be {attitude} - Clearly align with the {attitude} mindset."

For the CPR, `{state}` = ", history and state". `{attitude}` is one of eight synonyms. Collective = altruistic, benevolent, collective, prosocial; Selfish = individualistic, opportunistic, self-interested, selfish [40][41]. The prompt also states four assumptions: common knowledge, aggregate observation, no communication, and opponents are "independent strategies from other AI systems" [40].

### 1.4 How code is checked
- **Static check** (`validate_strategy_code`): exactly one class, the right method signatures, an AST whitelist, and no imports, `global`, `del`, `with`, `raise`, `eval`, `exec`, `open`, etc. [40].
- **Dynamic check** (`test_strategies.py`): the strategy plays every opponent-count history at n = 4, r = 5, then eight fixed opponent mixtures at n = 256. It must not crash and must finish in 2 s [40].
- **Retries.** A failed implementation is regenerated from the same description, up to 5 times. The paper says this "removes execution errors rather than reshaping the strategy distribution" and that logic which is strategically wrong but runs is kept [40][41].
- **Run-time fallback.** If a strategy throws, the player plays C (Collective) or D (Selfish). After 2 errors it raises [40].

### 1.5 Scale and models
3 models × 3 games × 8 synonyms × 128 = **9,216 strategies**, at low reasoning effort [41]. The grep counts agree: 1,024 classes per model per game [40]. The repo also holds 64 per game from `claude-fable-5`, which the paper does not use [40].

### 1.6 Three analyses
1. **Behavioural fingerprinting** (`scripts/diversity.py`). Each strategy plays against every possible opponent-cooperator-count trajectory in a 4-player, 7-round game, with 30 rollouts each. The vector of cooperation rates per history prefix has 5,461 entries. One PCA is fitted over all strategies. Reported per set: mean pairwise distance MPD\*, Collective-vs-Selfish centroid gap Δ, and participation ratio PR [40][41].
2. **Self-play robustness** (`scripts/run_tournament.py`, `MixtureTournament`). For n ∈ {4, 16, 64, 256} and every split n_c + n_s = n (every 4th split at n = 256), sample strategies without replacement from one model's two sets. 200 games per split [40][41].
3. **Cultural evolution** (`tournament/cultural_evolution.py`). 512 agents, genotype = (model, attitude). Each generation: random groups until each agent has played G = 4 games; fitness = total payoff, normalised by single-agent payoff bounds. Each agent then copies a random peer with probability 1/(1+exp(−β·Δfitness)), β = 1. Model and attitude mutate independently with probability 0.25%. Agents whose gene changed draw a new strategy from the pool. 100 runs; the last 100 generations are averaged. Sensitivity runs: G = 16, β = 0.25, β = 4 [40][41].

### 1.7 How results are computed
- **Welfare efficiency** = (U − U_min)/(U_max − U_min), using the game's analytic welfare bounds [40][41].
- For the CPR the bounds are as follows. U_max = everyone plays C for r − 1 rounds, then everyone plays D. U_min = everyone plays D in round 1 [41]. Per player, that is `0.5·(capacity/n)·(r+1)` and `capacity/n` [40].
- Cultural evolution reports genotype frequencies, Collective frequency and welfare efficiency, as mean (SE) over runs [40][41].

### 1.8 Discrepancies found between paper and repo (my analysis, from [40][41])
- The paper says G_max = 2000 generations [41]; `scripts/analyse.sh` passes `--n_generations 1000` [40].
- `analyse.sh` passes no `--models` filter to cultural evolution, so a rerun at this commit would also load the unused `claude-fable-5` strategies [40].
- The deposited descriptions were generated with an older 1-based round prompt, per the replication module's own docstring [40]. So the current `prompts.py` is not exactly the text that produced them.
- No random seed is set anywhere in `src/` [40]. The code uses NumPy's global RNG inside `multiprocessing` workers. *Unverified:* forked workers may start with the same NumPy state. Check this before reusing the batch runner for paired comparisons.

---

## 2. Results relevant to a commons overseer [41]
- **The stock is what CPR strategies respond to.** "the models primarily base their decisions on the current stock levels remaining, rather than the actions of their opponents". CPR has the lowest participation ratio of the three games.
- **Self-play.** Gemini is robust at n = 4, but at n = 64 "even a group consisting entirely of Collective strategies can only achieve a welfare efficiency of 50%". Claude and GPT stay high in Collective groups and fall in Selfish-majority groups as n grows.
- **Cultural evolution (β = 1, G = 4).** CPR welfare efficiency is 91% at n = 4 and 15% at n = 64; Collective frequency 56% vs 19%. In all three games, Selfish dominates at n = 64. The results hold under G = 16 but depend on β.
- The authors credit group selection for the n = 4 result: small groups are more often all-Collective.

---

## 3. How the CPR parameters were chosen, and whether the paper justifies them

| Setting | Value | Justification in the paper [41] | My assessment |
|---|---|---|---|
| Rounds r | 20 | "long enough for conditional strategies to express their behaviour, and to align with protocols used in literature (Fehr and Gächter, 2000)" | Justified by convention: Fehr & Gächter is a public-goods lab protocol, not a commons one. |
| Capacity k | 4n | "k enters the payoff and dynamics only as a multiplicative scale on S_t, so it acts as a unit of reward and does not affect welfare efficiency; we set k = 4n to keep numerical values convenient" | Correct: with S₁ = k, every quantity scales with k. The constraint `capacity ≥ 2n` is enforced in code [40]. |
| D takes double C's share (S/n vs S/(2n)) | fixed | none beyond the game definition; the CPR cites Levhari & Mirman 1980, Ostrom et al. 1994, Gordon 1954 | Not justified on its own. See the MSY point below. |
| Logistic regrowth, rate 2 | fixed | none beyond the citations | Not justified on its own. |
| Players observe | aggregate count | anonymity, motivated by the rate-limited-API story in the introduction | A modelling choice, stated as such. |

**My analysis: the numbers fit together around MSY.** Growth 2x(1 − x/k) peaks at x = k/2, where it adds k/2. Universal C at full stock harvests exactly k/2 and leaves exactly k/2. So **C is precisely maximum-sustainable-yield harvesting**, and all-C keeps the stock at k forever, as the paper says [41]. Any D pushes the post-harvest stock below the MSY level. Universal D harvests everything, and "The state S = 0 is absorbing" [41]. The paper never states this MSY link, but it explains why the shares are 1/(2n) and 1/n. This ties neatly to Ameer's "MSY harm" decision (8 Oct).

**How they evaluate.** Welfare efficiency on a 0–1 scale between analytic worst and best profiles [41]. Willis's MARL paper also uses Schelling diagrams (welfare on a second axis) to check that an environment really is a dilemma [44]. He did not do that for the CPR in [41].

---

## 4. What it takes to add an overseer to their CPR

### 4.1 Where the audit and fine go
**(my analysis, from the code in [40])**

**Game.** Add `AuditedCommonPoolDescription(CommonPoolDescription)` in `games/common_pool.py` with `audit_prob: float`, `fine: float` and optional switches: fine fixed vs proportional to stock, exempt final round, fines public or private. Add `AuditedCommonPoolGame._calculate_payoffs`, which:
1. computes `extraction` as now;
2. reduces the stock by `extraction.sum()`, **not** by the payoffs (today the stock is reduced by `np.sum(payoffs)`, so a naive fine would leak into the resource);
3. draws `audited = rng.random(n) < q`, sets `caught = audited & ~actions`, and computes `payoffs = extraction − F·caught`;
4. keeps the audit and fine arrays for the history.

Binary actions make an audit exact: a caught D is a D. This is simpler than graded extraction, where "slightly over" needs a tolerance.

**Bounds.** `min_payoff` and `max_payoff` feed fitness normalisation in cultural evolution (`cultural_evolution.py` L161–163). With fines, the minimum individual payoff falls (a caught D can net S/n − F, below zero for large F), so the bounds must be redefined. Welfare needs a decision too: count fines as transfers (welfare = extraction) or as deadweight (welfare = net payoff). Report both.

**Registration.** `games/helper.py` needs `get_game_type`, `get_description_type` and a `STANDARD_GENERATORS` entry. The generator lambdas take only `(n_players, n_rounds)`, so q and F need either named generators or a kwargs extension. CLI choices also need adding in `run_cultural_evolution.py`, `run_tournament.py`, `create_strategies.py` and `diversity.py`.

**Registry coupling.** `StrategyRegistry(strategies_dir, game_name)` loads `strategies/<game_name>/`. To run the *existing* CPR strategies in the audited game, add a separate `strategies_game` field to `BatchCulturalEvolutionConfig` (or symlink the folder).

### 4.2 How the strategy interface would change
Today a strategy cannot "know" about audits except indirectly: `my_payoffs` would drop when it is fined [40]. For audit-aware strategies:
- **History.** Add fields to `PlayerHistory`, for example `my_audited`, `my_fines`, and if fines are public `opponents_fined` (a count, which keeps anonymity). Extend `GameHistory.record` and `for_player`, or override `_play_round` in the new game. Existing strategies only read the old fields, so this is backward compatible.
- **Description.** Expose `audit_prob` and `fine` as dataclass fields. `print_definition()` puts them into the prompt automatically [40].
- **Prompt.** Add an "audited_common_pool" branch to `format_game_description` with the audit rule and worked examples, including an expected-fine example. Add an assumption line on what is observed about audits. Update the **hard-coded** `PlayerHistory` block in `get_interface_description` (`prompts.py` L249–280) [40].

### 4.3 Cultural evolution with fines
No change to the imitation code is needed. Fitness is total payoff, so fines enter selection directly once the game subtracts them [40].

**Two cheap designs.**
- **(a) Blind strategies.** Run the existing 3,072 CPR strategies in the audited game. Fines then act only through selection between (model, attitude) genotypes, not through in-game responses. This costs no LLM calls. It tests the paper's own claim that institutions are needed at n = 64: does an audit-and-fine overseer reverse the Selfish takeover?
- **(b) Audit-aware strategies.** Regenerate the pool with the rules in the prompt and compare.

Sweep q × F with F in units of F\* (section 4.5). Report Collective frequency, welfare efficiency (both welfare definitions) and the number of fines paid. Adding monitoring cost (cost per audit × q × n) answers the supervisor's monitoring-cost point.

### 4.4 Regenerating strategies via Ollama
This is supported already: `--llm_provider ollama` with `OLLAMA_HOST`, though `reasoning_effort` is ignored for Ollama [40]. Each strategy needs at least 2 calls, plus retries (up to 5).

**Cost, rough (my analysis).** 3 models × 2 families × 4 synonyms × 32 = 768 strategies ≈ 1,600–2,000 calls. These calls produce long outputs (a description, then a class), so each costs more than the short per-step calls measured in L4. I cannot verify a cost per call. Run a 20-strategy pilot on one Ollama model and scale from it.

Make q and F *parameters* the code reads from `game_description`, so one pool covers a whole sweep. Otherwise each (q, F) setting needs its own pool. Check with the fingerprint and Δ whether audit-aware and blind pools actually differ [41].

### 4.5 Becker deterrence threshold under their payoff rule (my analysis)
Payoff π = S_t/(2n)·(1 + 1[D]) [41].

- **One-round gain of D over C:** S_t/(2n). This equals the whole C payoff: defecting doubles your take that round. D does **not** change anyone else's payoff in the same round. Its harm passes entirely through next round's stock.
- **Myopic risk-neutral deterrence:** q·F ≥ S_t/(2n), so **F\* = S_t/(2nq)**. At full stock (S = k = 4n) the gain is exactly 2 for every n, so **F\* = 2/q**: 40 at q = 0.05, 20 at q = 0.1, 10 at q = 0.2, 8 at q = 0.25, 4 at q = 0.5, 2 at q = 1. A fine proportional to the stock, F = φ·S_t, gives φ\* = 1/(2nq).
- **Accounting for the defector's own future loss.** I re-implemented their rule independently, without running their code: `share = S/(2n)`; the stock falls by (n − d)·share + d·2·share; then `S ← min(S' + 2S'(1 − S'/k), k)`, with k = 4n and r = 20. The script is in the session scratchpad: `/private/tmp/claude-501/-Users-ameerfiras-invasion-commons-invasion-commons/cfe06c49-9d3e-4d81-b792-bdc11cc119da/scratchpad/calc/becker.py`. With the others always playing C and one deviation mid-game: the net gain is 1.30 (n = 4), 1.86 (n = 16), 1.97 (n = 64) and 1.99 (n = 256), against the immediate 2.00. In the final round it is exactly 2.00. So the forward-looking threshold is at most 2/q. It sits about a third lower in groups of 4 but is almost 2/q in large groups. That fits the paper's finding that large groups are where cooperation fails.
- **Social cost of one D** (same script): total welfare changes by −0.79 at n = 4 and −0.04 at n = 64 for a mid-game D, but by **+2.00** in the final round. In large groups a lone defector does almost no social damage while gaining almost 2. The harm comes from many defecting together.
- **Design implication.** In the final round, D is both privately and socially best, and the paper's U_max assumes all-D then [41]. An overseer that fines every D will cut measured efficiency by deterring that last-round grab. Exempt round r, define harm against the MSY stock instead of "any D", or report the effect explicitly.

### 4.6 Engineering size (my analysis)
| Version | Files touched | New or changed functions | Rough lines |
|---|---|---|---|
| A. Audit and fine, existing (blind) strategies | `games/common_pool.py`, `games/helper.py`, `games/__init__.py`, `tournament/configs.py` (strategy-folder field), `scripts/run_cultural_evolution.py`, `scripts/run_tournament.py` | new description + game classes; `_calculate_payoffs`; bounds (`min_payoff`, `max_payoff`, welfare bounds); generator entry; two CLI arg sets | 150–250 |
| B. Audit-aware strategies | A, plus `common/history.py`, `games/base_game.py` (or a game override), `generation/prompts.py` (new game text + interface block), `generation/create_strategies.py` (choices, `LOCAL_IMPORTS`), `scripts/diversity.py` (choices, names) | `PlayerHistory` fields; `record`/`for_player`; `format_game_description`; `get_interface_description` | 300–450 + prompt writing |
| C. LLM overseer as code (Ameer's L9b) | B, plus a new overseer interface (it sees true actions, chooses audits and fines) and a generation prompt for it, reusing `get_llm_response` and `validate_strategy_code` | new `Overseer` base class; game calls it after actions | +200–300 |

Version A is about 1–2 days of work. Version B is about 3–5 days plus generation time **(my estimate)**. The paper reports 1–2 days to generate 9,216 strategies through commercial APIs [41].

---

## 5. Richard Willis's other work

| # | Paper | Question | Method | Key finding | Relevance to an overseer study |
|---|---|---|---|---|---|
| [42] | *Will Systems of LLM Agents Cooperate* (AAMAS 2025 EA; arXiv 2501.16173) | Do aggressive or cooperative LLM strategies win under selection? | GPT-4o and Claude 3.5 Sonnet write IPD strategies under 3 attitudes and 3 prompt styles; GPT-4o codes them; checked by hand; 1000-round IPD on the Axelrod library; Moran process, population 12 | Aggressive strategies mostly lose. Self-refine makes them stronger. Noise makes aggressive outcomes more likely ("noise can mask aggression") | Origin of the method. Imperfect detection by an overseer acts like noise about who defected. |
| [43] | *Resolving social dilemmas with minimal reward transfer* (JAAMAS 38(2):49, 2024; ALA 2023 workshop version; AAMAS 2025 EA) | How much reward must agents share to make the social optimum dominant? | Game theory: symmetric and general "self-interest level"; a linear program for the minimal transfer matrix; new multi-player games | Similar-looking dilemmas need very different transfers; some scale with n, others need almost all reward shared | His institution is a contract, not policing; he argues "it can be difficult to detect when an opponent is shirking". The self-interest level s\* is a natural comparator for F\*. |
| [44] | *Quantifying the Self-Interest Level of Markov Social Dilemmas* (IJCAI 2025; arXiv 2501.16138) | Can s\* be estimated in Markov games? | PPO in Melting Pot; lower s step by step from selfish pre-trained policies; Dunnett's test; Schelling diagrams | Commons Harvest 0.29 ≤ s\* < 0.33; Clean Up 0.25 ≤ s\* < 0.29; Externality Mushrooms is not a social dilemma | Schelling-diagram check for environments; names "introducing a sanctioning mechanism" as future work. |
| [45] | *A Review of Cooperation in Multi-agent Learning* (arXiv 2312.05162; Du, Leibo, Islam, Willis, Sunehag) | Survey | Survey | (abstract only read) | Framing for transferring RL cooperation mechanisms to LLMs. |
| [46] | *SocialJax* (ICLR 2026; Guo, Shi, Willis, Tomilin, Leibo, Du) | Fast JAX social-dilemma substrates | JAX environments, MARL baselines, Schelling diagrams | At least 50× faster than Melting Pot RLlib baselines | Willis is a co-author on the 2D substrate Ameer may extend. |

**Jianing Zhao.** The only verifiable KCL paper is [41]. Other arXiv hits under that name have non-KCL co-authors and are probably a different person [47].

---

## 6. Strengths and limitations

### Stated by the authors [41]
- Binary actions, fixed strategies and anonymity were "deliberate modelling choices that enabled exhaustive behavioural analysis, pre-deployment inspection, and scaling to large populations, but they constrain generality".
- Fingerprints are computed at n = 4, r = 7 and "may not fully reflect strategy behaviour at the larger group sizes and horizons".
- Imitation ignores brand loyalty, switching costs and non-payoff preferences. The results are "indicative of risks … rather than … precise predictions".
- Future work they list: continuous actions, adaptive revision during play, communication, richer environments, cooperation-supporting institutions.

### Strengths (my assessment)
1. **Scale for little money.** Two generation calls per strategy (description, then code), then free simulation. This allows populations of 512 and groups of 256, which per-step LLM play cannot reach [41].
2. **Inspectable policies.** The code exists before deployment, which fits an oversight story: an overseer could audit the *policy*, not only the actions. That matches the paper's "the natural unit of human oversight is the policy" [41].
3. **Exhaustive fingerprinting** is possible because actions are binary and observations are anonymous counts [41].
4. **A clean evaluation harness**: analytic welfare bounds, a composition sweep, an evolutionary attractor analysis and sensitivity checks [41].
5. **The code is small and readable**, with Ollama already supported and an AST sandbox [40].

### Limitations (my assessment)
1. **Binary extraction.** D always takes exactly twice C's share. There is no "slightly over quota", so an overseer's tolerance, graded fines or allowances cannot be expressed. Shared allowances (a cut to everyone's share) would need the action to become a share of the stock.
2. **No institutions at all.** No enforcer, no fines, no communication, no reputation [41]. The paper calls for them; Willis's own institution work is reward transfer, not sanctions [43][44].
3. **Fixed strategies.** No learning within or across games beyond the conditional logic. Evolution acts on (model, attitude) labels, not on the code; strategies are re-drawn from a fixed pool [40][41]. Strategies cannot react to a new overseer unless they are regenerated.
4. **Anonymity limits enforcement.** Agents cannot sanction each other, but an external overseer that sees individual actions when it audits still can. This is exactly the niche an overseer fills.
5. **Harsh, knife-edge dynamics.** One round of all-D destroys the resource for good [41]. A single D in large groups costs almost nothing socially, and a final-round D is socially optimal (section 4.5). Both complicate a "harm" metric.
6. **The CPR parameters have little justification** beyond "convenient" and "align with protocols" [41]. There is no Schelling-diagram check of the CPR in [41], unlike [44].
7. **Reproducibility details:** no seeds; the paper says 2000 generations while the script says 1000; unused Fable strategies in the default path; deposited descriptions came from an older prompt [40][41].
8. **Small, cheap models at low effort** (Haiku, Flash Lite, Mini) [41]. That is fine for cost but may understate strategic sophistication.

---

## 7. How Ameer could build on it (my analysis)
1. **Overseer on their CPR, blind strategies (Version A).** No LLM cost. Sweep q × F/F\* at n ∈ {4, 64}. Question: does an audit-and-fine overseer turn the n = 64 Selfish attractor (15% efficiency) into a Collective one, and at what monitoring cost? This directly answers their stated future work [41].
2. **Audit-aware regeneration via Ollama (Version B).** Do LLM-written strategies respond to deterrence as Becker predicts? Compare where strategies switch with F\* = S_t/(2nq). This parallels Ameer's L4 results (thresholds fitted to the strategy-as-code paradigm).
3. **Public versus private fines** (`opponents_fined` in the history). This connects to payoff-biased imitation in [41], and to the L7 prediction that publicly visible fines slow the spread of cheating.
4. **Overseer as code (L9b)**, generated and checked like the agent strategies [40][42].
5. **Comparator institution:** compute the self-interest level s\* [43][44] for the same CPR. The comparison is reward exchange versus fines at matched welfare.
6. **Fix the final-round trap** and add a Schelling diagram to admit the audited game (the A1 admission test), following [44].
