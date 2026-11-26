# B. MARL overseers, regulators and planners; decentralised formulations; testbeds

Read 23 Nov 2026 for the overseer formulation (see `notes/formulation/00_plan.md`). Framing: game theory and multi-agent RL. For each paper: what the shaping agent observes, what it does, how it learns, and how the learners respond.

Sources: PDFs in `notes/icml27_lit_review/papers/`, text extracted with pypdf. Page numbers are the PDF's own printed page numbers where they exist (arXiv versions), otherwise section names. The Ma et al. (2024) Nature MI paper is not local; I read the open-access version on nature.com through a web fetch tool, which summarised the page for me, so I did not read every line myself (details in its section). Anything I could not check is marked **[unverified]**.

Notation used throughout: N learners indexed i; s global state; o_i local observation; a_i action; r_i extrinsic reward; θ_i learner parameters. The shaping agent (overseer, planner, regulator, incentiviser) is indexed p, with parameters η or θ_p.

---

## 1. Comparison table

| Paper | Who shapes | What it sees | What it does | Budget / cost | How it learns | How many | Timing / solution concept | Learners | SSD? | Objective |
|---|---|---|---|---|---|---|---|---|---|---|
| Sun 2019, regulation enforcement | No overseer. Compliant agents (≥80%) boycott, using a shared detector | Detector: each agent's reward and/or action sequence (global, full) | Compliant agents add a penalty: minus B times the mean observed reward of flagged agents | None; boycott ratio B costs compliant agents return | Detector: rule or supervised classifier. Agents: independent DQN | Detector is one global classifier | Ad hoc; checked with an empirical 2×2 payoff matrix (Nash before/after) | 5 (10 in EGTA), MAgent grid, DDDQN | Harvest-like, yes | Make complying a best response (V_C ≥ V_D) |
| Zheng 2021, AI Economist | Separate planner agent | Market prices, tax rates, all agents' inventories; not skills, not space | Sets 7 bracket marginal tax rates once per tax year; taxes redistributed equally | No explicit budget; anonymous taxes | RL (PPO), two-level, with curricula and entropy schedule | 1 | Co-learning approximating a leader-follower (nested) problem | 4 or 10 RL agents, egocentric 11×11 view | No (economy) | Utilitarian welfare, or equality × productivity |
| Yang 2020, LIO | Every agent is also an incentiviser | Own observation plus all other agents' actions | Pays bounded rewards [0, R_max] to each other agent | L1 cost on rewards given | Gradient through the recipients' next policy update (online cross-validation) | All N (2 or 3, up to 5) | Co-learning; one-step lookahead on recipients' learning | PG / actor-critic | Cleanup (2 agents), Escape Room, IPD | Each giver's own extrinsic return |
| Baumann 2020, adaptive mechanism design | Separate planning agent outside the original game | Its own observation plus all players' actions | Adds a reward or penalty to each player: vector in R^N, max abs ≤ c | Quadratic cost α‖V^p‖²; optional revenue-neutral (sum to 0) | First-order lookahead through learners' gradient step (LOLA-like); MLE opponent model if parameters hidden | 1 | Leader shaping followers' learning step | 2 (and 10) policy-gradient learners | Matrix games only | Sum of players' rewards |
| Vinitsky 2023, public sanctions | No overseer. Every agent sanctions (zaps) | Own egocentric view plus a public log of all sanction events and their context | Zap: freezes and marks the target; second zap removes and penalises | Zapping costs time; no explicit budget | Each agent learns a classifier of "would the group sanction this?" and gets intrinsic reward for agreeing with it | All 16 (AH) | Decentralised co-learning; norms as equilibria selected by learning | A3C agents | Yes (Allelopathic Harvest, Clean Up with start-up problem) | Collective return, coordination fraction |
| Wang 2019, GSG-I | Separate defender (patroller) agent | Local only: footprints in its current cell, its own history | Moves on grid; removes snares; catches attacker | One patroller (limited resource) | PSRO / double oracle with DQN best responses | 1 | Zero-sum, so Stackelberg = Nash; equilibrium computation | 1 attacker, best-responding (also learned) | No (adversarial) | Defender expected utility |
| Du 2022, DMPO | No overseer; cooperative networked agents | κ-hop neighbours' states | Control actions | Communication grows with κ | Decentralised model-based PPO with extended value functions | n = 8 to 25 | Cooperative team | PPO agents | No | Average reward |
| Ma 2024, NMI | No overseer; same template, larger scale | κ-hop neighbours; ξ-dependent dynamics | Control actions | Message size O(\|s_i\|) | Decentralised model-based policy optimisation | Up to ~430 agents | Cooperative team | Same | No | Average reward |

Skims (positioning only): LLM legislator (Gasztowtt 2024: LLM principal sets tax/incentive rates per episode, Stackelberg-Markov), IQ-Flow (Guresti 2023: central incentive regulator, offline Q-learning plus meta-gradient), Hughes 2018 (inequity aversion, intrinsic), Köster 2022 (environment labels taboo acts, agents punish), Christoffersen/Haupt 2022 (agents propose binding reward-transfer contracts), Tang 2026 (each agent learns when and how hard to punish).

---

## 2. Overseer, regulator and planner papers

### 2.1 Sun et al. 2019, A Regulation Enforcement Solution for MARL (arXiv 1901.10059v5)

**Players.** No overseer. The paper says outright that it does not assume "police" agents or any authority that can change the environment or punish (p. 2, Sec. 2.1). Enforcement comes from the population: at least M% (80%) of agents are compliant by design, and they punish the rest.

**Observations.** A detector D takes agent i's sequence of rewards and/or actions up to time t and outputs 1 (defective) or 0 (compliant) (Sec. 4.1, p. 4). In the experiments it is one global detector that sees every agent's sequence. The authors assume agents can observe how much resource others collect (p. 3–4). In the MAgent setup each agent's non-spatial input includes the absolute positions of all other agents and apples (p. 6), so the agents are close to fully informed.

**Actions.** Compliant agents train on a shaped reward: own reward minus B times the mean observed reward of all flagged agents (Eq. 9, p. 4). B is the "boycotting ratio". There is no separate sanction action; boycotting means grabbing the resources the defector would have taken.

**Learning.** Detector: a rule (quota check) in Exp. 1 and 3; a 4-layer classifier on reward sequences in Exp. 2, trained offline on 10,000 labelled trajectories (p. 7). Accuracy improves with longer sequences (Fig. 3). Agents: independent double dueling DQN (p. 6). Nothing accounts for agents learning in response to the detector; the detector is fixed after training.

**Number and timing.** One detector. No leader-follower structure. The check is an empirical game: fix 8 of 10 agents as compliant and fill a 2×2 payoff matrix for the remaining two by training each cell (Fig. 2, Sec. 5.3).

**Agents and setting.** 5 agents in a 20×20 grid. The rule is "never hold more than 3 apples" (a harvest quota; Exp. 1), or "use a diminishing reward function" (Exp. 2, a rule on the reward function itself, so only a learned classifier can detect it).

**Objective and results.** Goal: for all j, the value of complying ≥ the value of defecting (Eq. 4 and 6). B ≥ 1.0 (Exp. 1) or ≥ 1.5 (Exp. 2) achieves it (Tables 1–2). In the 2×2 game, mutual defection is the Nash equilibrium before and mutual compliance after (Table 3, p. 8). Everyone's return drops as B rises.

**Formal core.** Markov game M, regulation set splits policies into Π_C and Π_D. A detector D_t(j) ∈ {0,1}. Compliant reward: r'_i = r_i − B·mean_{j: D_t(j)=1} r^obs_j. Target: V_j(π_C^j, π_{−j}) ≥ V_j(π_D^j, π_{−j}).

**What we can borrow.** (1) The split "detector, then response" is exactly an overseer decomposed into an inspection policy and a sanction policy. (2) Rules defined in policy space (a state-conditioned action constraint) versus reward space (a hidden internal property) map onto "observable violations" versus "violations only a classifier can guess". (3) The EGTA 2×2 check is a cheap way to show an overseer moves the equilibrium. (4) They list open questions we can answer directly: the effect of detection errors, and online detector training (p. 8).

**What doesn't fit.** Full observability; no budget; no detection errors studied; the punishers are the peers, so punishing costs the compliant majority. Defectors are scripted, not learning to evade.

### 2.2 Zheng et al. 2021, The AI Economist (arXiv 2108.02755)

**Players.** A separate planner agent plus N economic agents. All are RL agents (Sec. 3, p. 7).

**Observations.** Each actor sees only part of the state (Sec. 2, p. 7). The planner sees market prices, tax rates, the cumulative bids and asks of all agents, and every agent's inventory (endowment). It does not see skills or the spatial map (Methods, p. 20–22, Fig. 7). Leaving out the map made no difference in pilots (p. 20). Not noisy, not costly.

**Actions.** At the start of each of 10 tax years (100 steps each, H = 1000) it picks a marginal rate for each of 7 fixed brackets. Fig. 7 says 22 options per bracket; the Methods list rates 0 to 1 in steps of 0.05, which is 21 values (p. 22). Small mismatch in the source. Taxes are anonymous (same schedule for all, p. 22) and collected tax is shared out equally (p. 6). No budget.

**Learning.** PPO for agents and planner. Two-level: the outer level (planner) maximises welfare, the inner level (agents) maximises utility given taxes (p. 8). Stabilised by (a) curricula that phase in labour costs and then taxes, and (b) an entropy schedule that makes the planner act near-randomly at first, so agents learn to condition on a wide range of tax rates before the planner starts optimising (p. 8). Agents' adaptation is accounted for only by this staging, not by differentiating through their learning.

**Number and timing.** One planner. Leader-follower in spirit (planner moves at tax-year boundaries, agents every step), solved by co-learning, not by computing a Stackelberg equilibrium. Gasztowtt 2024 later recasts it as a Stackelberg-Markov game (Sec. 4 below).

**Agents.** 4 or 10 heterogeneous agents (different build skill) in Gather-Trade-Build: move, gather wood and stone, trade, build. Observation: egocentric 11×11 window, own inventory and skill, market info, tax rates (p. 20). Not a sequential social dilemma in the SSD sense; the tension is equality versus productivity.

**Objective and evaluation.** Planner reward is the per-step change in social welfare (Eq. 3). Welfare is either inverse-income-weighted utilitarian, or equality × productivity. Compared against free market, US federal schedule and Saez formula. Recovers the Saez rates in a one-step economy.

**Formal core.** Agents: max over π_i of E[Σ_t γ^t r_{i,t}] given tax rates τ, where r_{i,t} is the change in isoelastic utility (Eq. 1–2). Planner: max over π_p of E[Σ_t γ^t (swf_t − swf_{t−1})] (Eq. 3), with π_p choosing τ from its partial observation.

**Borrow.** (1) The overseer acts on a slower clock than the agents (once per period); we can let an inspector choose its k targets per period. (2) The entropy-then-optimise schedule is a practical recipe for training a learned overseer against learning agents. (3) Observing aggregate or public signals (prices, inventories) rather than private ones (skill) is a natural "what the overseer can see" baseline.

**Doesn't fit.** The planner sees everyone every period; no per-agent targeting (anonymous); no rule violations or sanctions, only a smooth tax; no budget.

### 2.3 Yang et al. 2020, Learning to Incentivize Others (LIO) (NeurIPS 2020, arXiv 2006.06051)

**Players.** No separate overseer. Each agent has two parts: a normal policy, and an incentive function that pays rewards to the other agents (Sec. 3, p. 3). In the experiments every agent is both giver and recipient.

**Observations.** The incentive function of giver i maps its own observation o_i and all other agents' actions a_{−i} to a reward vector for the N−1 others (p. 3). So each giver sees everyone's action every step. The fully decentralised variant (LIO-dec) also needs each recipient's egocentric observations and rewards to fit an opponent model (p. 5–7).

**Actions.** Real-valued rewards r^j_{η_i} ∈ [0, R_max] to each other agent, every step (Sec. 4.2; R_max = 3, 2, 2 for IPD, Escape Room, Cleanup). Positive only, no punishment.

**Budget/cost.** The giver's objective subtracts α times the L1 norm of rewards given, discounted (Eq. 5–6). This cost is charged to the incentive function, not the policy.

**Learning.** Bilevel, solved online (Algorithm 1). Step 1: recipients update their policies with the incentives (Eq. 3–4). Step 2: generate new trajectories with the updated policies. Step 3: update η_i by the gradient of the giver's extrinsic return through the recipients' update (Eq. 7–9). So LIO explicitly models that recipients learn. It differs from LOLA in that it measures the effect after the recipients actually update (p. 5). Adding a plain "give reward" action and using RL learns poorly because the effect shows up only after many episodes (p. 2, Fig. 5).

**Timing.** Co-learning with a one-step lookahead on learners. Proposition 1: two exact LIO agents converge to mutual cooperation in IPD (p. 6). LIO paired with a plain PG agent ends at (defect, cooperate): LIO pays the PG agent to cooperate while it defects (p. 7). Worth remembering: an incentiviser optimising its own return can exploit the shaped agent.

**Agents and setting.** IPD; N-player Escape Room ER(2,1), ER(3,2), ER(5,3); Cleanup with 2 agents on 7×7 and 10×10 maps, where each agent sees the whole map (p. 6). LIO finds a division of labour: one cleans, the other harvests and pays the cleaner (p. 8).

**Objective.** Each giver's own extrinsic return (not welfare), yet collective return approaches the centralised upper bound.

**Formal core.** r_j^total = r_j^env + Σ_{i≠j} r^j_{η_i}(o_i, a_{−i}). Lower level: θ̂_j = θ_j + β ∇_{θ_j} J_j(θ_j, η_{−j}). Upper level: max_{η_i} J_i^env(θ̂) − α L(η_i).

**Borrow.** (1) The cleanest MARL statement of "a shaping agent must differentiate through, or at least wait for, the learners' updates". (2) The incentive function's inputs (o_i, a_{−i}) are where a budget would go: replace a_{−i} with actions of only the k inspected agents. (3) Scripted test agents (R/C/M in Fig. 6b) to check what the shaper rewards: a good diagnostic for an overseer.

**Doesn't fit.** Each giver sees all actions; only 2 to 5 agents; positive transfers only; incentivisers are self-interested, not welfare-seeking overseers.

### 2.4 Baumann, Graepel and Shawe-Taylor 2020, Adaptive Mechanism Design (arXiv 1806.04067v2)

**Players.** N learners plus one added planning agent. The Markov game is amended: the planner has its own observation function O_p and action set A_p ⊂ R^N (Sec. 4.1). The planner's action does not change the transition; it only changes rewards.

**Observations.** O_p(s) plus all players' actions a_1..a_N in the same step (Sec. 4.1). In the matrix-game experiments this is perfect monitoring. If the planner cannot see the learners' parameters, it fits them by maximum likelihood from observed actions (Eq. 7, Sec. 4.4). Not noisy, not costly to observe.

**Actions.** A reward or penalty for each player, r^p ∈ R^N, added to that player's reward: r_i^tot = r_i + r_i^p (Sec. 4.1). Bounded by max_i |r_i^p| ≤ c (c = 3). An optional "revenue-neutral" constraint makes the vector sum to zero, so the planner can only redistribute (Sec. 5).

**Budget/cost.** A quadratic cost α‖V^p‖² on the incentives' value (Eq. 8, α = 0.0002), plus the bound c.

**Learning.** Learners do one gradient step on their total value (Eq. 2). The planner maximises welfare after that step, V(θ + Δθ), via a first-order Taylor expansion. Its gradient is Σ_i η_i (∇_p ∇_i V_i^p)ᵀ ∇_i V (Eq. 4). Policy-gradient estimates replace exact gradients when needed (Eq. 6). So it accounts for learners by looking one learning step ahead, like LOLA but aimed at welfare.

**Timing.** Leader shaping followers' learning, one step at a time. No equilibrium is computed.

**Agents.** 2 learners with one parameter each (probability of cooperating), centralised critic, 4000 episodes (Sec. 5). Also a 10-player PD (Fig. 2). Not sequential.

**Objective and results.** Sum of rewards. Mutual cooperation reaches ~99% in PD, Chicken and Stag Hunt (Table 4). The planner first rewards cooperation, then switches to punishing defection once defection is rare, and total payouts fall towards zero (Fig. 1). Switching the planner off keeps cooperation only where mutual cooperation is already an equilibrium (Stag Hunt); PD collapses (Table 4). Using estimated rather than exact values drops PD cooperation to 61% (Table 5).

**Formal core.** max_{θ_p} V(θ_1 + Δθ_1, …, θ_N + Δθ_N) − α‖V^p‖², with Δθ_i = η_i ∇_i (V_i + V_i^p).

**Borrow.** (1) Clean template for an overseer as an extra player whose action enters only the reward channel. (2) The "turn the overseer off" test tells whether it changed the equilibrium or only props it up. (3) The finding that intervention can fall to near zero once behaviour shifts matters for a budgeted overseer. (4) The authors list partial observability of actions as future work (Sec. 7).

**Doesn't fit.** Sees every action; tiny games; needs learner gradients or a good opponent model; no sanctions targeted under uncertainty.

### 2.5 Vinitsky et al. 2023, A learning agent that acquires social norms from public sanctions (arXiv 2106.09012v5)

**Players.** No overseer. Every agent can sanction others with a short-range zap. Approval means not zapping when it could (Sec. 3.2).

**Formal additions.** (a) A Markov game with sanctions: a set J(s) ⊆ I² of sanctioning opportunities; (i, j) ∈ J(s) when i can zap j (in range). (b) A Markov game with public sanctions: every player also receives a shared sanction observation g = B(s), listing each opportunity, its context C and outcome Z ∈ {0, 1} (Sec. 3.3, p. 5–6). The context can be the sanctioner's current frame, because agents change colour to show their recent activity (planting colour, or grey for free-riding) (p. 6, Sec. 4).

**Observations.** Own egocentric view, plus the global log of all sanction events in the episode (the main setting; p. 6). The authors also test a local version, where each agent learns only from sanctions it saw itself.

**Actions.** Zap: freezes the target for 25 steps and marks it; a second zap while marked removes it for 25 steps with a penalty (Sec. 4). Zapping also has selfish uses (competing for berries).

**Learning.** Each agent learns a classifier Ψ_φ predicting whether the group would disapprove in a given context, trained by cross-entropy on the public log (Sec. 3.5). The classifier is frozen after 5×10⁷ steps to stop it forgetting suppressed behaviours (p. 7, 10). An intrinsic reward +α for zapping when Ψ ≥ 0.5 and −β for zapping when Ψ < 0.5 pushes agents to sanction like the group (Sec. 3.6). Policies: A3C with V-trace, no parameter sharing.

**Timing.** Decentralised co-learning. Norms emerge as equilibria picked by early random patterns (a "bandwagon" effect). The selected norm can be bad: agents can lock into the blue berry nobody prefers (p. 11).

**Agents.** Allelopathic Harvest, 16 agents with conflicting tastes (App. B); Clean Up with a start-up problem (fewer agents; exact number **[unverified]**). Both have start-up and free-rider sub-problems.

**Results.** CNM raises collective return and coordination. Zap counts stay flat or rise, so the gain comes from who gets zapped, not from less zapping (Fig. 5). Local-only sanctions improve on the baseline but do not match public sanctions, because each agent sees far fewer examples (Sec. 5.3, Fig. 8b).

**Borrow.** (1) J(s) is a ready-made formal object for "who can act on whom right now": an overseer with a budget is an agent whose sanctioning-opportunity set is chosen, size k. (2) The public sanction log is a natural thing an overseer can publish; it is the paper's only shared signal. (3) Visible markers of recent behaviour (agent colour) are what make violations observable from one frame. In our testbeds this sets what one inspection reveals. (4) Local versus public sanctions is an observation-budget ablation already done once.

**Doesn't fit.** No designated overseer; no rule exists outside the group's own pattern; no budget except opportunity; norms can be arbitrary, so there is no ground-truth "violation" to score an overseer against.

### 2.6 Wang et al. 2019, Deep RL for Green Security Games with Real-Time Information (AAAI 2019)

**Players.** Two: a defender (one patroller) and an attacker (poacher). The defender is the overseer-like player (GSG-I section).

**Observations.** Local only. Each player sees just the opponent's footprints in its own current cell, and remembers its whole history (GSG-I section). The defender's network input stacks local footprint channels, own footprints, own location, each cell's attack-success probability and time (best-response section).

**Actions.** Move up/down/left/right or stay. Moving onto a snare removes it (+r_tool); finding the attacker catches him (+r_catch); a snare that fires costs p_attack. The attacker moves and optionally places snares from a limited stock. The budget is implicit: one patroller can be in one cell.

**Learning.** DeDOL: double oracle / PSRO. Keep a restricted game of policies; solve it for a Nash mix; train a DQN best response for each side against the other's mix; add it; repeat. Enhancements: heuristic initial policies (a parameterised random-walk poacher; a "random sweeping" patroller that follows footprints), mixed exploration in best-response training, and "local modes" per attacker entry point trained in parallel (DeDOL section).

**Timing.** Zero-sum, so the Stackelberg (defender commits) and Nash solutions coincide; the paper computes an approximate Nash. The attacker best-responds.

**Agents.** One attacker, learned by DQN or heuristic. Grids 3×3, 5×5, 7×7 (Experiments). Not a social dilemma.

**Objective.** Defender's expected utility. On 3×3 it is compared with exact solutions (CFR); on 7×7 against heuristics.

**Borrow.** (1) The plainest formal model of an inspector with local, partial observation in a spatial grid, and of "traces" (footprints) that reveal past violations without catching them live. In Harvest the analogue is depleted patches. (2) PSRO with heuristic starting strategies is a workable way to compute a robust overseer against adaptive violators. (3) The overseer being one unit that must move to see is a natural spatial budget.

**Doesn't fit.** One attacker, not a population; zero-sum; the defender's own reward, not welfare; no learners' welfare to protect; the authors say general-sum is future work (Discussion).

---

## 3. Decentralised networked formulations (Du's group)

### 3.1 Du, Ma et al. 2022, Scalable Model-based Policy Optimization for Decentralized Networked Systems (DMPO, arXiv 2207.06559v2)

**System.** n agents on a fixed undirected graph G = (V, E). N_i is i's neighbours including i; N_i^κ is the κ-hop neighbourhood (graph distance ≤ κ) (Sec. III-A). Each agent has a local state s_i and action a_i; global state and action are concatenations. Policy π_i(a_i | s_{N_i}) depends only on 1-hop states. Reward r_i(s_i, a_i) is local; the team reward is the average (1/n) Σ r_i (Eq. 1–2). It is cooperative.

**Communication.** Agents exchange states with neighbours, and each agent broadcasts its model's predictions (abstract; Sec. IV-A). For the critic, an agent also uses its κ-hop neighbours' value estimates (Eq. 9).

**Locality.** Transitions are assumed to factorise as p(s′ | s, a) = Π_i p_i(s′_i | s_{N_i^κ}, a_i) (Eq. 4). The true neighbourhood is unknown, so κ-hop is used as an approximation.

**Decay property.** No ξ parameter in this paper. Locality is expressed through κ and the discount: the extended value V_i(s_{N_i^κ}) differs from the true V_i(s) by at most r_max γ^κ / (1 − γ) (Theorem 1, Eq. 12). The average of local values approximates the global value with the same bound (Eq. 13). The policy gradient computed with neighbours' extended values is close to the true one (Theorem 2, Eq. 14).

**Algorithm.** Each agent learns a local model p_ψi(s′_i, r_i | s_{N_i^κ}, a_i) (Eq. 11), generates short branched rollouts from real states (length 25), and trains a decentralised PPO policy on model data with the extended value function (Eq. 6–10).

**Testbeds.** CACC catch-up and slow-down (8 vehicles), Flow figure-eight (14), ring attenuation (22), traffic signal grid (25) (Sec. VI; κ per task in Table I). Larger κ: better model and policy, more communication and compute (Fig. 5).

### 3.2 Ma, Li, Du, Dong and Yang 2024, Efficient and scalable reinforcement learning for large-scale network control (Nature Machine Intelligence 6:1006–1020, doi 10.1038/s42256-024-00879-7)

**Source used.** The open-access article on nature.com, read through a web fetch tool that summarised the page (main text and part of the Methods; the capture was cut off around Theorem 4, and supplementary material was not read). I did not see the PDF. Treat the specific numbers below as **[checked against the fetched article only]**.

**System.** Same networked MDP as DMPO: undirected stationary graph, one node per agent, local s_i and a_i, policy π_i(a_i | s_{N_i}), κ-hop neighbourhoods, local rewards averaged into a global reward, messages only along edges. Minimum message size is O(|s_i|).

**Locality and the ξ parameter.** Definition 1: an *independent networked system* is one whose transition factorises exactly into local κ-hop terms. Definition 2: a system is *ξ-dependent* if there is an independent system p̄ whose transition is within total-variation distance ξ of the true one, for every state and action. Small ξ means "almost independent". This turns the DMPO assumption into a measurable approximation. Corollary 1 adds a term of order 2 r_max γ ξ / (1 − γ)² to the model-rollout error bound. Theorem 3 repeats the γ^κ value bound. Monotonic improvement is claimed only when policy updates stay inside a trust region. Fig. 2d reports ξ estimates per testbed (fetched summary: about 0.2 for Catch-up and 0.8 for Real Power-Net); how ξ was estimated is not explained in what I read **[unverified]**.

**Testbeds and scale.** CACC, Flow ring, traffic signal control on a synthetic grid, Monaco and New York (about 430 intersections; text and figure disagree on 436 vs 432), IEEE power grids and Real Power-Net (141, 322 and 421 buses; 199 agents in the largest), and pandemic networks (agents are societal units such as health and education systems; their exact states and rewards were not in what I read **[unverified]**).

**One formulation, many testbeds.** Every testbed is written as the same networked MDP. Only the graph (line, ring, grid, real road or power network), the agent types and the local state/action/reward definitions change. The same algorithm and κ setting run on all of them.

### 3.3 How Du's group builds a general formulation: a template

What both papers do, in order:

1. **One abstract object first.** A graph of agents G = (V, E); for each node a local state, action, and reward; neighbourhoods N_i and N_i^κ; a local policy class π_i(a_i | s_{N_i}). The team objective is written once (average reward).
2. **A structural assumption that makes scale possible, stated as a definition.** Factorised κ-hop dynamics (DMPO), then relaxed into a measurable quantity, ξ (NMI). The assumption has a knob (κ) and a mismatch measure (ξ).
3. **Bounds that say what you lose by being local.** The gap between local and global value is r_max γ^κ / (1 − γ); dependence adds a ξ term. These are short, readable results that justify the design.
4. **One algorithm that uses only local information** and whose cost grows with κ, not n.
5. **Many testbeds, each a table row.** For each: what an agent is, its local state, action, reward, the graph, κ. No testbed-specific theory.
6. **Measure the assumption in each testbed** (ξ in Fig. 2d) rather than only asserting it.

What this would require of an overseer formulation:

- **The overseer is a node (or nodes) in the same graph.** Define the extended system G⁺ = (V ∪ P, E ∪ E_P), where P is the set of overseers. Edges from an overseer to agents are not fixed; they are chosen each step (the k inspected agents). This makes "budget k" a degree constraint on the overseer's node.
- **The overseer's observation is a local observation.** o_p,t = {o_j or a summary of agent j : j ∈ K_t}, with |K_t| ≤ k. This parallels s_{N_i} for ordinary agents.
- **A locality assumption for oversight.** Something like: an agent's violation affects others' rewards only within κ hops (in space or in the interaction graph), or the dynamics are ξ-dependent. Then a bound like Theorem 1 can say how much welfare an overseer loses by inspecting k agents rather than all n, as a function of k, κ (or ξ) and γ. This is the natural "theorem slot" in the template.
- **Several overseers** are just several nodes in P, each with its own budget and possibly its own region; they communicate along edges like other nodes.
- **Each testbed becomes a row**: agents, local state, overseer observation per inspection, overseer actions, violation predicate, welfare and harm measures, the graph, κ or ξ. Measure the locality parameter per testbed.
- **Caveat.** Both papers are cooperative (shared average reward). Our learners are self-interested and the overseer's reward differs from theirs. The template carries over for the structure and the bounds, not for the solution concept, which must come from the shaping papers (Sec. 2): a leader-follower or co-learning game in which learners respond to the overseer.

---

## 4. Testbeds: what an overseer could see, define and measure

### 4.1 SocialJax (Guo et al., ICLR 2026)

**State and observations.** N-player partially observable Markov game (Sec. 3.1). Nine environments (Coins; Commons Harvest Open, Closed, Partnership; Clean Up; Coop Mining; Mushrooms; Gift Refinement; Prisoner's Dilemma Arena) (Sec. 3.2). Every agent gets an 11×11 egocentric grid window with objects as integer codes (Sec. 3.2, App. B.2). Default agent counts: Clean Up 7, Harvest 7, Coop Mining 6, Gift Refinement 5, PD Arena 5, Mushrooms 5, Coins 2; adjustable, tested at 20 (App. B.4–B.5, Table 6). Since the whole environment is a JAX state array, the full grid and every agent's position, action and reward are available to code. That is my inference from the implementation style, not a stated feature **[unverified]**.

**What an overseer could feasibly inspect.** Per inspected agent at time t: its action, its reward, its 11×11 window, its position. Cheaper than full state; a natural unit of budget. Aggregate public signals: apples left, river pollution level.

**Definable violations.** Harvest: taking the last apple in a patch, or harvesting above a local-density threshold (regrowth depends on apples within radius 2; App. B.2). Clean Up: free-riding (harvesting without ever cleaning). Coins: picking up the other agent's coin. Coop Mining: mining iron when gold needs a partner. Mushrooms: eating selfish mushrooms. Zapping others in Closed/Partnership Harvest.

**Welfare and harm.** Per-environment cooperation metrics (Sec. 3.4): own-colour coins, apples remaining (Harvest), cleaned tiles (Clean Up), gold mined, blue mushrooms eaten, tokens received, cooperative resources collected. Schelling diagrams verify each is a dilemma (Sec. 4). Plus collective return. Perolat's U, E, S, P metrics apply directly.

**Built-in shaping methods.** IPPO with individual or common reward, MAPPO, VDN, SVO, and reward exchange IPPO-RE (Willis et al. 2025, a different Willis paper from the IPD one below; Sec. 3.3). These are baselines an overseer must beat or complement.

### 4.2 OpenSocialJax (Guo, Fang, Leibo and Du 2026, arXiv 2610.13710)

**State and observations.** Each episode samples a hidden configuration z, which defines a POMG G_z (Sec. 3.1). OpenCleanup: 15 waste types and a hidden tool-pair system, 57,657,600 rule sets. OpenHarvest: apples with colour and ripeness stages and hidden reward and regrowth multipliers, 60,466,176 rule sets; zap sends the target back to spawn (Sec. 3.2). z is fixed over K = 5 trials of up to 400 steps. LLM agents get a text version of their 11×11 view, available actions, their own rule evidence, recent steps, what they saw others do, trial summaries and their own notes. They cannot see the hidden rule, others' prompts or scores, and cannot talk (Sec. 3.3). Five agents per team, homogeneous or mixed (Sec. 4.1).

**What an overseer could inspect.** Everything an agent sees, plus its notes and JSON reasoning. That makes text inspection of intent possible, not only of actions. An overseer could also know z when the agents do not (a capability asymmetry in the overseer's favour), or not know it (asymmetry against).

**Definable violations.** Free-riding on cleaning; harvesting the last apples; zapping. Possibly "stated intent to free-ride" in notes (the paper quotes agents planning to rely on others; Sec. 4).

**Welfare and harm.** Mean per-agent return per trial, effective number of cleaners, time at which the last apple is eaten (depletion), Gini of returns, hidden rules known (Fig. 3–5). Every team eventually depletes OpenHarvest: a clear harm that an overseer could target.

### 4.3 Leibo et al. 2017, MARL in Sequential Social Dilemmas (AAMAS)

**State and observations.** Two-player partially observable Markov games (Gathering, Wolfpack). RGB observation 15 cells ahead, 10 to each side; 8 actions including a beam; 1000-step episodes (Sec. 4).

**Key definition.** An SSD is a Markov game in which, for some states, the empirical payoff matrix induced by cooperative and defecting policies satisfies the social-dilemma inequalities. The two policy sets are made by thresholding a continuous behaviour metric, such as aggressiveness (beam-use rate) (Sec. 2). This is exactly the move an overseer needs: a measurable per-agent behaviour statistic, thresholded, defines "violation".

**Overseer view.** Beam use and apple collection are observable per agent and per step. Two players only, so oversight there is trivial; the definitional template matters more than the game.

### 4.4 Perolat et al. 2017, A MARL Model of Common-Pool Resource Appropriation (NIPS)

**State and observations.** N-player partially observable commons game: apples regrow faster with more apples nearby, never once a local patch is empty; 1000-step episodes (Sec. 2). Agents can tag others with a time-out beam that removes them for 25 steps; no direct reward or cost (Sec. 2). Agents see a local pixel window (Fig. 1). Number of agents: not checked in what I read **[unverified]**.

**Social outcome metrics** (Sec. 2.3), worth adopting as the standard harm and welfare panel:
- Utilitarian U = E[Σ_i R_i / T];
- Equality E = 1 − Gini of returns;
- Sustainability S = average time at which rewards are collected;
- Peace P = average number of untagged agent-steps.

**Overseer view.** Tagging and harvesting per agent are observable; S and P are system-level harm measures an overseer can be scored on. The paper reports that agents learn to exclude others (territories via tagging), a form of private enforcement an overseer could replace or regulate.

### 4.5 Willis, Du, Leibo and Luck 2025, Will Systems of LLM Agents Cooperate (arXiv 2501.16173)

**Setting.** LLMs (ChatGPT-4o, Claude 3.5 Sonnet) write iterated Prisoner's Dilemma strategies in natural language under three attitudes (aggressive, cooperative, neutral) and three prompt styles. ChatGPT-4o converts them to Python. Strategies are checked by hand, then play all-play-all tournaments of 1000 rounds, with and without 10% action noise, and a Moran process tests which attitude takes over a population (Sec. 3).

**Overseer view.** The point that matters for us: strategies exist as code before play, so they can be inspected in advance (Sec. 1). An overseer of LLM agents can inspect policies, not only actions. There is no spatial state; the history of moves is fully observable. Violations: defection, or attitude mismatches. Welfare: tournament scores, population share after selection.

---

## 5. Skims (positioning)

**Gasztowtt et al. 2024, Large Legislative Models (arXiv 2410.08345).** Defines a Stackelberg-Markov game: a principal commits to an action φ each episode, which induces a POMG M_φ; followers respond with policies; equilibrium = principal's best φ given follower equilibria (Sec. 2). Recasts the AI Economist in this frame. The LLM principal reads a prompt of past (action, payoff, observation) triples and outputs the next φ by in-context learning. Harvest (7 agents): three-tier tax on apple reward, tier set by apples collected in the last 10 steps. Clean Up (7 agents): incentives on harvest, clean and other actions. Also a Contextual Escape Room. Converges about an order of magnitude faster than AI Economist on Harvest and Clean Up, two orders on the Escape Room variant (results section). They also found that removing the planner's observations did not hurt AI Economist or MetaGrad, so in those environments the best tax is static (Sec. 1). Relevance: the only LLM overseer acting on MARL learners in SSDs here; acts per episode, sees aggregates, no per-agent inspection.

**Guresti et al. 2023, IQ-Flow (AAMAS).** A central incentive regulator pays incentives R_inc,η(s, a) ∈ U^N to all agents, with full state and joint action (Sec. 3). It trains offline critics (implicit Q-learning) for cooperative, environment and incentivised returns, and updates η by meta-gradient so that the incentivised greedy policy matches the cooperative one. It makes no assumption about agents' learning algorithms. Escape Room and 2-player Cleanup. Relevance: overseer that does not need learners' gradients; still fully observing.

**Hughes et al. 2018, Inequity aversion (NeurIPS).** Agents get an intrinsic reward that penalises differences between their own and others' (smoothed) rewards; this helps Cleanup and Harvest. Needs everyone's rewards to be public. Relevance: a reward-shaping alternative inside the agents, not an overseer; a baseline.

**Köster et al. 2022, Silly rules (PNAS; arXiv 2020 version read).** The environment, not the agents, labels which acts are taboo (eating certain berries); agents can punish taboo-breakers. Adding a harmless extra taboo makes agents learn enforcement and compliance faster and more stably (abstract, Sec. 1). Relevance: shows that a given classification scheme plus peer punishment works, and that more frequent enforcement opportunities aid learning. An overseer supplying the labels is the next step.

**Christoffersen, Haupt et al. 2022/24, Formal contracts (arXiv 2208.10469v4).** Agents may propose binding reward transfers conditioned on behaviour; others accept or not, and the environment enforces them. With a rich enough contract space and full observability, all subgame-perfect equilibria are socially optimal; under partial observability richer contracts still help (abstract, Sec. 2). Tested in Cleanup, traffic, pollution and common-pool games. Relevance: enforcement is assumed perfect; an overseer with limited observation is what makes contracts enforceable in practice.

**Tang et al. 2026, Adaptive punishment (APC).** Each agent learns a "defection awareness" module from local observation trajectories, then punishes with a learned probability scaled by the defection's severity, to cut wasted punishment (abstract, Sec. III). Iterated public goods game, Coin Game and a berry-harvest SSD (exact name **[unverified]**). Relevance: decentralised peers deciding when punishment is worth its cost; closest to a budgeted sanctioner, but without a separate overseer.

---

## 6. Design space: every way these papers model an agent that shapes others

Axes, and the values found:

| Axis | Values seen | Papers |
|---|---|---|
| **Who shapes** | (a) a separate agent added to the game | Baumann, Zheng, Gasztowtt, IQ-Flow, Wang (defender) |
| | (b) every agent, as an extra channel | LIO, Vinitsky, Tang, Hughes (intrinsic), contracts |
| | (c) a compliant majority guided by a shared detector | Sun |
| | (d) the environment itself (labels, enforcement) | Köster (labels), contracts (enforcement) |
| **What it sees** | full state and joint action | IQ-Flow, Baumann, contracts (theory) |
| | everyone's actions plus own view | LIO |
| | aggregate/public signals, not private types | Zheng, Gasztowtt |
| | public log of sanction events | Vinitsky |
| | per-agent reward/action sequences (all agents) | Sun |
| | local only, spatially limited | Wang, Tang, Vinitsky (local ablation), DMPO/NMI (κ-hop) |
| **What it does** | adds rewards/penalties per agent | Baumann, LIO, IQ-Flow |
| | sets anonymous rates (tax) applied to all | Zheng, Gasztowtt |
| | physical sanction (zap, freeze, remove) | Vinitsky, Köster, Tang, Perolat (tagging) |
| | reshapes peers' rewards to boycott | Sun |
| | moves to find and remove violations | Wang |
| | enforces agreed transfers | contracts |
| **Budget / cost** | none | Zheng, Vinitsky, IQ-Flow |
| | bound on payment size | Baumann (c), LIO (R_max) |
| | cost on payments (L1 or quadratic) | LIO, Baumann |
| | redistribution only (sum zero) | Baumann (revenue-neutral), Zheng (taxes returned) |
| | physical: one unit, one location | Wang |
| | opportunity cost to the punisher | Sun, Vinitsky, Tang, Perolat |
| **How it learns** | gradient through learners' next update | Baumann, LIO |
| | meta-gradient with offline critics | IQ-Flow |
| | two-level RL with curricula and entropy | Zheng |
| | in-context LLM over episodes | Gasztowtt |
| | equilibrium computation (PSRO) | Wang |
| | supervised detector, fixed | Sun |
| | classifier of group approval plus intrinsic reward | Vinitsky |
| | learned detection plus learned punishment probability | Tang |
| **How many** | one | Baumann, Zheng, Gasztowtt, IQ-Flow, Wang, Sun (detector) |
| | all agents | LIO, Vinitsky, Tang, contracts |
| | several coordinated overseers | none |
| **Timing** | leader-follower over episodes (commit then respond) | Gasztowtt (Stackelberg-Markov), Zheng (per tax year) |
| | one-step lookahead on learning | Baumann, LIO |
| | simultaneous co-learning | Vinitsky, Tang, Sun |
| | zero-sum equilibrium (Stackelberg = Nash) | Wang |
| **Learners** | policy-gradient / PPO / A3C / DQN; scripted defectors (Sun); best-responding attacker (Wang); LLM agents (OpenSocialJax, Willis, BazaarBench not covered here) | |

**Combinations nobody here has studied** (with a budgeted, partially observing inspector in an SSD):

1. **A separate overseer that chooses which k of N agents to inspect each step, and only sees those.** The closest are Wang (one patroller, local view, but zero-sum against one attacker), Sun (detector, but sees everyone) and Vinitsky's local ablation (no designated overseer). The choice of whom to inspect, as part of the overseer's action, appears in none of them.
2. **Sanctions that depend on noisy or inferred evidence.** Sun lists detection error as open (p. 8). Every shaping agent above acts on correct observations. Misses and false alarms, and how learners exploit them, are untouched in SSDs.
3. **Learners that adapt to the inspection policy itself** (evading the inspector, behaving well only when watched). LIO and Baumann model learners responding to incentives, not to observation. Only Wang has an adversary that reacts to the patroller, and only one.
4. **Several overseers with separate budgets or regions**, with or without communication. No paper has more than one designated shaper. Du's networked template (overseers as nodes with κ-hop locality, ξ-dependence) is the obvious way to formalise this, and nobody has done it.
5. **A weaker overseer than the agents.** All shapers here have equal or greater information than the agents (Zheng's planner sees all inventories; LIO sees all actions). Capability asymmetry against the overseer is absent, apart from Gasztowtt's LLM principal seeing only aggregates per episode.
6. **An LLM overseer acting per step on individual agents in an SSD.** Gasztowtt acts once per episode on everyone via a tax. Per-agent, per-step LLM inspection in SocialJax or OpenSocialJax is open.
7. **Budget as observation, not as payment.** Every explicit budget above limits what the shaper pays (Baumann, LIO). None limits what it can see. That is the scalable-oversight question.
8. **"Turn the overseer off" and EGTA checks under partial observation.** Baumann and Sun give these tests for full observers; repeating them for a k-of-N inspector would show whether partial oversight changes equilibria or only props up behaviour while active.

---

## Sources

- PDFs: `notes/icml27_lit_review/papers/` (Read properly/, Skim/).
- Ma et al. 2024, open-access article: [nature.com/articles/s42256-024-00879-7](https://www.nature.com/articles/s42256-024-00879-7). Metadata cross-checked with [ResearchGate listing](https://www.researchgate.net/publication/383702263_Efficient_and_scalable_reinforcement_learning_for_large-scale_network_control) and [Google Scholar](https://scholar.google.com/citations?hl=en&oi=sra&user=8J0OzuAAAAAJ) (search results only).
