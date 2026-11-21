# Decentralised overseers in sequential social dilemmas (draft, 23 Nov 2026)

**Why this exists.** Yali's feedback is that the formulation must be general, not tied to SocialJax. The direction is now **decentralised overseers**: several overseers, each covering part of the system, sharing one total budget. This note does three things:
1. it records how Yali's group sets up decentralised problems (section 1);
2. it writes the decentralised overseer problem the same way (sections 2–4);
3. it restates the claims and the scorecard for the decentralised case (sections 5–7).

It builds on `2026-11-21_general_formulation.md`, which is the one-overseer version, with claims C1–C4. The results so far are in `../OVERSEER_TRACKER.md`.

**Plain terms used throughout.**
- **Agent:** a learner (RL or LLM) playing the game. There are N of them.
- **Overseer:** something that checks agents for rule-breaking and penalises them. There are M of them.
- **Check:** one overseer reads one agent's recent record.
- **Region:** the part of the system one overseer is responsible for. It can be a map area or a set of agents.
- **Violation:** one act that breaks the rule.
- **Prediction:** not tested yet. Every claim in sections 5 and 6 is a prediction unless it says otherwise.

---

## 1. How Du's group sets up decentralised problems

I read two papers in full:
- **DMPO:** Du, Ma, Liu, Lin, Dong, Wang & Yang, "Scalable Model-based Policy Optimization for Decentralized Networked Systems", IROS 2022, pp. 9019–9026; arXiv 2207.06559.
- **The NMI paper:** Ma, Li, Du, Dong & Yang, "Efficient and scalable reinforcement learning for large-scale network control", *Nature Machine Intelligence* 6, 1006–1020 (2024); doi 10.1038/s42256-024-00879-7.

### 1.1 The networked MDP (both papers; NMI Methods "Networked MDP")

| Symbol | Meaning |
|---|---|
| G = (V, E) | an undirected graph that does not change over time |
| V = {1, …, n} | the agents, one per node |
| E | the edges; agents can only talk along edges |
| N_i | agent i's neighbours, including i itself |
| N_i^κ | the κ-hop neighbourhood: every node at graph distance ≤ κ from i |
| N_{−i}^κ | everyone else, V \ N_i^κ |
| s_i, a_i | agent i's local state and action |
| s = (s_1, …, s_n) | the global state, which is just all the local states side by side |
| π_i(a_i \| s_{N_i}) | agent i's local policy: it acts on its own and its neighbours' states only |
| r_i(s_i, a_i) | agent i's local reward |
| r = (1/n) Σ_i r_i | the global reward, the average of the local ones |
| V(s) = (1/n) Σ_i V_i(s) | the global value, the average of the local values |

Everyone maximises one shared objective, E[Σ_t γ^t r(s_t, a_t)], with γ ∈ (0, 1) the discount. **The setting is fully cooperative.** It has no social dilemma and no overseer.

### 1.2 The extended value function and its bound

The extended value function is agent i's value when it knows only its κ-hop neighbourhood. It averages over everyone further away:

  V_i(s_{N_i^κ}) = E_{s_{N_{−i}^κ}} [ Σ_t γ^t r_i^t | s^0_{N_i^κ} = s_{N_i^κ} ]

- The bound is DMPO's Theorem 1 and NMI's Theorem 3: |V_i(s) − V_i(s_{N_i^κ})| ≤ r_max γ^κ / (1 − γ). Here r_max is the largest reward.
- **Averaged** (Remark 1): |V(s) − (1/n) Σ_i V_i(s_{N_i^κ})| ≤ r_max γ^κ / (1 − γ).
- **Policy gradient** (DMPO Theorem 2, NMI Theorem 4). Agent i builds its value estimate only from its κ-hop neighbours' extended values: Ṽ_i = (1/n) Σ_{j ∈ N_i^κ} V_j. The gradient this gives is close to the true one:

  |g_i − g̃_i| ≤ γ^{κ−1}/(1 − γ) · [1 − (1 − γ²) |N_i^κ|/n] · r_max g_max

  Here g_max bounds the size of the gradient.

**Where the bound comes from.** DMPO's proof cites an earlier result for the Q-function (Qu, Wierman & Li 2020, L4DC; arXiv 1912.02906), which assumes that each agent's next state depends only on its neighbours.

The point is that **what happens far away matters exponentially less, in κ**. So local information plus a little communication recovers almost all of the central answer.

### 1.3 The ξ-dependent system (NMI only; DMPO has only the exact factorisation)

- **Definition 1: an independent networked system (INS).** The transition factorises into local pieces: p(s′ | s, a) = Π_i p_i(s′_i | s_{N_i^κ}, a_i).
- **Definition 2: ξ-dependent.** Some INS p̄ exists that is close to the true dynamics p, measured by total variation (TV) distance:

  sup_{s,a} D_TV( p(·|s,a) ‖ p̄(·|s,a) ) = sup_{s,a} ½ Σ_{s′} |p(s′|s,a) − p̄(s′|s,a)| ≤ ξ

  - TV distance is the largest gap in probability that the two systems give to any event. It runs from 0 (identical) to 1.
  - So ξ measures how far the real system is from one that splits cleanly into local parts.
- **The model error splits in two:** D(p‖p̂) ≤ D(p‖p̄) + D(p̄‖p̂). The first part is the "dependency bias" (at most ξ). The second is the "independence-approximation error".
- **ξ enters the return bound** (Corollary 1) as one extra additive term, 2 r_max γ ξ / (1 − γ)², on top of the model-error and policy-shift terms of Theorem 2.
- **They measured ξ** in two testbeds (Fig. 2d, 5 seeds). The caption says "ξ ≥ 0.2 in Catch-up and ξ ≥ 0.8 in Real Power-Net" and calls these "small positive values". The wording is ambiguous; I'd read the box plots before quoting the numbers.

### 1.4 How they present the work

1. **Theory first**, for the whole problem class, with no testbed in it: the definitions, then the theorems.
2. **One property that decides fit:** ξ, which says how close a system is to splitting into local parts.
3. **Then many testbeds with the same method**, growing in size ("in a progressive manner"):
   - DMPO: CACC Catch-up and Slow-down (8 agents), Flow Figure Eight (14) and Ring Attenuation (22), ATSC Grid (25).
   - NMI adds:
     - ATSC Monaco (28 heterogeneous traffic lights);
     - ATSC New York (436 intersections in the text, 432 traffic lights in the Fig. 5 caption);
     - Power Grid (20 and 40 agents);
     - Real Power-Net 141-, 322- (22 agents) and 421-bus (199 agents);
     - Pandemic Networks.
4. **Fixed reference points:**
   - a centralised learner, CPPO, which is "κ arbitrarily large" (DMPO §VI-A);
   - an independent learner, DPPO or IL, which ignores neighbours;
   - communication-based methods (IC3Net, CommNet, DIAL, NeurComm and others).
5. **Ablations on the knobs the theory names:** rollout length T (theorems 1–2) and neighbourhood order κ (theorems 3–4). Each figure says which theorem it checks.
6. **5 seeds, reported as box plots** (median, quartiles, range).

### 1.5 How they report communication cost

- **DMPO (2022):** in words only. "Larger κ leads to better results … but also more communication costs", and a κ ablation (κ = 1–3) on reward and model error.
- **NMI (2024):** three things.
  - **The size of a local message:** "the minimum local message size O(|s_i|)", meaning each agent sends roughly its own state and nothing more.
  - **Cost against centralised methods:** "only about 5–35% of that in centralized methods (more details in Supplementary Table 3)".
  - **A κ ablation:** neighbour order 1st to 5th, on reward and model error.

  Communication is reported next to performance, never instead of it ("Higher performance / Less communication", Fig. 3a).

**What we take from this.** We copy the structure:
- one problem class;
- one locality number that plays ξ's role;
- one κ-hop bound;
- central and independent reference points;
- many testbeds;
- communication reported as a share of the central cost.

We do not copy the setting. Theirs is cooperative control with no dilemma and no adversary. Ours has self-interested agents and an overseer layer on top.

---

## 2. The setting: N agents in a sequential social dilemma

Most of this repeats the one-overseer note, so it is brief.

| Symbol | Meaning |
|---|---|
| i = 1, …, N | the agents |
| t = 0, 1, … | time steps (Harvest: 1,000 per episode) |
| a_{i,t}, r_{i,t} | agent i's action and its own reward at step t |
| ℓ_{i,t} ∈ L | where agent i is at step t. L is the set of places: map cells in a gridworld, or nodes of a trading graph |
| v_{i,t} ∈ {0, 1} | 1 if agent i broke the rule at step t |
| g | what one violation gains the agent, in the game's units (Harvest: about 1 apple) |
| D_{i,t} ⊆ L | the **damage footprint**: the places whose resources the violation harms (Harvest: the 12 cells within two squares of the eaten apple, whose regrowth falls) |
| W | welfare: total reward of all agents per episode (Harvest: apples eaten; also, does the orchard survive?) |

A sequential social dilemma (Leibo et al. 2017, arXiv 1702.03037) is a Markov game in which each agent's own best policy, when all agents follow theirs, gives lower W than if all held back.

## 3. Definitions: M overseers sharing a budget

### 3.1 Regions, budgets and checks

| Symbol | Meaning |
|---|---|
| m = 1, …, M | the overseers |
| R_m ⊆ L | overseer m's **region**: the places it is responsible for. Regions do not overlap and together cover L |
| A_{m,t} = { i : ℓ_{i,t} ∈ R_m } | the agents inside region m at step t. In a non-spatial game, R_m is a fixed set of agents and A_m does not change |
| H | the **overseer graph**: m and m′ are joined if their regions touch (an agent can step from one to the other) |
| H_m^κ | overseers within κ hops of m in H, including m. This is the analogue of N_i^κ |
| k_m | overseer m's checks per step |
| K = Σ_m k_m | the **total budget**, the same for every design being compared |
| C_{m,t} ⊆ A_{m,t}, \|C_{m,t}\| ≤ k_m | the agents m checks at step t. **It can only check agents inside its region** |
| d_m = k_m / E[\|A_{m,t}\|] | **budget density**: checks per agent present in region m |
| q_{i,t} | the chance agent i is checked at step t. Under random checks it is min(1, k_m / \|A_{m,t}\|) for the region i is in |

### 3.2 What overseer m sees

A check reads the last w steps of the agent's record. **The overseer only sees the steps the agent spent inside its own region:**

  ρ_m(i, t) = { (o_{i,τ}, a_{i,τ}, v_{i,τ}) : t − w < τ ≤ t and ℓ_{i,τ} ∈ R_m }

| Symbol | Meaning |
|---|---|
| w | the window: how many past steps a check reads |
| o_{i,τ} | what agent i saw at step τ (needed to judge the rule; Harvest judges an eat on the map the agent saw) |
| ρ_m(i, t) | the part of i's last w steps that overseer m can see |
| α | miss rate: the chance a check fails to see a violation that is in its window |
| β | false-alarm rate: the chance a check wrongly flags an honest action |

A violation committed in region m, by an agent that then crosses into region m′ before m checks it, is **invisible to both overseers**: m no longer sees the agent, and m′ never saw the act. This is the new failure mode. Crossing a boundary works like erasing evidence, in Gans & Holden's (2026) sense.

### 3.3 Optional messages

Each step, overseer m may send messages to its neighbours in H.

| Symbol | Meaning |
|---|---|
| μ_{m→m′,t} | a message from m to a neighbour m′, for example "agent 4's unchecked record" or "agent 4's suspicion score is 0.8" |
| σ_m(i) | overseer m's **suspicion score** for agent i: a decaying count of past catches, as in the existing `history` chooser |
| κ | how far information may travel: a record or score survives at most κ hand-overs between overseers |
| c_msg | the cost of sending one number to one neighbouring overseer |

**Two simple protocols:**
- **Hand-over:** when agent i leaves R_m for R_{m′}, m sends i's unchecked record and score to m′. A record is dropped after κ hand-overs.
- **Broadcast:** each step, m sends its scores to every overseer in H_m^κ.

With κ = 0 there are no messages: these are independent overseers, the analogue of DPPO or IL in §1.4.

### 3.4 The consequence

The same as with one overseer. A caught agent loses F:

  r̃_{i,t} = r_{i,t} − F · caught_{i,t},  caught_{i,t} = 1 if any overseer catches i at step t

F is in the game's units (apples). Other consequences (a freeze, a blocked listing) fit the same slot. Harvest showed that a 50-step freeze does not work against learners (tracker finding 5).

### 3.5 Cost

  Cost = Σ_t Σ_m [ |C_{m,t}| · (c_check + c_hist · w) + c_msg · (numbers m sends at step t) ]

| Symbol | Meaning |
|---|---|
| c_check | the fixed cost of one check |
| c_hist | the cost of reading one step of history (for an LLM judge: tokens) |
| λ | how much one unit of cost is worth in welfare units |

**The central comparison point for communication.** A central overseer must first collect every agent's record: N · w record-steps per step, sent to one place. We report the decentralised message cost as a percentage of that, in the same way NMI reports "5–35% of centralised" (§1.5).

### 3.6 The objective

The overseer settings are

  θ_O = ( regions {R_m}, budgets {k_m}, how each C_{m,t} is chosen, w, F, message protocol, κ )

The design problem is:

  maximise over θ_O:  W(θ_O, π*) − λ · Cost(θ_O, π*)
  subject to:
  - π*_i ∈ argmax over π_i of E[ Σ_t γ^t r̃_{i,t} | π_i, π*_{−i}, θ_O ] for every agent i (each agent best-responds to the overseers and to the other agents);
  - Σ_m k_m ≤ K.

| Symbol | Meaning |
|---|---|
| π* | the agents' joint reply. For RL learners, what IPPO converges to; for LLMs, what they do when told the rules |
| W(θ_O, π*) | welfare when the agents play π* under settings θ_O |

**Two versions.**
- **(a) A team of overseers designed together** for global W. This is the version above.
- **(b) Each overseer maximises its own region's welfare**, W_m − λ · Cost_m. When welfare adds up over regions (W = Σ_m W_m), version (b) is DMPO's local-reward set-up, with overseers in place of DMPO's agents (§4).

The gap between (a) and (b) is the harm a violation in region m does elsewhere, which overseer m ignores in (b). Section 5 measures that with ζ.

### 3.7 The central overseer is the case M = 1

Set M = 1, R_1 = L (the whole map), k_1 = K and no messages. Then:
- every agent is always in region 1, so ρ_1(i, t) is i's full last w steps;
- the cost is the one-overseer cost;
- the problem is exactly the one in `2026-11-21_general_formulation.md`.

**So every Harvest result so far is the M = 1 baseline.** For example:
- 1 check per step with a perfect inspector, an average penalty of 2 apples and w = 1 gives 515 apples per episode, against 530 for checking all 7 agents and 68 for no overseer (tracker finding 4);
- at an average penalty of 0.5 apples nothing works at any k (finding 3).

Every decentralised design is compared with the M = 1 overseer at the same K.

## 4. The overseer layer as a networked system

This maps the decentralised overseers onto §1, so the formulation reads as a member of the same family as DMPO.

| DMPO / NMI | Decentralised overseers |
|---|---|
| node i of G | overseer m, node of H |
| local state s_i | region m's resource state, plus the records ρ_m and the scores σ_m |
| local action a_i | the check set C_{m,t} plus the outgoing messages |
| local reward r_i | W_m − λ · Cost_m (version b) |
| κ-hop neighbourhood N_i^κ | overseers within κ hops, H_m^κ |
| extended value V_i(s_{N_i^κ}) | overseer m's value using its κ-hop neighbours' information |
| ξ: TV distance to a factorised system | ξ_R: TV distance to a **region-factorised** system (below) |
| CPPO (central critic) | the central overseer, M = 1 |
| DPPO / IL (no communication) | independent overseers, κ = 0 |

**Region-factorised system.** In such a system:
- no agent ever leaves its region;
- no violation harms another region;
- each region's next state depends only on that region: p̄(s′ | s, a) = Π_m p_m(s′_{R_m} | s_{R_m}, a_{A_m}).

Define

  ξ_R = sup_{s,a} D_TV( p(·|s,a) ‖ p̄(·|s,a) )

the same quantity as NMI's Definition 2, with regions in place of agents.

**Two differences from DMPO** keep this from being a copy:
1. **The overseers' environment contains learners that react to them.** So the problem is two-level: overseers set the rules and the agents best-respond. That is a Stackelberg-style problem, not a plain networked MDP.
2. **DMPO's bound relies on exponential decay of influence with graph distance** (Qu et al. 2020). That holds when dynamics are local. Agents who roam carry their influence with them, so it may fail. ξ_R is what tells us how badly.

ξ_R is the clean definition, but it is hard to measure directly. Section 5 gives two numbers that can be measured.

---

## 5. Locality: the number that should decide whether decentralising works

### 5.1 Two measurable parts

| Number | Definition | How to measure | 0 means | 1 means |
|---|---|---|---|---|
| **χ (chi), crossing rate** | the chance per step that an agent moves into a different region | count region changes in evaluation rollouts, divided by agent-steps | agents never leave their region | agents change region every step |
| **ζ (zeta), harm containment** | the share of a violation's harm that lands in the region where it was committed | Harvest: share of the damage footprint D_{i,t} inside the same region; Clean Up: share of the orchard's lost regrowth inside the region; in general, counterfactual rollouts with and without the violation | all harm lands elsewhere | all harm stays in the region |

**Locality for a window of w steps.** Λ_w is the share of violations whose evidence stays in one region for the whole window (no crossing in the next w − 1 steps) and whose harm stays there too. With crossings independent of harm,

  Λ_w ≈ (1 − χ)^{w−1} · ζ

The **non-locality**

  ξ_loc = 1 − Λ_w

plays ξ's role:
- ξ_loc near 0 means the system splits cleanly into regions;
- ξ_loc near 1 means it does not.

Reported per testbed, it says in advance whether local overseers should be enough.

**Expected values.** None of these has been measured yet.
- **Harvest Open** (16×22 map, 6 apple patches, all 7 agents spawn along the bottom edge, 11×11 view): ζ close to 1 for a left/right split, because the damage footprint has radius 2 and each patch lies on one side. χ is probably high, because agents roam the whole map. To measure from stage 1 rollouts.
- **Harvest with rooms** (walls added with `map_ASCII`): ζ close to 1 and χ close to 0. This is the cleanest locality dial, because the mechanics do not change.
- **Clean Up:** ζ is low. Regrowth everywhere depends on one global dirt fraction, so not cleaning in the river harms every orchard region.
- **A non-spatial commons** (Willis et al. 2026; GovSim), split into M fixed groups: χ = 0, because groups are fixed, and ζ ≈ 1/M, because one shared pool takes all the harm.

### 5.2 A first result: what the window does to catching (derived, under stated assumptions)

**Assumptions:**
- random checks;
- every agent has the same chance q of being checked each step (budgets match occupancy, d_m = K/N);
- an agent leaves its region with chance χ each step and does not come back within the window;
- the checked window is cleared after a check, as in `harvest_overseer.py`.

Then a violation committed in region m is caught with chance:

  **Central overseer (M = 1):** P_C = (1 − α) · [1 − (1 − q)^w]

  **Independent overseers (κ = 0):** P_0 = (1 − α) · q · [1 − ((1 − q)(1 − χ))^w] / [1 − (1 − q)(1 − χ)]

| Symbol | Meaning |
|---|---|
| q | chance an agent is checked in a given step (K/N under matched random checks; 1/7 in Harvest with K = 1) |
| P_C, P_0 | chance a violation is ever caught, by a central overseer and by independent regional ones |

**Why.** Under the central overseer, the violation is caught at the first check in the w steps that read it. Under independent overseers, the check at lag j must also find the agent still in region m, which has chance (1 − χ)^j. Summing q(1 − q)^j (1 − χ)^j over j = 0, …, w − 1 gives P_0.

**What this says:**
- **With w = 1, P_0 = P_C = (1 − α) q.** Decentralising loses nothing from crossings when checks look only at the current step, which is the setting of every Harvest run so far. The only loss left comes from budgets not matching where the agents are (§6, D2).
- **With w > 1 and χ > 0, P_0 < P_C.** As w grows, P_C tends to (1 − α) while P_0 tends to (1 − α) q / (q + χ − qχ). When agents cross regions much more often than they are checked (χ ≫ q), independent overseers catch only about a q/χ share of what a central one catches.
- **This interacts with C4.** Violations that only show over many steps need a large w, and a large w is exactly where boundaries lose the most. **Prediction:** decentralising will be hardest in games like Clean Up, where cheating builds up slowly.

The formula is exact only under its assumptions. Agents that return to a region, or crowd unevenly, change it, so it should be checked against simulation before it is relied on.

---

## 6. The claims for the decentralised case

### 6.1 C1–C4 restated (each still a prediction for M > 1)

**C1-D. Deterrence, region by region.** Breaking the rule stops paying in region m when

  P_m · F ≥ g_m

| Symbol | Meaning |
|---|---|
| P_m | the chance a violation in region m is caught: (1 − α) d_m when w = 1, or P_0 in §5.2 when w > 1 |
| g_m | what a violation gains in region m |

Deterrence holds everywhere only if it holds in the **weakest region an agent can reach**: min_m P_m · F ≥ g. With M = 1 this is C1 exactly.

The evidence for M = 1 is that Harvest's switch lies between an average penalty of 0.5 and 1 apple, at any k (finding 3).

**C2-D. The cost to honest agents, by region.** An honest agent in region m loses d_m β F per step on average.
- If budget densities differ across regions, honest agents in heavily watched regions pay more. This is a fairness cost that a central overseer with uniform checks does not have.
- At a fixed average penalty d_m F, the loss does not depend on k_m, as in C2.

**C3-D. Targeting needs a good local signal.** Targeting inside region m beats random only if the catch rate E[q | v = 1] is above d_m.
- **Signals built from local damage mislead when ζ is low:** harm seen in region m was caused by agents of another region.
- This is the decentralised version of Harvest's location result (location targeting caught 5% of over-harvests, against 28% for random; finding 7).
- It is also the version of Makins et al.'s (2026) "fragmentation effect" for overseers: as more agents coordinate an attack, per-agent monitors become less likely to catch any of them.

**C4-D. Violations a check can't see.** Effective detectability is detectability multiplied by evidence locality, (1 − χ)^{w−1} under the §5.2 assumptions. Slow violations are hit twice, as §5.2 says.

### 6.2 New claims

**D1 (prediction). At equal total budget K, local overseers match the central one when cheating and its damage are local and agents rarely cross regions.**

Concretely, with matched budgets (d_m = K/N), the decentralisation gap

  Δ_dec = η_central(K) − η_dec(K)

should be:
- near 0 when ξ_loc is near 0;
- growing in χ when w > 1;
- growing in 1 − ζ under version (b), where each overseer optimises its own region.

η is defined in §7.

**Testable splits:**
- **Harvest Open, M = 2 vs M = 1, w = 1:** Δ_dec ≈ 0, from §5.2.
- **The same at w = 5 or 10:** Δ_dec > 0 if χ is large.
- **Harvest rooms at w = 10:** Δ_dec ≈ 0 again.
- **A non-spatial commons with M groups (χ = 0, ζ = 1/M):** Δ_dec ≈ 0 for a fixed rule (version a). Under version (b), overseers under-spend, because each sees only 1/M of the harm.

This last split separates the two parts of locality.

**D2 (prediction). Agents exploit boundaries and migrate to under-watched regions.** Three things should predict it:
1. **Unequal budget density.**
   - With fixed k_m and free movement, cheating should concentrate where d_m (1 − α) F < g.
   - Arrivals lower d_m further, because the same k_m is spread over more agents.
   - **Prediction:** fixed per-region budgets are fragile when travel between regions is cheap. Budgets that follow occupancy (k_m ∝ |A_{m,t}|, reallocated each step through messages) remove the effect.
   - **Measure:** the spread of d_m across regions, and the share of violations by region against each region's share of agents.
2. **Boundary-hopping (only when w > 1).**
   - Agents should violate more near borders, then cross.
   - **Measure:** violation rate against distance to the nearest border, and the share of violations that are never seen because of a crossing (from the §5.2 formula against simulation).
3. **Being able to see or predict checks.**
   - As in C3, a predictable chooser or a visible budget per region gives the agents a target.
   - This is the worst-case setting in §7.

**Travel time relative to w** should also matter. If crossing takes longer than w steps, hopping cannot launder a violation.

**D3 (bound under assumptions, plus a conjecture). Sharing between neighbouring overseers recovers most of the central performance, at a communication cost.**

*The part with an honest bound: catching.*
- **Protocol:** hand-over or broadcast with range κ (§3.3).
- **Assumption:** the chance of crossing a boundary in any step is at most χ, whatever happened before.
- **When a violation is lost:** only if the agent makes more than κ crossings before it is checked within the window. Then

  0 ≤ P_C − P_κ ≤ (1 − α) · P(more than κ crossings in w − 1 steps) ≤ (1 − α) · C(w−1, κ+1) · χ^{κ+1} ≤ (1 − α) · ((w − 1) χ)^{κ+1} / (κ + 1)!

- **Proof sketch:**
  - The difference is Σ_j q(1 − q)^j · P(more than κ crossings by lag j), which is at most P(more than κ crossings in w − 1 steps).
  - More than κ crossings needs some set of κ + 1 crossing steps among the w − 1.
  - There are C(w−1, κ+1) such sets, and each has chance at most χ^{κ+1}.
  - The union bound finishes it.

| Symbol | Meaning |
|---|---|
| P_κ | catch chance with sharing of range κ |
| C(w−1, κ+1) | the number of ways to choose κ + 1 steps from w − 1 |
| (w − 1) χ | expected crossings within one window |

**What the bound says.**
- This is the analogue of |V_i(s) − V_i(s_{N_i^κ})| ≤ r_max γ^κ / (1 − γ): **the loss falls geometrically in κ once (w − 1) χ < 1**, that is, when an agent makes fewer than one crossing per window on average.
- (w − 1) χ plays the role of γ.
- **Cost of hand-over:** about N χ messages per step, each carrying a record of up to w steps. That is at most N χ w record-steps per step, against N w for central collection, so **at most a χ share of the central cost** (for example, 5% if agents cross a boundary once every 20 steps).
- **Cost of broadcast:** it grows with |H_m^κ|, which is 2κ + 1 regions on a line and 2κ² + 2κ + 1 on a grid of regions.
- **Prediction:** κ = 1 recovers most of the central catch rate in Harvest Open at w = 10, at a few percent of central collection cost. To be checked against the measured χ.

*The part that is only a conjecture: the objective.*
- Suppose each overseer optimises the welfare of its κ-hop neighbourhood (Σ_{m′ ∈ H_m^κ} W_{m′}), the analogue of Ṽ_i.
- Suppose also that the harm a violation does more than d regions away is at most a ρ^d share of its total, for some ρ < 1.
- Then the welfare the decentralised team ignores should fall like ρ^{κ+1}, as in DMPO's Theorem 4.
- **I have not proved this.** It needs the exponential-decay property (Qu et al. 2020), which assumes local dynamics. Roaming agents and global resources (Clean Up's dirt) break it, and a two-level problem with learning agents is not covered by their proof.

---

## 7. The bounded evaluation scorecard (decentralised version)

The same scorecard for every testbed, as in `2026-11-21_way_forward.md` §2, extended for M > 1.

| Piece | What we measure | Comes from |
|---|---|---|
| **Floor** W_0 | welfare with no overseer (Harvest: 68 apples, orchard dead) | AI Control's "no protocol" reference points (Greenblatt et al. 2024) |
| **Ceiling** W* | a **central, all-seeing, perfect** inspector: M = 1, every agent checked every step, the full record, α = β = 0, at the same F (Harvest at w = 1: 530 apples) | same; NMI's CPPO plays this role ("highest theoretical performance") |
| **Score** η | η = (W − W_0) / (W* − W_0), the share of the floor-to-ceiling gap recovered. Harvest, M = 1, K = 1: (515 − 68) / (530 − 68) = 97% | our oversight efficiency; AI Control's safety/usefulness |
| **Decentralisation gap** Δ_dec | η_central(K) − η_dec(K), at the same K | NMI compares with CPPO at the same task |
| **Cost** | checks × (c_check + c_hist · w), plus messages × c_msg. **Also communication as a % of central collection cost (N · w record-steps per step)** | Monitored MDPs (Parisi et al. 2024): looking has a cost; NMI's "5–35% of centralised" |
| **Harm to honest agents** | wrongful penalties per honest agent per step (≈ d_m β F), by region; honest work lost | AI Control's usefulness; C2-D |
| **Weaker overseer** | η as one thing gets worse: K and its split k_m; w; α and β; model size of an LLM overseer; plus the decentralised axes, M, κ and region design | Engels et al. 2025: oversight success against the capability gap |
| **Worst case** | η against agents that can see or predict checks, know the per-region budgets, and are trained or told to dodge, including moving to unwatched regions and hopping boundaries. Also distributed violations split across regions | AI Control's red team; Wang et al. 2019 (DeDOL: train the attacker, then the defender, in a double-oracle loop); Makins et al. 2026 (distributed attacks) |

**Reporting rules, copied from §1.4–1.5:**
- show η and cost side by side;
- run the κ and w ablations and say which formula each one tests;
- give the measured ξ_loc (χ and ζ) for each testbed before its results;
- always include the M = 1 and κ = 0 reference points.

## 8. Literature tied to each piece

All entries below were checked against the arXiv abstract page or Crossref on 21 Nov 2026, for title, first author and year.

| Paper | What we use |
|---|---|
| Du, Ma, Liu, Lin, Dong, Wang, Yang 2022, *Scalable Model-based Policy Optimization for Decentralized Networked Systems*, IROS 2022; arXiv 2207.06559 | networked MDP, κ-hop neighbourhood, the extended value function and the γ^κ bound (§1) |
| Ma, Li, Du, Dong, Yang 2024, *Efficient and scalable reinforcement learning for large-scale network control*, Nat. Mach. Intell. 6:1006–1020; doi 10.1038/s42256-024-00879-7 | the ξ-dependent system; theory first, then many testbeds; communication as a % of centralised (§1, §5) |
| Qu, Wierman, Li 2020, *Scalable Reinforcement Learning for Multi-Agent Networked Systems*; arXiv 1912.02906 | the exponential-decay property behind the γ^κ bound (§1.2, D3) |
| Greenblatt, Shlegeris, Sachan, Roger 2024, *AI Control: Improving Safety Despite Intentional Subversion*, ICML 2024; arXiv 2312.06942 | floor, ceiling, safety/usefulness, red team (§7) |
| Engels, Baek, Kantamneni, Tegmark 2025, *Scaling Laws For Scalable Oversight*, NeurIPS 2025; arXiv 2504.18530 | oversight success as a function of the overseer–agent capability gap; the "model size" axis (§7) |
| Parisi, Mohammedalamen, Kazemipour, Taylor, Bowling 2024, *Monitored Markov Decision Processes*; arXiv 2402.06819 | rewards (here: violations) are seen only when a monitor is on, and monitoring can cost (§3.5, §7) |
| Gans & Holden 2026, *When Does Randomized Oversight Align AI Agents That Can Conceal?*; arXiv 2609.38262 | rare random audits deter if evidence survives and draws can't be learned; crossing a boundary as evidence erasure (§3.2, D2) |
| Makins, Angelini, Shams, Phuong 2026, *Multi-Agent AI Control: Distributed Attacks Hamper Per-Instance Monitors*; arXiv 2607.07368 | the fragmentation effect against per-agent monitors (C3-D, worst case) |
| Wang, Shi, Yu, Wu, Singh, Joppa, Fang 2019, *Deep Reinforcement Learning for Green Security Games with Real-Time Information*, AAAI 2019; arXiv 1811.02483 | spatial patrols against poachers, DeDOL double-oracle training of attacker and defender (worst case) |
| Avenhaus, von Stengel, Zamir 2002, *Inspection games*, Handbook of Game Theory vol. 3, ch. 51, pp. 1947–1987; doi 10.1016/S1574-0005(02)03014-X | the classic inspector with limited inspections; C1 is its learning, many-agent version |
| Leibo et al. 2017, *Multi-agent Reinforcement Learning in Sequential Social Dilemmas*; arXiv 1702.03037 | what an SSD is (§2) |

## 9. Open questions

1. **Region design.** Fixed regions, regions that follow the agents, or overlapping regions? This note assumes fixed, non-overlapping regions.
2. **Can the M overseers be learned with a DMPO-style method?** Each overseer would be a policy with a κ-hop extended value. The two-level structure (agents adapt) is the obstacle.
3. **Does ξ_R or ξ_loc predict Δ_dec across testbeds?** This is the paper's central test, and it needs at least 3 testbeds with different ξ_loc.
4. **Who pays when two overseers both catch the same agent for the same act?** Assumed here: F is paid once.
5. **Measure χ in the existing Harvest runs** before choosing w for the D1 experiments.
