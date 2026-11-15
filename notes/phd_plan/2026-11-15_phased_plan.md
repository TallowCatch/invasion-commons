# Phased plan: a budgeted overseer in SocialJax, then a weaker and costed overseer (15 Nov 2026)

This plan builds on the six steps in the 2 Nov deck (`notes/slides/2026-11-02_lit_review_progress/`, slide 13) and adds Yali's three notes from the meeting:
1. **How strong does the overseer need to be?** Test its power by size and intelligence: can a smaller or less capable overseer still keep the agents in check? Open question: how do we bound the evaluation?
2. **Look at BazaarBench** (a C2C market) and put a cost on the overseer in tokens (for example 1 unit against 10 units).
3. **Plan the agents and experiments in more detail:** how the overseer operates, and random against targeted checks.

It replaces the ordering in `2026-10-27_calibrated_overseer_plan.md`. Plain language, MARL framing.

---

## The project in two sentences

An overseer watches a group of learning agents in a shared-resource game but can only look at some of them, some of the time, and each look costs something. We build it, measure how much sight, intelligence and spending it needs to keep the group cooperating, and make it robust to agents that learn to dodge it.

## The three dials of overseer power (Yali's note 1)

Every experiment is a point on three dials. Keeping them separate answers "how do we bound the evaluation?".

| Dial | What it means | How we turn it |
|---|---|---|
| **Sight** | How many agents it can check per step | k of N (k = 0, 1, 2, 3 … 7) |
| **Smarts** | How capable it is compared with the agents | RL: network size and training budget. LLM: model size (a small model watching a large one). |
| **Spend** | What each check costs | 1 unit per check by default; for an LLM overseer, tokens read and written. Budgets of 1, 10, 100 units per step. |

**A proposed answer to "how do we bound the evaluation?"** (to agree with Yali):
- **Upper bound for the overseer:** an overseer that sees every agent every step (k = N, unlimited spend). Nothing should beat it, so it shows the most oversight can achieve.
- **Lower bound:** no overseer. The gap between the two is what there is to recover; report every result as the share of that gap recovered.
- **Worst case for the overseer:** agents trained against it (a red team), not scripted cheaters. This is the AI-control convention (Greenblatt et al. 2024): report safety against the best attack you can find.
- **Fix the currency:** cost per check in one unit, so "random with k = 3" and "targeted with k = 1 plus an expensive look" can be compared on one cost axis. The main plot is cooperation against cost (a frontier), not against k alone.
- **Bound the capability gap explicitly:** measure overseer and agent capability on one scale (Engels et al. 2025 use an Elo score per role) and report results against the gap.

## What the overseer does each step (the operating loop)

This is the same in every phase; only its parts change.

1. **Choose whom to check:** k agents, under a budget.
2. **Look:** see what each checked agent did in the last w steps. An LLM overseer reads a text summary, and that reading is what costs tokens.
3. **Decide:** did that agent break the rule? A cheap overseer can act on suspicion without looking closely, which causes wrong punishments.
4. **Apply a consequence:** a reward penalty, or freezing the agent for T steps. SocialJax Harvest already has a freeze mechanism, used for zapping, that we can reuse.
5. **Update what it believes:** a suspicion score per agent, used by targeted strategies in step 1.

**The rule, in Harvest:** in SocialJax Harvest Open an apple only regrows if there are other apples within two squares, and the more neighbours the faster it regrows (`harvest_open.py`, `regrow_apple`). An apple with no neighbours never comes back. So the rule the overseer enforces is: **don't eat an apple that has fewer than m apples within two squares.** This is concrete, checkable from the grid, and it is exactly the move that destroys the orchard. m is set in phase 1 from a Schelling diagram.

**What we measure, every time:**
- **Over-harvesting missed:** rule-breaking steps that went unpunished.
- **Normal harvesting wrongly stopped:** punishments of agents who followed the rule (borrowed from Casheekar & Tangirala 2026).
- **Orchard health:** apples alive over time, and whether the orchard survives the episode (Perolat et al. 2017's sustainability).
- **Group welfare:** total reward, and equality across agents.
- **Cost:** checks used, and tokens for an LLM overseer.

---

## The phases

### Phase 0: groundwork (this week)

- **CREATE:** check the smoke job (job 38280051, `scripts/create/smoke_harvest.sbatch`) ran; record steps per second on one GPU, which sets the size of every sweep.
- **Confirm the rule:** read `harvest_open.py` properly (rewards, freeze, `shared_rewards`), and pick individual rewards so it is a real dilemma.
- **Decide the cost unit** with Yali (1 check = 1 unit).
- **Read** for phases 1–2: AI Control (Greenblatt), Green Security Games (Wang 2019), Blindfold (Zhu). These are already on the reading list.

### Phase 1: the overseer in Harvest with scripted bots (deck step 1)

This runs on a laptop or one GPU, with no training.

- **Agents (scripted):**
  - **restrained:** follows the rule;
  - **greedy:** eats everything;
  - **sneaky:** greedy only when it was not checked in the last few steps, as a first stand-in for dodging.
- **Overseer strategies:**
  - none;
  - random k;
  - round-robin;
  - targeted (highest suspicion);
  - all-seeing (k = 7, the upper bound).
- **Consequences:** a reward penalty of size F, or freezing for T steps.
- **Sweep:** k × consequence size, for each mix of bots (0–7 greedy). This gives the Leibo-style heatmap planned on the 2 Nov slides.
- **Output:**
  - a Schelling diagram with and without the overseer, showing that it removes the dilemma;
  - the heatmap of orchard survival against k and penalty.
- **Reuse:** our Fishery results (L4, R1a) say the expected penalty (chance of a check × penalty) matters most. Phase 1 checks whether that holds in space.

### Phase 2: RL agents trained against it (deck step 2)

This needs CREATE GPUs.

- **Agents:** IPPO from SocialJax, 7 agents, individual rewards.
- **Two starts:**
  - (a) the overseer is on from the start;
  - (b) agents first learn to over-harvest, then the overseer is switched on ("recovery", Casheekar).
- **Question:** what is the smallest k (and lowest cost) that saves the orchard?
- **Switch-off test:** remove the overseer after training. Does over-harvesting come back? Baumann et al. 2020 found Prisoner's Dilemma collapses; our R1b found selfish strategies comply but don't change.

### Phase 3: the overseer sets itself up (deck step 3)

- Watch first, with no penalty, to estimate how much over-harvesting gains an agent. Then set the penalty just above that, as in our Fishery L3/L4, where the prediction held.
- **Question:** does the prediction still hold with learning agents and space?

### Phase 4: agents try to beat it, random against targeted (deck step 4; Yali's note 3)

This is the core contribution.

- **What agents can see about the overseer:**
  - nothing;
  - who was checked last step;
  - the full checking pattern.
  - (Blindfold, Zhu 2026: audits only stay effective if agents can't predict them.)
- **Overseer strategies:**
  - **random;**
  - **targeted by suspicion;**
  - **randomised targeting:** a mix that is hard to predict, the security-game idea (Sinha et al. 2018; Wang et al. 2019);
  - **learned:** an RL overseer that picks k agents, rewarded for welfare minus cost.
- **Red team loop:** train agents against the overseer, then improve the overseer against those agents, and repeat (the double-oracle method from Wang et al. 2019).
- **Expectation from our own data:** in T1, always checking the largest reporter caught 25% of liars while random checks caught 100%. Predictable targeting can backfire. The question is when targeting beats random once agents adapt.

### Phase 5: a weaker overseer and a costed overseer (Yali's notes 1 and 2)

**5a. Smarts, RL version:** shrink the overseer (a smaller network, less training) while the agents stay the same, and the reverse. Plot cooperation against the capability gap.

**5b. Smarts, LLM version:** use OpenSocialJax, with LLM players and an LLM overseer (deck step 5).
- Pair a **small overseer model** with **large player models**, and the reverse. This is the "trusted weaker monitor" setting from AI control (Greenblatt et al. 2024).
- Measure capability on one scale, following Engels et al. 2025:
  - they fit a "guard" and a "Houdini" Elo per game;
  - at a general Elo gap of 400, nested oversight succeeds 9.4–51.7% of the time depending on the game.
  - Our version asks the same question in a sequential social dilemma, which they list as a limitation ("single-step oversight").

**5c. Spend:** make every check cost tokens. The overseer reads a window of an agent's history, and a longer or more careful read costs more.
- Budgets of 1, 10 and 100 units per step.
- Plot the frontier of cooperation (or harm prevented) against tokens spent.
- Let the overseer choose between a cheap glance at many agents and a careful read of a few.

**5d. A second testbed, BazaarBench (Yali's group):**
- **The paper:** Wang, Shi, …, Du, Bibi 2026, arXiv:2610.06748.
- **The market:** 100 LLM agents trade eBay items for 30 simulated days.
- **Failures it records:** over-committing one item to several buyers, listing items an agent doesn't own, overstating condition, and others.
- **No live overseer today:** it has platform checks and an offline "idealized inspection audit" that changes only scores, not behaviour.
- **What we add:** an online overseer with a token budget that checks some listings and deals each simulated day and applies consequences (delisting, reputation penalty, freeze). This is the "cost at 1 unit, 10 units" experiment in a realistic market.
- **Cost warning:** their GPT-5 judge alone used about 2.3 billion input tokens (about $6,500 estimated) for 48 runs. We need small markets (for example 20 agents, 7 days) and open models served on CREATE GPUs, and we should talk to Ziyan Wang and Shuqing Shi before building.
- **Code:** github.com/ziyan-wang98/BazaarBench.

### Phase 6: agents that talk (deck step 6)

Does cheating spread through messages, and can the overseer use what agents say? This comes later, and links to the 13 Oct feedback on communication and contagion.

---

## What goes in which paper (proposal for Yali)

- **ICML 2027 (late January):** phases 1–4 in SocialJax Harvest, plus 5a if time allows. The claim: a budgeted overseer that chooses whom to check, tested against learning agents that adapt, with a cost frontier.
- **After ICML (NeurIPS, May 2027):** phase 5b–5d, the weaker LLM overseer and the token-costed overseer, with BazaarBench as the second domain.

## Papers to keep open while building

All are in the Zotero collection ICML unless marked.

| Phase | Paper | What we borrow |
|---|---|---|
| 1 | Guo et al. 2025, SocialJax | Environment, IPPO, Schelling diagrams |
| 1 | Perolat et al. 2017 | Sustainability, equality and peace metrics |
| 1 | Casheekar & Tangirala 2026 | Missed vs wrongly stopped; the recovery setting |
| 1 | Sun et al. 2019 | Enforcement has a cost for honest agents too |
| 2 | Baumann et al. 2020 | The switch-off test |
| 2 | Gasztowtt et al. 2024 | An overseer in Harvest with 7 agents; LLM-overseer prompting |
| 4 | Greenblatt et al. 2024, AI Control | Audit budget, red team, safety vs usefulness |
| 4 | Wang et al. 2019, Green Security Games | Patrols on a budget against adaptive attackers; double oracle |
| 4 | Sinha et al. 2018, Stackelberg Security Games | Randomised targeting |
| 4 | Zhu et al. 2026, Blindfold | Audits must stay unpredictable |
| 4 | Ye & Steinhardt 2026, Norm Enforcement | Agents exploit simple enforcement; tracking reliability fixes it |
| 4 | Ge-Wang et al. 2026; Terekhov et al. 2025 | Attackers choosing when to strike; adaptive attacks on monitors |
| 4 | Makins et al. 2026 | Distributed attacks beat per-agent monitors |
| 4 | Parisi et al. 2024, Monitored MDPs | Learning when rewards are only sometimes observed |
| 5 | Engels et al. 2025, Scaling Laws for Scalable Oversight (**matrix row 27, foundational**) | An Elo per role (overseer "Guard" vs evader "Houdini") plotted against general capability; their own limitation is that they only test single-step oversight, and ours is sequential; design lesson: the overseer should start ahead and gain more from capability than the evader |
| 5 | Willis, Du, Leibo & Luck 2025, Will Systems of LLM Agents Cooperate? (**matrix row 28, foundational**) | LLMs write inspectable strategies rather than acting move by move; evolution as the test of which behaviour wins; the lineage behind our R1a/R1b runs |
| 5 | Kenton et al. 2024; Burns et al. 2023; Kale et al. 2025 | Weak models judging or monitoring strong ones |
| 5 | Koran et al. 2026, Ensemble Monitoring | Monitoring cost: diverse signals beat more compute |
| 5 | Guo et al. 2026, OpenSocialJax | LLM players in SocialJax |
| 5 | Wang et al. 2026, BazaarBench (**matrix row 29, read; add to Zotero**) | C2C market with violations checkable from records (selling unowned items, promising one item twice); no live overseer; the judge alone used about $6,500 of tokens over 48 runs |

## Questions for Yali

1. **Bounding the evaluation:** is "all-seeing overseer as upper bound, none as lower bound, trained red team as worst case, one cost currency" the right frame?
2. **Cost unit:** should a check cost 1 unit for the RL phases, with tokens only for LLM overseers? Or tokens throughout?
3. **BazaarBench:** can we work with Ziyan and Shuqing on an online overseer? Is there compute for open models on CREATE?
4. **Split:** phases 1–4 for ICML, and weaker/costed overseer for the next paper?
5. **GPU credit:** we have the free `gpu` partition, but `er_prj_inf_du` has 0 credit and `/scratch/prj/inf_du` is 99% full.

## Exactly what to do, week by week (to ICML, about 22 Jan 2027)

**Week 1, 16–22 Nov: build the wrapper (phase 0–1).**
- [ ] Check the CREATE smoke job (`cat /scratch/users/k26108758/38280051.out`); note steps per second.
- [ ] Install SocialJax locally (CPU JAX is enough for scripted bots) and read `harvest_open.py`: how eating is recorded, the freeze mechanism, individual vs shared rewards.
- [ ] Write the overseer wrapper: the rule (eating an apple with fewer than m apples within two squares), k-of-N checks with four choosers (random, round-robin, targeted, all-seeing), two consequences (reward penalty F, freeze T), and per-step logs of missed over-harvesting, wrongly stopped harvesting, apples alive, welfare and checks used. Unit tests.
- [ ] Scripted bots: restrained, greedy, sneaky (greedy only when not checked recently).
- [ ] Email Yali the five questions below; ask to be introduced to Ziyan and Shuqing about BazaarBench.

**Week 2, 23–29 Nov: first results with bots (phase 1).**
- [ ] Schelling diagram with no overseer, then with it, to show the overseer removes the dilemma.
- [ ] Pick m from the diagram.
- [ ] Heatmap of orchard survival over k × penalty.
- [ ] Random vs round-robin vs targeted against the sneaky bot.
- [ ] One-page write-up for Yali.

**Weeks 3–4, 30 Nov–13 Dec: learning agents (phases 2–3, CREATE).**
- [ ] IPPO with individual rewards and no overseer; save the checkpoint once agents over-harvest.
- [ ] Switch the overseer on, from scratch and from that checkpoint, at k = 0, 1, 2, 3, 7.
- [ ] Switch-off test.
- [ ] The overseer sets its own penalty (watch first).

**Weeks 5–7, 14 Dec–3 Jan: agents that dodge, and a weaker overseer (phase 4, plus 5a in RL).**
- [ ] Agents that can see who was checked; random vs targeted vs randomised-targeted vs a learned overseer.
- [ ] Two or three rounds of the red-team loop.
- [ ] Shrink the learned overseer's network and plot cooperation against overseer size.
- [ ] The cost frontier (cooperation against checks spent).

**Weeks 8–10, 4–22 Jan: write the ICML paper.**
- [ ] Claim: a budgeted overseer that chooses whom to check keeps learning agents cooperating in Harvest, holds up when they learn to dodge it, and degrades in a measurable way as it gets weaker or cheaper.

**After ICML:** phase 5b–5d, with an LLM overseer smaller than the LLM players, token budgets of 1, 10 and 100 units, and BazaarBench, then phase 6.

**Biggest risk:** IPPO training time on CREATE. The smoke job's steps per second decide whether the weeks 3–7 sweeps fit. If they don't, cut k to {0, 2, 7} and use fewer seeds before cutting any phase.
