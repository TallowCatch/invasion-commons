# Budgeted Oversight Games: a formal model of partially capable overseers in multi-agent commons

*Draft 2 of the formulation (26 Nov 2026). It follows the structure of Griffin, Shlegeris & Abate 2024, "Games for AI Control": results in plain words first, then the formal game, then the model of our instances, then propositions with their proofs.*

*Every proposition carries a **prediction**, written before the runs in §7 are done. §6 lists which predictions existing data already tests and which need new runs. Once the runs are done, results replace the "predicted" lines and this becomes the paper's formulation section.*

*Sources for each piece are named where it is used. "Ours" means introduced here.*

---

## 0. Abstract

We study an overseer that must keep a group of self-interested learning agents from destroying a shared resource. It can only inspect a few agents each step, its judgement makes mistakes, and it can only sanction what it catches.

We introduce **Budgeted Oversight Games**, a formal model of this setting. It extends AI-Control Games (Griffin et al. 2024) from one untrusted model to many learning agents in a sequential social dilemma.

The overseer's action has four parts: **look, judge, respond, share**. Its capability is a class κ (budget k, look window w, judge, sanction F). It is evaluated in an ordinary mode and an adversarial mode, as in AI Control.

Simple propositions predict when a partially capable overseer keeps the commons safe and when it cannot. The failure cases are:
- the sanction a harmful violation actually meets is too low;
- the looks are predictable to the agents;
- the look is shorter than the rule's window;
- the rule itself can be obeyed while the commons dies.

We test each prediction in SocialJax Harvest and Clean Up with IPPO learners and with LLM judges of different sizes.

## 1. Introduction

**The problem.**
- AI Control (Greenblatt et al. 2024) asks how a weaker, trusted overseer can safely deploy a stronger, untrusted model.
- Scalable oversight asks how oversight scales as the overseen systems get stronger (Engels et al. 2025).
- Both study **one** overseen model and **single-step** tasks; Engels et al. name single-step oversight as a limitation.
- Multi-agent systems add two things:
  - the harm is collective (a commons collapses), not one bad output;
  - the overseen agents learn how the overseer behaves.

**What exists.**
- *MARL planners and sanctioners* act on many learners, but observe everything and budget only payments: Zheng et al. 2022 (AI Economist); Yang et al. 2020 (LIO); Baumann et al. 2020; Gasztowtt et al. 2024; Vinitsky et al. 2023; Köster et al. 2022.
- *Security games* budget a defender's coverage against adaptive attackers, but have one attacker and a perfect observation of attacks: Pita et al. 2008; Wang et al. 2019.
- *AI-Control Games* model an imperfect, budgeted monitor, but against one model (Griffin et al. 2024).
- *Networked MARL* gives the template of one problem class with theory, instantiated across many testbeds: Du et al. 2022 (DMPO); Ma et al. 2024.

The empty cell is an overseer that chooses whom to observe, judges imperfectly, and faces many learners in a sequential commons.

**Contributions.**
1. **Budgeted Oversight Games (§3):** a formal model of a partially capable overseer as a player in a partially observable Markov game with N learning agents. It has two evaluation modes, and Stackelberg timing with learning followers.
2. **Propositions (§5).** These say which part of the overseer's capability decides the outcome:
   - the realised sanction ρ(s);
   - the need to randomise looks when agents can see them;
   - zero capability gap for uniformly random looks;
   - the judge's miss rate on harmful situations, which gives a capability and cost condition for "a smaller overseer suffices";
   - the window condition;
   - a local-pass/global-fail bound;
   - regional budgets.
3. **Evidence (§6–7).** Every proposition has a written prediction and an experiment, in two SocialJax games with IPPO learners and LLM judges of increasing size.

## 2. Example: the Harvest overseer, in plain words

**The setting.**
- Seven agents pick apples in SocialJax Harvest (Guo et al. 2025). An apple only regrows if other apples are near it, so taking the last apples in a patch kills that patch for good.
- The rule: don't eat an apple with fewer than 3 apples in the 12 cells around it.
- Each step, the overseer looks at 2 of the 7 agents. If a judge says a looked-at agent broke the rule, the agent loses apples (the sanction).
- The agents are IPPO learners trained for 1e8 steps. They learn whatever pays.

**Figure 1** (to draw):
- (a) the loop: look, judge, respond;
- (b) the gpt-oss-20B judge's miss rate by situation. It misses 0–3% of violations that take the last apples, and up to 48% of borderline ones.

**What we found** (each line is a proposition in §5):
- **What matters is the sanction a violation actually meets.** That is the chance of being looked at, times the chance the judge catches it, times the sanction. Below a threshold of about 0.8–1 apple, the orchard dies. Above it, the orchard survives. This holds for 7 or 14 agents, and for any split between looks and sanction size.
- **If agents can see when they are looked at, predictable looks are worthless.** Round robin or repeat-offenders-first gives 69–72 apples, the same as no overseer. Random looks lose nothing.
- **Agents find the judge's blind spot.**
  - A judge that misses mostly mild violations keeps the orchard.
  - A judge with the same average miss rate, missing mostly harmful violations, does not.
  - What decides is the miss rate on harmful violations.
- **In Clean Up, the look must read the rule's whole window,** and the rule must protect the commons. Agents who never break Clean Up's rule still lose the orchard in 62% of games.

**Figure 2** (to draw), the analogue of Griffin et al.'s safety–usefulness frontier:
- x-axis: worst-case safety (adversarial mode);
- y-axis: welfare (ordinary mode);
- one point per look policy, at a fixed capability class.

## 3. Formalising Budgeted Oversight Games

A Budgeted Oversight Game is played between **one overseer O** and **N agents**. The game continues for H steps.

Each episode has a **mode** ⊛ ∈ {⊕, ⊗}:
- **ordinary (⊕):** the agents' observations carry no information about the overseer's past looks;
- **adversarial (⊗):** each agent also observes its own look history.

This is our analogue of Griffin et al.'s honest and scheming modes. There is one difference. In AI Control the scheming model is a single adversary (the red team) minimising safety. Here the agents are self-interested learners in **both** modes. The adversarial mode gives them the information a scheming model would exploit: when it is being watched. Whether they exploit it is then up to their learning, not assumed.

**Definition 3.1 (Budgeted Oversight Game).** A Budgeted Oversight Game G is an (N+1)-player, partially observable, finite-horizon Markov game defined by the tuple

  ⟨ S, S_⊥, {A^i}, A^O, {Ω^i}, Ω^O, I, P, {r^i}, v, J, F, κ, H ⟩.

- **S, S_⊥.** S is the state space. S_⊥ ⊆ S is the set of **collapse states**, where the commons is lost. A group harm predicate g : S → {0, 1} marks safe states (g = 1).
  - *Harvest:* S_⊥ = {no apples left}. It is absorbing, because apples cannot regrow without neighbours.
  - *Clean Up:* S_⊥ = {pollution ≥ 0.4}, where regrowth stops. It is not absorbing.
- **{A^i}.** The agents' environment actions (move, turn, eat, zap, clean).
- **A^O.** The overseer's action. Each step it chooses a **look** (K_t, w_t):
  - a set K_t ⊆ {1, …, N} of at most k agents;
  - a window w_t ≤ w_max of past steps to read for each.

  The **judge** J turns each window into a verdict. The **response** applies the sanction F to agents with a "violation" verdict. With M overseers, A^O also contains **share** messages (§5.7).
- **{Ω^i}, Ω^O.** The observations.
  - Agent i sees its local view of the map (SocialJax's egocentric window).
  - In mode ⊗ it also sees (was I looked at last step, steps since I was last looked at).
  - The overseer sees only what its looks return: the looked-at agents' windows, plus its own history of looks and verdicts.
- **I, P.** The initial state distribution and the transition function. P : S × A^O × ∏A^i → Δ(S × Ω^O × ∏Ω^i).
- **{r^i}.** Agent i's reward: the environment reward (apples eaten) minus sanctions received.
- **v (the individual rule).** v_i,t ∈ {0, 1} marks a violation by agent i at step t, as a function of i's last W steps.
  - *Harvest:* W = 1; eating an apple with fewer than 3 apples in its 12 neighbouring cells.
  - *Clean Up:* W = 100; eating while pollution > 0.2 without having cleaned in the last 100 steps.
- **J (the judge).** J maps a looked-at window to a verdict. It is characterised by:
  - its miss rate α(s) = P(verdict = pass | v = 1, situation s);
  - its false-alarm rate β(s) = P(verdict = violation | v = 0, s).

  A perfect judge has α = β = 0.
- **F.** The sanction per "violation" verdict, in reward units.
- **κ = (k, w_max, J, F, c).** The **capability class**: looks per step, longest window, judge, sanction size and cost per look. Each look costs c = c_0 + c_1·w (in tokens for an LLM judge).
- **H.** The horizon (1,000 steps).

**Policies.**
- The overseer has perfect recall and may randomise. Its policy is π^O : (A^O × Ω^O)^t → Δ(A^O).
- The agents' policies are π^i : Ω^i → Δ(A^i). In our experiments they are learned by IPPO with a shared network.
- Π^O(κ) denotes the overseer policies feasible within class κ.

**The process.**
- S_0 ∼ I.
- At step t the overseer draws its look A^O_t ∼ π^O(history).
- Each agent draws A^i_t ∼ π^i(O^i_t).
- Then (S_{t+1}, O^O_{t+1}, {O^i_{t+1}}) ∼ P(S_t, A^O_t, {A^i_t}).
- Sanctions are applied at the step of the verdict.

**Objectives.** The overseer has three:

  Safety(π^O, π) := P( S_t ∉ S_⊥ for all t ≤ H | π^O, π )    (1)
  Welfare(π^O, π) := E[ Σ_t Σ_i r^i_env,t | π^O, π ]          (2)
  Cost(π^O) := E[ Σ_t Σ_{i ∈ K_t} c(w_t) ]                    (3)

In words:
- safety is the chance the commons survives the episode;
- welfare is the apples eaten, sanctions not deducted;
- cost is what the looks cost.

Safety in Clean Up is reported as the share of time in safe states, because S_⊥ is not absorbing there.

**Timing (leader–follower).** The overseer commits to π^O first. The agents then learn in the game G[π^O] that it induces. This is Gasztowtt et al.'s Stackelberg-Markov game. It differs from Griffin et al., whose red team computes an exact best response. Our followers are N learners, and we take their outcome to be the joint policy π*(π^O, ⊛) that IPPO reaches at convergence. This is an approximation to a Nash equilibrium of G[π^O], checked by the levelling-off of the learning curves (F31).

**Definition 3.2 (Oversight equilibrium).**
- The worst-case safety is Safety^⊗(π^O) := Safety(π^O, π*(π^O, ⊗)).
- The ordinary welfare is W^⊕(π^O) := Welfare(π^O, π*(π^O, ⊕)).
- A policy π^O ∈ Π^O(κ) is in **oversight equilibrium within κ** if no other policy in Π^O(κ) has higher worst-case safety, higher ordinary welfare, and no higher cost.

This is Griffin et al.'s Control Equilibrium, with learning followers and a cost axis added.

**Definition 3.3 (realised sanction and capability gap). Ours.**
- Let p_i,t be the probability, given agent i's information at step t, that i is looked at. The **realised sanction** on a violation by agent i in situation s is

  ρ_i(s) := p_i(s) · (1 − α(s)) · F.    (4)

  In words: the chance of being looked at, times the chance the judge catches it, times the sanction.
- The **capability gap** of a policy is Δ(π^O) := W^⊕(π^O) − W^⊗(π^O). It is how much the same overseer loses when agents can see its looks. This is Ameer's Decision 2 of 23 Nov.
- The gap of a class, Δ(κ), is the gap of its oversight-equilibrium policy.

**What the definition rules out.**
- The overseer does not observe the commons directly. It sees only what its looks return. Decision 1 of 23 Nov keeps the commons view as an extension, now motivated by §5.6.
- The overseer cannot change the agents' rewards except by sanctions on verdicts.
- The agents cannot observe other agents' looks unless a communication channel is added (§5.8).

## 4. Model of the SocialJax instances

| Element | Harvest (7 or 14 agents) | Clean Up (7 agents) |
|---|---|---|
| Collapse states S_⊥ | no apples left (absorbing) | pollution ≥ 0.4 (regrowth stops; not absorbing) |
| Rule v, window W | eat with < 3 apples in the 12 neighbouring cells; W = 1 | eat while pollution > 0.2 without cleaning in the last 100 steps; W = 100 |
| Situations s for the judge | near ∈ {0, 1, 2} apples counted × outer apples (< 2, ≥ 2) | — |
| Sanction F | N·c/k apples, so random looks give ρ = c (stage 1 scaling) | same |
| Look cost c | 1 per look (Harvest, w = 1) | 1 + 0.01·w |

**The protocol class used throughout. All are memoryless in the overseer's suspicion state.**
- **Random:** K_t uniform over k-subsets, independent of history.
- **Round robin:** cycle through the agents.
- **Repeat offenders first ("history"):** look at the k highest suspicion scores; the score rises with each catch and decays by 0.95 a step.
- **ε-mixture:** with probability ε the whole step's looks are random, otherwise repeat offenders first. The randomised commitment of security games (Pita et al. 2008).
- **Regional (M overseers):** the map is cut into M strips, each with k_m looks. They optionally share suspicion scores.

**Judges.**
- perfect (α = β = 0);
- coin flip (constant α, β);
- tables α(near, outer), β(near) measured from LLM judges on 2,463 trained-agent situations (gpt-oss-20B; `llm_overseer/error_patterns/`).

## 5. Propositions

Each proposition is followed by its proof (or a sketch, when it is a modelling statement), the prediction it makes, and the evidence. "Myopic" means the agent compares a violation's immediate gain with its immediate expected sanction. Learners are not myopic, so thresholds are **measured**, not derived from the gain.

### 5.1 The realised sanction decides the regime

**Proposition 5.1.** Let g(s) > 0 be agent i's immediate gain from the violating action over its best legal action in situation s. A myopic, risk-neutral agent violates in s if and only if g(s) > ρ_i(s), with ρ_i(s) as in (4).

*Proof.* The expected immediate cost of violating is P(looked at) · P(caught | looked at, violation) · F = p_i(s)(1 − α(s))F. The agent violates when its gain exceeds this. □

**Corollary 5.2 (only ρ matters).** Under random looks p_i = k/N. So two overseers with equal (k/N)(1 − α)F induce the same myopic best response, whatever their k, N, α and F taken separately.

**Prediction.** There is a threshold ρ* such that the commons is lost when ρ < ρ* and kept when ρ > ρ*, for any combination of k, N, α and F.

**Evidence.**
- The threshold lies in (0.79, 1.0]. The commons is lost at ρ = 0.5 (c = 0.5; 50% misses at c = 1), 0.71 and 0.79 (coin-flip judges, α = 0.29 and 0.215, at c = 1, k = 1 and 7). It is kept at ρ = 1 (perfect judge, c = 1) and ρ ≥ 1.4.
- Doubling N and k together leaves the outcome per agent unchanged (F34).

**Known exception, the spread.** At the same ρ = 1, k = 1 gives fewer apples than k = 7: 453, 410, 457 against 484, 479, 468. Proposition 5.1 predicts no difference for risk-neutral agents. *Proposed explanation:* learning speed. Rarer, larger sanctions give noisier returns, so learners improve more slowly.
- **Prediction (X3):** at 3e8 steps the k = 1 and k = 7 arms meet, within the seed range.
- **Falsifier:** the gap stays at about 37 apples or more.

### 5.2 Looks that agents can see must be random

**Proposition 5.3.** In mode ⊗, suppose the look policy is a deterministic function of quantities in agent i's observation history. Then i can tell in advance at which steps p_i,t = 0. A best-responding agent then sees a realised sanction of 0 on its violations, whatever F.

If instead, with probability ε at each step, the looks are drawn uniformly at random and independently of the history, then p_i,t ≥ εk/N at every step. So

  ρ_i ≥ ε · (k/N) · (1 − α) · F.

*Proof.* In the deterministic case, i knows K_t before acting, so it violates only when i ∉ K_t, where p_i = 0. In the ε case, the random component puts i in K_t with probability k/N, whatever i knows. The targeted component can only add to this. □

**Corollary 5.4 (enough randomness).** Combining with Proposition 5.1, ε ≥ ε* := ρ* / ((k/N)(1 − α)F) guarantees the regime of a fully random overseer. With the stage 1 scaling (F = Nc/k, perfect judge), ε* = ρ*/c.

**Corollary 5.5 (a bigger sanction cannot buy back predictability).** In mode ⊗, a deterministic look policy loses the commons whatever the size of F.

**Predictions** (c = sanction level; ρ* ∈ (0.79, 1.0]):
- *At c = 1:* ε* ∈ (0.79, 1.0]. Survival rises with ε. ε = 0.9 survives if ρ* ≤ 0.9. ε = 0.5 and 0.7 are not guaranteed.
- *At c = 2:* ε* ∈ (0.40, 0.50]. ε = 0.5 and 0.7 survive. ε = 0.3 is not guaranteed.
- *Repeat offenders first (ε = 0) in mode ⊗:* dies at c = 2 and at c = 4. Caught share stays at 1–4%, as at c = 1.
- *In mode ⊕:* every ε survives at c = 1.

**Falsifiers:**
- an arm with ε ≥ ε* dies;
- repeat offenders first survives in mode ⊗ at a larger F.

**Evidence so far.** At c = 1 in mode ⊗:
- ε = 0 (repeat offenders first, round robin): dead, 69–72 apples;
- ε = 0.3: 265 apples [170, 417], alive 69% of the time;
- ε = 1: alive, 457 apples (F29).

This fits: the floor guarantees nothing at ε = 0.3, and the targeted part was only partly predictable.

### 5.3 Random looks have no capability gap

**Proposition 5.6.** If the looks are drawn i.i.d. uniformly over k-subsets, independently of the history, then for every agent p_i,t = k/N in both modes. The extra observation in mode ⊗ carries no information about current or future looks, so the induced games G[π^O] in the two modes have the same best responses, and Δ = 0 in expectation.

*Proof.* Independence makes i's look history independent of the event i ∈ K_t. So conditioning on it leaves P(i ∈ K_t) = k/N. The extra observation channel is then payoff-irrelevant. □

**Prediction.** Δ ≈ 0 for random looks, with one or several overseers. Δ is large for deterministic policies.

**Evidence.** Δ = 2 (central random) and 8 (two regional overseers, random). Δ = 386–405 for round robin, repeat offenders first, and shared repeat-offender lists (F29). **Holds.**

### 5.4 A judge's capability enters through harmful situations

Let 𝓗 be the **harmful situations**: those where a violation takes a part of the commons past the point of no return. In Harvest that is near = 0: the patch's last apples. Write α_𝓗 for the judge's miss rate on them.

**Proposition 5.7.** Under Proposition 5.1, agents violate in any situation where ρ(s) is below their threshold. If ρ(s) ≥ ρ* on every harmful situation, the violations that remain are non-harmful ones. The commons then survives if non-harmful violations alone cannot collapse it. In Harvest they cannot: patches with 1–2 apples left regrow (F27).

So, with random looks, survival requires

  (k/N) · (1 − α_𝓗) · F ≥ ρ*.

*Sketch.* This combines Proposition 5.1, applied situation by situation, with the measured fact that only harmful violations cause permanent loss (H5 in Harvest). □

**Corollary 5.8 (when a smaller overseer suffices; Yali's question).**
- Two judges with the same α_𝓗 give the same safety, whatever their size or average accuracy.
- The looks needed per step are

  k ≥ ρ* · N / ((1 − α_𝓗) · F).

**Corollary 5.9 (cost; Yali's 1-unit vs 10-unit question).** With a per-step budget B and a cost per look c_J (tokens, for an LLM judge), the overseer can afford k = B/c_J looks. Then ρ = (B/(c_J·N))(1 − α_𝓗)F. So the judge that needs the least budget is the one with the largest (1 − α_𝓗)/c_J, not the most accurate one.

**Predictions:**
- *Existing judges.* Let ρ_𝓗 = c(1 − α_𝓗):
  - coin flip α = 0.215: ρ_𝓗 = 0.79 at c = 1 → dies;
  - gpt-oss table, α_𝓗 = 0.03: 0.97 → survives;
  - inverted table, α_𝓗 = 0.39–0.62: 0.38–0.61 at c = 1 → dies; 0.76–1.22 at c = 2 → survives, with more harmful violations.
- *New judges (X5).* Each judge's α_𝓗 is measured on the situation bank first. Its outcome is then predicted from ρ_𝓗 **before** its RL runs.
- **Falsifier:** more than 1 of the 10 new cells lands on the wrong side, or two judges with the same α_𝓗 differ in survival.

**Evidence.** All four existing judge arms at c = 1 land on the predicted side, and so do the three survival cells at c = 2 (F24, F28, F30). The inverted arm at c = 2 survives but takes 30–44 apples less, with 6–8 harmful violations per episode against under 1.

### 5.5 The look must cover the rule's window

**Proposition 5.10.** Suppose the rule depends on W past steps (Clean Up: no cleaning in the last W), and the look reads w < W steps. Then the judge cannot tell a violator from a complier who cleaned between w and W steps ago.
- A judge that flags only proven violations has α = 1, so ρ = 0, the same as no overseer.
- A judge that presumes guilt when no cleaning is seen has α = 0 but a false-alarm rate β(w) = P(no cleaning in the last w | cleaned within W) > 0.

*Proof.* The two histories agree on the w steps read. □

**Prediction.** In Clean Up: w = 50 equals no overseer; w = 150 does better; presumed guilt at w = 50 protects, at the price of wrongful sanctions.

**Evidence** (F33):
- w = 50: 12 apples, no violation proved; no overseer gives 29;
- w = 150: 130 apples;
- presumed guilt: 312 apples, 35 wrongful sanctions per game.

**Holds.**

### 5.6 Local pass, global fail: what inspection of individuals cannot do

**Proposition 5.11.** An overseer whose responses depend only on rule verdicts can at best make every agent comply. Once all agents comply, no sanction is applied, so their choice among compliant behaviours is set by their own rewards alone. Hence

  Safety(π^O, π*) ≤ max { Safety(π) : π compliant and a best response for the agents }.

If compliant play can still reach S_⊥ (high local-pass/global-fail), no protocol in any capability class can guarantee safety. Only the commons view can: observing g, or a rule whose compliance implies safety.

*Proof.* Under compliance the sanction term is identically 0. So the agents' optimisation problem is the original one restricted to compliant policies, which the overseer does not affect. □

**Predictions:**
- *Clean Up, full perfect inspection* (k = 7, c = 2, w = 150; X4): violations near 0. But pollution crossings with no violation in the rule's window remain, and safety stays below 100%.
- *Clean Up with a stricter rule* (W = 25; X10): if never-violating scripted agents keep the orchard in at least 90% of games under it, then learners at ρ ≥ ρ* also keep it. That makes Clean Up a second game where all conditions hold, for Proposition 5.1's threshold.

**Evidence.**
- Harvest: 0 of 580 permanent losses came from compliant eats (F27), so the bound does not bind.
- Clean Up: 54–72% of pollution crossings had no violation in the rule's window. Never-violating scripted agents keep the orchard to the end in only 38% of games (F32). The bound binds.

### 5.7 Several overseers

**Proposition 5.12.** Take M regional overseers. Region m holds N_m(t) agents and has k_m looks. Then p = k_m/N_m(t) in region m, so

  ρ_m(t) = (k_m / N_m(t)) · (1 − α) · F.

- With fixed equal budgets, ρ_m(t) falls in crowded regions, and agents there violate when ρ_m(t) < ρ*.
- Budgets proportional to headcount, k_m(t) = k · N_m(t)/N, give p = k/N everywhere: the central overseer's regime.

*Proof.* Substitution into (4). □

**Predictions:**
- *X6:* with 7 agents and 4 regions at c = 1, headcount-proportional budgets survive. Equal budgets die (they are known to die: 11 apples per agent).
- *Already observed:* with more agents per region the fluctuation of N_m(t)/N shrinks, so equal budgets do better. With 14 agents: 40 apples per agent, alive 78% (F34).
- **Falsifier:** proportional budgets lose about as much as equal ones.

### 5.8 Communication between agents (remark)

Suppose agents share their observations (own looks, catches), as in the 13 Oct feedback on contagion.
- **Under random looks,** the shared information is independent of future looks (the argument of Proposition 5.6). So at a fixed ρ the model predicts **no contagion through information**. Contagion through the commons remains: fewer apples changes everyone's gains.
- **Under repeat offenders first in mode ⊕,** public catches reveal who the current suspects are. When others are the top suspects, i knows p_i is low.

**Prediction (X7).** Making catches public turns the ordinary mode into an adversarial one for predictable protocols, but not for random ones:
- repeat offenders first with public catches falls well below its hidden-catch result (474 apples);
- random with public catches stays near 460.

**Falsifier:** public catches change random's outcome, or leave repeat offenders first unchanged.

## 6. What the evidence already covers, and what needs new runs

| Proposition | Already tested | New experiment | Prediction (written 26 Nov) |
|---|---|---|---|
| 5.1, 5.2 threshold, only ρ | yes, ρ* ∈ (0.79, 1.0]; F34 | X1 seeds | holds at 5 seeds |
| 5.1 spread exception | k = 1 vs 7 gap, 37 apples | **X3**, 3e8 steps | gap closes |
| 5.3–5.5 randomness, F vs predictability | ε = 0, 0.3, 1 at c = 1 | **X2**: ε sweep at c = 1 and 2; repeat offenders first at c = 2 and 4; mode ⊕ sweep | ε* = ρ*/c; ε = 0 dies at any F |
| 5.6 zero gap | yes (F29) | X1 seeds | holds |
| 5.7–5.9 judge capability, cost | 4 judges (F24, F28, F30) | **X5**: tables for Qwen 2.5 3B–72B on CREATE; RL at k = 1, c = 1 and 2 | predicted per judge before RL; ≤ 1 of 10 cells wrong |
| 5.10 window | yes (F33) | X1 seeds | holds |
| 5.11 local pass, global fail | offline (F27, F32) | **X4**: full inspection in Clean Up; **X10**: stricter rule | safety below 100% under full inspection; stricter rule restores it |
| 5.12 regional budgets | equal budgets (F23, F34) | **X6**: headcount budgets | proportional budgets survive |
| 5.8 communication | none | **X7**: public catches | changes repeat offenders first, not random |
| cost frontier (5.9) | tokens per look measured | **X8**: analysis | ranking by (1 − α_𝓗)/c_J |
| generality | Harvest, Clean Up | **X9**: BazaarBench instance of Definition 3.1 | written; live overseer scoped separately |

## 7. Experiments

| ID | What | Arms × seeds | Runs | Where |
|---|---|---|---|---|
| X1 | Seeds 4 and 5 on the key arms of H1, H2, H3, H3b and H9 | 28 arms × 2 | 56 | CREATE |
| X2 | ε sweep in mode ⊗ (c = 1: ε = 0.5, 0.7, 0.9; c = 2: ε = 0.3, 0.5, 0.7); repeat offenders first in ⊗ at c = 2 and 4; mode ⊕ ε = 0.5, 0.7, 0.9 at c = 1 | 11 × 3 | 33 | CREATE |
| X3 | k = 1 vs k = 7 at c = 1, 3e8 steps | 2 × 3 | 6 (= 18 normal) | CREATE |
| X4 | Clean Up full inspection, k = 7, w = 150, c = 1 and 2 | 2 × 3 | 6 | CREATE |
| X5 | Judge tables: Qwen 2.5 Instruct 3B, 7B, 14B, 32B, 72B on the 2,463-item bank (vLLM); then RL at k = 1, c = 1 and 2 | 5 judges × 2 × 3 | 30 + judging | CREATE (no Ollama spend) |
| X6 | Headcount-proportional regional budgets (code change), 4 regions and 2 regions, c = 1 | 2 × 3 | 6 | CREATE |
| X7 | Public catches channel (code change), repeat offenders first and random, mode ⊕, c = 1 | 2 × 3 | 6 | CREATE |
| X8 | Cost frontier from existing runs and X5's token counts | — | 0 | laptop |
| X9 | BazaarBench as an instance of Definition 3.1 (tuple, rule, judge, cost), plus Corollary 5.9 applied to the replay's judge data | — | 0 | laptop |
| X10 | Clean Up with W = 25: scripted bots first (CPU), then learners at k = 2, w = 50, c = 1 | 1 × 3 | 3 | laptop + CREATE |

Total: about 146 GPU runs (about 160 normal-length runs), roughly 110 GPU-hours. At our quota of 6–8 GPUs at a time, that is about 15–20 hours of wall time. No Ollama spend unless the CREATE judges fail. In that case the fallback is gpt-oss-120b, about $0.77 of the $8.2 left.

## Notation

| Symbol | Meaning | Range or units | Example | Source |
|---|---|---|---|---|
| N | number of agents | integer | 7 | SocialJax |
| k | looks per step | 0…N | 2 | ours |
| K_t | agents looked at at step t | subset, size ≤ k | {3, 6} | ours |
| w, W | look window; rule window | steps | 150; 100 | ours (June) |
| α(s), β(s) | judge's miss and false-alarm rates in situation s | [0, 1] | 0.03 at near = 0 | ours; Griffin's monitor distributions |
| α_𝓗 | miss rate on harmful situations | [0, 1] | 0.03 | ours |
| F | sanction per "violation" verdict | apples | 3.5 | ours |
| c | sanction level, F = Nc/k | apples | 1 | stage 1 |
| p_i(s) | chance agent i is looked at, given its information | [0, 1] | 2/7 | ours |
| ρ_i(s) | realised sanction p_i(s)(1 − α(s))F | apples | 1.0 | ours |
| ρ* | threshold realised sanction | apples | (0.79, 1.0] | measured |
| ε | share of steps with random looks | [0, 1] | 0.3 | Pita et al. 2008 |
| ⊕, ⊗ | ordinary and adversarial modes | — | — | Griffin et al. 2024 |
| Δ | capability gap W^⊕ − W^⊗ | apples | 403 | ours (Decision 2) |
| S_⊥, g | collapse states; group harm predicate | — | no apples left | ours (June) |
| κ | capability class (k, w_max, J, F, c) | — | — | ours |
