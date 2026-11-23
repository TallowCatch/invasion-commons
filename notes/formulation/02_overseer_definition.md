# Phase 2: the overseer, defined (draft 1, 23 Nov 2026)

**What this is.** A definition of an overseer for sequential social dilemmas, written as a player in a multi-agent game. It brings back the June–July ideas (observation scope, a capability gap, harm that only shows at the group level) and keeps the November setting (k-of-N inspection against learning agents in SocialJax). **The overseer's action is the centre: what it chooses to look at, how it judges what it sees, and how it responds.**

**Status of each part.**
- §1–§7 are the **model**: what we assume.
- §8 is the **choice and its justification**.
- The **claims** that follow from the model, with falsifiers, are Phase 3 (`03_hypotheses.md`, not yet written).
- **Evidence** is quoted only as worked examples, with its source.

Literature references point to `01_literature_map.md` and the reader notes in `lit/`.

---

## 1. The game the overseer joins

**The agents.** N agents play a game where each sees only part of the world and acts to raise its own return. Formally, a partially observable Markov game (POMG):
- states s ∈ S;
- each agent's action a_i and observation o_i;
- transitions P(s′ | s, a);
- rewards r_i(s, a);
- episodes of T steps.

*Harvest example:* N = 7 agents, an 11×11 view each, T = 1,000 steps (Guo et al., SocialJax). *Source:* standard POMG, as in SocialJax and Leibo et al. 2017. *Assumed.*

**The social dilemma.** The game is a dilemma when policies that are good for each agent alone are bad for the group. Leibo et al. (2017) test this with an empirical payoff matrix built from "cooperative" and "defecting" policies. SocialJax checks it with Schelling diagrams.

*Harvest example:* scripted bots that all hold back collect 71.5 apples each; bots that are all greedy collect 9.6 (tracker finding 1). *Assumed per testbed, checked empirically.*

**Two levels at which things can go wrong.** This is the June idea brought back. We separate them because an overseer may see one level and not the other.

1. **The individual rule.** A rule says whether agent i's recent behaviour is a violation: v_i,t = 1 or 0. It is defined on the agent's own trajectory, so in principle it can be checked by looking at that agent.
   - *Harvest:* eating an apple that leaves fewer than 3 apples within two squares (`harvest_overseer.py`; regrowth is 2.5% / 0.5% / 0.1% per step with 3+ / 2 / 1 apples near, and never with none).
   - *BazaarBench:* failures F1–F6, such as selling an item not owned.
   - *Source:* Leibo et al. 2017 (thresholding a per-agent behaviour statistic); Sun et al. 2019 (rules on observable behaviour). *Assumed.*
2. **The group outcome.** A safety predicate on the shared system says whether the commons is in a safe state: g_t = 1 or 0. It depends on everyone's behaviour together.
   - *Harvest:* the orchard still has apples (`orchard_alive`).
   - *Clean Up:* pollution below a level.
   - *BazaarBench:* harm reaching buyers (stage S5).
   - *Source:* our June draft (global safety predicate); Perolat et al. 2017 (sustainability). *Assumed.*

**Local pass, global fail (LPGF).** This happens at step t when no agent's behaviour breaks the individual rule, yet the group outcome is unsafe:

  LPGF_t = 1 if v_i,t = 0 for every i, and g_t = 0.

In words: every agent passes its own check, and the commons still fails.
- Its episode rate is the fraction of steps where this happens.
- It measures how well the individual rule tracks the real harm. If LPGF is common, an overseer that only checks individuals cannot protect the group, however many it checks.
- *Harvest:* not yet measured in SocialJax. The rule protects local regrowth, but seven agents each eating "legal" apples can still strip the orchard.
- *Source:* ours (June–July draft). The same idea appears as attacks split across agents in Makins et al. 2026. *Assumed definition; its size per testbed is an open measurement.*

---

## 2. The overseer is a player, and this is its action

The overseer, O, is an extra player in the game, as in Griffin et al. 2024 (AI-Control Games) and Baumann et al. 2020. Like any player it has an observation, an action, a policy and an objective. Unlike the agents, it does not harvest. **Each step, it acts in up to four stages:**

> **look → judge → respond → (share)**

### 2.1 LOOK: choose what to observe (the core of the action)

**Which agents to inspect.** The overseer cannot see everything. Each step it chooses a set K_t of agents to inspect, at most k of them:

  K_t ⊆ {1, …, N}, |K_t| ≤ k.

In words: the overseer picks at most k agents this step. Everyone else goes unseen.
- *Harvest:* k = 2 of N = 7.
- *BazaarBench:* k traders of 100 per day.
- k is the **inspection budget**. This is the observation budget that no MARL shaper has (literature map, gap 1).
- *Source:* audits as a scarce action in Griffin et al. 2024 and Greenblatt et al. 2024. The k-of-N form is ours. Vinitsky et al.'s "who can sanction whom now" set is the nearest MARL object. *Assumed.*

**How far back to read.** Each inspection reads the last w steps of the inspected agent's record.
- *Harvest:* w = 1 means only this step.
- *LLM inspector:* larger w means a longer prompt; input tokens grow by about 10 per step for summaries and about 16 for raw maps (LLM ladder).
- *Source:* ours. Ye & Steinhardt check an agent's recent history. *Assumed.*

**Which channels it has: observation scope.** This is the June "architecture" brought back. An overseer can have one or both of two channels:
- an **inspection channel**: records of the agents in K_t (individual level);
- a **system channel**: public signals about the group outcome, such as apples left, pollution level or the failed-patch fraction.

June's four architectures are the four combinations: none, inspection only ("local"), system only ("global signal"), and both ("hybrid").
- *Harvest:* the inspection channel is each checked agent's last w moves; the system channel is the apple count.
- *Source:* our June draft. Zheng and Gasztowtt's planners see only aggregates; Sun et al. see every agent. *Assumed.*

**What the overseer receives.** Its observation at step t is

  o^O_t = ( records of agents in K_t over the last w steps ; system signals z_t ; messages from other overseers ),

restricted to the channels it has. In words: it sees the records it chose to read, the public signals if it has that channel, and nothing else.

What this rules out: an overseer that sees every agent's actions for free, which is the assumption in Baumann, LIO, IQ-Flow and Sun.

### 2.2 JUDGE: turn what it saw into a verdict

**The inspector.** Reading a record does not guarantee a correct verdict. An inspector J maps a record to a verdict: flag or clear. How good it is depends on what it reads and who is reading. It is described by two error rates:
- the **miss rate** α: a real violation in the window is cleared;
- the **false-alarm rate** β: a clean record is flagged.

More generally, α and β can depend on the situation, α(s) and β(s).

- *Harvest, measured:* a perfect code check gives α = β = 0. gpt-oss-20B reading raw maps of trained agents misses α ≈ 0.22 on average, ranging from 0.00 to 0.48 by situation; it misses most on borderline violations with other apples visible nearby. Its false alarms are β ≈ 0.04 (F26).
- *Source:* the trusted monitor's score distributions for safe vs unsafe outputs in Griffin et al. 2024 (α, β is one threshold on that ROC curve); Engels et al. 2025 for capability; ours for the patterned form. *Assumed form; values measured.*

**Inspector capability.** This is the part of the overseer that LLM size, prompt and record format change. It is the most direct handle on Yali's "size and intelligence" question.
- *Example:* at w = 1 on summaries, Qwen ≤ 3B gives the same answer every time, while Qwen 7B and gpt-oss are perfect. On raw maps only gpt-oss works (F22).

### 2.3 RESPOND: act on the verdict

**Response to flagged agents.** For each flagged agent, the overseer applies a response u. Options:
- a **sanction**: remove F units of reward;
- a **freeze**: hold the agent still for some steps;
- **removal**;
- nothing.

- *Harvest:* the sanction F removes apples' worth of reward. Freezing caught agents was tried and failed (finding 5).
- *Source:* sanctions in Vinitsky et al. 2023 and Casheekar 2026; freeze and remove in Greenblatt et al. 2024 and Ye & Steinhardt 2026. *Assumed.*

**Response to the system signal** (if the overseer has that channel). A system-level response, such as closing part of the commons, setting a quota, or concentrating inspections where the signal is bad.
- *Source:* our June draft (global signal and hybrid); Gasztowtt and Zheng (one rule for everyone). *Assumed; not yet in our SocialJax code.*

**Delay.** The response may come d steps after the violation.
- *Harvest:* d = 0 today.
- *Source:* our June draft. *Assumed.*

### 2.4 SHARE (several overseers only)

With M > 1 overseers, an overseer can send other overseers a message, for example who it flagged or its suspicion scores, at a cost per message (§4).
- *Harvest:* "share suspicion" between 2 regional overseers costs 500–1,500 message units per game (DECENTRALISED_RESULTS).
- *Source:* communication along edges in Du et al. 2022; Makins et al. 2026 (pooled vs per-agent monitors). *Assumed.*

### 2.5 The whole action, in one line

  a^O_t = ( K_t, w ; verdicts from J ; responses u ; messages ),

restricted by the budget k, the channels the overseer has, and its inspector J. In words: whom to look at and how far back, what the inspector says, what to do about it, and what to tell other overseers.

---

## 3. How the overseer decides, and how capable it is

**Policy.** The overseer's policy π^O maps everything it has observed so far to its next action. Classes we use or could use:
- **fixed rules** for LOOK: random, repeat offenders first, round robin, mixed;
- **learned** policies (Baumann, LIO);
- an **LLM** choosing whom to inspect (design note only: `llm_overseer/live_overseer_design.md`).

*Source:* Griffin et al. (threshold and solver policies); Baumann and LIO (learned); Gasztowtt (LLM in context). *Assumed.*

**Capability vector.** This is the June tuple, extended. An overseer's capability is what it *can* do, separate from what its policy *chooses* to do:

  κ^O = ( k, w, channels, J (α, β), response strength F, delay d, cost per look ).

- *Harvest example:* κ^O = (2 of 7, w = 1, inspection only, perfect code check, F = 1-apple average sanction, d = 0, 1 unit per check).
- The June draft's tuple was (detection recall 1 − α, delay d, targeting capacity k/N, intervention cost).
- *Source:* ours (June draft), with k from Griffin's budget and J from Engels. *Assumed.*

**What it rules out:** capability as a single number. Two overseers with the same k can differ completely in J or channels.

---

## 4. Several overseers

**Overseers and assignment.** M overseers, each responsible for part of the system: a map region, or a subset of agents. Overseer m has its own budget k_m (with Σ k_m = k when comparing against one central overseer), its own channels, and its own inspector.

**What each one sees.** Each sees only records of agents it inspects in its part, plus local system signals, plus messages.

**The central case.** M = 1 is the central overseer. Every result for one overseer is the special case.

- *Harvest:* M = 2 vertical strips with k_m = 1 each, or M = 4 strips.
- *Source:* Du et al. 2022 and Ma et al. 2024. Agents are nodes in a graph with local neighbourhoods. Here the overseers are nodes whose links to agents are *chosen* each step (the K_t), so the budget is a limit on how many links an overseer node has. *Assumed.*

**Locality: when splitting works.** How much an agent's behaviour, and the harm it causes, stays inside one overseer's part. We measured two parts:
- χ, how often agents cross between parts: 0.011–0.015 per agent-step for trained Harvest agents on 2 strips;
- ζ, the share of a violation's harm that stays in its part: about 1.0 on 2 strips (F21, locality results).

This plays the role of ξ in Ma et al. 2024: a measured number per testbed that says whether the decentralised version should work. *Source:* Ma et al. 2024 (ξ); ours (χ, ζ). *Defined; measured.*

---

## 5. The agents' response and the timing

**Timing: the overseer commits, then the agents learn.** The overseer fixes its policy π^O. That defines a game G(π^O) in which the agents' rewards include the overseer's responses. The agents then learn in that game.
- This is a **Stackelberg-Markov game** (Gasztowtt et al. 2024; the AI Economist recast).
- An overseer is judged by what the agents learn to do in G(π^O), not by whether good behaviour is an equilibrium (Casheekar & Tangirala 2026).
- *Harvest:* the overseer is a fixed rule; 7 IPPO agents train for 1e8 steps. *Assumed.*

**Two modes of agent: ordinary and adversarial.** This follows the honest and scheming modes of Griffin et al. 2024.
- **Ordinary agents** learn with their normal observations. They are not told how the overseer works.
- **Adversarial agents** are the worst case. They get information about the overseer (who was inspected; their own chance of being inspected), or the overseer's blind spots to exploit, and still maximise their own return.
- *Harvest:* `--reveal-checks`; the `dodger` and `just_checked` bots; the patterned-inspector runs.
- *Source:* Griffin et al. 2024; Zhu et al. 2026 and Gans & Holden 2026 (agents who know when they are watched); Parisi et al. 2024 (learners treat unseen as unpenalised). *Assumed.*

**Agent capability.** The agents' side of the capability gap is set by the class of agent:
- scripted bots;
- learners with one shared network;
- learners with separate networks;
- adversarial learners that see the inspections;
- LLM agents (OpenSocialJax, BazaarBench).

*Source:* ours. Engels et al. vary both sides. *Assumed ordering; to be measured.*

**The capability gap Δ.** This is the June idea brought back, now measured rather than a difference of ranks. Decided by Ameer, 23 Nov: **Δ is how much worse the same overseer does against stronger agents than against ordinary ones.**

For an overseer policy π^O, an outcome J (safety S or welfare W from §6) and an agent class c:

  Δ_c(π^O) = J(π^O, ordinary learners) − J(π^O, agent class c).

In words: run the same overseer twice, once against ordinary learners and once against a stronger class of agent. Δ is how much the outcome drops. Δ near 0 means the overseer holds up when agents get stronger. A large Δ means stronger agents get round it.

- The classes c run from adversarial learners that see the inspections, to agents that exploit the inspector's blind spots, to LLM agents.
- *Harvest examples, from bots:*
  - With inspections visible, round robin and repeat-offenders-first are fully dodged (no catches), while random inspection holds (caught share ≈ 0.30 in both modes). So Δ is large for predictable overseers and about 0 for random ones (worst-case bot preview).
  - The learner version is in the held worst-case runs.
- *Why this definition:*
  - It needs no rating fit, and every term is something we already measure.
  - It matches Griffin et al.'s comparison of honest-mode and worst-case performance.
  - It keeps the scalable-oversight meaning: a capability increase on the agents' side, and how much of it the overseer absorbs.
- *What it does not give:* a single scale on which overseers and agents both sit, which Engels et al.'s Elo approach would. That stays available as an extension.
- *Source:* Griffin et al. 2024 (honest vs worst-case modes); our June draft (Δ); Engels et al. 2025 (the capability-gap framing). *Defined.*

---

## 6. What the overseer is trying to achieve

There is not one number. As in Griffin's safety–usefulness frontier and Perolat's outcome panel, the overseer is scored on a vector:

| Objective | Plain meaning | Harvest measure |
|---|---|---|
| **Safety S** | how often the group outcome is safe | fraction of games the orchard survives (`orchard_alive`); Perolat's sustainability |
| **Welfare W** | what the agents collect | apples collected per episode (`apples_eaten`); Perolat's utilitarian metric |
| **Harm to compliant agents H** | what the overseer's mistakes cost agents who follow the rule | sanctions on clean records, per legal apple |
| **Burden B** | what oversight uses | inspections, tokens, messages per episode |

Diagnostics, reported but not optimised: the violation rate, the share of violations flagged, and the LPGF rate.

**The overseer's problem.** Choose a policy π^O within the capability κ^O. Each choice gives a point (S, W, H, B) once the agents have learned in G(π^O). The good choices form a frontier. Any point on it can be found by maximising a weighted sum, as in Griffin et al. 2024 (Corollary 5.2).

*Harvest worked example:*
- No overseer: about 68 apples per episode, orchard dead in every run.
- 2 random inspections per step with a 1-apple average sanction: about 460 apples, orchard alive.
- Burden: 2,000 inspections per episode.
- (Findings 2–3.)

*Source:* Griffin et al. 2024; Perolat et al. 2017; our June draft (welfare, unsafe rate, burden). *Assumed.*

---

## 7. How an overseer is evaluated

**Bounds.**
- **Lower bound:** no overseer.
- **Upper bound:** inspect everyone (k = N) with a perfect inspector.
- **Worst case:** adversarial agents (§5).

A useful overseer sits near the upper bound with a fraction of its burden, and degrades gracefully in the worst case.

**Switch-off test.** Turn the overseer off after training and see whether behaviour persists. This tells whether the overseer changed what agents learned, or only props up behaviour while it is on (Baumann et al. 2020).
- *Harvest:* behaviour reverts in 18 of 18 runs within about 5–12M steps (finding 8). Our overseers prop up; they don't change norms.

**Scorecard per overseer and testbed.**
- (S, W, H, B) in each mode.
- The LPGF rate.
- The switch-off result.
- The capability gap Δ against each agent class.

---

## 8. What makes a game a fair testbed

A game can host this overseer when:
1. it is a social dilemma (payoff-matrix or Schelling test);
2. it has an **individual rule** that can be decided from what one inspection reveals;
3. it has a **group safety predicate** that can be measured;
4. the gap between the two can be measured (the LPGF rate);
5. a violation stays visible within the inspection window w;
6. for M > 1, its locality (χ, ζ) can be measured.

| Testbed | Individual rule | Group predicate | What one inspection reveals | Look cost unit |
|---|---|---|---|---|
| Harvest (SocialJax) | eating below 3 apples nearby | orchard survives | last w moves, map patch | 1 unit, or LLM tokens |
| Clean Up | not cleaning (a slow violation) | pollution below a level | last w moves | units |
| Coins | taking the other's coin | (two players; weak) | last w moves | units |
| BazaarBench | F1–F6 failures | harm reaching buyers (S5) | a day's listings, offers, messages | GPT-5 tokens ($6.5k for 48 offline runs) |
| OpenSocialJax | free-riding, depletion | last apple eaten time | moves *and* agents' notes and reasoning | LLM tokens |

---

## 9. The formulation we choose to try, and why

**The choice.** An overseer that is a player in a Stackelberg-Markov game over a sequential social dilemma, defined by:
- its **action**: look (choose k of N agents and a window, from the channels it has), judge (an inspector of measured quality), respond (sanction or freeze), and share (with other overseers);
- its **capability vector** κ^O;
- its objective, scored as (S, W, H, B) against ordinary and adversarial learners.

The single-overseer case is M = 1.

**Why this point in the design space.**
1. It sits in the empty cell of the literature map: an observation budget, imperfect inspection, many learners, a sequential commons.
2. It puts the scalable-oversight question in the model, not in the prose. The capability vector and the gap Δ make "a weaker overseer watching stronger agents" a measured quantity.
3. It keeps the multi-agent part. The individual-vs-group levels and LPGF capture harm that per-agent inspection can miss, and M overseers with locality follow Du's template.
4. It is runnable now. Every piece except the system channel and the Δ ratings exists in our SocialJax code.

**Alternatives and why not (for now).**
- **A fully observing shaper** (Baumann, LIO, the AI Economist). Rules out the observation budget, which is the question.
- **A learned overseer policy.** A natural extension inside this formulation (a policy class), not a different formulation. It comes after fixed policies are understood.
- **Peer enforcement only** (Vinitsky, Ye). Different who-acts; it could be added as a channel (peer reports) later.
- **A single scheming agent** (AI Control). Not multi-agent; no commons.

**What is new compared with each core paper (one line each).**

| Paper | What it has | What we add |
|---|---|---|
| Griffin et al. 2024 | an overseer player with a budget and a noisy monitor; two modes | many learning agents in a commons; choosing *which* agents to look at |
| Greenblatt et al. 2024 | a weak trusted monitor, audits, shutdown | a population and repeated responses instead of one model and one shutdown |
| Makins et al. 2026 | per-agent vs pooled monitors; split harm | learners, a commons, and M budgeted overseers with locality |
| Engels et al. 2025 | an overseer–agent capability gap on an Elo scale | the gap measured in sequential social dilemma match-ups |
| Parisi et al. 2024 | observation as part of the problem; one agent | many agents, and the overseer choosing whom to observe |
| Baumann 2020; LIO 2020; Zheng 2021 | a shaping agent over learners in SSDs | a budget on what the overseer sees; imperfect verdicts |
| Gasztowtt et al. 2024 | Stackelberg-Markov game; an LLM principal; aggregates only | per-agent, per-step inspection; inspector capability |
| Vinitsky et al. 2023 | sanction opportunities; local vs public sanctions | a designated overseer with a budget of k opportunities |
| Du 2022; Ma 2024 | networked agents, locality, ξ | overseers as nodes whose links to agents are chosen each step |
| Sun et al. 2019 | a regulator in MARL that sees every agent | partial observation; detection error, which they list as open |

---

## 10. What this commits us to, what it leaves open, decisions for you

**Commits us to:**
- The overseer is defined by its action (look, judge, respond, share) and its capability vector.
- Two levels of harm (the individual rule and the group outcome), with LPGF measured.
- Evaluation as a vector, against ordinary and adversarial agents, with bounds and a switch-off test.

**Leaves open:**
- How to measure Δ in practice.
- Whether the system channel is in the first paper.
- Whether the overseer's policy is ever learned.
- The theory "slot" of Du's template: a bound on what inspecting k rather than N costs. This belongs to Phase 3.

**Decisions for you:**
1. **The system channel.** Include the June "global signal" channel (and so the four architectures) in the first formulation, or keep the first paper to the inspection channel and add it later? My suggestion: include it. Without it, LPGF has nothing to compare against.
2. ~~The capability gap Δ.~~ **Decided 23 Nov:** Δ is the drop in outcome for the same overseer from ordinary learners to a stronger agent class (§5).

---

## Notation

| Symbol | Meaning | Range / units | Harvest example | Source |
|---|---|---|---|---|
| N | number of agents | integer | 7 | SocialJax |
| T | episode length | steps | 1,000 | SocialJax |
| v_i,t | agent i broke the individual rule at t | 0 or 1 | ate an apple leaving < 3 nearby | Leibo 2017; ours |
| g_t | group outcome safe at t | 0 or 1 | orchard has apples | June draft; Perolat 2017 |
| LPGF | rate of steps where all pass the rule but g = 0 | fraction | not yet measured | June draft |
| O | the overseer (a player) | | | Griffin 2024; Baumann 2020 |
| K_t | agents the overseer inspects at t | subset, size ≤ k | 2 agents | Griffin 2024 (audits); ours |
| k | inspection budget per step | integer ≤ N | 2 | ours |
| w | steps of record one inspection reads | ≥ 1 | 1 | ours |
| z_t | system signals (system channel) | | apples left | June draft |
| o^O_t | the overseer's observation | | records of K_t, z_t, messages | ours |
| J | the inspector | | code check, or gpt-oss-20B | Griffin 2024; Engels 2025 |
| α, α(s) | miss rate (possibly by situation) | [0, 1] | 0 (code); 0.22 avg, 0–0.48 (gpt-oss raw) | Griffin 2024; F26 |
| β, β(s) | false-alarm rate | [0, 1] | 0 (code); 0.04 (gpt-oss raw) | Griffin 2024; F26 |
| u | response to a flagged agent | sanction, freeze, none | sanction | Vinitsky 2023; Greenblatt 2024 |
| F | sanction size | reward units | per catch 3.5–7 apples (1-apple average at k = 2 or 1) | ours |
| d | response delay | steps | 0 | June draft |
| a^O_t | the overseer's full action | | (K_t, w, verdicts, responses, messages) | ours |
| π^O | the overseer's policy | | random, history, round robin, mixed | Griffin 2024 |
| κ^O | the overseer's capability vector | | (2/7, w = 1, inspection only, perfect, 1-apple avg, d = 0, 1 unit) | June draft; ours |
| M | number of overseers | integer | 1, 2 or 4 | Du 2022; ours |
| k_m | budget of overseer m | Σ k_m = k | 1 each (M = 2) | ours |
| χ, ζ | locality: crossing rate; share of harm kept local | per agent-step; [0, 1] | 0.011–0.015; ≈1.0 (2 strips) | ours (F21); Ma 2024 |
| G(π^O) | the game agents face once π^O is fixed | | | Gasztowtt 2024 |
| Δ_c(π^O) | capability gap: drop in outcome J for the same overseer from ordinary learners to agent class c | same units as J | bots: large for round robin, ≈ 0 for random | Griffin 2024; June draft |
| S, W, H, B | safety, welfare, harm to compliant agents, burden | | survival rate, apples, wrongful sanctions, inspections | Griffin 2024; Perolat 2017; June draft |
