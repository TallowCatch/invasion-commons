# Plan around the 13 October feedback

Written 14 October 2026. This turns the feedback into studies. Each study below is a **draft design**. It becomes a frozen protocol (`studies/<ID>_<name>/protocol.md`, timestamped, never edited after results; only amendments appended) before any code is written, as for L1–L6.

Background and literature: `next_steps.md`, `literature_review_ai.md`, `spatial_game_options.md` in this folder.

Labels used: **[verified]** read from data files; **[notes]** from earlier notes; **[estimate]** a planning number; **[post hoc]** decided after seeing results.

---

## What we learned (up to 14 October)

### From the experiments

| Study | Finding | Status |
|---|---|---|
| L3 | From each model's one-round gain, the first fine that deters was predicted to be 30 t (q = 1/6). Nemotron stopped at 30 t; gpt-oss one step early, at 24 t. | [verified] |
| **L4** | The 30 t prediction held on 10 new populations (over-taking 0.2% gpt-oss, 1.8% Nemotron) and under reworded rules (1.1%, 1.9%). gpt-oss again stopped at 24 t. | [verified] |
| **L4** | At the same expected fine, the mix of audit chance and fine mattered only where deterrence was partial. At e = 2 t: rare audit with a large fine 27.9% against frequent audit with a small fine 40.7%, d = −12.9 points [−16.4, −9.3]. At e = 3 t: 9–11% in every mix. | [verified] |
| **L4** | Rules that never mention audits gave *less* over-taking than audits with a 0 t fine: 30.2% against 57.5% (gpt-oss), 46.0% against 64.3% (Nemotron). | [verified] |
| L5 | DeepSeek over-took in 80% of decisions with no fine; its one-round gain was 5.41 t, so its predicted stop is 36 t. That stage was not run, for cost. | [verified]; prediction untested |
| **L6** | One programmed over-cutter among five gpt-oss foresters raised their over-taking by +25.1 points [+20.2, +30.6], at every ring distance. Seeing neighbours' cuts added nothing (+2.2, n.s.). | [verified] |
| L6 | The spread ran through the reviewer's shared cut. The cut factor fell from 0.76 to 0.22–0.36, roughly halving allowances, while the foresters' actual cuts stayed about 2.3 units. | [post hoc] |
| Code check | A fine only lowers the fined agent's score. It is not paid back to the group and does not touch the stock. | [verified] (`run_l2_llm_agents.py`) |

### From the literature (`literature_review_ai.md`; 58 papers, each checked)

- **Close papers that the paper must cite and position against:**
  - Bell and Madisetti, ICML 2026: one adversarial fisher, fixed by a voted sanction that returns its catch to the pool;
  - Okamoto et al., AIES 2026: stating a penalty can turn a duty into a cost-benefit choice;
  - OpenSocialJax (Guo, Fang, Leibo, Du; arXiv, 9 Oct 2026): LLMs in SocialJax Harvest and Clean Up, no overseer, every team depletes the orchard;
  - CoopEval, ICML 2026: benchmarks cooperation mechanisms, with no random audits among them.
- **Still open, and held by this project:**
  - a deterrence threshold predicted in advance and confirmed;
  - audit chance separated from fine size;
  - the never-mentioned control;
  - harm spreading through the overseer's own rule.
- **Not yet studied by anyone:**
  - an overseer with random audits inside a spatial commons;
  - contagion through an institution;
  - deterrence, as distinct from detection, in AI-control-style monitoring.

---

## The feedback, item by item

| # | Feedback (as given) | Our answer | Study |
|---|---|---|---|
| 1 | Cost of monitoring: how to count expenses, and is monitoring good or bad? | Welfare net of audit cost; a break-even cost per audit; fines counted both as lost and as transfers | **W1** (no games) |
| 2a | Can agents communicate? If one finds out another cheated and wants to cheat too… | A talk channel × planted cheater experiment that separates imitation from the shared-cut route found in L6 | **L7 Stage A** |
| 2b | …how can the overseer react to that? | Three responses compared under talk with a cheater: shared cut, targeted allowances, leniency for reports | **L7 Stage B** |
| 2c | Ground it in Richard and Janny's paper and the literature | Willis, Zhao, Du and Leibo (arXiv:2602.16662) list communication as future work, and their mechanism is payoff-biased imitation. L7 tests it with an enforcer present. | L7 protocol, related work |
| 3a | Individual utility and social welfare | Payoffs split by role (cheaters, LLM agents, honest agents) next to total welfare | **W1** |
| 3b | How to reject or choose a good environment | An admission test every game must pass before LLM runs: a real dilemma, room to improve, enforcement feasible | **A1** (CPU only) |
| 3c | How agents interact with each other | Interaction structure as a named factor: who sees whom (L6), who talks to whom (L7), who is near whom (L8) | L6, L7, L8 |
| 3d | A metric to know the overseer is working right | Oversight efficiency (the share of the welfare gap closed, net of cost) plus five diagnostics | **W1** |
| 4 | Consider making the overseer an LLM | An LLM chooses the shared cut and is compared with the rule on the same populations; then a version that writes the policy as code | **L9** |
| 5 | Make it an actual game (SocialJax-like, environment destruction, proximity) | First the ring with plots that die for good and audits that depend on position; the grid game in the next paper | **L8**, then **G1** |
| 6 | More AI papers, fewer economics papers | 58 checked AI papers; related work rewritten around them; Becker kept as the source of the prediction | Writing |

---

## The studies

### W1. Welfare, oversight cost and an overseer metric (no new games)

- **Question.** Is the oversight in L4 and L6 worth its cost, and does a single number capture whether the overseer works?
- **What will be done.** Recompute from stored games (L4: 190 gpt-oss and 70 Nemotron games; L6: 40 games):
  1. **Efficiency** = total harvest ÷ (T · rK/4), as in the paper. The first-best is 1.
  2. **Oversight efficiency** = (efficiency of the cell − efficiency of N0) ÷ (1 − efficiency of N0). N0 is the never-mentioned control. Fines are counted two ways, as lost and as transfers.
  3. **Break-even audit cost** c\* = (welfare of the cell − welfare of N0) ÷ (number of audits). Above c\*, no auditing is better.
  4. **Payoffs by role:** LLM agents, programmed honest agents and the planted cheater (L6).
  5. **Diagnostics:** over-taking against N0; collateral (what honest agents lose); calibration (distance between the observed stopping fine and the predicted one).
- **What is fixed.** Data, cells and populations are as already run.
- **Caveat.** This is post hoc for L4 and L6, so it must be labelled post hoc in the paper. The metric's definition is fixed in this plan before computing it.
- **Output.** `studies/W1_welfare_and_oversight_cost/results.md`, a table for the paper, and one figure: oversight efficiency by cell, with the zero-fine cell below zero if that holds.
- **Cost.** $0. CPU minutes.

### A1. Environment admission test (programmed agents, CPU only)

- **Question.** Do Fishery, Forest and River each pose a real oversight problem, and which settings would fail?
- **Three conditions, each with a numeric test:**
  1. **A real dilemma.** One agent gains by over-taking while the others comply (gain > 0), and all over-taking gives lower total welfare than all complying.
  2. **Room to improve.** With no oversight, harm occurs in 10–90% of rounds and collapse is not certain.
  3. **Enforcement can work.** The predicted deterring fine F\* = gain ÷ q is within the allowed fine range.
- **Also run on:** the planned L8 ring (plots that die for good) and any future grid game, *before* LLM runs. This is the gate the SocialJax Clean Up attempt failed in September.
- **Group size** is varied (4, 6, 12 agents), because Willis et al. found welfare falls with group size.
- **Cost.** $0.

### L7. Talk, contagion, and how the overseer responds (LLM agents, Fishery)

**Stage A: does talk make cheating spread?**

- **Question.** When agents can message each other and one agent visibly over-takes, do the LLM agents start over-taking more than they do without messages?
- **Design.** A 2 × 2 grid:
  - talk off or on: one message of up to 40 words per agent per round, broadcast to all, before requests;
  - planted over-taker absent or present: one of the two programmed agents takes its full request, as in L6.
- **What stays fixed.**
  - Every cell shows a public log of last round's catches per agent. Agents can always find out who over-took, so talk is the only change.
  - q = 1/6 and F = 12 t (e = 2 t). In L4 this gave 32.1% over-taking for gpt-oss, mid-range, so effects can show in both directions.
  - Model gpt-oss-120b, 4 LLM agents, 2 programmed, 20 rounds; the same populations in every cell.
- **Why 2 × 2.** The planted over-taker also lowers everyone's allowance through the shared cut (L6). That route is present with and without talk, so the **interaction** (cheater effect with talk minus cheater effect without talk) isolates what talk adds.
- **Hypotheses and falsifiers.**
  - **H1 (primary).** The interaction is positive: talk amplifies the cheater's effect on LLM over-taking. *Falsified* if the 95% interval of the interaction lies below +3 points or includes 0 with the point estimate ≤ 0.
  - **H2.** Without talk, the cheater still raises over-taking, through the shared cut, as in L6. *Falsified* if the simple effect is ≤ 0.
  - **H3 (descriptive).** In talk-plus-cheater games, messages name the over-taker and some propose matching it. Coded by a fixed keyword scheme, written into the protocol before any game, plus a blind second coder (a different LLM) on a sample.
- **Secondary outcomes.**
  - Over-take *tonnes*, not only the rate. The rate is measured against the allowance, which the shared cut moves.
  - Stock below MSY.
  - Efficiency and oversight efficiency, as in W1.
- **Independent unit.** The population. There are 4 LLM agents and 20 rounds inside each game, so decisions are not independent; rates are averaged per population first.
- **Sample size** [estimate, from the statistical-analyst calculator]:
  - Between-population SD of the over-take rate at mid-range rates in L4 is about 0.06 [verified from L4 intervals].
  - With 12 populations per cell, the minimum detectable effect is about 7 points for a simple effect and about 10–11 points for the interaction (80% power, α = 0.05).
  - Talk may add variance. If the pilot SD is above 0.09, n rises to 20 per cell when credit allows; otherwise the protocol reports the larger minimum detectable effect before the full run.
  - Pairing populations across cells should reduce variance, so these numbers are conservative.
- **Pilot.** 2 populations per cell, not analysed. Its gates:
  - messages parse;
  - comprehension checks pass;
  - the public log is used.
- **Stop rule.** If the pilot shows talk-off over-taking below 10% or above 90% at F = 12 t, change F once by the frozen rule (to 6 or 18 t) and say so. No reruns for significance.

**Stage B: how should the overseer respond?** (after Stage A, under talk plus cheater)

| Arm | Overseer response | Prediction |
|---|---|---|
| B0 Shared cut | As now: the cut is set from total requests and the stock (= Stage A's talk-plus-cheater cell) | Baseline |
| B1 Targeted allowances | The cut for compliant agents ignores the known cheater's excess, and the cheater's own allowance falls (S5's `memory_cut`) | Less over-taking than B0; removes the L6 route |
| B2 Leniency | Any agent may report another. A report triggers an audit of that agent, and a reporter who reports truthfully pays no fine that round | Reports appear in messages, and over-taking falls. Cartel-leniency logic (Motta and Polo, 2003) |

- **Hypotheses.** B1 < B0 and B2 < B0 on LLM over-take rate, Holm-corrected.
  - *Falsified* for an arm if its 95% interval of the difference includes 0 or is positive.
  - **Watch for:** B2 creating false reports. Count reports that the audit does not confirm.
- **n.** 12 populations per new arm (B1, B2). B0 is reused.

**Cost** [estimate]: about 240 calls per game with messages, roughly $0.06 per game.
- Stage A pilot: 8 games.
- Stage A: 48 games.
- Stage B: 24 games.

That is about 80 games, **about $5**.

### L8. A ring commons with proximity and plots that die (LLM agents, Forest)

- **Question.**
  1. Do LLM foresters over-cut more when their own chance of being audited is lower because of where they sit?
  2. Does a known risk of permanently losing a plot reduce over-cutting?
- **Design.**
  - **Proximity** (within each game): the reviewer's station sits at plot 0. Audit chance falls with ring distance: q = 1/3 at distance 0–1, 1/6 at 2, 1/12 at 3, so the average stays 1/6. Each agent is told its own q.
  - **Irreversibility** (between cells): reversible plots (as now) or plots that die for good below a health line, with agents told so. 2 cells.
  - Fixed fine F, set by a pilot rule so that the middle distance gives partial deterrence. L6 ran at F = 0, so a Forest fine has not yet been calibrated.
- **Hypotheses and falsifiers.**
  - **H1 (Becker).** Over-cutting rises as the agent's own q falls (distance 0–1 < 2 < 3). *Falsified* if over-cutting at distance 3 is not above distance 0–1. This comparison is within population, so it has more power.
  - **H2 (competing).** Two outcomes are possible, and the frozen protocol states both:
    - *forward-looking:* irreversibility lowers over-cutting;
    - *myopic:* no difference, because L3 and L4 found the models respond to the one-round gain, not the whole-game gain (Update 8).
- **First:** A1 must admit the irreversible ring with programmed agents.
- **n.** 12 populations per cell [estimate]. Minimum detectable effect about 7 points between cells; smaller within games.
- **Cost.** 5 LLM agents × 2 calls × 20 rounds ≈ 200 calls, about $0.05 per game. 24 games plus a pilot: **about $1.5–2**.

### L9. An LLM overseer (programmed agents first)

- **Question.** Can an LLM do the reviewer's job as well as the rule?
- **Stage 1** (programmed agents, so the only LLM is the overseer):
  - Fishery with the S-series populations (MSY target; 0, 2 or 4 cheaters).
  - **L9a: LLM chooses the cut.** Each round the LLM chooses the shared cut from {1, ¾, ½, ¼, 0}, given the stock, the requests and the growth rule. Compared with the rule on the same 32 populations.
  - **L9b: LLM writes the overseer policy as code** once per population (Willis et al.'s strategy-as-code), which then runs. The code is inspectable before use.
- **Outcomes.**
  - harm rate;
  - efficiency;
  - oversight efficiency (W1);
  - agreement with the rule's choice;
  - consistency across repeated calls.
- **Hypotheses.**
  - **H1.** The LLM overseer's harm rate is within 5 points of the rule's. *Falsified* if it is worse by more.
  - **H2.** L9b matches L9a at a fraction of the calls.
- **Stage 2** (next paper):
  - LLM agents under an LLM overseer;
  - the same model or a different model as overseer (untrusted monitoring, Greenblatt et al., 2024);
  - whether agents can message the overseer (persuasion). SovSim warns that giving one LLM authority can hurt the commons.
- **Cost.**
  - L9a: 32 populations × 2 mixes × 20 rounds ≈ 1,300 calls, **about $0.35**.
  - L9b: about 64 calls.

### G1. A small spatial grid game (next paper, design only for now)

- A 12×12 to 15×15 text grid with 3–5 apple patches and the SocialJax regrowth rule, so a patch eaten bare is lost.
- 4–6 LLM agents with high-level moves ("go to patch B") carried out by a path planner. The overseer acts every K steps.
- **Gates before any LLM oversight run:**
  - A1 admission;
  - a navigation competence check (agents told to be sustainable, with no overseer, must harvest well and keep patches alive);
  - a random-removal control for any zap action.
- **In parallel:** propose putting the overseer into OpenSocialJax with its authors.

---

## Order, time and credit

| Week | Work | Credit |
|---|---|---|
| 14–20 Oct | **W1** and **A1**; add the new sources to the literature ledger; fix the SocialJax citation (now ICLR 2026); rewrite related work and the novelty claims | $0 |
| 21–27 Oct | Freeze the **L7** protocol, write code and tests, run the pilot; **L9a** in parallel | about $1 |
| 28 Oct–3 Nov | **L7 Stage A** in full, run remotely on GitHub lanes; freeze **L8** | about $3 |
| 4–10 Nov | **L7 Stage B**; **L8** pilot and run if credit allows | about $3–4 |
| 10–20 Nov | Write the workshop paper: current results plus W1, with L7 if finished and clean | $0 |
| After 20 Nov | L8 and L9 for the January paper; G1 design and the OpenSocialJax conversation | from the next monthly credit |

- **Credit** [estimate]: about $7 of this month's $60 is left. The plan needs about $8–9 in total, so L7 runs first and L8 may move to the next month's credit.
- **What goes into the 20 November paper:** current results, W1, A1, and the repositioning against the new papers. L7 goes in only if its full run is finished and analysed by about 10 November. Otherwise it goes to January.

---

## Where we stand

- **High confidence:**
  - LLM agents in Fishery stop over-taking at the fine predicted from their one-round gain, out of sample and under rewording;
  - a zero-fine audit announcement raises over-taking compared with silence.
- **Moderate confidence** (one model, 20 populations): where deterrence is partial, fine size counts for more than audit chance.
- **Moderate confidence for the effect, lower for the mechanism** (one model; mechanism post hoc): one cheater raises others' over-taking through the shared cut.
- **Untested:**
  - contagion through talk;
  - which overseer response contains it;
  - whether proximity-dependent audits shift behaviour as Becker predicts;
  - whether irreversibility matters to myopic agents;
  - whether an LLM can do the reviewer's job;
  - DeepSeek's 36 t prediction.
- **Next concrete step:** W1 and A1, this week, at no credit cost.
