# Next steps after the 13 October catch-up

This note covers the feedback from the 13 October 2026 catch-up and how I would answer each point. It says what the project already has, what is missing, and what an experiment would look like. It also has a literature review focused on AI papers and an overall verdict.

"Richard and Janny's paper" is taken to be Willis, Zhao, Du and Leibo, *Evaluating Collective Behaviour of Hundreds of LLM Agents* ([arXiv:2602.16662](https://arxiv.org/abs/2602.16662)).

## Final feedback

**The feedback holds together.** Every comment moves the project from "does deterrence work on language-model agents?" to "how should an overseer of AI agents be designed, and how do we know it is working?" That is a thesis arc, not a single paper. It is a good direction because the current results already answer the first question.

**1. Protect the current paper.**
- The 20 November workshop and the January full paper should take only what costs no new games:
  - welfare net of oversight cost and a break-even audit cost (section 1);
  - the oversight-efficiency metric (section 4);
  - payoffs split by role (section 3);
  - the new citations.
- Everything else is the next study or the next paper. Adding a talk channel, a grid game and a language-model overseer to the current paper would weaken it.

**2. The literature has moved, so narrow the novelty claims.** Three recent papers sit close to this work:
- **Bell and Madisetti (ICML 2026)** have one adversarial fisher in a language-model commons, and a voted sanction that returns its catch to the pool.
- **Okamoto et al. (AIES 2026)** show that stating a penalty can turn a duty into a cost-benefit choice, which is close to the zero-fine finding.
- **OpenSocialJax** (Guo, Fang, Leibo and Du; arXiv, 9 October 2026) already puts language models into SocialJax Harvest and Clean Up.

None of them has what this project has:
- **(a)** a quantitative deterrence threshold, predicted before the games and confirmed on new games and reworded rules;
- **(b)** audit chance separated from fine size at a fixed expected fine;
- **(c)** a no-audit control showing that announcing audits with no fine makes over-taking worse, in a multi-agent commons;
- **(d)** harm spreading through the overseer's own rule (L6), a route the contagion literature has not named.

The paper should claim these four things clearly and cite the three papers above.

**3. "More AI papers, fewer economics papers": yes for positioning, no for the hypothesis.**
- AAAI, IJCAI and ICML reviewers judge the work against AI papers, so the related-work section should be mostly AI. The review below gives 58 checked papers for that.
- Keep Becker, though. The deterrence threshold comes from it, and a prediction frozen in advance and then confirmed is the strongest thing in the paper.
- The framing to use: economics supplies the testable prediction; AI supplies the setting and the gap.

**4. The spatial game is worth doing, as the next paper.**
- The novelty of putting language models into SocialJax is gone (OpenSocialJax). An overseer inside a spatial commons is still open.
- A full grid game would make the threshold depend on each agent's position, so it could only be estimated as a band. It would also cost roughly $80–200 per 40-game cell.
- Start with the ring of plots: plots that die for good, audits that depend on position, and reports from neighbours. Move to a small grid with high-level moves after that.
- Yali Du co-wrote both SocialJax and OpenSocialJax. Proposing an overseer for OpenSocialJax is a natural collaboration.

**5. The strongest next study is talk × planted cheater × overseer response** (section 2), including leniency for reporting. It tests the route L6 could not, imitation through messages. It extends Willis et al. directly (communication is their stated future work) and fits the remaining credit.

**6. A language-model overseer is a good question, but keep the rule as the benchmark** (section 5). The policy-as-code version connects to Richard's paper. SovSim warns that giving a language model authority over others can hurt the commons, which is a hypothesis worth testing directly.

**7. Scientific traps to keep in mind:**
- a metric that rewards low harm alone, which is met by banning all harvest;
- zapping that protects the resource only by removing harvesters (Perolat et al., 2017);
- poor navigation that looks like restraint;
- SocialJax's default shared reward, which removes the dilemma;
- not saying whether fines are lost resources or transfers.

## 1. The cost of monitoring: is it worth it?

**Already in the project.**

- **S4** (programmed agents) charged a cost *c* for each audit. As *c* rises, the best audit rate falls. A fine combined with the reviewer's memory deters at half the audit rate a fine alone needs (0.05 against 0.10). This combination also gives the most harvest once audits cost something.
- **L4** (language models): at the same expected fine, rare audits with a large fine deterred more than frequent audits with a small one (28% against 41% over-taking, at an expected fine of 2 t).

**Grounding.** The classic results on optimal enforcement say the same thing. Audits cost real resources, while fines are close to free, so enforcement should be cheap: audit rarely and fine heavily. These are Becker (1968) and Polinsky and Shavell (2000). L4 shows the cheap policy is also the one that works better on these models, which is a useful practical point.

The standard limits still apply:

- a fine cannot be larger than what an agent can pay;
- audit errors make very large fines unfair.

**Missing.**

1. **Where fines go.** In the runner (`experiments/oversight/run_l2_llm_agents.py`), a fine is only taken off the fined agent's score. It is not paid back to the group and does not change the fish stock. The paper has to state which welfare count it uses:
   - fines as **lost resources** (welfare = harvest − fines);
   - fines as **transfers** to the reviewer (welfare = harvest).

   At deterrent fine levels, fines are rarely paid, so the two counts mostly differ in the cells where cheating continues.
2. **The full cost of oversight.** This has three parts:
   - the audit cost;
   - the harvest honest agents lose to an over-cautious shared cut (L6 shows this can be large);
   - for a language-model reviewer, the tokens it uses.
3. **A single answer to "good or bad".** Auditing at rate *q* is worth it when it saves more welfare than it costs: W(q) − W(none) > c · q · N · T, with N agents and T rounds. Rearranged, this gives a **break-even cost per audit**. If audits cost more than that, it is better not to audit. One number per game, model and fine answers the question.

**Cost.** No new games. Everything above can be computed from the L4 and L6 data already stored.

## 2. Communication: does cheating spread, and how should the overseer respond?

**Already in the project.**

- **L6:** showing foresters their neighbours' cuts did *not* make them copy the over-cutter (+2.2 points, not significant). The harm spread through the reviewer's shared cut, which roughly halved everyone's allowance.
- **S1** `peer_collude`: a scripted neighbour that covers for a cheater. This is a narrow, programmed form of collusion. Lying paid only in this setting (+110 in Fishery).

**Two separate questions.** The comment links two mechanisms, and they need different tests.

1. **Payoff-biased imitation**, the mechanism in Willis et al.: agents copy strategies that earn more. On this view, cheating spreads when others can *see that it pays*. Testable prediction: if fines are announced publicly, cheating should spread less, because everyone can see it does not pay. This connects deterrence to cultural spread.
2. **Conditional cooperation and erosion of the norm**: people cooperate when others do, and stop when others visibly break the rule. These are Fischbacher, Gächter and Fehr (2001) and Keizer, Lindenberg and Steg (2008). A message such as "agent 3 over-fished" is far more noticeable than a number in a table, which may be why L6 found no imitation.

**Communication cuts both ways.**

- Ostrom, Walker and Gardner (1992) found that talk alone raises cooperation over a shared resource, and talk combined with sanctions raises it more.
- On the other side, language models can collude, even in hidden ways (Motwani et al., 2024), and they collude in markets (Institutional AI, 2026).

**An important design point.** With perfect audits, talk can only coordinate what agents do. It cannot hide over-taking. Collusion becomes possible only when the reviewer relies on what agents *report*, as in S1. The design must decide which case it is testing.

**Proposed study.**

- Fishery with gpt-oss, as a 2 × 2 design:
  - talk channel off or on;
  - a planted over-taker absent or present.
- About 10 populations per cell.
- Code the messages for accusations, imitation and cover-ups.
- A third factor, how the reviewer responds:

| Response | What it does | Expected effect |
|---|---|---|
| Shared cut (current) | Lowers everyone's allowance after harm | L6 suggests it spreads the harm (collective punishment) |
| Targeted memory | Lowers only the caught agent's allowance | Already built; should contain spread |
| Graduated sanctions | Fines rise with repeated offences (Ostrom's design principles) | Deters persistent cheaters |
| Leniency | A reduced fine for reporting yourself or another agent | Breaks collusion by making agents race to report, as in cartel policy (Motta and Polo, 2003) |

**Cost.** About $3–5. This is an estimate: messages make the prompts longer.

## 3. Individual utility, social welfare, and choosing environments

**Already in the project.** Efficiency, equality (1 − Gini), survival and harm, following Perolat et al. (2017) and GovSim.

**Additions.**

- **Split payoffs by role:** the cheaters' gain, the honest agents' loss, and total welfare net of oversight cost. L6 is the example: harm was contained, but honest agents paid for it.
- **An admission test for environments.** This is how to decide whether to accept or reject an environment, and an answer to the "toy setting" criticism. An environment is informative only if all three conditions hold:
  1. **A real dilemma:** cheating pays the individual, and everyone cheating is worse than everyone complying.
  2. **Room to improve:** with no oversight, the resource is damaged but collapse is not certain.
  3. **Enforcement can work:** some allowed fine deters.

  All three can be checked numerically for Fishery, Forest and River. Willis et al. show that welfare falls sharply with group size, so group size belongs in the test. The old Harvest results are a warning: cheating never paid there, so deterrence could not be tested (S2). That game would have failed condition 1.
- **Interaction structure as an explicit factor:** who sees whom, who talks to whom, and who harms whom (shared stock, ring of plots, river chain).

## 4. A metric that shows the overseer is working

**The trap.** A reviewer judged only on harm would set the shared cut to zero. Nobody would fish, so there would be no harm and no welfare. Any metric must pair harm with welfare.

**Headline metric: oversight efficiency.** It is the share of the welfare gap the reviewer closes, after its own costs:

> oversight efficiency = (welfare with reviewer − welfare with no oversight) ÷ (best possible welfare − welfare with no oversight)

- 1 means the best achievable outcome.
- 0 means no better than no oversight.
- Below 0 means it does harm.

L4 already contains a negative case. Audits announced with a 0 t fine gave more over-taking than never mentioning audits: 58% against 30% for gpt-oss, and 64% against 46% for Nemotron. So the metric catches real failures, and the right baseline is the no-audit control (N0), not the zero-fine cell (E0).

**Diagnostics alongside it.**

| Diagnostic | Question | Already measured in |
|---|---|---|
| Detection | Precision and recall of catches | S1, T1 |
| Deterrence | Over-taking compared with no oversight | L4 (N0) |
| Collateral cost | What compliant agents lose | L6 (post hoc) |
| Robustness | Performance against cheaters who adapt to the rule | S4, S5 |
| Calibration | Does behaviour match what the reviewer's model predicts? | L3/L4: the frozen 30 t prediction held out of sample |

## 5. A language-model overseer

**Caution.** Do not just swap the reviewer for a language model. The fixed rule is what makes L4 interpretable: the audit chance and the fine are known exactly. Treat it as its own question, "can a language model do the reviewer's job?", with the rule kept as the benchmark.

**Steps, from least to most autonomy:**

1. The language model chooses the shared cut each round.
2. It chooses whom to audit, compared against T1.
3. It sets fines.
4. **It writes the reviewer's policy as code before the game.** This is Willis et al.'s strategy-as-code idea applied to the overseer: the policy can be read and tested before it is used.

**Factors to cross with it.**

- **Same model or a different model** for reviewer and agents. This is the "untrusted monitoring" question in AI control (Greenblatt et al., 2024), and the weak-judge, strong-agent question in scalable oversight.
- **Whether agents can message the reviewer**, which tests whether it can be persuaded. This links to section 2.

**Cost.** One extra call per round, roughly +15–20% per game.

## 6. Making it an actual game: space, proximity, and destroying the environment

**How SocialJax does it** (Guo et al., ICLR 2026; details from its source code; full notes in `spatial_game_options.md`):

- **Commons Harvest.** An empty cell regrows an apple with a chance set by the number of apples within two steps:
  - 0.025 per step with 3 or more nearby;
  - 0.005 with 2;
  - 0.001 with 1;
  - **0 with none.**

  So a patch eaten bare never comes back. That is the "destroying the environment" mechanic: the harm cannot be undone.
- **Agents.** Seven agents move on a grid, see an 11 × 11 window around themselves, and can zap another agent, which sends it back to a spawn point.
- **Episodes** last 1,000 steps.
- **A trap for this project.** The code's default reward is *shared*: every agent gets the group's total. That removes the dilemma. A study of audits must switch to individual rewards.
- **Clean Up.** Apples grow only while a river is kept clean, and growth stops at 40% dirt.

**What already exists here.**

- **Harvest** (`fishery_sim/harvest.py`, March 2026): a *ring of patches*, not a grid, with no movement. Over-harvest damages the two neighbouring patches, and there are numeric messages and credit transfers. A patch at zero can still recover through weather noise, so there is no true destruction yet.
- **Forest** (L6): the same engine, with messages off.
- **A SocialJax Clean Up adapter** (`fishery_sim/cleanup_oversight.py`, September 2026). It passed the mechanics check but failed policy admission. The starting map was already dirty, and neither policy reached the safe state or harvested an apple in 180 steps. So it never produced a usable contrast between safe and risky behaviour.

**Language models in grid games so far.**

- **OpenSocialJax** (9 October 2026) gives eight language models a labelled 11 × 11 text grid and calls each agent once per step, up to about 10,000 calls per episode. Every team ate the orchard bare. There is no overseer.
- **Mosquera et al. (2024)** found a symbol grid failed and switched to listing nearby objects with coordinates. One simulation cost $0.78–15.63 and took 14–213 minutes. The agents still ate the last apple.
- **Hypothetical Minds (ICLR 2025)** cost about $10 and 45 minutes per 1,000-step episode with GPT-4.

**What a game would add scientifically.**

1. **Irreversible harm.** Does a known risk of losing a patch for good deter more than a fine?
2. **Proximity.** If the overseer sees only nearby agents, or neighbours can report each other, do agents take more when no one is close?
3. **Displacement.** Do audits move over-taking to patches the overseer cannot see?
4. **Peer sanction.** Does zapping or reporting by neighbours replace or complement central audits? In the human version, costly punishment did not raise harvests without communication (Janssen et al., 2010, *Science*).

**What it would cost scientifically.**

- **The threshold stops being a single number.** In a grid, the gain from over-taking depends on position, travel time, the apples left and nearby competitors. The fine that should deter becomes a band estimated state by state, so a clean test like L4's 30 t prediction is lost.
- **Navigation looks like restraint.** An agent that cannot reach apples looks restrained, and one that stumbles onto the last apple looks greedy. Agents must first pass a competence check: harvesting well with no overseer.
- **Zapping protects the stock by removing harvesters** (Perolat et al., 2017). It needs a control in which agents are removed at random.
- **More noise per game**, so more games are needed for the same power.

**Three options.**

| Option | What changes | Questions it answers | Clean threshold? | Calls per game | Cost per game |
|---|---|---|---|---|---|
| **(i) Extend the ring** | Patches that die for good below a health level; detection that depends on position; neighbour reports or sanctions; optionally cutting an adjacent plot (poaching) | Irreversibility, proximity, peer sanction, displacement | **Yes** | 200–300 | about $0.05–0.08 |
| **(ii) Small text grid** | 12×12 to 15×15 grid, 3–5 apple patches, SocialJax regrowth rule; high-level moves ("go to patch B") run by a path planner; overseer acts every K steps | All of (i), plus real movement and seeing others eat the last apple | As a band only | 100–300 (750–1,000 if the model picks every step) | about $0.10–0.30 |
| **(iii) Full SocialJax or Melting Pot** | The native game with an overseer layered on top; trained agents as a fixed background population | All of (ii), plus direct comparability with the multi-agent RL literature | No | about 7,000 | about $2–5, so $80–200 per 40-game cell |

**Recommendation.** Option (i) for now, because it keeps the measurement that makes this project distinctive. Option (ii) as the next paper. Option (iii) only as a collaboration with the OpenSocialJax authors.

The ring can honestly be framed as a *network commons*. Many real forests, fisheries and irrigation systems are governed this way: by who neighbours whom, not by grid coordinates.

## 7. Literature review (AI papers)

The full review is in `literature_review_ai.md`: 58 papers, each checked against its arXiv, proceedings or publisher page, grouped by theme, with how each bears on this project.

**The papers closest to this project, and how it differs.**

| Paper | What it does | How this project differs |
|---|---|---|
| Bell and Madisetti, *Restitution Over Deterrence* (ICML 2026) | One adversarial fisher collapses a language-model commons; a voted rule returning its catch to the pool restores cooperation (2.7–4.3× better) | A central overseer with random audits; tests fine size and audit rate against a predicted threshold |
| Okamoto et al., *Why Do AI Agents Break Rules?* (AIES 2026) | Stating a penalty can turn a duty into a cost-benefit choice ("enforcement information paradox"); single-agent procurement task | Shows the zero-fine version in a shared resource with many agents, against a never-mentioned control |
| Piatti et al., *GovSim* (NeurIPS 2024) | Language-model fishers talk between rounds; most collapse the resource | Adds an external enforcer; GovSim has none |
| Guzman Piedrahita et al., *Corrupted by Reasoning* (COLM 2025) | Reasoning models opt out of a peer-sanctioning group and free-ride | Sanctions are imposed, so there is no opt-out; isolates deterrence |
| Tewolde et al., *CoopEval* (ICML 2026) | Benchmarks repetition, reputation, mediators and contracts | Random audits with fines are not among their mechanisms |
| Willis et al., arXiv:2602.16662 (2026) | Hundreds of agents with strategies written as code; selfish strategies take over in large groups | Supplies the institution they call for; communication is their future work and the next study here |
| Guo et al., *OpenSocialJax* (arXiv, October 2026) | Eight language models in SocialJax Harvest and Clean Up; every team depletes the orchard | No overseer; that is the open slot |
| Paglieri et al., arXiv:2609.04170 (2026) | Cheating spreads through a swarm of 100 research agents via a shared library and messages; peers then expose it | L6 finds a different route: the overseer's shared rule |
| Greenblatt et al., *AI Control* (ICML 2024); Hua et al. (NeurIPS 2025) | Monitors with a budget: how much bad behaviour do they catch? | Measures whether the threat of being caught changes behaviour |
| Das et al., *Optimally Auditing Adversarial Agents* (AAAI 2026); Blocki et al., *Audit Games* (IJCAI 2013) | Theory for choosing audit probability and punishment | Tests that theory on language-model agents |

**The gaps, theme by theme.**

- **Language models in commons:** no paper tests a central overseer with random audits and fines, compares stopping points with a quantitative threshold, or separates audit chance from fine size.
- **Sanctions in multi-agent RL:** the work is on learned peer punishment, learned incentives or contracts, with agents trained over many episodes. Audit-game theory has not been tested on agents that only read the rules in their prompt.
- **Contagion:** studies show bad behaviour spreading through messages, shared memory or shared files. None looks at spread through the enforcement rule itself.
- **Oversight and monitoring:** monitors are treated as detectors with a budget. Deterrence, meaning whether being watched changes what agents do, is not measured. Evaluation awareness (Needham et al., 2025) and alignment faking (Greenblatt et al., 2024) offer possible explanations for the zero-fine result.
- **Metrics:** existing measures score outcomes (welfare, equality, survival), not the mechanism. This project can add three:
  - the distance between where agents stop and the predicted fine;
  - deterrence per unit of expected fine;
  - the share of harm passed on through the overseer compared with through peers.

**Corrections for the bibliography.**

- SocialJax is now ICLR 2026, so update the 2025 arXiv entry.
- "SanctSim" is the code name of *Corrupted by Reasoning*, not a separate paper.
- "Haupt et al." and "Christoffersen et al." on formal contracts are one paper (AAMAS 2023).

## 8. Plan

The plan built on this note, with study designs, sample sizes, costs and a week-by-week order, is in `plan.md`.
