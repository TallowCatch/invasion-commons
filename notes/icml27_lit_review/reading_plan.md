# Reading plan for the Yali meeting (Thu 29 Oct to Mon 2 Nov 2026)

Built from SciSpace's screening table (`~/Downloads/Screening of Lit.md`, 278 rows) and the abstracts in Zotero.

## What the screening already suggests (abstract level; reading must confirm it)

SciSpace marked each paper against five criteria:
- **(a)** learning agents (RL or LLM) in a social dilemma;
- **(b)** an outside party (planner, regulator, monitor) that intervenes;
- **(c)** limits on what that party can observe;
- **(d)** agents that adapt to it or evade it;
- **(e)** outcome metrics.

| Combination | Papers |
|---|---|
| (a) + (b): learning agents in a dilemma, with an outside party intervening | **29** |
| …of those, also (c), limited observation | **0** |
| …of those, also (d), agents adapting or evading | **2** |
| (b) + (c): an outside party with a monitoring budget | **16**, all AI-control, auditing and security-game papers. **None has (a).** |

So the candidate gap is: **the MARL planner and enforcement papers let the intervener see everyone; the budgeted-monitoring papers have no learning agents in a social dilemma.** The reading checks whether that holds.

### Where SciSpace got it wrong (fix before quoting the numbers)

- **The costly-observation RL papers** were marked MAYBE with no (c): *Monitored MDPs*, *RL with State Observation Costs*, *Act-Then-Measure*, *Active Measure RL*, *OCMDP*. Limited observation is exactly what they study, but each is single-agent. They are (c) without (a), which supports the gap.
- **Learning-agent papers given MAYBE or no (b):**
  - *Oracles & Followers* and *Inducing Stackelberg Equilibrium in SSDs* (a leader steering learning followers) should have (a)(b).
  - *LIO* is marked (e) only; it should also have (a).
- **13 papers had no abstract** (mostly economics classics). They're marked MAYBE for that reason only.

## 8 to read properly (≈75 min each; write the five-line note straight after)

| # | When | Paper | Why it's in | Check (line 3 of your note) |
|---|---|---|---|---|
| 1 | Thu night | Guo et al. 2025, **SocialJax** (arxiv.org/abs/2503.14576) | Your platform and the vocabulary: sequential social dilemmas, Harvest, Schelling diagrams | How "defection" is defined in Harvest; what is measured |
| 2 | Fri | Baumann, Graepel & Shawe-Taylor 2020, **Adaptive Mechanism Design** (doi 10.1109/IJCNN48605.2020.9207690) | A planner that learns rewards and penalties for learning agents in dilemmas; the closest RL version of your overseer | Does the planner see every agent's action? Does cooperation last when it stops? |
| 3 | Fri | Gasztowtt et al. 2024, **Large Legislative Models** (arxiv.org/abs/2410.08345) | **An LLM as planner, taxing RL agents in Commons Harvest** | It taxes by each agent's apples over the last 10 steps, so it observes everyone. What does it cost? |
| 4 | Fri | Guresti et al. 2023, **IQ-Flow** (arxiv.org/abs/2302.14604) | An "incentive regulator agent" in sequential social dilemmas | What the regulator observes; whether agents can exploit it |
| 5 | Fri | Sun et al. 2019, **A Regulation Enforcement Solution for MARL** (arxiv.org/abs/1901.10059) | Detecting and punishing agents that break regulations, without seeing their states | How defectors are detected; whether every agent is checked |
| 6 | **Sat, first** | Casheekar et al. 2026, **Learning to Report Unsafe Tasks in a Multi-Agent Game** (arxiv.org/abs/2610.09002) | **Closest to the gap:** random audits plus policy-gradient learners, with a condition on the audit rate | Is it a social dilemma? A budget? Do agents evade? **Does it already do your idea?** |
| 7 | Sat | Vinitsky et al. 2023, **Norms from public sanctions** (doi 10.1177/26339137231162025) | Sanctions in Melting Pot-style games, learned by agents; the MARL view of sanctions | Peer sanctions, not an overseer; what agents learn from seeing sanctions |
| 8 | Sat | Wang et al. 2019, **Deep RL for Green Security Games** (arxiv.org/abs/1811.02483) | A patrol with a limited budget against **best-responding attackers** on a grid (poaching, overfishing): budget plus adaptive adversaries | Not a dilemma among agents; borrow its red-team method (double oracle) |

**If one runs short or you have spare time,** read Ye & Steinhardt 2026, *Norm Enforcement for AI Agents* (arxiv.org/abs/2607.09766). Agents exploit simple enforcement there, and tracking each agent's reliability fixes it.

## 15 to skim (≈15 min each; Saturday afternoon). Fill in line 3 only.

**Setting:**
1. Leibo et al. 2017, *MARL in Sequential Social Dilemmas* (1702.03037)
2. Perolat et al. 2017, *MARL model of CPR appropriation* (1707.06600): zapping and the four metrics
3. Guo et al. 2026, *OpenSocialJax* (2610.13710): LLMs in SocialJax, no overseer

**Other fixes for the dilemma:**

4. Yang et al. 2020, *LIO* (2006.06051): agents learn to pay each other
5. Zheng et al. 2022, *The AI Economist* (doi 10.1126/sciadv.abk2607)
6. Christoffersen, Haupt et al., *Formal Contracts* (2208.10469)
7. Hughes et al. 2018, *Inequity aversion* (1803.08884)
8. Köster et al. 2022, *Spurious normativity* (doi 10.1073/pnas.2106028118)
9. Tang et al. 2026, *Adaptive Punishment for Cooperation* (2605.24516)

**Oversight with a limited budget:**

10. Greenblatt et al. 2024, *AI Control* (2312.06942): audit budget, safety vs usefulness, red team
11. Makins et al. 2026, *Multi-Agent AI Control* (2607.07368)
12. Parisi et al. 2024, *Monitored MDPs* (2402.06819): rewards not always observable
13. Gans & Holden 2026, *When Does Randomized Oversight Align AI Agents That Can Conceal?* (2609.38262)
14. Zhu et al. 2026, *Blindfold* (2610.37055): audits stay valid only if agents can't tell when they're audited

**LLM commons enforcement:**

15. Bell & Madisetti 2026, *Restitution Over Deterrence* (ICML 2026)

You already know Willis et al. 2026 and GovSim, so put them on the map without rereading.

## The five-line note (one per paper)

1. **Question:** what they ask, in one sentence.
2. **Setup:** environment, agent type, who intervenes.
3. **What the intervener sees, and whether agents adapt to it.**
4. **Result:** one sentence with a number.
5. **What it leaves open for me:** their limitations, and anything bearing on the gap.

## Schedule

| When | Do |
|---|---|
| Thu night | Paper 1 |
| Fri | Papers 2–5 |
| Sat | Paper 6 first, then 7 and 8, then the 15 skims |
| Sun | Write the gap in one sentence and try to break it (SciSpace step 5); send the notes, and the map and 3 slides get built |
| Mon morning | Fix the slides and rehearse |
