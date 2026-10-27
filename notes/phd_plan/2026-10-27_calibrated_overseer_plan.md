# PhD contribution plan: a calibrated overseer for populations of AI agents

Written 27 October 2026. The workshop paper is paused (Ameer's decision, 27 Oct). This plan replaces the paper-driven ordering of `research/commons_platform/2026-10-14_decision.md` §6. The platform decision itself stands: the overseer is the contribution, and existing games are the test environments.

---

## 1. The problem (as the literature states it)

- **AI control and scalable oversight.** Only a small share of an AI system's actions can be checked by a trusted party: audit budgets of 0.5–2% (Greenblatt et al., ICML 2024; Bhatt et al. 2025; Kutasov et al. 2025). This work measures how much bad behaviour a monitor **detects** under such a budget. Whether the *threat* of a check **deters** misbehaviour, and how to set the consequence so that it does, is not evaluated. Gans (2026) argues in theory that rare random audits can deter AI agents.
- **Multi-agent LLM systems.** LLM agents sharing a resource mostly fail to sustain it without enforcement (GovSim, Piatti et al. 2024; Willis, Zhao, Du & Leibo 2026). Willis et al. call for "cooperation-supporting institutions" and leave them untested. Existing LLM enforcement studies use peer reports and escalating penalties (Ye & Steinhardt 2026), voted restitution (Bell & Madisetti, ICML 2026) or stated penalties for a single agent (Okamoto et al. 2026). None calibrates the consequence, and none evaluates the overseer under a budget, against adaptive agents, or across environments.
- **Optimal auditing.** Inspection and audit games give the audit probability and punishment a rational auditee should face (Avenhaus et al. 2002; Blocki et al., IJCAI 2013; Das, Yu & Zhang, arXiv:2604.25085, 2026). They assume known, rational agents and are untested on LLM agents, whose gains and responses are unknown in advance.

**The problem, in one sentence.** An overseer of many AI agents can check only a few of their actions. How should it choose *whom* to check, *how often* and *what consequence* to impose, so that misbehaviour stays below a target at the lowest cost, when it does not know in advance what the agents gain from misbehaving or how they respond?

## 2. The contribution: a calibrate-then-deploy overseer, and a protocol to evaluate it

The idea comes directly from what has already worked: L3/L4 and R1a predicted the deterring fine from behaviour without a fine, then confirmed it. The PhD turns that one-off procedure into a general overseer.

**The overseer** (built on the existing `overseer/` package):
1. **Probe.** Observe the agents for a short period with no consequence, in *shadow mode* (rules not announced, because a zero-fine announcement backfires; L4, F4). Estimate each agent's gain from a violation and how often it violates.
2. **Design.** Choose the audit rate, the targeting and the fine (or memory, when fines are capped) at the lowest expected cost such that the expected consequence exceeds the estimated gain for the target share of agents. This step has three inputs:
   - the theory: Becker, inspection games and the budget;
   - the known corrections: the one-round gain for LLMs (L3); lone vs group cheaters (G1b, F1); the shared cut spreading harm (L6);
   - the budget and fine caps.
3. **Deploy and announce** the rule, committing to it as in an inspection game.
4. **Verify and recalibrate.** Random audits give an unbiased estimate of the violation rate, with a confidence bound. If the bound exceeds the target, raise the consequence (graded); if it is far below, lower the cost. Targeted audits, for example by an LLM monitor, are added only on top of a random floor, because aiming audits at reports failed (T1).
5. **Report** safety (violations, harm), usefulness (welfare of compliant agents) and cost, as in AI control. It also reports the behaviour of agents, not the share of "good" types, because compliance is not conversion (R1b, F16).

**The evaluation protocol:**
- propensity tests with normally prompted LLMs, and worst-case tests with a red-team ladder;
- several environments: the text commons, Willis et al.'s CPR, and SocialJax later;
- held-out populations and model families;
- the audit budget reported as % of actions;
- every prediction frozen before the run.

This is released as code: the `overseer` package plus the protocol.

**Claims the thesis must establish** (each falsifiable):
- **C1, calibration out of sample.** On held-out populations, models and environments, the deployed overseer keeps violations at or below the target, within a stated tolerance.
- **C2, cost.** At the same safety, it is cheaper (fewer audits, smaller consequences) than fixed rules and than an AI-control-style monitor-only protocol.
- **C3, robustness.** It holds, or recovers through recalibration, when agents are not told the audit odds, adapt to the overseer, or change during deployment (red-team ladder).
- **C4, scarce budgets.** It works at 1–5% audit budgets, and it reports where it stops working (fine caps, detection limits).
- **C5, who oversees.** An LLM can run the design and recalibration steps, and the thesis reports where it is worse than the rule: a weaker overseer, a same-family overseer, and agents that talk.

## 3. Thesis chapters

| Chapter | Question | Status |
|---|---|---|
| 1. When does checking deter AI agents? | Can the deterring consequence be predicted from behaviour? Where does the classical prediction fail for LLMs? | **Largely done:** L2–L4, R1a, R1b, Phase 0, with the findings F1–F16. It needs only writing, plus L5b if credit allows. |
| 2. The calibrated overseer | Does calibrate-then-deploy meet a target out of sample, cheaply, and robustly (C1–C4)? | **To do** (Nov–Mar). The core contribution. |
| 3. Who oversees? | An LLM overseer, the capability gap, collusion and talk (C5) | To do (Mar–May) |
| 4. Generalisation | Scale (Willis), space and irreversible harm (SocialJax); the benchmark release | To do (Jun–Sep); spatial needs a GPU |

## 4. Schedule (November 2026 – September 2027)

Each study gets a frozen protocol before code. Costs are estimates in Ollama credit (gpt-oss about $0.04–0.06 per game) at $60 per month, assuming the Pro credit refills monthly; check the renewal date at ollama.com/settings.

| Month | Work | Studies | Credit | Gate at the end |
|---|---|---|---|---|
| **Nov 2026** | Build the calibrated overseer (probe, design, verify and recalibrate) in `overseer/`. Validate it offline with programmed agents, and by **replaying stored LLM games** (L2–L4 data give gains and responses for free). Apply for closed-model research credits. | O1 (design and recalibration on programmed agents, $0); O2 (offline replay on L2–L4, $0) | ~$0–5 | The overseer meets its target on programmed agents out of sample. If not, fix the design before spending credit. |
| **Dec 2026** | The first live test with LLM agents. Then the test that matters most: **agents not told the odds** (they must learn them), and red-team rungs 1–3 (programmed best response; LLM told the rule; LLM told to evade). | O3 (live calibration on held-out populations, gpt-oss and Nemotron); O4 (unknown odds); O5 (red team 1–3) | ~$25–35 | **The key gate.** If calibration fails when odds are unknown, Chapter 2 becomes "why and how to recalibrate", which is still a result, but the plan changes. |
| **Jan 2027** | Scarce budgets: 1–5% audits, with fine scaling, a random floor and LLM-monitor targeting (cheap screen plus a few true audits). Drift: agents that change mid-deployment. | O6 (scarce budgets, C2/C4); O7 (recalibration under drift) | ~$30–40 | Is it cheaper than the baselines at the same safety? |
| **Feb 2027** | Robustness: more families (DeepSeek's untested 36 t from L5; one or two more open; closed if credits arrive); red-team rungs 4–5 (strategy search, optimised agents); the R1b replication with a second model. | O8 (families); O9 (red team 4–5); R1b-rep | ~$40–60 | C1 and C3 across ≥ 3 families |
| **Mar 2027** | Scale, on CPU and free: Willis CPR with group size 4–256 (R1c), the overseer written as code by an LLM at scale (R1d), and removing the overseer after evolution (compliance is not conversion, directly). Start writing the main paper. | R1c, R1d, R1e | ~$1–5 | — |
| **Apr 2027** | An LLM as overseer: it runs the design and recalibration steps. Same vs different family; weaker overseer than agents (capability gap). | O10 (LLM overseer); O11 (same family); O12 (capability gap, small grid) | ~$40–60 | Does C5 hold, or where does it break? |
| **May 2027** | **Main paper submitted** (NeurIPS 2027, about mid-May, estimate): "A calibrated overseer for populations of AI agents", Chapters 1–2 plus the core of 3. | — | — | Submitted |
| **Jun–Jul 2027** | Talk and collusion: L7 talk × planted cheater, leniency, and a monitor that reads messages. | L7, O13 | ~$30–50 | — |
| **Jul–Sep 2027** | Spatial generalisation: SocialJax/OpenSocialJax port (S0), RL agents trained against the overseer, LLM agents with high-level moves. Needs a GPU. Benchmark release. | S0–S4 | GPU, ~$40–80 | The thesis chapter on generalisation |

**Running alongside, every month:** keep the trackers current (checklist in `notes/claude_audit_20261005/README.md`), and check novelty before each submission (`research/commons_platform/refresh_targets.md`).

## 5. What existing work becomes

| Existing | Use |
|---|---|
| L2–L4, L3 frozen predictions | Chapter 1 evidence; probe data for O2 (offline replay) |
| Phase 0 (G1/G2, W1) | Grounding for every chapter; W1's oversight efficiency and break-even cost become the cost metrics |
| G1b (lone vs group) | A design correction the overseer must handle (heterogeneous gains) |
| L6 (shared cut spreads harm) | A design rule: target allowances, never spread the cost |
| T1 (targeting at reports fails) | A design rule: keep a random audit floor |
| R1a, R1b | Chapter 1 at scale; evidence that the probe-predicted consequence works through both selection and compliance |
| `overseer/` v0 | The base the calibrated overseer is built on |
| The paused workshop draft | Chapter 1 text; can be revived as a separate paper later |

## 6. Risks and how they are handled

| Risk | Handling |
|---|---|
| "The LLMs just do the arithmetic you gave them" | O4 (unknown odds) and shadow-mode probes in Dec, early, because this is the most damaging criticism |
| Too few model families | O8, and research credits for closed models applied for in Nov |
| Agents not trying to beat the overseer | Red-team ladder O5 and O9 |
| Budgets unrealistically generous | O6 at 1–5% |
| "Classical economics applied to LLMs" | Lead with what classical theory does not predict (F4, F16, F1, L3's one-round gain, L6), and with the overseer as a working system |
| Credit | Monthly allocation above; heavy-reasoning models only when budgeted |
| No GPU | Spatial work scheduled last; everything before it is CPU or LLM-API only |

## 7. First steps (this week)

1. Ameer: confirm this plan, or change it.
2. Write the design spec for the calibrated overseer: the probe, the design rule, and the recalibration test with its confidence bound.
3. Freeze the O1 and O2 protocols (programmed agents and offline replay; $0).
4. Apply for closed-model research credits (Anthropic's External Researcher Access, Google's Gemini Academic Program).
