# Project history: from commons governance to scalable oversight

**This is the project's running tracker.** It records each piece of work: what
was done, why, and what came out, with numbers. Update it whenever an
experiment finishes.
- Last updated: 7 October 2026 (after R3 and T1).
- February to September are summarised in the timeline below.
- The October experiment series has its own section, with every number
  verified against saved run files.

This is a research history, not a list of every code commit. Dates through
June come from Git; later dates come from dated experiment notes. A feature
being implemented does not mean that its proposed scientific claim has been
demonstrated. The linked closeouts and the current manuscript take precedence
over older plans and promotional descriptions.

## The question in plain language

Several agents use a resource that can be damaged by their combined actions.
Someone reviews their requests and may restrict them. The research question
has become: **what must the reviewer know, and how much checking can it afford,
to stop harmful combined requests without needlessly reducing useful work?**

This remains connected to **scalable oversight**: supervision becomes difficult
when there are many interacting actions, limited checking time, and potentially
more capable actors. But the strongest *completed* September experiment isolates
the information-and-checking part. It has not yet demonstrated that the same
reviewers remain effective as actor capability increases. Earlier experiments
tried to represent that pressure with progressively stronger strategy search;
the simple actor-rank-minus-overseer-rank score was not a validated measure of
their relative ability. Calling the current result a full solution to scalable
oversight would overstate it.

The [current manuscript](paper/paper_v5_scalable_oversight_commons/main.tex)
uses the narrower title **Evaluating Oversight in Sequential Commons**. A
defensible one-sentence description is: *we use repeated shared-resource games
to test when a reviewer with limited information accepts harmful joint actions
or restricts safe ones.* A stronger future paper about **scalable oversight
under capability pressure** needs an additional, controlled actor-strength
test against the same fixed reviewer protocol.

## Terms used here

- **Agent:** one participant making a resource-use request.
- **Strategy or policy:** a rule for choosing requests from observations. In
  Harvest it is usually a small, inspectable set of numeric settings, not a
  language model choosing every action live.
- **Reviewer or overseer:** a rule that inspects some evidence and may leave a
  request unchanged or scale it down.
- **Local check:** evaluates a participant or patch using limited information.
  In later experiments, local reviewers still aim at the *same system safety
  target* as the joint reviewer; "local" does not mean a different target.
  The later bounded-local Harvest rule combines patch reports before its final
  decision, so it is not a fully decentralized institution.
- **Joint check:** can use more of the agents' current requests together.
  It is a diagnostic comparison, not necessarily a real institution or a
  deployable all-knowing AI.
- **Sequential commons:** decisions happen repeatedly and today's extraction
  changes the resource available tomorrow. This follows the established
  sequential-social-dilemma line of work, not a new game genre invented here.

## Timeline: what happened and why

| When | What was done | Why it was done | What it did and did not establish |
| --- | --- | --- | --- |
| 24-25 Feb 2026 | Created the repository and a minimal Fishery simulation: one renewable stock, agents requesting harvest, noisy stock observations, collapse and population-composition sweeps. | Start with a setting where individual gain can damage a shared stock and failure is measurable. | A tractable experimental substrate, not an oversight result. The initial Git commit itself contained almost no research implementation. |
| 27-28 Feb | Added repeated replacement of less successful strategies by new entrants, mutation, monitoring/quotas/sanctions, held-out conditions, confidence-interval summaries, and JSON strategy input including live-model adapters. Calibrated easy/hard conditions and compared mutation with live model-generated Fishery policies. | A fixed, cooperative population is too easy; governance should face strategies that change and seek advantage. Harsh conditions were tuned because near-certain collapse cannot distinguish interventions. | Made adversarial turnover testable. The presence of an LLM adapter was not itself evidence that live LLMs were the central paper result. |
| 2 Mar | Built matched Fishery matrices, higher-repetition medium runs, result manifests, and the first paper drafts. Added resource, collapse, welfare, and governance-cost comparisons. | Replace one-off demonstrations with repeated, matched comparisons and visible trade-offs. | Fishery evidence about intervention effects in that simulation; not evidence that one architecture wins everywhere. |
| 11-16 Mar | Added Harvest as a second, more spatial commons, with patch health, neighbour effects, messages and credit transfers; then strategy turnover, local-only, system-level and combined interventions, held-out conditions, remote sweeps, and paper figures. Added PPO-based learned-policy checks. | Fishery's single stock could not test the difference between what a participant sees locally and what happens across patches. A learned-policy check could expose weaknesses missed by scripted policies. | Broadened the governance comparison. PPO was supporting validation, not the central trained-agent contribution. Earlier Harvest rankings were conditional and some differences were small. |
| 2-21 Apr | Added Harvest scenario presets, institutional-friction settings such as missed detection, delay, limited targeting and intervention cost, per-agent burden analysis, and structured LLM strategy-bank tooling. Reworked proposal/manuscript framing toward governance under imperfect oversight. | Yali's question required specifying what the monitor can observe and do. Edward's concerns pushed the framing away from treating "central control" as a literal universal solution. | More realistic limits and better measurement, but the scenario labels were parameter packages, not validated irrigation or forestry field models. The limits also moved together in some comparisons, so a single cause could not be inferred. |
| 10 Jun | Implemented Harvest Stage A: local action checks, system safety checks, a local-pass/global-fail diagnostic, three actor-generation presets, three bundled overseer presets, and four governance conditions. | Give the oversight proposal measurable actors, monitors, failure conditions, and a benchmark grid. | The completed summary contained 72 condition rows and 18 architecture-ranking cells: hybrid led 13, local 4, global signal 1. These are descriptive rankings of the chosen settings, not a universal winner or a validated capability-gap scale. |
| 17-18 Jun and subsequent closeout | Built resumable, sharded threshold replay and merged a full grid over nearby safety definitions. | Check whether local-pass/global-fail depended on one convenient threshold without rerunning an enormous local job. | The completed audit has 9,000 grid rows with no missing keys or duplicates. These are re-evaluations of saved trajectories at different thresholds, **not 9,000 independent ecological simulations**. |
| After Stage A, before the September review | Generated and validated structured Harvest strategies using local open-weight LLMs, including Qwen2.5 3B and Llama 3.2 3B banks with cooperative and exploitative instructions. Evaluated saved strategies under the same Harvest pipeline. | Check whether model-produced strategy *artifacts* can enter the benchmark without changing its action interface or relying on live inference every step. | A small offline strategy-source pilot. It is not evidence about live LLM autonomy or about stronger frontier models. Later direct numeric-template controls reproduced the broad pattern, weakening any claim that the pattern is specifically due to LLM reasoning. |
| By 13 Sep | Audited Stage A's causal story: kept requested actions separate from allowed actions, distinguished a new unsafe transition from remaining unsafe, tested state-aware local rules, weather and spillovers, strategy search, and LLM-template controls. | Find out whether the proposed failure mechanism survives better local checks and whether "higher capability" really means stronger behavior. | A fixed local cutoff can miss failure; a state-aware local rule is a strong baseline. Some local-pass/global-fail traces reflect already-damaged states rather than a new collapse. Search strength was not reliably monotonic in every stress setting. The original simple ranking of actor/overseer capability was withdrawn as an empirical conclusion. |
| 21-22 Sep | Defined a matched reviewer protocol. Local and joint reviewers received the same one-step global safety target and the same ability to scale requests. Made uncertainty about neighbours and resource renewal explicit. Added direct logs for harmful requests approved, safe requests restricted, checking work, resource state and useful activity. | Older comparisons changed both the information and the rule/authority. That could not isolate the value of seeing more requests. Direct reviewer decisions are closer to the oversight problem than winner maps alone. | A fairer *mechanism test* in Fishery and Harvest, with controls and diagnostic examples. It still assumes the monitors know the simulation rules. |
| 22-23 Sep | Ran constructed safe/risky cases, then fresh policy rollouts and a budgeted-reviewer pilot. Tested reviewers inspecting 0, 3 or 6 current requests. | Ensure the games actually contain both decisions a reviewer must make, then measure the safety-versus-activity trade-off under a limited checking budget. | The development pilot produced 400 episodes, 1,716 frozen proposals and 15,444 matched decisions. Its few policy contexts per setting made it a pilot, not broad confirmation. |
| 23 Sep | Ran a predeclared fresh-seed confirmation in selected Fishery and Harvest settings: 1,920 episodes, 11,249 original proposals and 101,241 matched reviewer decisions. | Test the immediate decision trade-off on new population contexts after piloting the protocol. | In these settings, joint review spared safe requests relative to a conservative local rule without approving more resolved-risky requests. An optimistic local rule exposed a different trade-off and was competitive in Harvest. Immediate safe decisions did **not** automatically improve long-run stock or return. This is conditional on two pilot-selected settings. |
| 22-23 Sep | Built a native adapter and attempted policy admission for SocialJax Clean Up. Stopped the transfer after the chosen policies/starting state did not produce the required reachable, useful safe/risky comparison. | Try a genuinely different mechanism: agents may need to clean a shared area before others can benefit. A third game is useful only if it supplies meaningful oversight decisions. | Adapter and tests exist; **Clean Up is not a validated third result**. More game rows would not fix the missing policy/outcome coverage. |
| 23 Sep | Ran a bounded Harvest actor-search pilot: nested 1/8/32-candidate search chose one entrant, then fixed reviewers inspected 0/3/6 requests from new partner populations. | Test the missing link between strategy-generation resources and reviewer decisions without changing the reviewer at the same time. | Search changed held-out payoff unevenly, but no-review failure was often saturated and all reviewers rejected the resolved-risky requests in four contexts. This is a useful negative design check, **not** a validated rising-capability oversight result. |

## October 2026: the oversight experiment series

**Where it lives.**
- Notes: `notes/claude_audit_20261005/`.
- Code: `experiments/oversight/`.
- Raw runs: `results/runs/claude_*` (git-ignored). Small tables are copied
  under `notes/claude_audit_20261005/runs/`.

**How each experiment is run.** Each one has its own folder,
`notes/claude_audit_20261005/studies/<ID>_<name>/`, holding:
- `protocol.md`, written and committed **before** any code ran;
- `results.md`, which follows the same structure every time.

**How to read the numbers.**
- Intervals are 95% paired context-bootstrap intervals.
- A *context* is one freshly generated population of 6 agents. Every study
  uses 64 test contexts and 80 steps unless stated.
- Fishery uses the productive-level (MSY) target from S1b on.

### What was run and what came out

| ID (date) | Question | What came out | Verdict on its hypotheses |
| --- | --- | --- | --- |
| **Audit** (5 Oct) | Re-check the 23 Sept confirmation. | Harvest results were driven by a weather buffer 3.6× larger than needed. With a buffer sized correctly, joint review cut **0 of 3,687** safe requests, not 2,521 **[post hoc]**. | Led to R1. |
| **R1** (5 Oct) | Repair the reviewer comparison and test the Fishery safety target. | **Harvest:** joint review approved 0.6% of risky requests and cut 0.2% of safe ones. Optimistic local approved **12.6%** of risky requests; bounded local cut **90%** of safe ones. **Fishery:** the MSY target gave **1,393 against 555** total harvest [difference +838, interval +800 to +865]. "More information gave less harvest" only appeared under the old line *and* when unseen requests were assumed to be the maximum. | Confirmed on fresh seeds. |
| **S1** (5 Oct) | What if agents under-report and the reviewer trusts reports? | Unsafe actions rose: **Harvest 0.35% → 3.0%**, **Fishery 0% → 24.8%** (the stock collapsed in 61 of 64 contexts). Audits and neighbour reports brought this back to about 0%, but **by excluding caught cheaters**, not by better decisions. Lying paid only in Fishery with colluding neighbours (+110). | Mixed; see results. |
| **S1b** (5 Oct) | Do audits help by correcting the reviewer or by removing the cheater? | A reviewer with no memory, using 1–2 audit results per step, barely improved: **Harvest 1.9% → 1.7%** of steps unsafe. Exclusion produced the safety. | Belief-correction hypothesis falsified. |
| **S2** (5 Oct) | Does checking deter cheaters who gain privately by taking more than allowed? | With no checks, harvest per honest agent fell **106 → 45** (Fishery). One random audit per step plus any fine **≥ 6** stopped cheating. In Harvest, cheating never paid. | Supported in Fishery; untestable in Harvest. |
| **S3** (5 Oct) | Where exactly does deterrence start? What about imperfect audits, timed cheaters, and a reviewer with memory? | **Threshold:** cheaters stop at expected fine per agent-step **e ≈ 0.30–0.40**, against a predicted break-even of **0.356**, for audit rates from 1/6 down to 0.02 and audits that miss up to ¾ of cheats. **Known schedule:** worse than no checks (honest agents **28.8 against 40.9**; the stock collapsed in 12 of 64 contexts). **Memory:** Fishery **69% → 1.5%** of steps unsafe (fixed liars) and **75% → 32%** (noisy liars); Harvest **1.74% → 0.98%** (0.61% when everyone is honest). | 8 supported, 2 falsified (one only partly), 1 empty by construction. |
| **S4** (5 Oct) | Does memory still help against cheaters who know the reviewer remembers? What if audits cost something? | Memory alone **does not deter**. Cheaters kept 41% of their gain even at the highest audit rate. In 98% of caught-cheater steps the shrunken allowance was already at its floor of 0 **[post hoc]**. **Memory + fine** deterred at audit rate **0.05 instead of 0.10**, and gave the most harvest once audits cost something. Misreporting still did not pay against memory. | 6 of 7 supported; 1 falsified by a 0.04% near-tie. |
| **R2** (5 Oct) | Do the findings hold in 8 never-piloted settings? What if the reviewer's model is wrong? | **Memory** was safer in **9 of 9** testable settings. The **MSY target** won in 5 of 6; the exception is a setting where the one-step line never binds. **Wrong model:** joint review's risky approvals rose from **0.9% to 26%** (weather noise underestimated by half) and **82%** (regrowth overestimated by 25%). A reviewer that learned the model recovered within about 60 steps. Under a hidden collapse point, the old line collapsed **78%** of contexts and MSY **0%**. | Mostly supported. The deterrence formula as written failed; it matched once measured correctly **[post hoc]**. |
| **S5** (5 Oct) | Do the conclusions survive stronger attackers (per-agent, state-dependent, PPO)? Do two new memory rules fix S4's floor? | Fines plus random audits still deterred at **q ≥ 0.10**. **memory_cap** (check a caught agent every step) cut the gain to **7%** of no checks, but used **4.1 audits per step**, against 0.6 for a fine. **memory_cut** (tighten the shared cut) protected the stock but was unfair: cheaters gained **more** than with no checks (142 against 113), and honest agents got **36** each against 104. Richer search mostly produced cheaters competing with each other, not a stronger team. | H1 and H3 supported; H2 split (memory_cut failed, memory_cap passed); H4 not supported. |
| **C1** (5 Oct) | Does the reviewer ranking hold in a third game, where harm comes from *combining* two reagents? | Yes. Joint approved **0.3%** of risky requests, optimistic local **100%**. A fair-share quota cut **82–89%** of safe requests in *both* games, so its failure is not specific to compositional harm. Targeting audits by report size backfired: **+3.4 points** more unsafe steps than random audits. | 2 of 4 supported. |
| **L1 pilot** (6 Oct) | Can a language model (gpt-oss:120b) act as a fisher in this simulator? Does it over-take? | The interface works: **100%** valid answers in 1,968 calls, about **$0.41** in total. The model **never** over-took unless the rules said it could. One sentence of permission raised over-taking to **23%** with no checks, and random checks with a large fine cut it to **0%**. 3 contexts per cell, so descriptive only. | Pilot; proposes L2. |
| **R3** (7 Oct) | Does deterrence start at the break-even in every Fishery setting, with the gain per cheating step fixed **before** the search? | Yes: in **5 of 5** testable settings the threshold fell within one grid step of the prediction. The predicted thresholds range from 0.19 to 1.20. The sixth setting gives cheating no gain, so it is not testable. | 2 of 3 supported; 1 badly posed (the gain is zero at exactly the break-even). |
| **T1** (7 Oct) | Do audits aimed at the largest report miss under-reporters, in all three games? | Aiming at the largest report was worse than random in **Fishery** (+89 points of harm) and **River** (+3.2), with no difference in **Forest**. The reason is **coverage**, not hiding: the aim keeps auditing the same agent, so only **25%** of liars were ever caught in Fishery, against 100% at random **[post hoc diagnostic]**. In Forest, aiming at the plot doing worst against prediction was best: harm fell to **26%** of the no-audit level, against 65% at random. | 2 of 3 supported; the mechanism hypothesis was falsified and revised. |

**7 Oct: paper exhibits.**
- Built for claims 1–5, following how the closest papers present their results.
- In `paper/paper_v6/`:
  - a TikZ protocol diagram;
  - 5 multi-panel figures with 95% intervals;
  - 2 booktabs tables.
- Rebuild with `experiments/oversight/make_paper_exhibits.py`.

### What we now believe, and how confident we are

1. **With true inputs and a known model, joint review is near-exact, and
   local rules err in fixed directions.** Confidence: high.
   - Shown in R1 and C1.
   - Largely arithmetic (see `propositions.md`, P5).
   - **It breaks with a wrong model (R2).** This is now the main weakness of
     the "joint review" story.
2. **The safety target matters as much as the information.** The MSY target
   beat the old line wherever the resource was under pressure (R1, R2).
3. **Random audits plus a fine deter at the expected-penalty break-even.**
   - Shown in S3, S5 and R2, and **confirmed prospectively in R3** (5 of 5 settings).
   - Mostly arithmetic (P1).
   - A **known schedule** cannot deter (P2).
   - A **fixed, report-based aim** fails because it leaves most agents unchecked (T1). Audits must reach everyone.
4. **Memory improves decisions without sanctions, but does not deter (P4).**
   - Memory plus a fine needs half the audit rate.
   - Robust across settings (R2).
   - Its effect is about half as large when lies vary.
5. **Making a stronger attacker is still unsolved.** Richer search and PPO did
   not give a validated ladder of attacker strength (S5, and the September
   actor pilots).
6. **LLM actors are feasible and cheap.** Whether they cheat depends heavily
   on how the rules are worded (L1 pilot).

### Open decisions (for Ameer, Yali and Edward)

- **Definition of harm:** keep MSY? The evidence supports it.
- **Which paper:**
  - a short paper on monitor design with honest agents;
  - or a full paper on misreporting, costly checking and deterrence.
  - The notes recommend the full paper.
- **Main game:** Harvest, with Fishery as a sanity check? Most deterrence
  results are Fishery-only, because cheating never paid in Harvest.

### Next steps already proposed

- **L2, the last experiment:** LLM actors across a fine grid, with two model families. The protocol is drafted in `studies/L2_llm_agents/protocol.md`; it waits for the compute and model decisions.
- **A coordinated group attacker** with per-agent strategies (S5 gap).
- **A reviewer that hedges against uncertainty in its own model** (R2 gap).
- **A direct test** of report-based against signal-based audit targeting (C1
  conjecture).

### Where the October files are

| Item | Path (in `notes/claude_audit_20261005/`) |
| --- | --- |
| Per-study protocol and results | `studies/<ID>_<name>/protocol.md` and `results.md` |
| Arithmetic results stated once | `propositions.md` |
| Running "what we learned" log | `what_we_learned_and_next.md` |
| Every check that was run | `verification_log.md` |
| Figures, with one-line takeaways | `figures_guide.md`, `figures/` |
| Sources and terms | `literature_ledger.md` |
| Background (September audit, design issues, caveats) | `background/` |

## Why these games, specifically?

The names are mnemonics for different **structures**, not claims about actual
fisheries, farms or sanitation policy.

| Game | Distinct question it can test | Present status |
| --- | --- | --- |
| **Fishery** | If several requests draw on *one stock*, can a reviewer recognize when individually plausible requests add up to overuse? This isolates shared-total accounting. | Implemented and used in the recent matched-reviewer confirmation. |
| **Harvest** | If agents have *different local patches*, can a reviewer handle neighbour spillovers and uncertain renewal without restricting too much safe work? This adds a genuine local-versus-joint information issue. | Main historical governance/strategy testbed and used in the recent matched-reviewer confirmation. |
| **SocialJax Clean Up** | If useful work depends on maintaining a shared environment, can review distinguish maintenance from free-riding or damage? This could test a mechanism Fishery and Harvest do not. | Native integration was attempted, but the selected policies never met the admission criteria. Not a result in the current cross-game comparison. |

The "community irrigation" and "forest co-management" labels in older
Harvest plots are **stress presets within Harvest**, not two independent game
families. Adding five game names would not prove generality. A new game earns
its place by exposing a different oversight error, having enough safe and
harmful proposals to compare reviewers, and supporting the same decision
question without pretending its actions or safety thresholds are identical.
Delayed harm might eventually justify another game; a larger count alone does
not. Fishery and Harvest already provide two different mechanisms, but they
are still custom research environments rather than a broad external suite.

## What the work has really shown

1. **A simple personal limit is not the full safety test.** Even if each
   request passes a fixed cutoff, damage already present, unpredictable
   renewal, or interactions can leave the whole system unsafe. The more useful
   diagnostic distinguishes a *new* unsafe transition from continued time in
   an unsafe state. A state-aware local rule often performs much better than
   the original fixed cutoff.
2. **Information matters, but its value depends on the game and the rule.**
   Under matched targets and actions, seeing requests together helped spare
   safe Fishery work compared with a conservative local rule. In the tested
   Harvest setting an optimistic local rule was similarly effective on the
   immediate labels, so the joint reviewer did not uniquely dominate.
3. **A correct immediate decision is not the same as a good long-run policy.**
   In the Fishery confirmation, full-information joint review left all 694
   resolved-safe original requests unchanged and approved none of 63
   resolved-risky ones. Yet at full inspection it had lower mean 80-step
   return (551.5 versus 784.7) and lower mean stock (16.88 versus 25.58)
   than the bounded local rule. Allowing an immediately safe request can
   still use stock needed later. These are outcomes of different trajectories,
   not proof that the local rule is generally better.
4. **Oversight is a trade-off, not a winner badge.** The reviewer may miss
   harmful requests, constrain safe requests, spend more effort, or change
   future resources. The paper should show those quantities separately.
5. **Model-generated strategies can be tested in the same pipeline.** The
   offline LLM banks demonstrate compatibility and some behavioral variation,
   but small local models and a direct numeric-template control prevent a
   broad claim about LLM-agent populations or unique model capability.

## Important corrections to earlier interpretations

- The old "actor capability minus overseer capability" rank compressed
  different things (search count/horizon versus detection/delay/capacity/cost)
  into one number without validating that they share a scale. It is a preset
  label, not a scientific measurement of relative capability. Search may
  produce stronger entrants in some settings but not monotonically everywhere.
- The older 29-step local-pass/global-fail case shows a real recorded
  disagreement, but an extended trace can include *persistence* after the
  system was already unsafe. It is not 29 independent new failures.
- Hybrid's Stage A ranking does not isolate which component helps, because
  some architectures and overseer presets changed several mechanisms at once.
- A joint reviewer with full simulator knowledge is an informative upper
  reference. It is not a weaker human or learned AI supervising a stronger
  actor. The latest matched comparison fixes safety target and intervention
  menu, but total model checks and communication still differ and are logged.
- A threshold replay checks how labels change on saved trajectories. It does
  not show how policies would adapt to a changed threshold.
- A result from Fishery cannot be numerically pooled with a Harvest return;
  they use different resource units and policies. The recent confirmation
  reports them side by side.
- The current results do not establish universal superiority of local, joint
  or hybrid oversight, real-world policy effectiveness, stable learned norms,
  live LLM-agent safety, or a five-game benchmark.

## Where this leaves the first paper

*Written 23 September. The October series above changes the picture: see "What we now believe" and "Open decisions" there.*

The present paper can make a **specific, defensible contribution**: define a
repeated multi-agent resource-use oversight problem, show why local/global
safety labels need careful interpretation, and compare reviewers on the same
safety target while varying how much of the current joint action they inspect.
That is an oversight-information paper using sequential commons. It is
relevant to scalable oversight, but its confirmed result is narrower than the
original ambition. The honest choice for Yali and Ed is whether that narrower
result is the first paper, or whether to add one focused experiment before
submission to make rising actor strength central.

If the **scalable-oversight-under-capability-pressure** title is essential,
the next necessary experiment is not five games or more model names. Hold the
reviewers, safety target, checking budgets and action menu fixed. Generate
actor policies with different search resources on training settings; verify
on held-out settings that those resources actually produce distinct, stronger
requests by a predeclared behavioral measure. Then test how often each fixed
reviewer approves harmful requests, restricts safe ones, and preserves useful
activity as that measured actor pressure changes. Report actor search budget,
reviewer inspection, model checks and communication **separately**, not as
one subtraction score. If stronger policies do not emerge, report that as a
failed capability manipulation rather than renaming the presets.

Only after that gate should another game or larger LLM bank be considered.
For another game, first show a working safe/risky policy contrast and a
distinct failure mechanism. For an LLM extension, use held-out generated
strategies under the *same reviewer protocol*, retain simple numeric-template
and search controls, and do not call it live LLM oversight unless models
actually act live. Before release, package the code, configurations, results
or reproducible result-generating route: much of the recent `results/` data
is ignored locally and is not in a clean checkout.

## Evidence trail

- Git chronology: `git log --reverse --date=short --pretty=format:'%h %ad %s'`;
  early project summary in `git show 46f524d:PROJECT_CHAT_SUMMARY.md`.
- March Fishery/Harvest closeout:
  [paper v2 upgrade](notes/cycle_logs/paper_v2_harvest_upgrade_closeout.md).
- [Stage A closeout](notes/harvest_oversight_gap_stageA_closeout.md),
  [threshold-grid completeness audit](notes/THRESHOLD_SWEEP_COMPLETENESS_AUDIT.md),
  and [LLM strategy-bank closeout](notes/harvest_llm_bridge_stageB32_v3_closeout.md).
- [September validation and direction](notes/research_review/RESULTS_AND_DIRECTION.md)
  and [research-review index](notes/research_review/README.md).
- [Matched-reviewer implementation](notes/research_review/MATCHED_OVERSIGHT_IMPLEMENTATION_CLOSEOUT.md),
  [development pilot](notes/research_review/BUDGETED_REVIEWER_PILOT_CLOSEOUT.md),
  [fresh-seed confirmation](notes/research_review/BUDGETED_REVIEWER_CONFIRMATION_CLOSEOUT.md),
  [actor-search pilot](notes/research_review/ACTOR_PRESSURE_PILOT_CLOSEOUT.md),
  and [Clean Up admission failure](notes/research_review/CLEANUP_PARSER_REPAIR_CLOSEOUT.md).
- [Current manuscript](paper/paper_v5_scalable_oversight_commons/main.tex)
  and [reproducibility manifest](notes/PAPER_REPRODUCIBILITY_MANIFEST.md).
- External context: [Leibo et al., sequential social dilemmas](https://arxiv.org/abs/1702.03037)
  establishes the repeated-game lineage; [Kenton et al., weak judges and strong
  LLMs](https://arxiv.org/abs/2407.04622) illustrates the direct capability-gap
  question that this project has not yet fully tested.
