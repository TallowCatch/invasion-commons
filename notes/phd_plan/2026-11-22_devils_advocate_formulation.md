# Devil's advocate: the general and decentralised overseer formulation (22 Nov 2026)

This applies the `devils-advocate` skill to `2026-11-21_general_formulation.md` and `2026-11-23_decentralised_overseer_formulation.md`, using the results in `../OVERSEER_TRACKER.md`.

## Level 1: decision critique

**Steel-man.** The formulation has the right shape for Yali's question.
- It copies her group's own template: one general problem class, one measurable number that says whether a testbed fits (the locality ξ_loc), then many testbeds.
- The central overseer is the special case M = 1, so no result so far is wasted.
- It makes falsifiable predictions:
  - with a 1-step look, regional overseers catch exactly as much as a central one;
  - catching falls with how often agents cross regions;
  - sharing with neighbours closes the gap at a cost.
- The deterrence claim already holds in two very different settings: Fishery with LLM agents, and Harvest with RL agents. Misses scale it exactly as predicted (479 apples against 477).

**The three biggest risks**

| | Risk | Probability (evidence) | Impact | Detectability | Mitigation |
|---|---|---|---|---|---|
| 1 | **Not novel enough: "deterrence economics with RL agents".** | High. C1 is the classic fine × chance ≥ gain rule. Monitored MDPs, inspection games and Gans & Holden already formalise checks with a cost. Our derived results are a geometric series and a union bound. | Reviewers call it incremental. | High: the first reviewer or supervisor will say it. | Make the contribution the locality number predicting the central-vs-regional gap across 3 or more testbeds, plus the effects deterrence theory doesn't predict: with mistakes, few big checks lose to many small ones (342 vs 479); repair speed depends on k; removing the overseer undoes everything. |
| 2 | **The experiments don't test what the formulas say.** | High (these are facts). The agents are meant to best-respond individually, but 7 agents share one network. Coins has 2 players. The Clean Up rule (100 steps, pollution 0.2) is ours. BazaarBench offline traders never react. | Targeting and migration effects may be artefacts. History targeting already converges to random because no agent is a persistent cheater (finding 6). | Medium: only visible when rerun with separate networks. | Separate networks or mixed populations for targeting and migration; more agents (Harvest 14, Willis's commons up to 256); rules stated as fixed before runs (done); BazaarBench used for detection only. |
| 3 | **Building the wrong "decentralised".** | Medium. Yali's meaning is unconfirmed. On a 16×22 map with roaming agents, regions lose by construction (bots: 22–29% of checks wasted; migrators 37 over-harvests against 7). | Weeks on the wrong axis, or a "don't decentralise" result nobody needed. | High if asked on Monday. | Ask Yali. Motivate regions by scale and communication cost, with numbers, as the Du papers do. Include a testbed where locality is high (the "rooms" map) so the positive case is shown. |

**The best alternative.** Keep one overseer, but make it learned or an LLM, and study how capable it has to be against agents that adapt. That was Yali's original size and intelligence point. It's the right choice if she meant "LLM overseer" rather than "several overseers", or if locality turns out low in every spatial testbed.

**Reversibility: easily reversible.** The formulation is a document, regions = 1 reproduces the central overseer in code, and the 159 decentralised runs cost about 50–100 GPU-hours.

**Verdict.** Proceed, but redefine the contribution, and confirm "decentralised" with Yali before building more.

## Level 2: scope critique

**What did we forget?**
- **χ and ζ have never been measured.** The formulation's central number has no data yet.
- **The worst case,** agents that see the checks, hasn't been run. Two of the scorecard's three anchors exist.
- **Only 3 seeds,** and no intervals.
- **The overseer isn't optimised.** The "maximise over settings" problem is answered by a grid, not solved.

**What breaks when this fails?**
- If χ is high everywhere, regional overseers simply lose, and there's no positive result.
- If the pilot fails, the 90 non-Harvest runs never start.

**What does Ameer need that he didn't ask for?**
- A stated reason to decentralise (scale, communication, privacy), with numbers.
- A clear line between derived, conjectured and measured.
- A sharp "what's new" against Monitored MDPs, inspection games and Gans & Holden.

**What would a reviewer flag?**
- Becker in RL.
- Parameter sharing.
- Invented rules.
- Arbitrary costs (0.25 per message, λ).
- η depending on the penalty level.
- Offline BazaarBench can't test deterrence.

## Level 3: first day on the job

**Can a new collaborator set it up?** Mostly: `scripts/socialjax_local_setup.sh`, `notes/socialjax/README.md` and `DECENTRALISED_README.md`.

**Do they know which doc to read?** No. The formulation, plan, tracker, results and READMEs are spread across `notes/phd_plan`, `notes/socialjax` and `notes/bazaarbench`, with no single "start here".

**Are the gotchas highlighted?** Partly. These are scattered:
- CREATE MFA;
- broken nodes comp035 and comp040;
- SocialJax's default shared reward (it must be switched off);
- the 21 Nov change to how false alarms work;
- k × P being 7 times the average penalty.

## The gaps, redefined (in priority order)

1. **Measure locality.** Compute χ (crossing rate) and ζ (harm containment) from rollouts in Harvest, the 14-agent Harvest, Clean Up and a rooms map. Test whether ξ_loc predicts the central-vs-regional gap. This is the contribution.
2. **Separate agents.** Rerun the targeting and migration comparisons with one network per agent, or mixed populations.
3. **The worst case.** Agents that can see who was checked; random vs repeat offenders vs a randomised mix.
4. **A reason and a scale.** Many agents (Willis's commons up to 256; Harvest 14 or more), with communication cost reported as a share of central collection.
5. **Say what's new.** One paragraph against Monitored MDPs, inspection games, Gans & Holden and AI Control. Lead with the effects deterrence theory doesn't predict.
6. **Confirm "decentralised" with Yali.**
7. **Start here.** One index page linking every doc, with the gotchas listed.
