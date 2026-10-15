# Novelty check, 8 October 2026: summary and contribution wording

Two literature searches were run, one per claim group, after all claims except EM had results:
- `claims1to5_audits.md`: the reviewer, audits and deterrence;
- `claim6_llm_deterrence.md`: LLM agents.

The five sources that most change the framing were re-opened by hand, so that check is first-hand:
- Okamoto et al. 2026: full text, the passages on enforcement, Gneezy–Rustichini and the model list.
- Gans & Holden 2026: abstract.
- Ye & Steinhardt 2026: abstract and the experimental setup.
- UNCLOS Art. 61 and UNFSA Annex II: secondary, through the cited commentaries.

Every other source is at the depth recorded in the two files.

## The bottom line

**The principles behind every claim are already known, mostly from economics and fisheries science.** These include:
- deterrence by expected penalty: Becker 1968, Allingham & Sandmo 1972;
- enforcement that depends on past behaviour: Harrington 1988;
- random inspection, the inspector committing in advance, and audit cutoff rules: Avenhaus et al. 2002, Reinganum & Wilde 1985;
- stock-based harvest rules;
- "a fine is a price": Gneezy & Rustichini 2000.

The paper must not present any of these as new.

**What is new** is that these principles are tested together, in one AI-oversight setting:
- a reviewer sets allowances for several agents on a shared renewable resource, with a limited audit budget;
- the deterrence threshold is predicted in advance;
- two specific audit-design failures appear that no source reports;
- language-model agents are tested in the same setting, against their own measured gain.

## Closest prior work, which must be cited

| Source | What it already shows | How we differ |
| --- | --- | --- |
| Okamoto, Erol & Erol 2026, arXiv:2608.12323 | 12 LLMs as procurement chatbots, including gpt-oss-120b, Nemotron 3 Super and Gemma 4 31B. One fine "below break-even" with an "unlikely" audit, and one above it with a "likely" audit. The Gneezy–Rustichini effect in LLMs (Nemotron 81% → 50% compliance with a small fine). Compliance spans 46 points across models. | Single or two-turn decisions, with no shared resource. Two enforcement levels, with audit odds given in words. No per-model measured gain. No oversight protocol. |
| Ye & Steinhardt 2026, arXiv:2607.09766 | LLM agents (Qwen3-Next-80B; gpt-oss-120b in an ablation) in three environments, **including a fishery commons**. An LM judge detects violations, and the penalty is removal. Tracking each reporter's reliability, with escalating penalties, resists exploitation. | No random audits, no fine size and no threshold. Their memory tracks the reliability of the agents doing the reporting; ours tracks how much each agent was caught taking. No MSY harm measure. |
| Gans & Holden 2026, arXiv:2609.38262 | Theory: rare random audits can deter AI agents that can conceal misconduct, provided audit draws cannot be learned in advance. | A single agent, no experiments, no memory, no shared resource. It supports claims 4–5 as theory. |
| Makins et al. 2026, arXiv:2607.07368; Hu & Wang 2026, arXiv:2607.11751 | Monitors that check one agent or one action at a time miss harm that only appears in combination (LLM code settings). | Our harm builds up through a renewable resource with random dynamics. |
| Piatti et al. 2024, GovSim, arXiv:2404.16698 | LLM agents in a commons; sustainability measured as survival. | No reviewer, no audits and no fines. |
| Greenblatt et al. 2023, arXiv:2312.06942; Griffin et al. 2024, arXiv:2409.07985 | AI control: audit budgets, and random against suspicion-targeted auditing. | A catch means shutdown, so audits without consequences, memory or fines are never tested. One untrusted model, no resource. |
| Reinganum & Wilde 1985; Kamijo 2014 | In static settings, audit cutoff rules weakly dominate random audits; auditing the lowest reporter can beat random. | A reviewer will raise these against claim 5. Our setting is repeated, the agents differ in size, and the audit is aimed at the single largest report. The paper must explain why the result differs. |
| Harrington 1988; Friesen 2003; Hansen, Jensen & Nøstbakken 2014 | Enforcement leverage from records of past behaviour, including in quota-managed fisheries. | Supports "memory plus a fine needs fewer audits" as known in principle. Our part is the measured size of the effect in a dynamic commons. |

## Verdict per claim, and the contribution wording to use

| # | Verdict | Wording to use |
| --- | --- | --- |
| 1 | Partly shown in fisheries; not found in AI oversight | "In AI oversight of a shared resource, what the reviewer aims for (an MSY limit) matters more than what it can see, echoing fisheries practice (UNFSA Annex II)." |
| 2 | Partly shown: Ludwig, Hilborn & Walters 1993 | "An exact joint reviewer fails under a wrong resource model; a learning reviewer recovers, and an MSY limit protects against a wrong model form." |
| 3 | The principle is known; the breakdown is new for AI oversight | "Audits without consequences barely reduce harm. Memory of caught over-takes or a fine is what works." Cite Becker, Harrington and Ostrom's principle 5. |
| 4 | The threshold is known and must not be claimed. **New: prospective prediction (5 of 5), and "a tighter shared cut makes cheating pay"** | "The classic expected-penalty threshold governs these oversight protocols, and can be predicted in advance from runs without audits. Tightening everyone's allowance to allow for known cheaters backfires." |
| 5 | Partly shown. **New: report-aimed audits lose coverage** (25% of liars ever caught) | "Audits must be unpredictable and reach every agent. Aiming at the largest report keeps auditing the same agent." Discuss Reinganum & Wilde, Kamijo and Ensign et al. 2017. |
| 6 | Partly shown: Okamoto 2026 | "A dose–response of deterrence for LLM agents in a repeated multi-agent commons, with explicit audit odds, set against each model's measured gain." Present fines backfiring as a replication of Okamoto and Gneezy–Rustichini in a new setting (post hoc), not as a first. Treat the wording effect as a control. |

## Update after L3 (8 Oct, 18:30Z)

L3 tested the gap named here.
- The fine at which each model stops was predicted from its own no-fine gain. The prediction held exactly for Nemotron and within one grid step for gpt-oss.
- Deterrence is graded, and it starts at about 0.4 g, earlier than a risk-neutral agent would stop.
- Okamoto et al. have one fine on each side of break-even, so they cannot show either result.
- **Claim 6 can now be stated as:** "the fine at which LLM agents stop over-taking can be predicted from their own no-fine behaviour (within one grid step), and LLMs are easier to deter than a risk-neutral agent."

## Update after the consistency check (8 Oct, about 21:00Z): the "myopic offender" claim

Details: `claim6b_myopic_offenders.md`. Its two key new sources were re-opened by hand: Bracale Syrnikov et al. 2026, and the penseralabs repository.

| Part of the claim | Verdict |
| --- | --- |
| LLM agents in a commons are myopic about the shared stock | **Partly shown, close to known.** GovSim ("long-term effects"); Bracale Syrnikov et al. 2026, arXiv:2607.22188, LLM collectives in an energy commons that "behave like impatient optimizers", with no enforcement. Do not claim it as new. |
| The one-round gain, not the whole-game gain, predicts the fine at which LLM agents stop | **No prior work found.** The theory is standard (Becker; the one-shot deviation principle). What is new is the empirical test. The one-round prediction was made in advance (L3); the comparison with the whole-game gain is post hoc. |
| Graded deterrence starting at about 0.4 of the one-round gain | **No prior work found** for LLMs. Modest, and seen in two models only. |
| Size penalties to the one-step gain | **Partly shown:** sizing fines to the gain is standard. The LLM-specific "immediate gain is the one that counts" version was not found. |

**Scooping watch:** penseralabs/becker-agents (re-checked 8 Oct) plans "deterrence curves" for LLM agents, but has no results, paper or preprint. Re-check before submission.

**Final wording for claim 6:**
- "In a repeated commons with random audits, the fine at which LLM agents stop over-taking can be predicted in advance from their one-round gain: within one grid step for both models, and exactly for one.
- Their whole-game gain from over-taking is small or negative, yet they over-take until the fine outweighs the one-round gain. As with the myopia reported in LLM commons (Piatti et al. 2024; Bracale Syrnikov et al. 2026), they ignore the cost to the shared stock, and here that myopia sets the deterrence threshold (post hoc)."

## What this means for the remaining work

- **Claim 6 is the weakest on novelty.** The one thing no prior work has is the threshold located at the model's own measured gain, predicted in advance. The current L2 grid cannot show it: e = 1.33 t gives no deterrence and e = 6 t gives full deterrence. A threshold-location experiment (fines between 8 and 36 t, with the prediction written first) is what makes claim 6 ours.
- **Scooping risk:** the repository penseralabs/becker-agents (September 2026) builds a harness that varies audit probability and penalties for AI agents. It has published no results. A dated preprint matters.
- **Still to read before quoting:** the full texts of Harrington, Sutinen & Andersen, Segerson, Landsberger & Meilijson and Alm & McKee (publisher pages were blocked), and the treaty texts of UNCLOS and UNFSA.
