# Formulation: index and record of decisions

The work to define the overseer properly before running more experiments. Started 23 Nov 2026.

## Files, in reading order

| File | What it is | Status |
|---|---|---|
| `00_plan.md` | the plan: phases 0–4, framing, devil's advocate | done (revised 23 Nov after Ameer's framing correction) |
| `01_literature_map.md` | how the literature models an overseer: design space, gaps, templates, reading list | done; checkpoint answered (below) |
| `lit/A_oversight_and_control.md` | reader notes: AI Control, AI-Control Games, multi-agent control, scalable oversight, Monitored MDPs, norm enforcement, BazaarBench, learning to report; skims of Gans & Holden and Zhu | done |
| `lit/B_marl_overseers_and_decentralised.md` | reader notes: Sun, AI Economist, LIO, Baumann, Vinitsky, green security games, DMPO, Ma et al. (Nature MI), SocialJax, OpenSocialJax, Leibo, Perolat, Willis; skims | done |
| `lit/C_our_prior_formulations.md` | our own June–July and Oct–Nov formulations and what to keep from each | done |
| `02_overseer_definition.md` | the overseer defined as a player: look, judge, respond, share; capability; several overseers; timing; objectives; evaluation; testbed criteria; choice and justification; notation | draft 1, awaiting Ameer's two decisions (§10) |
| `02a_one_pager.md` | the definition on one page, for Yali | draft 1 |
| `03_hypotheses.md` | hypotheses H1–H9 with predictions, falsifiers, evidence and needs; every experiment mapped (keep, drop, gap, redesign) | draft 1, awaiting Ameer's four decisions |

The writing style for all of these is in the `formulation-explainer` skill (`~/.claude/skills/formulation-explainer/SKILL.md`).

## Record of phases and decisions

### Phase 0: pause (23 Nov 2026)

- **Why.** The separate-network runs were still learning at 1e8 steps: shared networks level off by mid-training, while separate networks were still improving at the end. Results from undertrained agents would mislead (analysis in conversation, 23 Nov; data in `results/create_pull/SN_*/metrics.npz`).
- **What was paused.** The 18 worst-case runs (CREATE job 38845977) were put on hold with `scontrol hold`. They had been resubmitted that day restricted to large GPUs after all 18 failed out of memory on A100-40GB nodes.
- **What kept running.** Clean Up (38811468, 38832339), 14-agent Harvest (38811476), the error-pattern runs (38836358). They need no decisions.
- **Rule.** No new experiments until Phase 3 has agreed hypotheses.
- **Still to decide** (in Phase 3): release the held runs at 3e8 steps, rerun the 15 separate-network runs at 3e8, or drop both.

### Phase 1: literature (23 Nov 2026)

- **Framing, set by Ameer.** The overseer is a player in a multi-agent game, in the game-theory, MARL and scalable-oversight direction. Not economics: no "deterrence", "expected fine" or Becker as the explanation.
- **Read.**
  - In full: 19 papers, by two reader agents, each with the same questions (who acts, observation, actions, budget, policy, number, capability, timing, agents, objective, formal core, borrow, missing).
  - Skimmed: 8 papers.
  - Our own earlier formulations: read directly.
  - The new paper Ameer supplied: Griffin, Shlegeris & Abate 2024, *Games for AI Control*.
- **Main finding.**
  - AI Control has budgeted, imperfect, weaker overseers, but against one model.
  - MARL shapers act on many learners, but see everything and budget only payments.
  - The empty cell: an overseer that chooses what to observe, inspects imperfectly, and faces many learners in a sequential commons.
- **Checkpoint answer (Ameer, 23 Nov).** Bring back the June ideas (observation scope and architectures, the capability vector, the capability gap Δ, local-pass/global-fail). Start Phase 2. Highlight the overseer's action. Document every phase precisely.
- **Caveats.** Ma et al. 2024 was read from the web page, not the PDF, so its numbers are marked unverified. The local Willis 2025 PDF is the LLM iterated Prisoner's Dilemma paper.

### Phase 2: definition (started 23 Nov 2026)

- **Draft 1** written (`02_overseer_definition.md`, `02a_one_pager.md`).
- **Built around the overseer's action:**
  - **look:** choose k of N agents and a window w, from the inspection and/or system channels;
  - **judge:** an inspector with miss and false-alarm rates, possibly situation-dependent;
  - **respond:** sanction, freeze, system-level action, delay;
  - **share:** messages between overseers.
- **The June ideas restored:** two levels of harm (individual rule, group predicate) and LPGF; observation channels (June's four architectures); the capability vector κ^O; the capability gap Δ, proposed as Elo-style ratings from match-ups (Engels et al. 2025).
- **Borrowed structure:** the overseer as a player (Griffin; Baumann); Stackelberg-Markov timing with learning followers (Gasztowtt); ordinary and adversarial modes (Griffin); overseers as graph nodes with chosen links, plus measured locality (Du; Ma).
- **Decision 2 (Ameer, 23 Nov):** the capability gap is Δ_c(π^O) = J(ordinary learners) − J(agent class c), for the same overseer. Engels-style ratings are kept only as a possible extension.
- **Decision 1 (Ameer, 23 Nov):** the overseer looks at **individual agents only** in the first formulation.
  - Ameer had asked why this returned to "architectures". It was reframed as a question of what the overseer can observe.
  - My earlier argument that the commons view was needed for local-pass/global-fail was withdrawn as weak: LPGF can be measured from outside as a property of each testbed and rule.
  - Viewing the state of the commons is a **planned next step**, justified if LPGF turns out large.
- **Phase 2 closed** (draft 1 with both decisions applied). Moved to Phase 3.

### Phase 3: hypotheses (started 23 Nov 2026)

**Draft 1** (`03_hypotheses.md`). One quantity ties the overseer's action together: ρ = (k/N)(1 − α)F, the sanction a violation actually meets.

Nine hypotheses:

| | Claim | Status |
|---|---|---|
| H1 | learners respond only through ρ; threshold ρ* | supported in Harvest; H1b refinement: spread matters with errors |
| H2 | the look policy matters only against agents who see the looks | supported for bots; learners missing |
| H3 | learners exploit patterned misses | running |
| H4 | misses change behaviour; false alarms only harm compliant agents | supported |
| H5 | high LPGF defeats individual inspection | gap |
| H6 | separate networks exploit more | inconclusive at 1e8 |
| H7 | splitting costs come from crowding; sharing or reallocation recovers them | partly; extensions later |
| H8 | the overseer props up rather than changes behaviour | supported |
| H9 | slow violations need longer looks | Clean Up running |

**Experiments mapped:**
- Coins: dropped as a testbed.
- The held worst-case runs: proposed redesign with shared networks at 1e8.
- The separate-network runs: proposed reduced rerun at 3e8 (6 runs).
- LPGF: proposed first measurement, offline with no GPU.

**Decisions awaiting Ameer:**
1. H5 first.
2. H2 redesign.
3. H6 reduced rerun.
4. H7 extensions deferred.

### Phase 4: experiments (later)

Only those Phase 3 requires.
