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
| `03_hypotheses.md` | claims with falsifiers, mapped to existing and needed experiments | not started |

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
- **Open decisions for Ameer** (§10 of the definition):
  1. Include the system channel in the first formulation?
  2. Measure Δ by match-up ratings, or by the drop from ordinary to adversarial mode?

### Phase 3: hypotheses (not started)

To do once Phase 2 is agreed:
- Precise claims, each with the part of the definition it tests, a prediction, a falsifier, the cheapest experiment, and whether existing results already answer it.
- Map every past experiment to a claim (keep, drop or gap).
- Decide the fate of the held worst-case runs and the 3e8-step reruns.
- Candidate theory: a bound on what inspecting k rather than N costs (Du's template).

### Phase 4: experiments (later)

Only those Phase 3 requires.
