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
| H2 | the look policy matters only against agents who see the looks | **answered 24 Nov**: Δ ≈ 0 for random, about 400 apples for predictable looks (F29) |
| H3 | learners exploit patterned misses | answered: outcome refuted, mechanism supported (F28); H3b supported (F30) |
| H4 | misses change behaviour; false alarms only harm compliant agents | supported |
| H5 | high LPGF defeats individual inspection | Harvest: LPGF low (0 of 580 patch losses legal-only); Clean Up open |
| H6 | separate networks exploit more | **refuted 24 Nov** at 3e8 for random looks (F31) |
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

- **Decisions (Ameer, 23 Nov): all four recommendations accepted.**
  1. **H5 first.** An agent is measuring LPGF offline on branch `claude/h5-lpgf`.
  2. **H2 redesign.** The held separate-network worst-case job 38845977 was cancelled. `worstcase_shared` was submitted:
     - 18 `WS_` runs (visible checks, shared network, large-GPU constraint), job 38847109;
     - 6 `OS_` ordinary-mode baselines for round robin and mixed, job 38847112.

     Protocol amendment: `notes/socialjax/worstcase/protocol.md`, Amendment 1.
  3. **H6 reduced rerun.** `separate_long`: 6 `SL_` runs (separate networks, 3e8 steps, 5 h limit), job 38847113.
  4. **H7 extensions deferred.**
- **Already running:** error patterns (H3, job 38836358, nearly done); Clean Up (H1 second game, H9); 14-agent Harvest. CREATE is limiting us by our per-user GPU quota (`QOSMaxGRESPerUser`), so batches run in turn.

**H5 measured in Harvest (23 Nov 2026; `h5_lpgf/`).** Protocol written before measuring. 5 trained runs and 3 bot populations, 32 episodes each, CPU only.
- Every permanent loss (a patch losing its last apple; 580 in all) involved a rule violation. Perfect full inspection lost no patch.
- Temporary local damage often passes the rule in overseen runs (60–93% of newly dead cells), but 92–95% of those cells grow back.
- So Harvest gives no reason to add the commons view. Clean Up is the test; the script is ready for its params.

- **H5 and H3 answered (23 Nov).**
  - H5: local-pass/global-fail is negligible in Harvest (finding 26 / F27).
  - H3: outcome refuted, mechanism supported. The patterned inspector protects better because its blind spot is on mild violations (finding 27 / F28). H1 refined to the situation-specific sanction ρ(s).
- **H3b added and submitted (Ameer, 23 Nov):**
  - the inverted gpt-oss inspector (blind spot on harmful violations, matched average miss rate);
  - 12 runs, job 38848488;
  - hypothesis written in `03_hypotheses.md` before the runs.
- **Ameer's instruction (23 Nov): no new directions outside the agreed plan.** Every run must map to an agreed hypothesis (H1–H9, H3b). Anything else is raised with him first, not added.

- **H2, H3b and H6 answered (24 Nov; `experiments/socialjax/analyze_formulation_runs.py`, summary `phase3_runs_summary.json`).**
  - H2 (F29): with visible checks, predictable looks (repeat offenders first, round robin, shared repeat-offender lists) are dodged completely: 69–72 apples, the same as no overseer. Random looks lose nothing (Δ = 2–8 apples). Mixing in 30% random recovers part (Δ = 222, wide spread).
  - H3b (F30): the inverted inspector draws violations onto the last apples (6–8 per episode against under 1) and costs 30–44 apples against the coin flip at c = 2; at c = 1 it collapses like the coin flip while the gpt-oss table survives.
  - H6 (F31): at 3e8 steps separate networks level off at the shared-network result (447 vs 460; 438 vs 439). The 1e8 gap was slow learning.
- **Still running or to analyse:** Clean Up (H1 second game, H9, H5 in Clean Up), 14-agent Harvest (H1, H7; 9 runs still queued).

### Phase 4: experiments (later)

Only those Phase 3 requires.
