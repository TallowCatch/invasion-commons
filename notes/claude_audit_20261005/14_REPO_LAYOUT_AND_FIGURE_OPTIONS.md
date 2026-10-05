# Repository layout, and which result figures are ready (5 October 2026)

Nothing was moved or rerun for this note. The data checks below read saved
files only. They are descriptive, and they were run after the results were
known **[post hoc]**.

---

## Part 1. How the repository is organised now

### Top level

| Path | Size | Status | What it is |
| --- | ---: | --- | --- |
| `fishery_sim/` | 0.8 MB | tracked | Simulation package. It holds Fishery, Harvest and the shared reviewer code. |
| `experiments/` | 1.9 MB | tracked | About 105 scripts in one flat folder, plus `configs/`. |
| `paper/` | 28 MB | tracked | Paper versions v2, v3, v4 and v5. **v5 is current.** |
| `notes/` | 18 MB | 148 of 248 files tracked | Notes from every phase, with about 70 loose files at the top. |
| `tests/` | 0.8 MB | tracked | 20 test files. |
| `results/` | 2.3 GB | git-ignored | All raw runs. Only the v5 bundle is reproducible from a clean clone. |
| `notebooks/`, `scripts/` | small | tracked | Historical Fishery and Harvest notebooks, plus cloud run scripts. |
| `external/GovSim` | 1.1 MB | submodule | Not used by current work. |
| `tmp/`, `tmp_supervisor_meeting_brief_2026-03-31.md` | 13 MB | untracked, not ignored | Scratch files and a stale March brief. |

### `experiments/`, grouped by the work it serves

1. **Builds paper v5.** `export_reviewer_paper_data`, `plot_reviewer_decisions`,
   `run_/analyze_budgeted_reviewer_confirmation`, `analyze_reviewer_longrun`,
   `replay_coupled_local`, `analyze_harvest_validation`,
   `audit_research_evidence`, `package_paper_sources`, `check_paper_inputs`.
   Provenance also lists the Stage A, threshold-replay and overseer-ablation
   analysers. `plot_scalable_oversight_paper_v5` draws only the older v5
   figures (fig01, fig03–07).
2. **Current oversight line (this folder).** `claude_oversight_common`,
   `run_/analyze_r1_repaired_reviewer`, `run_/analyze_s1_reporting_audit`,
   `run_s1b_ablation_msy`, `run_s2_compliance_deterrence`, `analyze_s1b_s2`,
   `make_progress_figures`. **The last four are untracked**, so they are not
   in git yet.
3. **Earlier Harvest and oversight work** (Stage A, LLM bridge, invasion
   matrix, RL, actor pressure, Clean Up). About 40 scripts.
4. **Earlier Fishery study** (`run_single`, `run_sweep`, `run_invasion`,
   `run_governance_ablation`, `summarize_paper_v1`, `generate_paper_v2_artifacts`,
   Fishery RL and others). About 20 scripts.
5. **One-off scripts:** GIF makers, `showcase_project`, `organize_results`,
   `check_llm_setup`.

### What makes moving files risky

- Everything is imported as `experiments.<name>`, and `tests/conftest.py`
  puts the repository root on the path. Moving a script into a subfolder
  changes its import name.
- Many scripts are libraries for other scripts. The most-imported one is
  `run_matched_oversight` (about 10 importers). Others include
  `run_budgeted_reviewer`, `run_heldout_oversight`, `claude_oversight_common`,
  `run_s1_reporting_audit` (used by S1b), `run_governance_ablation` and
  `run_harvest_study`.
- Tests import about 25 experiment scripts directly.
- The GitHub workflows in `.github/workflows/` call about 12 Harvest
  scripts by module name.

### Suggested target layout (not done; for later)

```
fishery_sim/                 reusable simulation and reviewer code (unchanged)
experiments/
  common/                    shared libraries: run_matched_oversight, claude_oversight_common, ...
  paper_v5/                  the v5 data -> figures route
  oversight/                 R1, S1, S1b, S2 runners, analysers and figures
  archive/harvest_2026q2/    Stage A, LLM bridge, invasion matrix, RL
  archive/fishery_2026q1/    the earlier Fishery study
  archive/oneoff/            GIFs and showcase scripts
notes/
  current/                   claude_audit_20261005, research_review
  archive/                   cycle_logs, old plans, closeouts
  meetings/                  group_meeting_assets, interview_presentation, proposals
paper/paper_v5_.../          unchanged
```

**Safe order.** Each step can be checked with `pytest -q tests` before the
next.

1. **No import risk.**
   - Commit the four untracked S1b/S2 scripts and notes 11–14.
   - Add `tmp/` to `.gitignore`, and move the March brief into `notes/archive/`.
2. **Leaf scripts that nothing imports.** Move the GIF/showcase scripts and
   the Fishery-only scripts into `experiments/archive/`, after checking
   with `grep`.
3. **Shared libraries last.** Move them in one commit that also updates the
   tests and the workflows.

---

## Part 2. Line graphs that join separate conditions

Your guess is right for one paper figure. It holds only partly for the
others.

| Figure | x-axis | Problem | Fix |
| --- | --- | --- | --- |
| Paper `fig03_capability_gap` | "Capability gap Δc" | **Serious.** Δc adds up ranks of unrelated settings, and file 01 Phase 3 shows the actor ranking ran backwards. The lines suggest a smooth trend along a scale that does not exist. | Drop it. If needed, show overseer settings as separate panels or groups. |
| Paper `fig08_reviewer_decisions` | inspected requests k = 0, 3, 6 | **Moderate.** k is a real count, but only 3 values were run. The lines and shaded bands imply that k = 1, 2, 4, 5 were measured. The Harvest drop between 3 and 6 is guesswork. | Plot the 3 points with error bars and no connecting lines (or faint dotted ones), and label k as "tested values". |
| Progress `fig2_fishery_target` | k = 0, 3, 6 | **Mild.** Same issue as fig08. Here the story is the gap between the target lines, not the slope. | Plot points only, or a grouped dot plot by target × fill. |
| Progress `fig3_limited_checking` | trade-off plane, path through k | **Acceptable.** The path is labelled with k. | Keep it. Say that points between the markers were not tested. |
| Progress `fig6_s2_deterrence` (left) | fine = 0, 6, 12, 24 | **Correct already.** The points are not joined. | Keep it. |

---

## Part 3. Candidate figures: what the saved data supports

### What data exists (checked)

| Run | Unit saved | Per-step trajectories? |
| --- | --- | --- |
| R1 closed loop | 252,400 step rows, each with `state` and `requests` | **Yes**, for every game, reviewer, k, fill and target. |
| R1 open loop | 77,598 decision rows | Decisions only. |
| S1, S1b, S2 | one row per episode, 64 contexts per arm | **No.** Only per-context totals. |

### Ready now (data exists; the claim is tested or by construction)

**A. A method schematic of one step of the game.**
Agents send requests. The reviewer sees k of them and fills in the unseen
ones. It predicts the next state and scales everyone down by one shared
factor. Then the state updates.
- Draw it once and reuse it in every talk.
- No data is needed. Two versions of a schematic already exist in
  `notes/group_meeting_assets/`.

**B. "What each reviewer adds up."**
Draw the same 6 requests three times:
- joint: sums the actual requests;
- bounded local: assumes every other agent takes the maximum;
- optimistic local: ignores the other agents.
Show where each prediction lands relative to the safety line. This explains
Finding 2 (errors follow from arithmetic) better than any chart. **Label it
"by construction"**: it explains why the result holds, rather than showing a
discovery.

**C. Fishery stock over time, old line vs MSY target.** *Strongest new figure.*
- Data: R1 closed loop, joint reviewer, k = 6, 64 contexts. The stock at the
  start of each step, averaged over contexts (verified from
  `closed_loop_decisions.jsonl.gz`):

  | Target | t = 0 | t = 10 | t = 40 | t = 79 |
  | --- | ---: | ---: | ---: | ---: |
  | Old line (stock ≥ 10) | 70 | 17.5 | 14.0 | 14.4 |
  | MSY target | 70 | 70.3 | 70.1 | 70.5 |

- Plot each context as a faint line, the median as a bold line, and dashed
  lines at 10 and 50. Add the k = 0, fill = max case (it levels off near 37)
  to show "accidental conservation".
- This shows how the 1,393 vs 555 harvest result happens. Time is a real
  axis here, so joining the points over time is correct.

**D. Per-context spread instead of bar means (S1).**
- Fishery, trusting reports: the stock collapsed in 61 of 64 contexts
  (verified). The 24.8% average hides that this happens almost every time.
- Harvest, trusting reports: 58 of 64 contexts had at least one unsafe
  executed action, at most 6 per context (verified). With 1 random audit,
  the figure is 48 of 64, at most 11.
- A dot or strip plot per arm shows that the context is the independent
  unit, and that the Harvest effect is broad rather than driven by a few
  outliers.

**E. Paired slope plot per context for S1b.**
- Show trust → audit used → exclude for each of the 64 contexts as thin
  lines.
- Joining here is legitimate: the *same* context is followed across arms,
  and the lines show pairing, not a trend along a scale.
- This makes "exclusion, not better decisions, produced the safety" visible
  context by context.

**F. Evidence status map.**
A table-figure with one row per claim. Columns:
- tested on fresh seeds;
- by construction;
- post hoc;
- untested.

Supervisors can see at a glance what is solid. The source is the "honesty
labels" already in files 04, 09, 10 and 12.

### Possible, but say clearly what it is

**G. Error trade-off plane with all conditions.**
- Extend fig1/fig3: safety (risky approved) against usefulness (safe cut),
  one point per reviewer × k × fill.
- Use small panels per fill. Show the points unjoined, or join them only
  within one reviewer with k marked.
- This is the field's safety–usefulness framing (AI control). Keep it, with
  the caveat that the reviewer knows the game's rules.

**H. Project timeline.** Fishery → Harvest → Stage A → validation → R1 →
S1 → S1b/S2, with what each phase established or overturned. This is
useful for an interview or progress talk. It is a narrative figure, not a
result.

### Premature: do not make these yet

| Idea | Why not yet |
| --- | --- |
| Any "capability gap" or "stronger actor" axis | No working actor-strength ladder exists (file 01, 6b and 6e). |
| A deterrence-threshold curve (cheating against fine) | Only fines 0, 6, 12 and 24 were run. The threshold near 2 comes from post-hoc arithmetic. Wait for the finer grid in file 04, Update 2. |
| Deterrence in Harvest | Cheating never paid there, so there is nothing to deter. |
| "Audit memory helps" | A reviewer that remembers caught lies has not been tested. |
| Limited-checking or audit-rate *curves* | Only k = 0, 3, 6 and 1–2 audits were run. Show them as points, not curves. |
| Clean Up anything | No valid run exists. |
| Harvest stock-over-time figures for S1, S1b or S2 | These runs saved per-episode totals only. A figure would need a rerun that saves trajectories (CPU only, minutes). |

---

## Summary

**What we now believe.**
- The current results support about five honest figures: A, B, C, D and E.
  Add F for supervisors.
- C (stock over time by target) is the most persuasive new one, and its data
  is already saved.
- The paper's capability-gap figure should go.

**Confidence.** High that the data supports these figures. Each is
descriptive of runs that have already been analysed.

**Still untested.** Deterrence thresholds, audit memory, imperfect audits,
and actors who adapt to the reviewer.

**Next step and cost.** Write C, D and E into `make_progress_figures.py`.
This reads saved files only and takes about an hour of work.

---

## Amendment, 5 October 2026 (later the same day)

**C, D and E are now built** as `fig7_fishery_stock_over_time`, `fig8_s1_per_context`
and `fig9_s1b_paired` (see `13_FIGURES.md`).

**Two changes from the plan above:**
- **D became a count per arm, not a dot strip.** The Harvest values are
  whole numbers of steps (0–11 per context), so dots would pile up. The plot
  shows the number of contexts with at least one unsafe executed action.
- **E became a better / unchanged / worse count per context, not slope
  lines.** Many Harvest contexts are 0 in every arm, so slope lines would
  overlap. I first checked the pairing: S1b's "trust" arm (no belief update,
  no sanction) matches S1's trust-reports arm in 64 of 64 contexts, and S1b's
  exclusion arm matches the S1 arm in 64 of 64.

**Two findings from building them** (verified against the saved files):
- Exclusion was not safer in every context. With 2 random audits, 3 Harvest
  contexts were less safe than when reports were trusted.
- Even under full verification, 16 of 64 Harvest contexts had an unsafe
  executed action, at most 2 steps per context.
