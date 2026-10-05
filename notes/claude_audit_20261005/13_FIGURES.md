# Progress figures

All figures come from `experiments/make_progress_figures.py` and read only
saved run tables. The PDF and PNG versions are in `figures/`. Colours follow
one fixed palette; the three reviewers are always:

- **blue** = joint;
- **orange** = bounded local;
- **aqua** = optimistic local.

| Figure | What it shows | One-line takeaway | Source |
| --- | --- | --- | --- |
| `fig1_calibration_harvest` | Harvest at full inspection: unsafe approvals against usefulness loss for the three reviewers, 23 Sept vs R1 | The old weather buffer hid the real differences. With a calibrated 5% rule, joint review is nearly exact, optimistic local lets 12.6% of risky requests through, and bounded local cuts 90% of safe ones. | 09 §3a; Sept `primary_decision_quality.csv` |
| `fig2_fishery_target` | Fishery long-run harvest vs number of requests inspected, for each target × fill rule | The productive-level (MSY) target gives about 1,390 against 555. The "more information → less harvest" pattern needed both the low target and the "assume the maximum" fill. | 09 §3c, §4c |
| `fig3_limited_checking` | Harvest joint reviewer with 0/3/6 inspections, under two assumptions about unchecked requests | Assuming the worst cuts almost everything; assuming "same as last time" lets 23–36% of risky requests through. The assumption decides which mistake is made. | 09 §3a, §4b |
| `fig4_s1_protocols` | Unsafe executed actions by checking protocol when some agents under-report | Trusting reports is unsafe (Harvest 3.0%, Fishery 24.8%). Checks bring it to about 0, mostly by excluding cheaters. | 10 §4b–c |
| `fig5_s1b_ablation` | Share of the horizon unsafe when misreporters under-report: trusting reports vs using the audit result in the decision (memoryless, no sanction) vs excluding caught agents | With this memoryless reviewer, one or two audit results barely helped. Excluding caught cheaters (which also removes their extraction) produced the safety; complete neighbour information reached the honest level. | 12 Part A |
| `fig6_s2_deterrence` | Fishery: cheating level chosen at each tested fine, and per-agent outcomes | Unchecked hidden over-extraction moved harvest from honest agents (106 → 45 each) to cheaters. With perfect random audits plus a flat fine of 6 or more, fixed-level cheaters chose not to cheat; the threshold is probably near 2 (untested). | 12 Part C |
| `fig7_fishery_stock_over_time` | Fishery stock at the start of each step, 64 contexts per line, for MSY vs the old line (all 6 inspected) and the old line with none inspected | Under the old line the joint reviewer lets the stock fall to just above 10 within about 10 steps and keep it there (harvest 555). The MSY target holds it near 70 (1,393). With nothing inspected, assuming the maximum held it near 37 by accident (1,073). | R1 `closed_loop_decisions.jsonl.gz`, `closed_loop_outcomes.csv` |
| `fig8_s1_per_context` | S1 arms as in fig4, counted per context: contexts (of 64) with at least one unsafe executed action | Trusting reports harmed 58 of 64 Harvest contexts and 61 of 64 Fishery contexts (all 61 collapsed), so fig4's averages are not driven by a few outliers. In Harvest, 16 contexts had an unsafe action even under full verification. | S1 `episodes.jsonl.gz` |
| `fig9_s1b_paired` | S1b: for each context, whether each checking rule made it safer, unchanged or less safe than the same context with trusted reports | Excluding caught agents made 42–61 of 64 contexts safer; only 3 Harvest contexts (2 random audits) were less safe. Using audit results without memory made 4–9 contexts less safe in four of the five settings, and changed nothing in 22–40 Harvest contexts. | S1b `episodes.jsonl.gz` (paired: same contexts in every arm, checked) |

**Suggested order for a short progress talk:** 1, 7, 2, 4, 8, 9, 6. Figure 7 shows how
figure 2 happens. Figures 8 and 9 are the per-context versions of 4 and 5, which can
become backup. Figure 3 is backup.

**Caveats to state with the figures:**
- Two simulated games only.
- Fixed-rule agents.
- The reviewer knows the game's rules.
- Several conditions were chosen after earlier pilots, and each was then
  run on fresh seeds.
