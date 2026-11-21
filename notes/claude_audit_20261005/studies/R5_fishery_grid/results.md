# Results R5: the Fishery memory test on the common 2×2 grid

**Protocol:** `protocol.md` in this folder, frozen on 2026-10-09 before any code (commit `a4e5bf1`). There are no amendments.

**Run:** `results/runs/claude_r5_v1/`, in 5 seconds. The summary is in `notes/claude_audit_20261005/runs/claude_r5_v1/r5_summary.json`.

**Gates:** all passed.
- `pytest` passes: 362 passed, 11 skipped.
- With R2's seeds, R5's episode function reproduces R2's saved memory episodes exactly: `exec_risky` and `scored_steps` for 4g at r = 0.7, contexts 0–1, all three conditions.
- The smoke run passed. With 2 populations its intervals were too wide to support the hypotheses, as expected.

## What came out

Share of rounds in which the executed catch breaks the MSY limit, fixed liars, 64 new populations:

| Setting (regrowth r) | No audits | Audits without memory | Audits with memory | Memory − without memory [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 4g 1.0 (0.70) | 98.5% | 83.8% | **2.9%** | −80.9 points [−82.5, −79.3] |
| 4g 0.85 (0.595) | 96.8% | 79.6% | **3.0%** | −76.6 points [−78.0, −75.2] |
| 2g 0.85 (0.595) | 96.5% | 65.3% | **1.5%** | −63.8 points [−65.5, −62.1] |
| 2g 1.0 (0.70) | 43.1% | 23.4% | **0.3%** | −23.1 points [−29.5, −17.1] |

**Hypotheses:**
- **R5-H1** (memory below audits without memory): holds in 4 of 4 settings.
- **R5-H2** (memory below no audits): holds in 4 of 4.
- **R5-H3** (descriptive): audits without memory removed 15–46% of the harm.

## What it means

- Every game now uses one design in Figure 2(a): 2 or 4 greedy agents × a regrowth multiplier of 1.0 or 0.85, giving 4 settings each.
- Across the 12 settings, audits with memory gave less harm than audits without memory in 12 of 12, and removed most of the harm wherever harm was common.
- Audits without memory removed 10–46% of the harm. The 10% is Forest with two greedy agents, where harm was rare (1–2% of rounds).
- R2's wider Fishery grid (regrowth 0.5–0.9, 5 testable settings) gives the same result. It is now supplementary Table S5.
