# L03 The reviewer (all three games)

Local audit source, 14 Oct 2026. Read only.

## Code

`fishery_sim/calibrated_oversight.py`:

```
23	SCALES = (1.0, 0.75, 0.5, 0.25, 0.0)
...
33	def harvest_unsafe_mask(cfg: HarvestCommonsConfig, futures: np.ndarray) -> np.ndarray:
35	    mean_ok = futures.mean(axis=-1) >= cfg.global_min_mean_patch_health
36	    frac_ok = (futures < cfg.local_patch_failure_threshold).mean(axis=-1) < cfg.failure_fraction_threshold
...
52	def harvest_choose_scale(cfg, health, believed, reviewer, seed, draws=400, tolerance=0.05):
...
93	def fishery_target_ok(cfg, stock, total, target):
94	    if target == "one_step":
95	        return fishery_next(cfg, stock, total) >= cfg.collapse_threshold
96	    if target == "msy":
97	        return fishery_residual(stock, total) >= cfg.stock_max / 2
...
101	def fishery_predicted_total(cfg, believed, scale, reviewer):
102	    b = np.asarray(believed, float) * scale * cfg.max_harvest_per_agent
103	    if reviewer == "joint":
104	        return float(b.sum())
105	    if reviewer == "local_bounded":
106	        return float(cfg.n_agents * b.max())
107	    if reviewer == "local_optimistic":
108	        return float(b.max())
...
112	def fishery_choose_scale(cfg, stock, believed, reviewer, target, collapsed=False):
113	    if collapsed:
114	        return 0.0, len(SCALES)
115	    for count, scale in enumerate(SCALES, start=1):
116	        if fishery_target_ok(cfg, stock, fishery_predicted_total(cfg, believed, scale, reviewer), target):
117	            return scale, count
118	    return 0.0, len(SCALES)
```

River uses the same menu and tolerance: `fishery_sim/two_reagent.py:32` imports `SCALES`, `:42` `TOL = 0.05`. The LLM studies call the joint reviewer with `TARGET = "msy"` (`run_l2_llm_agents.py:34`, `:281`); L6 calls `harvest_choose_scale(... "joint", ...)` (`run_l6_llm_forest.py:211`). R4 sets River's target to 50 (`run_r4_river_grid.py:28`, `TARGET = 50.0`, applied through `river_target`, `:40-48`).

Note on the task brief: the "tolerance 0.06" is not a reviewer tolerance. The reviewer's chance-constraint tolerance is 0.05 (above). 0.06 t is the LLM over-take detection margin (`run_l2_llm_agents.py:60`, see L09).

## Recorded justifications (verbatim)

- Menu: `notes/research_review/MATCHED_OVERSIGHT_PILOT_PROTOCOL.md:36-38`: "Every monitor chooses one uniform request multiplier from `[1, .75, .5, .25, 0]`. The first predicted-safe option is selected. The finite menu is shared; its resolution is a limitation, not a separately tuned advantage." Kept in R1 (`R1_repaired_reviewer/protocol.md:57`: "the action menu {1, 0.75, 0.5, 0.25, 0}").
- Chance constraint: `R1_repaired_reviewer/protocol.md:37`: "The reviewer's 5% rule is a **chance constraint** (Charnes & Cooper 1959)." `:45`: "Each one approves the largest scale whose *estimated* one-step risk is ≤ 5%. The estimate comes from 400 simulated weather draws applied to that reviewer's own prediction." The 5% value itself: no source; it predates R1 (`HELDOUT_POLICY_PILOT_PROTOCOL.md:41-42`: "128 independent weather draws, Wilson 95% uncertainty and a 0.05 risk threshold"). Ledger caution, `literature_ledger.md:90`: "A Monte Carlo estimate of a 5% violation probability is a sample approximation with sampling error. ... do not call it a guaranteed 5% bound."
- MSY target: `R1_repaired_reviewer/protocol.md:38-39` (Schaefer 1954, secondary), and `:49`: "A2: Fishery safety line far below the productive level | **Fishery target is a factor.** Either `one_step` (next stock ≥ 10, the old line) or `msy` (at least 50 left after harvest, i.e. K/2)." `:120`: "Note: this tests the post hoc explanation from the September reviewer write-up §6d." Made the default after R1: `S1b_S2_ablation_and_deterrence/protocol.md:110`: "**Fishery:** `mix4`, **MSY target**, the default from now on because of R1." Paper: `body.tex:39` cites `may1978exploiting`.
- One-step line: `literature_ledger.md:70`: "one-step safety line | no standard literature name; it is a myopic, one-step reachability-style constraint. Contrast with MSY. | (none verified; describe in our own words)".
- Joint vs local reviewers: design (supplementary.tex:65-66). The ledger, `:91`, says "Calling our reviewer a "shield" ... implies guarantees it lacks. Say "a joint reviewer, analogous to centralized shielding"."
- The reviewer knows the true model unless R2 Part B changes it: `R2_robustness_and_reviewer_model/protocol.md` Part B ("The environment always uses its true configuration. Only the reviewer's copy changes."). Reviewer misspecification factors ×0.5/×2 noise, ×0.75/×1.25 regrowth and K, learned, Allee A = 20: no source for the sizes.

## Flags

- **Chosen after results:** the MSY target was introduced to test a post hoc explanation of the September result, and became the default after R1's results.
- The 5% tolerance and 400 draws are conventions with no cited source for the number.

## Verdicts

| Setting | Value | Verdict |
|---|---|---|
| Shared-cut menu | {1, ¾, ½, ¼, 0} | C ("its resolution is a limitation") |
| Rule | largest passing scale, else 0 | C |
| Reviewer types | joint, local bounded, local optimistic | C (design contrast) |
| Fishery target | residual ≥ K/2 (MSY) | G (Schaefer secondary; May 1978), adopted post hoc |
| One-step line | next stock ≥ 10 | U |
| Chance-constraint tolerance (Forest, River) | 5% | U for the number (concept: Charnes & Cooper) |
| Monte Carlo draws | 400 reviewer; 2,000/4,000 labels | C |
| Reviewer model | exact; R2 misspecification sizes | C |
