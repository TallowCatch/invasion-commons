# L01 Fishery: resource model, size and horizon

Local audit source, 14 Oct 2026. Read only. Repository: `/Users/ameerfiras/invasion-commons/invasion-commons`, branch `claude/r1-s1-oversight-experiments`.

## Where the settings are set

`experiments/oversight/claude_oversight_common.py` (the setup every current Fishery study uses):

```
33	def fishery_setup(context: int, pop_base: int, horizon: int = 80, n_stress: int = 4):
34	    cfg = FisheryConfig(n_agents=6, horizon=horizon, stock_init=70, stock_max=100,
35	                        regen_rate=.7, obs_noise_std=0, max_harvest_per_agent=6)
```

Docstring, lines 3-5: "Population generators and game settings are copied from the 23 September confirmation (`experiments/common/run_matched_oversight.py`)". The same line appears at `experiments/common/run_matched_oversight.py:164-165` and in R2 at `experiments/oversight/run_r2_robustness_reviewer_model.py:111-112` (with `regen_rate=regen`).

Not overridden, so taken from `fishery_sim/config.py`:

```
16	    collapse_threshold: float = 10.0  # below this is "danger"
17	    collapse_patience: int = 5        # K consecutive rounds below threshold => collapse
```

(The class defaults are stock 100/200, r = 1.5, noise 20 and 8 agents, lines 9-20; all overridden.)

Dynamics, `fishery_sim/env.py`:

```
174	        # Can't take more than exists
175	        if total_harvest > self.stock and total_harvest > 0:
176	            harvests *= self.stock / total_harvest
...
183	        # Logistic growth
184	        growth = self.regen_rate * remaining * (1.0 - remaining / self.stock_max)
185	        self.stock = float(np.clip(remaining + max(0.0, growth), 0.0, self.stock_max))
...
188	        if self.stock < self.collapse_threshold:
189	            self.below_count += 1
...
193	        if self.below_count >= self.collapse_patience:
194	            self.collapsed = True
195	            self.stock = 0.0
```

`fishery_sim/fishery_oversight.py:25-28`: "Matched Fishery disables legacy governance and observation noise"; "Monotone accounting bounds require regen_rate <= 1".

Horizon: 80 for programmed studies (`fishery_setup` default, line 33); 20 for LLM studies (`experiments/oversight/run_l2_llm_agents.py:36` `HORIZON = 20`).

Effective values printed with `fishery_setup(0, 1_500_000_000, 20)`:
`FisheryConfig(n_agents=6, horizon=20, stock_init=70, stock_max=100, regen_rate=0.7, collapse_threshold=10.0, collapse_patience=5, obs_noise_std=0, max_harvest_per_agent=6, monitoring_prob=0.0, quota_fraction=0.0, ...)`.

Derived (my arithmetic, not in the notes unless stated): MSY = rK/4 = 17.5 t per round (stated in `notes/claude_audit_20261005/background/september_reviewer_experiment.md:27-29`); equal share 2.9 t per agent; unconstrained demand up to 6 × 6 = 36 t per round, about 2× MSY; the pre-harvest stock at the MSY steady state is 50 + 17.5 = 67.5, close to the initial 70 (this coincidence is not recorded as a reason anywhere).

## Recorded justifications (verbatim)

- `notes/research_review/MATCHED_OVERSIGHT_PILOT_PROTOCOL.md:57-59`: "Fishery uses the existing `FisheryEnv`, with six users, stock 70/100, regrowth 0.7, maximum withdrawal 6, danger threshold 10 and the existing collapse patience. These are declared pilot choices, not externally calibrated values."
- Same file, lines 60-61: "No weather variants or repeated weather seeds are invented for this deterministic game."
- `notes/research_review/MIXED_POLICY_REPAIR_PROTOCOL.md:18`: "Use six agents." (no reason given).
- `PROJECT_HISTORY.md:66` (24-25 Feb 2026): "Created the repository and a minimal Fishery simulation ... Start with a setting where individual gain can damage a shared stock and failure is measurable. | A tractable experimental substrate, not an oversight result."
- Logistic growth / MSY: `notes/claude_audit_20261005/studies/R1_repaired_reviewer/protocol.md:38-39`: "The sustainability target is the logistic **MSY level**: stock K/2 left after harvest (Schaefer 1954, secondary source)." Ledger `notes/claude_audit_20261005/literature_ledger.md:42` (Schaefer 1954, depth "Secondary") and `:43` (May, Beddington, Horwood, Shepherd 1978, "Abstract only"). Paper `paper/paper_v6/draft_aaai/body.tex:39`: "Regrowth $r y(1-y/K)$ peaks at $y=K/2$ ... the level behind maximum sustainable yield \citep{may1978exploiting}."
- Ledger caution, `literature_ledger.md:89`: "rK/4 at K/2 comes from the deterministic logistic model. Under weather noise and patch spillovers, a stock-at-MSY target is a reference heuristic, not an optimum."
- Horizon 20 for LLMs: `notes/claude_audit_20261005/studies/L1_llm_actor_pilot/protocol.md:37`: "**Horizon: 20 steps**, not 80, to keep the pilot cheap." L1 results `:112`: "About 113,000 tokens per 20-step episode. An 80-step episode is about 450,000 tokens".
- Horizon 80: `R1_repaired_reviewer/protocol.md:52-60` lists "the 80-step horizon" under "Kept from the original design"; no reason.
- Regrowth robustness: R2 grid r ∈ {0.5, 0.7, 0.9} (`R2_robustness_and_reviewer_model/protocol.md:28`, "Piloted level 0.7"); R5 multiplier {1.0, 0.85} (`R5_fishery_grid/protocol.md:22`).
- GovSim comparison (ledger `:122`): GovSim's remaining stock "double[s] (capped by 100)"; "GovSim does not use MSY ... Our MSY limit (keep ≥ 50 of 100 under logistic regrowth) is the fisheries-standard alternative." K = 100 is not justified by GovSim anywhere; the match is noted only as a contrast.

## Verdicts

| Setting | Value | Verdict |
|---|---|---|
| Growth law | logistic, regrowth after harvest | G (Schaefer 1954 secondary; May et al. 1978 abstract only) |
| r | 0.7 (grid 0.5-0.9; ×0.85) | U ("declared pilot choices, not externally calibrated values") |
| K | 100 | U (same quote) |
| Initial stock | 70 | U |
| Max catch per agent | 6 | U |
| Observation noise / weather | none, deterministic | C (needed for exact reference labels) |
| Collapse rule | < 10 for 5 rounds | U |
| Rationing when demand > stock | proportional | U (no record) |
| r ≤ 1 | constraint | C (monotone bounds) |
| N | 6 | U |
| T | 80 programmed | U (inherited) |
| T | 20 LLM | C (cost) |
