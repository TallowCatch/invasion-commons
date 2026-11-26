# L02 Fishery (and shared): programmed agents

Local audit source, 14 Oct 2026. Read only.

## Code

`experiments/oversight/claude_oversight_common.py`:

```
33	def fishery_setup(context: int, pop_base: int, horizon: int = 80, n_stress: int = 4):
...
36	    rng = np.random.default_rng(pop_base + context)
37	    ranges = {"useful": ((.05, .20), (.20, .40)), "stress": ((.45, .65), (.75, .95))}
38	    stressed = sorted(rng.permutation(cfg.n_agents)[:n_stress].tolist())
39	    low = np.array([rng.uniform(*ranges["stress" if i in stressed else "useful"][0]) for i in range(cfg.n_agents)])
40	    high = np.array([rng.uniform(*ranges["stress" if i in stressed else "useful"][1]) for i in range(cfg.n_agents)])
41	    thresholds = rng.uniform(25, 65, cfg.n_agents)
...
45	def fishery_requests(policy, stock):
46	    return np.where(stock < policy["thresholds"], policy["low"], policy["high"])
```

So an agent requests its "low" fraction of the 6 t maximum when the stock is below its threshold, otherwise its "high" fraction. Honest ("useful"): low U(0.05, 0.20), high U(0.20, 0.40). Greedy ("stress"): low U(0.45, 0.65), high U(0.75, 0.95). Thresholds U(25, 65). Greedy count: 4 by default (`mix4`), 2 or 4 in the grids.

Misbehaviour:
- Under-report factor: `run_r2_robustness_reviewer_model.py:55` `LIE = 0.5`; `run_t1_audit_targeting.py:33` `LIE = 0.5`; `run_c1_compositional_harm.py:34` `LIE = 0.5  # stress agents report (1 - LIE) * p`; `run_r4_river_grid.py:27`. S1 fixed levels `run_s1_reporting_audit.py:30` `FIXED_D = (0.0, 0.25, 0.5)`, search `:31` `SEARCH_D = (0.0, 0.25, 0.5, 0.75)`.
- Over-take level grid: `run_s2_compliance_deterrence.py:34` `SEARCH_D = (0.0, 0.25, 0.5, 0.75, 1.0)`; `run_s4_adaptive_memory_cost.py:37` and `run_s5_stronger_attackers.py:39` `LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)`; `run_r3_breakeven.py:23` `LEVELS = (0.25, 0.5, 0.75, 1.0)` (for g*).
- Training contexts for the search: `run_s1_reporting_audit.py:40` and `run_s2_compliance_deterrence.py:44` `train_contexts=8` (S3 and S4 reuse `s2.profile`: `run_s3_threshold_timing_memory.py:323`, `run_s4_adaptive_memory_cost.py:239`); `run_r2_robustness_reviewer_model.py:66` `train_contexts=32` (R3 reuses it, `run_r3_breakeven.py:79`); `run_s5_stronger_attackers.py:54` `train_contexts=32`.
- Reaction to a catch (S4, S5): `run_s4_adaptive_memory_cost.py:38` `REACTIONS = ("continue", "stop")`.
- R2 `CHEAT_LEVEL_FOR_G = 0.75` (`run_r2_robustness_reviewer_model.py:56`).

## Recorded justifications (verbatim)

- `notes/research_review/HELDOUT_POLICY_PILOT_PROTOCOL.md:20-25`: "Fishery uses independent threshold draws, with low/high harvest fractions from [0.05, 0.20]/[0.20, 0.40] for useful and [0.45, 0.65]/[0.75, 0.95] for stress. Thresholds are uniform on [25, 65]. These are design choices, not measured actor-capability levels."
- `notes/research_review/MIXED_POLICY_REPAIR_PROTOCOL.md:18-24`: "Two fixed composition strata are `mix2` (two aggressive, four cooperative) and `mix4` (four aggressive, two cooperative). ... Fishery uses the previously declared useful/stress threshold-policy fraction ranges ... The policy generators are not fitted to the observed monitor results."
- Choice of mix4 (pilot): `notes/research_review/BUDGETED_REVIEWER_CONFIRMATION_PROTOCOL.md:5-6`: "The earlier pilot selected the two informative settings and the intervention rules". `notes/claude_audit_20261005/background/design_issues.md:199-204`: "### B5. The two settings were chosen after a pilot (acknowledged) ... Fishery with 4 aggressive agents and Harvest with 2 under slow renewal were chosen because they produced both safe and risky requests. ... Generality, however, rests on two hand-picked points."
- Under-report by half: no reason beyond inheritance. `R4_river_grid/protocol.md:24`: "each greedy agent reports half of its request (LIE = 0.5, as in C1, T1 and R2's fixed liars)." `T1_audit_targeting/protocol.md:33`: "the greedy agents under-report by half (d = 0.5, a fixed lie), as in S1 and C1." In S1, 0.5 is the largest of the fixed levels {0, 0.25, 0.5} (`S1_reporting_and_audits/protocol.md:101`).
- Over-take family: paper `body.tex:67`: "over-take with a level $d\in\{0,\frac14,\frac12,\frac34,1\}$, taking $a+d(p-a)$, which a search chooses to maximise their payoff on 32 training populations. These agents commit to one level per game". Limitation, `body.tex:131`: "The programmed cheaters optimise over a one-parameter family, a lower bound on an adaptive adversary."
- 8 → 32 training contexts: `notes/claude_audit_20261005/background/caveats_assessment.md:135-137`: "the cheaters' strategy is chosen on only 8 training contexts. In S3 Part C this over-fitted, and the chosen strategy collapsed the stock in 12 of 64 held-out contexts." Option 4c (`:180`): "Use 32 instead of 8".
- S1 search language: `S1_reporting_and_audits/protocol.md:33`: "Misreporters choosing their lie level against the rule | red team / attack policy (a best response to the blue team's protocol) | Greenblatt et al.; Avenhaus et al. ("inspector leadership")". This grounds the *role* (red team), not any value.

## Flags

- **Paper/code mismatch:** body.tex:67 says the search uses 32 training populations; S1, S1b, S2, S3 and S4 used 8 (code above). Only R2, R3 and S5 used 32. The pilot deterrence result in body.tex:96 ("cheaters stopped at $e\approx0.30$--$0.40$") is S3 Part A, 8 training contexts.
- **Pilot-chosen:** mix4 (Fishery) and mix2 (Forest) were picked after the September pilot because they gave both safe and risky requests.

## Verdicts

| Setting | Value | Verdict |
|---|---|---|
| Threshold rule form | low below threshold, high above | U ("design choices") |
| Thresholds | U(25, 65) | U |
| Honest ranges | 5-20% / 20-40% of 6 t | U |
| Greedy ranges | 45-65% / 75-95% | U |
| Greedy count | 4 of 6 (grid 2, 4) | C (pilot: both safe and risky requests) |
| Under-report factor | 0.5 | U (inherited; largest fixed level) |
| Over-take form and grid | a + d(p − a), d ∈ {0, ¼, ½, ¾, 1} | C (one-parameter red team, limitation stated) |
| Search training set | 8 (S1-S4) or 32 (R2, R3, S5) | C (32 after S3 over-fitting); paper misreports as 32 throughout |
| Reaction to catch | continue / stop | C |
