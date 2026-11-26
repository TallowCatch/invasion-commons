# L08 River (two-reagent, compositional game; C1, T1, R4)

Local audit source, 14 Oct 2026. Read only.

## Code

`fishery_sim/two_reagent.py`:

```
12	  compositional game ("comp"):  D = kappa * X * Y + lam * (X + Y)
13	  additive control   ("add"):   D = lam_add * (X + Y)
14	  Q' = clip(Q - D + r Q (1 - Q/100) + eps, 0, 100),  eps ~ N(0, sigma^2)
15	  a step is unsafe if Q' < 30.
...
34	N_AGENTS = 6
35	X_IDX = (0, 1, 2)
36	Y_IDX = (3, 4, 5)
38	Q_INIT = 80.0
39	Q_UNSAFE = 30.0
40	Q_MAX = 100.0
41	P_MAX = 1.0
42	TOL = 0.05
...
50	RANGE_SETS = {
51	    "fishery": dict(normal=((0.05, 0.20), (0.20, 0.40)), stress=((0.45, 0.65), (0.75, 0.95)), thresholds=(35.0, 70.0)),
52	    "wide": dict(normal=((0.00, 0.10), (0.35, 0.65)), stress=((0.15, 0.35), (0.75, 0.95)), thresholds=(35.0, 75.0)),
53	    "wide_low": dict(normal=((0.00, 0.10), (0.25, 0.50)), stress=((0.10, 0.30), (0.70, 0.95)), thresholds=(35.0, 75.0)),
...
57	LAM_RATIO = 0.05
...
254	    stressed = [int(rng.integers(0, 3)), int(3 + rng.integers(0, 3))]  # one X and one Y stress agent
...
280	FROZEN_RANGES = "wide_low"
281	FROZEN = make_params("comp", 32.0, 1.0, 3.0)
```

Printed: `Params(game='comp', kappa=32.0, lam=1.6, lam_add=22.917041234151657, r=1.0, sigma=3.0)`.

Calibration, `experiments/oversight/run_c1_compositional_harm.py`:

```
30	CALIB_SEEDS = dict(pop=1_490_000_000, weather=1_495_000_000, contexts=8, horizon=80)
31	CALIB_GRID = dict(ranges=tuple(TR.RANGE_SETS), kappa=(16, 20, 24, 28, 32), r=(0.7, 0.85, 1.0, 1.2), sigma=(3, 5, 7))
34	LIE = 0.5  # stress agents report (1 - LIE) * p
...
70	def feasible(pt):
71	    return all(0.30 <= pt[g]["risky_share"] <= 0.60 and pt[g]["presafe_share"] >= 0.5 for g in TR.GAMES)
74	def objective(pt):
75	    return abs(pt["comp"]["risky_share"] - 0.45) + abs(pt["add"]["risky_share"] - pt["comp"]["risky_share"])
...
123	                audited = int(np.random.default_rng(stable_seed(SEEDS["audit"], game, ctx, t)).integers(0, TR.N_AGENTS))
```

R4 grid, `experiments/oversight/run_r4_river_grid.py:23-29`: `GREEDY = (2, 4)`, `REGROWTH = (1.0, 0.85)`, `CONDITIONS = ("trust", "memoryless", "memory")`, `LIE = 0.5`, `TARGET = 50.0`; 4 greedy = two X and two Y (`:51-62`); one random audit per round (`:79-80`).

## Recorded justifications (verbatim), `notes/claude_audit_20261005/studies/C1_compositional_harm/protocol.md`

- Why the game (`:9-21`): "Fishery and Harvest share three features: one number per agent; harm that grows with the total taken; one shared cut. ... C1 breaks the second feature. Harm comes from a **combination** of actions, the case Hu & Wang (2026) call "compositional harm"".
- Parameters (`:42-46`): "fixed by a calibration on 8 calibration contexts ... before any test context is run ... They are chosen so that, with no reviewer: about 30–60% of steps are risky in the compositional game; the additive control has a similar share."
- Amendment 1 table: "λ | 1.6 (λ = 0.05 κ, ratio fixed before the search)"; "**Grid:** 3 request-range sets × κ ∈ {16, 20, 24, 28, 32} × r ∈ {0.7, 0.85, 1.0, 1.2} × σ ∈ {3, 5, 7}, which is 180 points." "**Result:** 9 of 180 points were feasible, and the rule picked the point above."
- "**Coarse look before the grid** (same calibration contexts, not saved): With the Fishery request ranges, the additive control was risky at about 90% or more of steps at every κ, r and σ tried. ... This is why two request-range sets with a wider low/high gap were added."
- "**How compositional the harm is.** The κXY term is 94% of the damage at the mean high requests and 75% at the mean low requests."
- Part B (`:55-61`): "The stress agents under-report by half. One audit per step ... A caught liar is corrected using memory, as in S3 Part D".
- Naming and grounding of the damage form, `literature_ledger.md` (7 Oct): Cedergreen (2014) "mixture toxicity" and "synergy" ("Abstract and definition"); Segerson (1988) non-point source pollution ("Secondary sources only").
- `literature_ledger.md:123` (TMDL): "This supports River's harm line as a water-quality standard (Q ≥ 30) ... A "half-capacity" (Q ≥ 50) line for River would be our modelling analogue of MSY, not a regulatory standard." Kept only as background (`:108`).
- R4 target (`R4_river_grid/protocol.md:18`): "Its target is **quality ≥ 50 (half capacity)**, not C1's 30. In all three games, the reviewer then aims at the same line the harm is measured against."

## Flags

- **Pilot/calibration-chosen:** κ, r, σ and the request ranges were picked by a recorded rule aimed at a no-reviewer risky share near 45%. An unsaved "coarse look" preceded the grid, and two range sets were added because of it. Calibration used 8 separate contexts and preceded the test contexts.
- λ = 0.05 κ, Q₀ = 80, the 30 line and the per-agent maximum of 1 have no recorded reason.

## Verdicts

| Setting | Value | Verdict |
|---|---|---|
| Recovery law | logistic in quality, K = 100 | G (same logistic sources) |
| Damage form | κXY + λ(X+Y), 3 X and 3 Y agents | C (built to make harm compositional; concept cited: Hu & Wang 2026, Cedergreen 2014) |
| κ, r, σ, ranges, thresholds U(35, 75) | 32, 1.0, 3, `wide_low` | C (recorded calibration rule) |
| λ/κ | 0.05 | U |
| λ′ (additive control) | matching rule, 22.92 | C |
| Q₀, Q_max, per-agent max | 80, 100, 1 | U |
| Unsafe line | 30 | U |
| Stress agents | 1 X + 1 Y (2g); 2 X + 2 Y (4g) | C |
| Lie | 0.5 | U |
| Audits | one per round, random or targeted | C |
| R4 grid and target | ×1.0/×0.85; quality ≥ 50 | C (consistency across games) |
