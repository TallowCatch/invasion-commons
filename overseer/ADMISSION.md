# `overseer` v0: admission tests

14 October 2026. The overseer package is admitted for the text Fishery if it reproduces the current reviewer **exactly**, on stored LLM games and on programmed games. Both tests pass with no exceptions and no loosened tolerances.

## Test A: exact replay of the stored LLM games

**File:** `tests/test_overseer_replay_l4.py`. It skips a data set whose folder is absent.

**Data:** the `l2-results` branch extract, `results/runs/l2_results_branch/<study>/<model>/episodes/*.json`.

**Set-up.** Each episode gets an overseer configured like its cell:
- `SharedCut()` with the MSY target, through `adapters/fishery_text.FisheryResource`;
- `UniformRandom(q, SEEDS["audit"], prefix=(context,))`, which is the runner's `stable_seed(audit, context, t)`;
- `Perfect(tol=OVER_T / MAX_CATCH)`, i.e. 0.06 t;
- `Fine(F)`, `Memory(rule)` or no sanction (N0, with q = 0).

The seeds are L4's `SEEDS` for L4 and L5, and L2's `SEEDS` for L2, the L2 EM re-run and L3.

**Inputs fed in:** the logged requests and catches (`taken`). The stock is **not** taken from the log. The adapter advances it from the catches, so the stock path is checked too.

**What is compared, round by round:**
- stock;
- scale;
- allowances;
- audited and caught (exact booleans);
- the MSY-break flag.

**What is compared at the end:** fines per agent, payoffs per agent, final stock, and collapse. Floats use `np.allclose(atol=1e-12, rtol=0)`, and the test also counts which episodes are bit-for-bit equal.

| Set | Cells | Episodes matched | Bit-for-bit |
|---|---|---|---|
| **L4** (`claude_l4_v1`, gpt-oss 190 + nemotron 73) | E0, P0, N0, E12–E30, P24, P30, Q3_6, Q3_9, Q12_24, Q12_36 | **263 / 263** | 263 |
| L2 (`claude_l2_v1`) | E0–E36, S0, S36, P0, P36, EM (`share` rule) | 402 / 402 | 402 |
| L2 EM re-run (`claude_l2_em_v2`) | EM (`tonnes` = `excess` rule) | 40 / 40 | 40 |
| L3 (`claude_l3_v1`) | E12–E30 | 80 / 80 | 80 |
| L5 (`claude_l5_v1`) | E0 | 10 / 10 | 10 |

**Notes on the data.**
- L4 has fine cells and the no-audit cell, but no memory cell. Memory is admitted on L2's EM games: 81 catches in total, and 200 allowances that memory lowered.
- 11 L4 episodes collapsed. The replay stops where the log stops, and the collapse matches.
- L4 had no fallback catches (`fallbacks = 0` in every episode). A fallback would be logged as the allowance anyway, so it would replay the same way.
- L2/L3 episodes have no `q` field, so L2's Q = 1/6 is used. `claude_l2_v1`'s EM games have no `memory_rule` field; that run used `share` (L2 Amendment 3).

**The test is sensitive.** Changes that should break it do break it:
- detector tolerance 1e-12 instead of 0.06 t: 83 / 263 L4 episodes fail;
- audit seed + 1: 243 / 263 fail;
- the memory rules swapped: 14 / 40 EM re-run games fail, and 1 of the 2 L2 EM games.

## Test B: programmed re-run of S3 Parts A and B

**Files:** `tests/test_overseer_admission_s3.py` and the runner `experiments/oversight/run_overseer_admission.py`.

**What the runner keeps from S3:** S3's programmed agents (the S2 cheater takes `a + d (p − a)`), its populations, seeds, cheating-level search, and its output format.

**What the overseer supplies:** the reviewer, audits, detection and fines:
- `SharedCut()`;
- `UniformRandom(q, audit, prefix=(context,), suffix=(round(q, 6),))`, which is S3's seed key and includes q;
- `Perfect(1e-12)` at s = 1, or `Noisy(miss = 1 − s, 0, detect, prefix=(context,))` at s < 1 (Part B);
- `Fine(e / (q s))`.

**Profile:** full. Part A is 2,944 test episodes (4 audit rates × 11 expected fines) plus training searches. Part B is 1,536 test episodes (s = 0.5 and 0.25).

**Compared against the stored runs** (`results/runs/claude_s3_partA_v1`, `results/runs/claude_s3_partB_v1`):

| What | Result |
|---|---|
| `episodes.jsonl.gz` and `adaptive_search.json`, A and B | **byte-identical** (sha256 `8d4a49bd…`, `2e3e265d…` for A; `123d8b29…`, `a8f5e9ce…` for B; the same as the stored manifest) |
| every episode record and every search score | equal (2,944 / 2,944 and 1,536 / 1,536) |
| the stored analysis, `notes/claude_audit_20261005/runs/claude_s3_partA_v1/s3_summary.json` part `AB`, recomputed by the unchanged `analyze_s3.parts_ab` (its bootstrap seed 20261008) on the re-run | equal in every field, bootstrap intervals included |

The headline numbers reproduced:
- e* = 0.30, 0.35, 0.40, 0.30 at q = 1/6, 0.10, 0.05, 0.02;
- e* = 0.30 at s = 0.5 and 0.25;
- d* (no audits) = 0.75;
- held-out gain per cheater = 28.47;
- predicted e threshold = 0.356;
- A_H1 true, A_H2 within one step false, B_H1 true.

## Exceptions

None. No tolerance was loosened.

## Differences from DESIGN.md (v0)

1. **`ResourceModel` has more than the three methods in DESIGN.md.**
   - `within_target(state, total)`: the exact test. "Total ≤ `safe_total`" (stock − K/2) and the runners' `max(0, stock − total) ≥ K/2` can differ in the last bit, so `SharedCut` calls the game's own test. `safe_total` is still provided.
   - `total(takes)`: a game-unit total with the runners' order of operations, `(take × 6).sum()`.
2. **`harm(state, takes)` instead of `harm(state_after)`.** The Fishery harm line is on the stock *after fishing, before regrowth* (L2's `msy_break`), and the next state does not hold that.
3. **The over-taking tolerance lives in the detector**, as `Perfect(tol)` and `Noisy(..., tol)`. The runners use two values: 0.06 t in L2–L5, and 1e-12 in S2–S5.
4. **Audit seed keys are `(seed, *prefix, window, *suffix)`.** S3/S4 put q in the key, so their audits at different q are *not* nested. L2–L5 do not, so theirs are nested.
5. **`Memory` is a sanction that the allowance policy reads.** The overseer passes the first `Memory` in `sanctions` to `SharedCut` or `TargetedCut`. `Memory("tonnes")` is accepted as the episode files' name for `excess`.
6. **`Leniency` is a stub.** It raises `NotImplementedError` until a report channel exists (L7).
7. **`Budget` caps** (per window or per episode) keep the first audits in agent order. No admitted study uses a cap.
