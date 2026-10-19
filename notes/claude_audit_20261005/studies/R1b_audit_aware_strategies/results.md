# Results R1b: do LLM-written strategies respond to audits when told the rule?

**Status:**
- **Part 1 (in-game test) and Part 3 (what the code does): complete.**
- **Part 2 (cultural evolution): running** on GitHub Actions (r1b-evolve run 39275589964).
- Analysed 19 October 2026 from the `r1b-results` branch with `analyze_r1b` (summary `runs/claude_r1b_v1/r1b_summary.json`, `r1b_part1.csv`). All numbers are **[verified]** from those files. The no-audit level comes from `part1/pool_*_none.json.gz`, read directly (descriptive, Amendment 3).

**Protocol:** `protocol.md` (frozen 15 Oct; Amendments 1–3, all before any real LLM call).

## Generation (verified)

- gpt-oss-120b on Ollama: **254 of 256 strategies valid** (127 per pool; one failure per pool after 3 attempts), **516 calls, estimated $0.58** (cap $3).
- **The pilot** (20 strategies, $0.05) projected $0.60 for 256 and passed the $2.5 gate.
- **Pool A** (audit-aware) got Willis et al.'s prompts plus one "AUDITS AND FINES" section. The worked example only computes the expected fine (Amendment 3). **Pool B** (blind) got their prompts unchanged.

## Part 1: audit-aware strategies cut defection sharply as the expected fine rises

**What was done.** Each strategy played 20 games as one of n players, with the other seats drawn from its own pool, at q = 0.1 and expected fine e = qF ∈ {0, 0.5, 1, 2, 3, 4}, plus a no-audit level (q = 0).
- **Outcome:** the strategy's defection rate outside the final round.
- **Unit:** the strategy (63–64 per attitude per pool).

**Defection rate, outside the final round:**

| Pool, group size, attitude | no audits | e = 0 | 0.5 | 1 | **2** | 3 | 4 |
|---|---|---|---|---|---|---|---|
| A (told the rule), 64, Selfish | 0.74 | 0.74 | 0.35 | 0.25 | **0.06** | 0.04 | 0.03 |
| A, 64, Collective | 0.34 | 0.34 | 0.29 | 0.26 | 0.03 | 0.02 | 0.02 |
| A, 4, Selfish | 0.43 | 0.43 | 0.25 | 0.21 | 0.04 | 0.03 | 0.03 |
| A, 4, Collective | 0.28 | 0.28 | 0.26 | 0.24 | 0.04 | 0.03 | 0.03 |
| B (blind), 64, Selfish | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 |
| B, 64, Collective | 0.63 | 0.63 | 0.63 | 0.63 | 0.63 | 0.63 | 0.63 |
| B, 4, Selfish | 0.56 | 0.56 | 0.56 | 0.56 | 0.56 | 0.56 | 0.56 |
| B, 4, Collective | 0.36 | 0.36 | 0.36 | 0.36 | 0.36 | 0.36 | 0.36 |

**H1 (in-game deterrence): holds at both group sizes.**
- **n = 64:** Selfish defection at e = 2 is **0.08 of its e = 0 level** [0.02, 0.16], against the falsifier of 0.5. It never rises from one level to the next.
- **n = 4:** the ratio is 0.08 [0.01, 0.18].
- Deterrence starts well below the myopic full-stock benchmark (e = 2). At e = 0.5, Selfish defection at n = 64 already halves (0.74 → 0.35), because the gain from defecting is smaller when the stock is below capacity.

**H2 (Pool B unaffected): holds.** This is true by construction: the blind strategies never see q or F. It confirms the code.

**H3 (Collective strategies at e = 0): Pool A defects less than Pool B.**
- **n = 64:** 0.34 against 0.63, difference −0.28 [−0.42, −0.15].
- **n = 4:** 0.28 against 0.36, difference −0.08 [−0.16, +0.01].

So strategies written for a game with audits are more restrained **even when the audit is switched off**. The prompt changes what gets written, not only how the code reacts to q and F.

**No audits = audits with a zero fine (descriptive, Amendment 3).** Pool A's rates are identical at q = 0 and at q = 0.1, F = 0. The written code responds only to the expected fine, so a zero-fine announcement changes nothing. This contrasts with the live LLM agents of L4 (F4), where announcing a zero-fine audit raised over-taking from 30% to 58%.

## Part 3: what the code does (verified, heuristic)

| Pool A | All | Collective | Selfish |
|---|---|---|---|
| Reads q or F | 96% | 92% | **100%** |
| Explicit expected-value comparison (product of q and F against a gain) | 62% | 50% | **75%** |
| Reads its own audit history | 7% | 9% | 5% |

Pool B: 0% on all three, as expected. Three-quarters of the Selfish strategies implement Becker's rule explicitly (defect only if the gain beats q × F). Hardly any learn from being audited: they use the stated probability, not experience. A hand check of 20 randomly drawn strategies (seed `R1B_BASE + 2`) is still to do. The list is in `runs/claude_r1b_v1/r1b_part3_strategies.csv`.

## Part 2: cultural evolution

**Pending, running.** The balance fine e_pred(B) = **0.63** per defection was computed and pushed at 15:11 UTC, before any evolution run. The grid is 2 pools × (5 fine levels + no-audit baseline) × 20 runs at n = 64. H4 (does Pool A reach higher welfare than Pool B at 1.25 × e_pred?) and H5 follow when the runs are in.

## What it means so far

- **When told the rule, gpt-oss writes strategies that obey Becker.** Selfish defection falls by 92% at an expected fine of 2, most strategies compute the expected fine explicitly, and the blind control does not move.
- Together with R1a, both channels of deterrence now work on Willis et al.'s game: **selection** (R1a, blind strategies) and **in-game response** (R1b).
- **Code-writing agents and acting agents can respond differently to the same announcement.** The zero-fine announcement that backfired with live agents (L4) had no effect on written strategies, which respond only to the expected fine.

## What it does not show (yet)

- Whether in-game deterrence adds to selection in evolution (H4, pending).
- Anything beyond one model (gpt-oss-120b) and one game.
- **Selection after the pilot:** the pilot's 20 strategies are part of the pools. They were generated with the same prompts, and nothing about them was analysed before the full run except that they were valid and could read the audit fields.
