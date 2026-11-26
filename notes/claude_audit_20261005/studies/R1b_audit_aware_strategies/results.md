# Results R1b: do LLM-written strategies respond to audits when told the rule?

**Status: complete.** Parts 1 and 3 were analysed 19 October; Part 2 (240 evolution runs, r1b-evolve run 39275589964) was analysed 20 October. Still to do: the hand check of 20 strategies (Part 3).
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

## Part 2: in evolution, oversight works by changing what Selfish strategies do, not by removing them

**What was done.** Willis et al.'s cultural evolution at n = 64, q = 0.1, with each pool alone (the genotype is the attitude), 20 runs per condition. Fines e ∈ {0, 0.5, 1, 1.25, 2} × e_pred(B), where e_pred(B) = **0.63**, frozen before any run, plus a no-audit baseline (Amendment 3).

| Expected fine ÷ e_pred(B) | no audits | 0 | 0.5 | 1.0 | **1.25** | 2.0 |
|---|---|---|---|---|---|---|
| **A (told the rule):** share Collective | 0.14 | 0.14 | 0.20 | 0.17 | **0.19** | 0.17 |
| A: welfare efficiency, fines as transfers / as lost | 0.03 / 0.03 | 0.03 / 0.03 | 0.22 / 0.17 | 0.37 / 0.28 | **0.45 / 0.34** | 0.64 / 0.51 |
| A: defections per agent per game (outside the final round) | 13.6 | 13.7 | 6.3 | 5.6 | 5.1 | 4.1 |
| **B (blind):** share Collective | 0.15 | 0.15 | 0.27 | 0.41 | **0.60** | 0.87 |
| B: welfare efficiency, as transfers / as lost | 0.04 / 0.04 | 0.04 / 0.04 | 0.06 / −0.05 | 0.09 / −0.12 | **0.13 / −0.11** | 0.42 / 0.16 |
| B: defections per agent per game | 15.3 | 15.3 | 13.6 | 12.6 | 11.8 | 7.9 |

**H4: holds.**
- At 1.25 × e_pred, Pool A's welfare efficiency exceeds Pool B's by **+0.32 [+0.29, +0.34]**: 0.45 against 0.13 (fines as transfers).
- Counting fines as lost, it is 0.34 against −0.11. In Pool B, much of the "harvest" goes to fines, because the blind strategies keep defecting.

**H5 (descriptive): the two pools reach welfare in opposite ways.**
- **Blind strategies (B): oversight works by selection.** The Selfish strategies cannot react, so they pay the fines and get copied less. The Collective share rises from 0.15 to 0.87, crossing half at 1.25 × e_pred, as in R1a.
  - But welfare stays low until 2 × e_pred (0.42), because gpt-oss's blind *Collective* strategies still defect often in this game (0.63 of rounds at n = 64 in Part 1).
- **Audit-aware strategies (A): oversight works by compliance.** The Selfish strategies stop defecting when the fine is high enough, so they are no longer worse off and selection never removes them. The Collective share stays at 0.14–0.20 at every fine. Defections fall by 63% (13.7 → 5.1 at 1.25×), and welfare rises to 0.45 and 0.64.
- Without audits, Pool A falls back to 0.03 efficiency: the same strategies, the same Selfish majority.

## What it means

- **When told the rule, gpt-oss writes strategies that obey Becker** (Part 1). Most compute the expected fine explicitly.
- **There are two routes to deterrence, and they leave different populations behind:**
  - against agents that cannot react, oversight changes *who* is in the population (R1a, Pool B);
  - against agents that can, it changes *what they do* while leaving them Selfish (Pool A).

  The second gives far higher welfare at the same fine.
- **Compliance is not conversion.** Under the overseer, Pool A stays four-fifths Selfish. Switch the audits off and its defection returns at once (no-audit level: 0.74 in Part 1, 0.03 efficiency in Part 2). So an overseer judged by the *share of cooperative types* would look like it failed, while one judged by behaviour and welfare would look like it worked. **Oversight metrics have to measure behaviour.** For AI deployment, observed compliance under audits says little about how the same agents behave without them.
- **Code-writing and acting agents differ.** The zero-fine announcement that backfired with live agents (L4) had no effect on written strategies.

## What it does not show

- Anything beyond one model (gpt-oss-120b) and one game.
- **Selection after the pilot:** the pilot's 20 strategies are part of the pools. They were generated with the same prompts, and nothing about them was analysed before the full run except that they were valid and could read the audit fields.
