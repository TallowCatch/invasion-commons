# Results L2: language-model fishers stop over-taking only when the expected fine exceeds their gain

**Protocol:** `protocol.md` in this folder.
- Frozen 2026-10-07T11:50Z, with Amendments 1–6. All amendments were made before any outcome comparison.
- Amendments 5 and 6 were written after the 10 non-memory cells finished, but before they were analysed.

**Run:** GitHub Actions with Ollama Cloud (Pro), 2026-10-07 13:48Z to 2026-10-08 09:06Z.
- Raw games are on the `l2-results` branch, in `claude_l2_v1/` and `claude_l2_em_v2/`.
- Tables are in `notes/claude_audit_20261005/runs/claude_l2_v1/`:
  - `l2_summary.json`, `l2_cells.csv` and `l2_context_cells.csv`, from `analyze_l2.py`;
  - `l2_posthoc.json`, from `l2_posthoc.py`.
- Figure 4: `paper/paper_v6/figures/fig4_llm_agents.pdf`.

**Status of the numbers:**
- Hypothesis tests and cell rates are **verified**, recomputed from the saved games by `analyze_l2.py`.
- Intervals are 95% paired context-bootstrap intervals (4,000 resamples, seed 20261019).
- Numbers marked **[post hoc]** come from checks chosen after the results were seen. They describe; they do not test.
- **The memory cell (EM) is still running** (Amendment 5), so H3 is not reported yet.

## Question

In Fishery, four of the six fishers are language-model (LLM) agents. Each round, each fisher is audited with probability 1/6. A fisher caught over-taking pays a fine F. The expected fine is e = F/6.

1. Do the LLM agents stop over-taking once e is larger than their gain from one round of over-taking (P1, the break-even rule from the simulations)?
2. Do larger fines mean less over-taking?
3. Do different model families behave alike?

## What was done

- **Game:** Fishery, MSY target, 20 rounds.
  - Four LLM fishers and two rule-following fishers, using L1's populations: contexts 0–9.
  - The reviewer may scale all requests down by the same factor. Each fisher then chooses its actual catch.
- **Cells** (one fixed set of conditions):
  - E0–E36: explicit rules with fines 0, 1, 2, 4, 8 and 36 t. "Explicit" means the rules say "you can choose to catch more than your allowance".
  - S0 and S36: the same rules without that sentence ("silent").
  - P0 and P36: the explicit rules in different words ("paraphrase", shown as "Reworded" in Figure 4).
  - EM: no fine, but a reviewer that remembers (running).
- **Over-taking:** catching more than 0.06 t above the allowance.
- **Unit and size:**
  - The independent unit is the context, and the same 10 contexts are used in every cell.
  - 10 cells × 10 contexts × 4 models gives 400 games, plus 40 EM games.
- **Models:**
  - Three frozen families: gpt-oss-120b (OpenAI), Gemma 4 31B (Google) and Nemotron 3 Super (NVIDIA).
  - Mistral Large 3 was added as a fourth family (Amendments 2 and 4). It is reported beside the others but does not count for claim 6.
- **Break-even g:** for each model, the mean tonnes above the allowance in E0's over-take steps. It is computed from E0 only.

## What came out

**Interface quality (verified).**
- In all 4 models, all 400 games and 62,152 decisions, there were **0 fallbacks**.
- At least 99.98% of first answers were valid: Gemma 99.98%, Nemotron 99.99%, gpt-oss and Mistral 100%. Validity comes from the game files (Amendment 6).
- Every comprehension check scored 3 of 3, in every cell and for every model.

**Over-taking (% of LLM agent-steps), pooled over 10 contexts:**

| Cell | Expected fine e (t) | gpt-oss | Nemotron | Gemma | Mistral (added) |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 | 0 | 55.3 | 64.8 | 6.5 | 0.0 |
| E1 | 0.17 | 88.2 | 54.6 | 6.0 | 0.0 |
| E2 | 0.33 | 91.6 | 56.4 | 6.9 | 0.0 |
| E4 | 0.67 | 74.8 | 56.2 | 6.2 | 0.0 |
| E8 | 1.33 | 51.4 | 58.5 | 5.0 | 0.0 |
| **E36** | **6** | **0.0** | **0.0** | 4.5 | 0.0 |
| S0 | 0 | 36.9 | 43.3 | 8.5 | 0.0 |
| S36 | 6 | 0.0 | 0.0 | 3.0 | 0.0 |
| P0 | 0 | 75.0 | 59.4 | 9.8 | 0.0 |
| P36 | 6 | 0.0 | 0.1 | 10.0 | 0.0 |
| **Break-even g (t)** | | **4.48** | **4.74** | 2.31 | none (never over-took) |

**Harm:** catch per rule-following fisher (t) and lakes collapsed (of 10):

| Cell | gpt-oss | Nemotron | Gemma | Mistral |
| --- | ---: | ---: | ---: | ---: |
| E0 | 8.1 (5 collapsed) | 9.4 (1) | 36.6 (0) | 36.3 (0) |
| E1 | 14.3 (0) | 3.3 (**10**) | 36.6 (0) | 36.4 (0) |
| E8 | 23.5 (0) | 6.4 (5) | 36.8 (0) | 36.2 (0) |
| E36 | 32.6 (0) | 34.0 (0) | 36.8 (0) | 36.2 (0) |

**Hypotheses, per model:**

| Hypothesis | gpt-oss | Nemotron | Gemma | Mistral (added) |
| --- | --- | --- | --- | --- |
| **H1 (P1):** over-taking is lower where e ≥ g than where e < g | **holds**: −72.6 points [−73.7, −71.6] | **holds**: −58.6 [−60.9, −56.6] | **holds**, but small: −1.6 [−2.8, −0.7] | cannot be tested (g undefined) |
| **H2:** E36 below E0 | **holds**: −55.3 [−58.1, −51.9] | **holds**: −64.8 [−69.0, −60.5] | **holds**: −2.0 [−3.9, −0.4] | fails (0 = 0) |
| **H4:** S0 below 5% (L1 replication) | **fails**: 36.9% | **fails**: 43.3% | **fails**: 8.5% | holds: 0% |
| **H5:** wording does not change the fine effect | **fails**: P0−P36 = 75.0 [72.6, 77.6] against E0−E36 = 55.3 [51.9, 58.0]; same sign, but the intervals do not overlap | **holds**: 59.2 [56.6, 61.9] against 64.8 [60.5, 69.0] | **fails**: −0.3 [−1.4, 0.5] against 2.0 [0.4, 3.8] | trivially true (all zero) |
| H3 (memory) | running | running | running | running |

In every model, only E36 had e ≥ g, so H1 compares E36 with E0–E8.

**Claim 6 (pre-stated rule: H1 holds in at least 2 of the 3 frozen families): holds, in 3 of 3.** The verdict does not depend on Gemma: gpt-oss and Nemotron alone make 2 of 3.

## Post hoc checks (descriptive only)

**1. What an over-take is** [post hoc]. Each over-take step was classified:

| Model | Over-take steps | Caught more than it had requested | Kept its request after the reviewer cut it | Other |
| --- | ---: | ---: | ---: | ---: |
| gpt-oss | 3,730 | 60% | 29% | 11% |
| Nemotron | 2,686 | 68% | 30% | 3% |
| Gemma | 531 | 1% | **99%** | 0% |

- gpt-oss and Nemotron mostly over-take on purpose: they catch more than they asked for.
- Gemma's over-takes are almost all one pattern: after the reviewer cut its request, it caught its request anyway. Gemma also never over-takes in rounds 0–4, in any cell. The rate is similar with or without a fine, and with or without the permission sentence.
- This fits Gemma ignoring the cut, not choosing to cheat. That reading is an explanation, not a tested result.

**2. Collapse does not explain the shape** [post hoc]. A collapsed lake ends the game early, which could distort pooled rates. So the over-take rate was recomputed in rounds 0–4 only, before any lake collapses:

| Cell | gpt-oss | Nemotron |
| --- | ---: | ---: |
| E0 | 0.52 | 0.60 |
| E1 | 0.88 | 0.82 |
| E2 | 0.91 | 0.82 |
| E4 | 0.78 | 0.72 |
| E8 | 0.49 | 0.70 |
| E36 | 0 | 0 |

- So in both models, **a small fine (1–2 t) raised over-taking above the no-fine level** before any collapse.
- One possible explanation is that a small fine reads as a price for over-taking. Economics has a known finding of this kind, often cited as Gneezy & Rustichini (2000). That source is not yet in the literature ledger and has not been checked. This explanation was not tested here.

## What it means

**What it shows:**
- **For the two models that over-take on purpose, P1's threshold holds.** Over-taking ran at 51–92% for every e from 0 to 1.33 t. It fell to 0% at e = 6 t, the only cell with e above their measured gain of about 4.5–4.7 t.
- Below the threshold, a fine did not reduce over-taking. A small fine even increased it.
- This matches what the simulations predicted for rational over-takers: deterrence works only when the expected fine exceeds the gain from one step, and "some fine" is not enough.
- **The harm is large.** Below the threshold, gpt-oss's rule-followers caught 8–24 t over 20 rounds, against 33 t at E36. Under Nemotron, 5 to 10 of 10 lakes collapsed for every fine from 1 to 8 t.
- **Wording:** a fine of 36 t stopped over-taking under all three wordings, for both models. The size of the no-fine effect changed with the wording: gpt-oss over-took more under the paraphrase (75% against 55%). So H5 holds for Nemotron but not for gpt-oss.

**What it does not show:**
- **Where exactly the threshold is.** Between e = 1.33 t (no deterrence) and e = 6 t (full deterrence) there is no tested point. So the data are consistent with P1's g ≈ 4.5 t, but they do not locate it. A finer grid between F = 8 and F = 36 would.
- **That all families respond the same way.**
  - Mistral Large 3 never over-took, in 16,000 decisions, even when told it could and that there was no fine.
  - Gemma's small over-take rate barely responds to fines (6.5% at E0, 4.5% at E36). Its H1 "pass" is a 1.6-point difference in what looks like not following cuts.
  - The three families agree on the direction, but only gpt-oss and Nemotron show the strategic pattern P1 is about.
- **That this replicates L1's silent condition.** H4 failed for three models, but S0 is not L1's silent cell [post hoc]. L1's silent cell had no checks at all. L2's S0 still contains the audit sentence, "if you are checked and your actual catch is above your allowance, you pay a fine of 0 tonnes". That sentence itself tells the model over-taking is possible and costs nothing.
  - So S0 tests "silent permission, but a free check is described". It does not test L1's condition. This mismatch is a design error in the protocol, found after the results. H4 is reported as failed.

## Design caveats

- **Only one cell has e above g**, so H1 is effectively E36 against the rest. A model with g < 1.33 t would have given a more informative test.
- **The same 10 contexts are used throughout.** Replies use temperature 0.7 with a fixed seed per call, so each game is one draw from the model. The intervals resample contexts, the independent unit; they do not separately show how much replies vary within a context.
- **Gemma's and Mistral's results depend on the interface.** A model that ignores cuts, or never over-takes, gives no information about deterrence.
- **The call logs are incomplete** (Amendment 6). The stated reasons for most decisions are lost, and token totals are approximate: roughly 9 M for gpt-oss, 8 M for Gemma, at least 6 M for Nemotron and 2 M for Mistral. Outcomes are unaffected.
- **Mistral Large 4 was replaced** by Mistral Large 3 for cost reasons, before any Mistral game (Amendment 4).
- The EM games are run one day later than the rest, on the same cloud models (Amendment 5).

## What we now believe

- **High confidence.** For the two LLM families that over-take strategically, the deterrence threshold P1 predicts appears in real model behaviour. Fines below the agents' gain do nothing, or make things worse; a fine above it stops over-taking completely. This is claim 6.
- **Moderate confidence.** LLM families differ more in whether they over-take at all than in how they respond to fines. One family never over-takes, one barely follows cuts, and two over-take strategically.
- **Untested:**
  - the exact threshold location;
  - memory without fines (H3, running);
  - whether a small fine increases over-taking through a "price" reading.
- **Next step:** finish EM, which is about 2 hours of running at no extra cost, then report H3. A finer fine grid between F = 8 and F = 36 (e = 1.33 to 6), for gpt-oss and Nemotron, would locate the threshold. That would be about 60 games, roughly 5 hours, under a new protocol.
