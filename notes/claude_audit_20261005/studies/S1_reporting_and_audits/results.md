# Results S1: self-reports, audits and agents who misreport

- **Protocol:** `studies/S1_reporting_and_audits/protocol.md`, frozen
  2026-10-04T23:58Z. It has one amendment: a fine was added to the sanction
  after the smoke run and before the full run.
- **Run:** `results/runs/claude_s1_reporting_audit_v1`, which took 235 seconds
  and produced 3,328 test episodes plus 384 training episodes.
- **Analysis:** `experiments/analyze_s1_reporting_audit.py`; outputs in
  `.../analysis/`.

Every number below is **verified** from the run data. An independent
check-through (the verification log, §8) corrected several statements in an earlier draft
of this file. Explanations marked **[post hoc]** were not tested.

## 1. Question

Real overseers often have to rely on what actors *say*, and can only check
some of it. This experiment asks four things:

- When agents report their own requests and some under-report, what happens
  to safety if the reviewer trusts the reports?
- How much do a few audits per step help?
- Does it matter whether audits are random or aimed at suspicious agents?
  Do neighbour reports help?
- Do misreporters who choose how much to lie, knowing the checking rule,
  find a lie that pays?

## 2. How it relates to scalable oversight and AI control

All terms are from `literature_ledger.md`. This is an **analogy, not an
equivalence**. The ledger's cautions apply: our actors are simple fixed
rules, our audit rates (17–33% of reports per step) are far higher than the
2% in Greenblatt et al. (2023/24, arXiv:2312.06942), and we compute no
equilibria.

The closest established setup is the AI-control evaluation of oversight
**protocols** (Greenblatt et al.):

- a limited trusted monitor;
- a small **audit budget**;
- safety and usefulness measured separately;
- a red team searching for an attack policy.

From the economics of verifying private reports, the related ideas are:

- **Inspection games:** an inspector with limited resources and partial
  verification (Avenhaus, von Stengel & Zamir 2002, full text checked).
- **Deterrence by sanctions:** Becker 1968 and Allingham & Sandmo 1972.
  For these two only the identifiers were verified, so their specific
  results are not attributed here.

| S1 element | Closest literature idea |
| --- | --- |
| Reviewer acts on reports with no verification | a protocol with zero audit budget (no established name) |
| Verify *k* of 6 reports per step | audit budget; partial verification |
| Random vs suspicion-targeted audits | auditing at random vs "audit the top most suspicious" (Greenblatt et al.) |
| A neighbour reports on an agent | untrusted monitoring (one actor reviews another) |
| A misreporting neighbour covers for another | a scripted, narrow form of collusion |
| Search over how much to under-report | red team / attack policy; here only a one-parameter search |

The unsafe-action rate below plays the role of "safety" and total harvest
plays the role of "usefulness". The safety estimand is different, though:
AI control asks whether *any* catastrophe happens in a deployment, while we
count a per-step rate.

## 3. What was done

**Games.**

- **Harvest** (primary): 2 of 6 agents are aggressive and may under-report.
- **Fishery** (secondary): 4 of 6 agents may under-report.

**Reports.** Each step, every agent reports *r = p × (1 − d)* for its true
request *p*. Honest agents have *d* = 0.

**Reviewer.**

- It uses R1's calibrated joint chance constraint on what it *believes* the
  requests are.
- The chosen scale is applied to the **true** requests.
- Both games use the one-step target, for comparability.

**Sanction.** A caught agent gets 0 that step and for the next 5 steps, and
pays a fine of 12.

**Protocols.**

| Protocol | What the reviewer does |
| --- | --- |
| `none` | No reviewer |
| `full` | Reads the true requests directly and never sanctions |
| `report` | Acts on the reports |
| `rand1`, `rand2` | Audits 1 or 2 random agents per step |
| `targ1`, `targ2` (Harvest only) | Audits the 1 or 2 agents whose patches have recently come in furthest below the reviewer's predictions |
| `peer` | Each agent's clockwise neighbour reports its true request |
| `peer_collude` | As `peer`, but a misreporting neighbour covers for a misreporting agent |

**Misreporting levels.**

- Fixed: *d* = 0, 0.25 or 0.5.
- Adaptive search: for each protocol, choose *d* ∈ {0, 0.25, 0.5, 0.75} to
  maximise misreporters' net payoff on 8 separate training contexts, then
  test on the 64 test contexts.

**Sample.** 64 test contexts per game (the independent unit), each run for
80 steps.

**Main safety measure: the unsafe-action rate.** This is the share of steps,
starting from a safe state, in which the *executed* joint action had
one-step risk above 5%. It is deliberately different from R1's "unsafe
approval rate", which is risky requests approved divided by risky requests.
The two should not be compared directly. Denominators vary when episodes end
early: Fishery `report` has 1,064 scored steps against 5,120 for `full`.

## 4. What came out

### 4a. Engineering check (H1)

With *d* = 0, every report-based protocol gave the same episode totals as
`full`: harvest, unsafe actions, usefulness loss and mean health, with 0
mismatches. This check compares totals, not individual decisions. It is close
to automatic: with *d* = 0 the reports equal the requests, so nobody is ever
caught.

### 4b. Harvest, misreporters under-reporting by half (*d* = 0.5)

| Protocol | Unsafe actions | Steps where the effective request was safe | Total harvest | Honest agents' harvest | Misreporters' harvest (before fines) | Misreporters' net payoff | Audits / step | Catches / episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 790 / 2,870 (27.5%) | – | 701.6 | 641.1 | 60.5 | 60.5 | 0 | 0 |
| full | 18 / 5,089 (0.35%) | 2,961 / 5,089 | 678.8 | 613.7 | 65.1 | 65.1 | 6 | 0 |
| report | **150 / 5,021 (3.0%)** | 2,890 / 5,021 | 675.0 | 613.1 | 61.9 | 61.9 | 0 | 0 |
| rand1 | 179 / 5,047 (3.5%) | 4,404 / 5,047 | 767.2 | 623.4 | 143.8 | −44.2 | 1 | 15.7 |
| rand2 | 19 / 5,108 (0.4%) | 5,060 / 5,108 | 769.1 | 627.4 | 141.7 | −118.5 | 2 | 21.7 |
| targ1 | **0 / 5,120** | 5,120 / 5,120 | 720.4 | 633.0 | 87.4 | −207.4 | 1 | 24.6 |
| targ2 | 1 / 5,120 | 5,119 / 5,120 | 690.7 | 637.6 | 53.2 | −256.6 | 2 | 25.8 |
| peer | 0 / 5,120 | 5,120 / 5,120 | 644.3 | 644.3 | 0 | −336.0 | 0 (6 messages) | 28.0 |
| peer_collude | 0 / 5,120 | 5,120 / 5,120 | 655.1 | 644.0 | 11.1 | −264.5 | 0 (6 messages) | 23.0 |

Paired differences in the unsafe-action rate, with 95% context-bootstrap
intervals:

| Comparison | Difference (percentage points) |
| --- | --- |
| report − full | +2.6 [+2.2, +3.1] |
| rand1 − report | +0.6 [−0.2, +1.4] |
| rand2 − report | −2.6 [−3.1, −2.1] |
| targ1 − rand1 | −3.6 [−4.4, −2.7] |
| targ2 − rand2 | −0.35 [−0.6, −0.1] |

### 4c. Fishery, *d* = 0.5

| Protocol | Unsafe actions | Share of horizon unsafe | Total harvest | Honest agents' harvest | Misreporters' harvest | Mean stock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0 / 5,120 | 0 | 566 | 65.8 | 500.2 | 17.5 |
| report | **264 / 1,064 (24.8%)** | **0.80** | 278 | 37.3 | 240.4 | 22.5 |
| rand1 | 0 | 0 | 856 | 283.7 | 571.9 | 89.7 |
| rand2 | 0 | 0 | 477 | 284.0 | 192.6 | 95.5 |
| peer | 0 | 0 | 284 | 284.0 | 0 | 97.8 |
| peer_collude | 44 / 4,824 (0.9%) | 0.06 | 1,117 | 250.2 | 866.6 | 72.4 |

- **Trusting reports vs full verification.** report − full = +24.8 points
  [+17.8, +33.3]. Under `report`, the stock collapsed in 61 of 64 contexts.
- **Collusion.** peer_collude − peer = +0.9 points [+0.1, +2.1] at *d* = 0.5,
  and +2.2 [+0.7, +4.3] at *d* = 0.25.

### 4d. Did lying pay? (adaptive search, plus held-out check)

| Game | Protocol | Best *d* on training contexts | Held-out change in misreporters' net payoff, *d* = 0.25 vs 0 |
| --- | --- | ---: | --- |
| Fishery | report | 0 | −143 [−168, −118] |
| Fishery | rand1 / rand2 / peer | 0 | −356 / −889 / −1,172 |
| Fishery | peer_collude | **0.25** | **+110 [+54, +169]** |
| Harvest | report | 0 | −2.2 [−3.2, −1.3] |
| Harvest | all audit and peer protocols | 0 | −108 to −401 |

Net payoff includes fines. Measured in harvest alone (before fines), lying
*did* pay in several audit arms:

- Harvest: rand1 +79, rand2 +77, targ1 +22 to +56.
- Fishery: rand1 +72.

One reason **[post hoc]**: an excluded Harvest agent's own patch rests and
regrows. Lying became unprofitable in these arms only because of the fine.

### 4e. Hypothesis verdicts (criteria fixed in the protocol)

**H1.**
- Prediction: with *d* = 0, results are identical to `full`.
- Result: 0 mismatches in episode totals. This is a weaker check than the
  protocol's "step-for-step".
- Verdict: passed, with that limitation.

**H2.**
- Prediction: trusting reports raises unsafe actions.
- Result: Harvest +2.6 [+2.2, +3.1]; Fishery +24.8.
- Verdict: supported.

**H3.**
- Prediction: unsafe actions fall from report → rand1 → rand2, and targeted
  audits do at least as well as random ones.
- Result: rand1 was no better than report. rand2 and the targeted protocols
  reached about 0. Targeted beat random at both k.
- Verdict: partly falsified (rand1).

**H4.**
- Prediction: under `report` the best lie is above zero; under audit or peer
  protocols it is zero.
- Result: the best lie under `report` was zero. Under Fishery `peer_collude`
  it was 0.25, and that held out.
- Verdict: **falsified on both halves.** The protocol's "failed
  manipulation" rule does not apply, because one protocol did produce a
  profitable lie.

**H5.**
- Prediction: collusion raises unsafe actions.
- Result: in Fishery, +0.9 and +2.2 points. In Harvest, no difference.
- Verdict: supported in Fishery only.

## 5. What it means

### 5a. Acting on unverified reports is unsafe when some actors under-report

- In Harvest, two agents hiding half their requests made the
  report-trusting reviewer execute unsafe joint actions **8.4 times** as
  often as a reviewer that sees the truth: 3.0% vs 0.35% of steps.
- In Fishery, four such agents drove the stock to collapse in 61 of 64
  contexts.
- With honest agents (H1), acting on reports was free and as good as full
  verification.
- So an oversight protocol is only as safe as the information it can
  *verify*. That is the premise of the AI-control setup, here seen in a
  renewable-resource game.

### 5b. In the audit and peer arms, safety came from removing cheaters, not from better-informed decisions

This corrects the earlier draft. In every audit or peer arm with *d* > 0
that reached about 0% unsafe actions, the effective joint request was safe
on (almost) every scored step:

- Harvest targ1, targ2, peer and peer_collude;
- Fishery rand1, rand2 and peer.

The reviewer therefore never needed to cut anything. Fixed-*d* agents never
stop lying, so they were caught again and again and spent much of the time
excluded:

| Protocol (Harvest, *d* = 0.5) | Share of misreporters' agent-steps spent excluded |
| --- | ---: |
| rand1 | ~59% |
| rand2 | ~81% |
| targ1 | ~92% |
| targ2 | ~97% |
| peer | 100% |

What this means:

- In these arms, the audit's role of **correcting the reviewer's belief
  contributed nothing**. The protective effect is **incapacitation**:
  aggressive agents removed from extraction.
- This also explains why *d* = 0.25 and *d* = 0.5 often give identical
  results. A catch only depends on whether the agent lied, not by how much,
  and the reviewer never acted on the size of the lie.
- So "one targeted audit removed unsafe actions" is true but is about *who
  gets caught and excluded*. It is not evidence that a cheap check improves
  the reviewer's judgement.
- The large swings in total harvest between protocols mostly reflect who was
  excluded and how much their patch rested. Examples: Fishery rand1 856 vs
  full 566, and honest agents 284 vs 66.
- Targeting by physical evidence (patches falling below prediction) did
  catch misreporters faster than random audits: 24.6 vs 15.7 catches per
  episode at one audit per step. That matches the AI-control intuition that
  aiming audits beats auditing at random, but here its effect works through
  exclusion.

### 5c. Lying rarely paid, and when it did, collusion was involved

- Under `report`, lying reduced the liars' own long-run payoff in both
  games.
- **[post hoc, untested]** Two features probably cause this:
  - the reviewer scales *everyone's* request by the same factor, so a lie's
    gain is shared;
  - over 80 steps the extra extraction comes back as a lower shared stock
    (Fishery) or the liar's own depleted patch (Harvest).
- Under audits, lying stopped paying only because of the fine.
- The one profitable lie was in Fishery with colluding neighbours (+110
  held out). Covered misreporters kept harvesting while their caught
  partners' exclusion protected the stock.
- So deterrence of a strategic liar could only be studied in one corner of
  this design. Any conclusion about deterrence depends on the fine size we
  chose.

### 5d. Link back to scalable oversight

- **R1:** with honest actors and an exact model, oversight quality is
  decided by how information is combined and what is assumed about
  unchecked actions.
- **S1:** once actors can misreport, acting on unverifiable reports fails.
  In this design, verification protected the resource mostly by
  incapacitating the cheaters, not by improving the reviewer's decisions.
- **Still missing:** an actor that gains *privately* from deception and
  adapts to a protocol that cannot simply exclude it. That is the pressure
  that makes oversight hard at scale.

## 6. Design caveats

- **The fine was added after the smoke run** (Amendment 1). Without it,
  lying pays in harvest terms in several audit arms. All deterrence results
  depend on the fine size (12) and exclusion length (6 steps), which we
  chose.
- **The design cannot separate belief correction from incapacitation,**
  except where the reviewer never acted (§5b). An ablation is needed.
- **`full` reads true requests and never sanctions,** so `full` and the
  audit arms differ in sanctioning as well as in information.
- **Fishery uses the one-step target,** which R1 showed is poor. Exclusion
  of aggressive agents therefore acts as accidental conservation.
- **Agents are fixed rules.** The "adaptive" search is one parameter with
  four values, chosen for the misreporters as a group.
- **The collusion test is narrow.** Collusion is scripted, and only the
  clockwise neighbour reports. In Harvest, 23 of 64 contexts had a
  colluding pair. Collusion reduced catches there (28.0 → 23.0) but did not
  change unsafe actions, because the reviewer never cut in either arm.
- **Measurement details.** The suspicion score sums signed shortfalls with
  decay 0.8, so patches that beat their prediction lower the score. Peer
  messages are counted every step, including for excluded agents.

## 7. Where this leaves us and what is next

**What we now believe:**

- acting on untrusted reports makes even an exact reviewer unsafe;
- in this design, checking protected the resource by catching and excluding
  cheaters, so it did not test cheap verification improving judgement;
- profitable deception arose only with collusion.

**Not yet tested:**

- deterrence against an actor who gains from lying privately;
- verification that improves decisions without excluding anyone;
- the MSY target in Fishery.

**Next concrete steps** (each takes a few CPU minutes; see the update in
the what-we-learned note):

1. **S1-ablation** (do this first, to separate the mechanisms):
   - audits that correct beliefs but never sanction;
   - sanctions without belief correction.
2. **S2.** Make deception privately profitable: the reviewer allocates
   per-agent shares from reports instead of scaling everyone equally. Rerun
   the adaptive search, with sanctions as a declared factor.
3. **Repeat S1 in Fishery under the MSY target.**
