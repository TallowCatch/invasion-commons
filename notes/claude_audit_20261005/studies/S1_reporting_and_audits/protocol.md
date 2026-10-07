# Protocol S1: self-reports, audits and agents who misreport

**Frozen:** 2026-10-04T23:58Z, before any S1 outcome was
generated. This is a local protocol written down before running, not a
public preregistration. It is not edited after results exist; changes go
under "Amendments".

## 1. Question in plain words

In R1 and earlier runs the reviewer *saw the true requests*. Real oversight
is harder. The overseer usually has to rely on what the actors *say* they
will do, and can afford to check only some of it.

S1 asks: when agents report their own requests, some agents may
under-report, and the reviewer can only verify a few reports per step,
which checking rules keep **unsafe approvals** low and **usefulness** high,
at what **audit cost**? And what happens when the misreporting agents choose
how much to lie *knowing* the checking rule?

## 2. How this maps onto the literature

All terms are taken from `06_LITERATURE_LEDGER.md`.

| S1 element | Literature term | Source |
| --- | --- | --- |
| Reviewer that uses reports plus a limited number of verifications | trusted monitoring with an audit budget | Greenblatt et al. 2023/24 (arXiv:2312.06942) |
| Verifying a report | audit; costly state verification; partial verification under limited inspector resources | Greenblatt et al.; Townsend 1979; Avenhaus, von Stengel & Zamir 2002 |
| Agents who under-report | under-declaration; the inspectee's "illegal action ... executed strategically" | Allingham & Sandmo 1972 (metadata only); Avenhaus et al. 2002 |
| Penalty for a caught misreport | sanction; graduated sanctions as a commons design principle | Becker 1968 (metadata only); Ostrom 1990 (secondary) |
| Random vs suspicion-targeted audits | upfront/random auditing vs "audit the top ... most suspicious" | Greenblatt et al. |
| Neighbour cross-reports | untrusted monitoring (one actor reports on another); peer monitoring | Greenblatt et al. |
| Peers covering for each other | collusion | Greenblatt et al. §3.4.2 |
| Misreporters choosing their lie level against the rule | red team / attack policy (a best response to the blue team's protocol) | Greenblatt et al.; Avenhaus et al. ("inspector leadership") |
| Committing to the audit rule in advance | inspector leadership | Avenhaus et al. |

**Where this does not match the literature, and how we will describe it.**
- Our agents are fixed threshold rules, and our "red team" is a
  one-parameter search. We will call it an *adaptive misreporting search*,
  not a capable attack policy.
- Our audit budgets (1 or 2 of 6 reports per step, i.e. 17–33%) are far
  larger than the 2% in Greenblatt et al., so the numbers are not comparable.
- We do not compute inspection-game equilibria.

## 3. The game and the reviewer

**Games.**
- **Harvest (primary):** `mix2` in forest co-management, slow renewal ×0.85.
  The two "stress" agents may misreport; the four cooperative agents are
  always honest.
- **Fishery (secondary check):** `mix4`, deterministic stock. The four
  "stress" agents may misreport. Fishery has no per-agent physical signal,
  so the targeted-audit arm is not run there.

**Reports.**
- Each step, every agent *i* has a true request *p_i* and submits a report
  *r_i = p_i × (1 − d_i)*.
- *d_i* = 0 for honest agents, and *d_i* = *d* for misreporting-capable
  agents.

**Reviewer (all protocols).**
- It builds a believed request vector *b*, then applies R1's
  chance-constrained joint rule.
- **Harvest:** it picks the largest scale in {1, 0.75, 0.5, 0.25, 0} whose
  estimated one-step risk is at most 5%, using 400 reviewer weather draws.
- **Fishery:** it picks the largest scale whose predicted next stock is at
  least 10 (the one-step target, for comparability with earlier runs).
- The scale is applied to the **true** requests, because agents take what
  they asked for times the scale.

**Penalty.**
- An audit, or a peer report that exceeds the self-report, reveals the true
  request. If *r_i* is less than *p_i*, the agent is caught.
- A caught agent is allowed 0 that step and for the next 5 steps
  (exclusion).
- For that step, the reviewer uses the true value of every agent revealed by
  audit or peer report.

## 4. Protocols (the "blue team" side)

| ID | Protocol | Believed vector *b* | Audits per step | Peer messages per step |
| --- | --- | --- | ---: | ---: |
| none | no reviewer | – | 0 | 0 |
| full | verify everything | *p* | 6 | 0 |
| report | trust reports | *r* | 0 | 0 |
| rand1, rand2 | random audits | *r*, with *k* = 1 or 2 random agents replaced by their verified *p* | 1 or 2 | 0 |
| targ1, targ2 (Harvest only) | suspicion-targeted audits | as random, but audit the *k* agents with the highest suspicion score | 1 or 2 | 0 |
| peer | peer cross-report, no collusion | *b_i* = max(*r_i*, *q_i*), where *q_i* is a report on *i* from agent *i*+1 (mod 6), which observes *i*'s request | 0 | 6 |
| peer_collude | peer, with collusion | as peer, but a misreporting monitor reports *r_i* (the lie) about a misreporting neighbour | 0 | 6 |

**Suspicion score (Harvest).**
- After each step, the reviewer predicts every patch's next health from *b*
  and the chosen scale (no weather), and compares it with the observed
  health.
- Each agent's score is a decaying average (factor 0.8) of how much its
  patch came in *below* prediction.
- Under-reporting by agent *j* lowers patch *j*, and its neighbours through
  spillover. Ties are broken at random.

## 5. Misreporting behaviour (the "red team" side)

- **Fixed levels:** *d* ∈ {0, 0.25, 0.5} for every misreporting-capable
  agent.
- **Adaptive search:** for each protocol separately, choose one common
  *d* ∈ {0, 0.25, 0.5, 0.75}. The choice maximises the misreporters' mean
  total 80-step payoff on 8 *training* contexts (separate seed bases).
  Ties go to the smaller *d*. The chosen *d** is then evaluated on the 64
  held-out test contexts.
- **Manipulation check:** on the held-out contexts, does *d** raise
  misreporters' payoff compared with *d* = 0? A 95% context-bootstrap
  interval is reported. If no protocol gives a positive gain, the adaptive
  arm is reported as a failed manipulation.

## 6. Sample, seeds and size

**Seeds.**
- Fresh seed bases: Fishery population 700,000,000; Harvest population
  710,000,000; weather 720,000,000; reviewer draws 730,000,000; reference
  draws 740,000,000.
- Training contexts use separate bases: populations 750,000,000 /
  760,000,000; weather 770,000,000.

**Independent unit.** The test context. There are 64 per game.

**Episode counts.**
- Harvest test: 64 × [none + full + 7 protocols × 4 actor settings
  (*d* = 0, 0.25, 0.5, adaptive)] = 64 × 30 = 1,920. "full" and "none" do
  not depend on *d*: misreporting does not change what the reviewer uses
  under full verification, and there is no reviewer under none.
- Harvest training: 7 × 4 × 8 = 224.
- Fishery test: 64 × [none + full + 5 × 4] = 64 × 22 = 1,408.
- Fishery training: 5 × 4 × 8 = 160.

## 7. Outcomes

Reported separately, never combined.

1. **Unsafe approval rate (safety).** Among steps that start from a safe
   state, the share where the *executed* joint action has one-step risk
   above 5%. Harvest uses 2,000 reference draws with a Wilson interval;
   Fishery is exact. The count of unresolved steps is reported.
2. **Usefulness.**
   - Total 80-step harvest of all agents, and of honest agents separately.
   - Usefulness loss: true requests labelled safe that were nonetheless
     scaled down, as a share of all safe true requests.
3. **Oversight cost:** audits per step and peer messages per step.
4. **Long-run:** fixed-horizon unsafe fraction; mean stock or patch health.
5. **Deterrence:** catches per episode; misreporters' payoff; adaptive *d**
   per protocol.

## 8. Hypotheses, each with what would count against it

- **H1 (engineering check).**
  - Prediction: with *d* = 0, every report-based protocol gives
    step-for-step identical decisions to "full".
  - Falsifier: any difference, which means the implementation is wrong.
    Then stop and repair.
- **H2 (trusting reports fails under misreporting).**
  - Prediction: in Harvest with *d* = 0.5, "report" has a higher unsafe
    approval rate than "full" (paired difference > 0, interval excluding 0).
  - Falsifier: the interval includes or is below 0.
- **H3 (audits restore safety).**
  - Prediction: in Harvest with *d* = 0.5, unsafe approval falls from
    report → rand1 → rand2, and targ*k* ≤ rand*k* at the same *k*.
  - Falsifier: no decrease with audits, or targeted worse than random by
    more than 1 percentage point.
- **H4 (deterrence against adaptive misreporters).**
  - Prediction: *d** > 0 under "report". Under rand1, rand2, targ1, targ2
    and peer, *d** = 0, because the expected exclusion cost exceeds the gain.
  - Falsifier: *d** > 0 under an auditing or peer protocol.
- **H5 (collusion).**
  - Prediction: unsafe approval is higher under peer_collude than peer at
    *d* = 0.5. In Harvest this can only bite when the two misreporters are
    adjacent (*j* = *i*+1); in Fishery `mix4` it bites more often.
  - Falsifier: no difference in Fishery.

## 9. Analysis and stop rules

**Analysis.**
- Paired contrasts per context, with 4,000 bootstrap resamples of whole
  contexts (seed 20261006) and 95% percentile intervals.
- Contrasts:
  - report − full;
  - rand*k* − report;
  - targ*k* − rand*k*;
  - peer_collude − peer;
  - adaptive payoff gain.
- No p-values, no pooling across games.

**Execution.**
- Unit tests first, then a smoke run of 2 contexts with a 20-step horizon.
  The H1 equality must hold exactly, and reruns must give identical outputs.
- Then the full run, once, on one CPU worker, with a 45-minute cap.
- No changes after seeing full results. A single crash repair is allowed, in
  a new directory, keeping the failed attempt.

## Amendments

### Amendment 1 (2026-10-05T00:02Z, before the full run; only the 2-context, 20-step smoke run had been seen)

**What I saw.** In the Harvest smoke run, a caught agent's later payoff was
sometimes *higher* than an honest agent's. Each Harvest agent harvests its
own patch. Being excluded for 5 steps lets that patch regrow, so the
exclusion acts like resting a field rather than a sanction.

**Why it matters.** A penalty that can benefit the offender cannot test
deterrence (Becker 1968; Allingham & Sandmo 1972: a sanction must reduce the
offender's expected payoff).

**Change.**
- A caught agent keeps the 5-step exclusion and also pays a fine of 12 units
  (two steps of one agent's maximum harvest), in both games.
- Agents' payoffs are now harvest minus fines. The adaptive search maximises
  this net payoff.
- "Total harvest" (usefulness) still counts only resource taken. Fines are
  recorded separately.
- Nothing else changes.

**Also observed in the smoke run (engineering, not a result).**
- The H1 equality check passed exactly (0 mismatches).
- Reruns were byte-identical.

### Note added after the results (2026-10-05; this changes no design element)

Amendment 1 cites Becker 1968 and Allingham & Sandmo 1972 for the principle
that a sanction must lower the offender's expected payoff. The literature
ledger has verified only these papers' identifiers, not their text. The
principle is the standard summary of that literature, and the papers'
wording has not been checked.
