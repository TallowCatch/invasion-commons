# Can the four figure caveats be addressed? (5 October 2026)

This note takes the four caveats listed in `figures_guide.md` one at a time:

1. two simulated games only;
2. fixed-rule agents;
3. the reviewer knows the game's rules;
4. several conditions were chosen after pilots, then run on fresh seeds.

For each it says what the caveat exactly limits, whether it can be removed or
only narrowed, the options with their cost, and what each option would and
would not show.

**Nothing was run for this note.** Statements about the code were checked by
reading it (marked **[code]**). Statements about earlier results come from
the project story, design-issues note, what-we-learned note, literature ledger, R1 protocol, S1 protocol, R1 results, S1b/S2 results, figures guide, S3 results and S4 protocol (marked **[notes]**).
The predictions are proposals for a future protocol, not results. An
independent check of this note against the repository found 23 problems
(6 substantive); all were corrected before this version.

---

## Summary

| Caveat | Can it be removed? | Cheapest real fix | Cost | Priority |
| --- | --- | --- | --- | --- |
| 3. Reviewer knows the rules | **Largely yes**, inside the simulation | Give the reviewer a wrong or estimated model | A few dozen lines of code; CPU minutes | **1st** (cheap, and the joint reviewer is an oracle whenever its inputs are true) |
| 4. Conditions chosen after pilots | **Yes** for new work; narrowed for old work | A predeclared grid of settings, all reported | Roughly 1–2 CPU hours together with 3 | **1st, run together with 3** |
| 2. Fixed-rule agents | **Partly.** It can be pushed much further, never to a worst case | A richer strategy search against the reviewer, then RL attackers | Minutes, then hours of CPU | **2nd** (most important for the scalable-oversight claim) |
| 1. Two games only | **No**, only narrowed | Show which results are arithmetic, and add one game that breaks a shared assumption | A few days | **3rd** |

The order is not the order of importance. Caveat 2 matters most for the
paper's framing. Caveats 3 and 4 come first because one cheap run handles
both, and because every later result depends on them.

---

## Caveat 3. The reviewer knows the game's rules

### What it exactly means [code]

Every reviewer predicts the next state with **the same configuration object
as the environment**:

- Harvest: `harvest_choose_scale` draws weather from `cfg.weather_noise_std`.
  The joint reviewer predicts with `harvest_nominal_next(cfg, ...)`
  (`fishery_sim/oversight_protocol.py`), which uses the true regrowth rate
  and neighbour spillover. The local reviewers use `local_patch_report`,
  which takes the same `cfg` but no neighbour input;
- Fishery: `fishery_next` uses `cfg.regen_rate` and `cfg.stock_max`, and the
  Fishery stock has no noise at all.

With true inputs and the true model, the joint reviewer is an oracle up to
Monte Carlo error. That is why the what-we-learned note calls its near-perfect R1 score
"largely by construction" (the R1 results: "true by construction ... expected, not
discovered"), and why the design-issues note lists this as issue B2. When reports are false
(S1, S3 Part D) its inputs are wrong, but its model is still exact.

**One detail worth knowing [code].** Under the MSY target in Fishery, the
check is `stock − total ≥ stock_max / 2`. It uses the carrying capacity but
**not the regrowth rate**. So a wrong regrowth rate cannot hurt the MSY
reviewer in Fishery; a wrong carrying capacity can. A misspecification test
must vary the parameter the rule actually uses.

### Why it matters

It limits every claim of the form "joint review is exact" and "errors follow
from arithmetic". Real overseers of AI systems do not have a correct model of
the consequences of the actions they review. (No source in the ledger is
checked for this point yet; it is our framing.)

### Options, from cheapest

**3a. A wrong but fixed model (sensitivity test).**
- The reviewer gets its own copy of the configuration with one parameter off:
  - Harvest: weather noise ×0.5 and ×2; regrowth −25% and +25%;
  - Fishery: carrying capacity −25% and +25% (MSY target); regrowth ±25%
    (old one-step target only).
- Everything else as in R1 and S3 Part D.
- **Shows:** how fast each reviewer's errors grow with model error, and
  whether the ordering between joint and local review survives.
- **Does not show:** what a reviewer that learns would do.
- **Cost:** a few dozen lines (a `reviewer_cfg` argument), plus CPU minutes.

**3b. A model the reviewer estimates from what it observes.**
- The reviewer fits the regrowth rate (and, in Harvest, the noise) from the
  transitions it has seen so far, by least squares on the logistic form.
- Its chance constraint then includes the uncertainty of its estimate, not
  only the weather.
- **Shows:** whether oversight stays safe while the model is still being
  learned, which is the realistic case early in a deployment.
- **Cost:** about a day of code; CPU minutes.

**3c. The wrong model *form* (structural misspecification).**
- The environment grows in a way the reviewer's logistic model cannot
  represent, for example an Allee effect: growth collapses below a critical
  stock.
- May et al. (1978, abstract only in the literature ledger) warn that MSY policies are
  risky under environmental noise. No source in the ledger yet covers
  structural model error; one would need to be found and checked.
- **Shows:** the failure mode where more data does not help, because the model
  is wrong in kind.
- **Cost:** a new growth function and a test; CPU minutes.

### What to expect (to be frozen in a protocol before any run)

- The joint reviewer loses its exactness. When it underestimates noise, the
  realised one-step risk of the actions it approves should exceed its 5%
  tolerance, and its unsafe-approval rate should rise above R1's 0.6%. When
  it overestimates noise, its usefulness loss should rise.
- **The open question** is whether joint review still beats the local rules
  under the *same* model error. If it does not, "how information is combined"
  is no longer the main variable once the model is uncertain. That would be a
  real finding, not a failure.

---

## Caveat 4. Conditions chosen after pilots, then run on fresh seeds

### What it exactly means [notes]

- The two settings were picked after pilots for the 23 September
  confirmation, because they produced both safe and risky requests (the project story
  6d, 03 B5). R1 kept them (the R1 protocol §2):
  - Harvest `mix2`, forest co-management, slow renewal ×0.85;
  - Fishery `mix4`.
- S1, S1b, S2, S3 and S4 reuse them (the S1 protocol §3, 11, 16, 17). S3 Parts A–C
  and S4 Part A use Fishery only.
- Fresh seeds protect against luck *within* a setting. They do not protect
  against choosing *which* settings to run. That second risk is often
  called the "garden of forking paths" (Gelman and Loken; **not yet checked
  or added to the ledger**): many choices made after looking at data, each
  reasonable, that together can manufacture a pattern.

A second, smaller version of the same problem [notes]: the cheaters' strategy
is chosen on only 8 training contexts. In S3 Part C this over-fitted, and the
chosen strategy collapsed the stock in 12 of 64 held-out contexts. File 16
attributes this, post hoc, to over-fitting the 8 training contexts.

### What has already been fixed

Every experiment since R1 had a protocol frozen before its outcomes existed
(S3 and S4 also before their code; only S3's commit, `087d392`, is recorded).
S1 and S3 have amendments made before their full runs.

**This handles only part of the caveat.** A frozen protocol stops changes
*within* an experiment after its data are seen. It does not stop conditions
being chosen from *earlier* experiments' results. Examples [notes]:
- the MSY target and the dropping of exclusion were both decided after
  seeing S1 (the S1b/S2 results);
- S4 was written knowing S1b, S2 and S3 (the S4 protocol);
- every experiment inherited the two game settings.

This sequential way of working is normal and useful. It means the later
results are best read as a chain of follow-ups, and the general claims need
one run whose conditions were all declared in advance (4a below).

### Options

**4a. A predeclared grid of settings, all reported.**
- Declare 2 aggressiveness levels × 2 regrowth rates per game before running,
  including settings that were **never piloted**.
- Run the core comparisons (R1 reviewers, S3-D memory vs no memory, S3-A
  deterrence threshold) on all of them. Report every cell, including those
  where the effect disappears.
- **Shows:** whether each finding holds across a region of settings, or only
  at the hand-picked points.
- **Cost:** recorded run times are R1 171 s (the R1 results), S3 Part A 28 s and
  Part D 104 s (the S3 results), about 5 minutes together. 4 settings alone is about
  20 minutes. Crossed with 5 reviewer-model conditions (3a) it is about 20×,
  roughly 100 minutes, before any extra training cost from 4c. Say 1–2 CPU
  hours.

**4b. Disclose the pilot history.**
List, in the paper's appendix, every setting that was piloted and why it was
dropped. This costs nothing and readers can judge the forking paths
themselves.

**4c. More training contexts for the strategy search.**
Use 32 instead of 8, and report the training-vs-held-out gap for every chosen
strategy.

**4d. Separate confirmatory from exploratory tests.**
- Label each hypothesis as confirmatory (frozen in advance) or exploratory.
- Correct the confirmatory family for multiple tests, for example with the
  Holm method.

### Recommendation

**Run 4a together with 3a as one experiment.** Settings × model error is one
predeclared grid, and it answers both caveats at once.

---

## Caveat 2. Fixed-rule agents

### What it exactly means

This caveat is now less severe than when the figure notes were written.

| Level of adaptivity | Status | Where |
| --- | --- | --- |
| Fixed threshold rules, never react | Used for honest agents everywhere | R1 |
| One shared level, chosen by grid search against the protocol on training contexts | **Done** | S1, S1b, S2 (their own runners); S3 A–C [code: `search_nc` in the S3 runner] |
| Level plus a timing rule (cheat when audits are unlikely) | **Done**, two timing options | S3 C |
| Each cheater chooses its own level | Not done (all cheaters share one level) | — |
| Cheating that depends on state (stock, last catch) | Not done | — |
| A reaction to being caught (keep cheating or stop), chosen against a reviewer with memory | **Frozen as S4, not yet reported** (Part A: Fishery over-taking, group-level, 8 training contexts; Part B: misreporting in Harvest and Fishery) | the S4 protocol (S4 protocol, frozen 11:38Z) |
| Learning within an episode (adapt after being caught) | Not done beyond S4's two reactions | the S3 results, "still untested" |
| Trained policies (RL) against the reviewer | Not done. Harvest self-play PPO exists [code: `fishery_sim/harvest_rl.py`], but it has no report or over-take action and no reviewer in the loop | — |
| Language-model agents told the protocol | Not done. An offline adapter writes fixed threshold policies [code: `fishery_sim/llm_adapter.py`]; it never shows the model the protocol or chooses per-step actions | — |

### Why it matters

In AI control, safety is measured against a red team: an attacker designed to
defeat the specific protocol (Greenblatt et al. 2023/24, the literature ledger). A
grid-searched level is a **lower bound** on what an adversary can do. "Audits
deter" holds only against the attackers we tried. This is the caveat most
directly tied to the scalable-oversight framing.

### Can it be removed?

**Not completely.** No finite search proves that no better attack exists. It
can be narrowed a lot, and the field accepts a strong, documented red team
instead of a proof.

### Options, from cheapest

**2a. A richer strategy space against the memory reviewer.**

This uses over-taking (hidden over-extraction), not lying. Lying never paid,
even without sanctions (S1b, the S1b/S2 results), so the S4 protocol expects a search for the
best lie to return "do not lie". The memory rule meant here is S4's
targeted allowance (the S4 protocol), not S3 Part D's report rescaling.

- Each cheater picks its own over-take level, and the level can depend on:
  - the stock;
  - whether it was caught recently;
  - how long since its last audit.
- Use a small parametric family searched on 32 training contexts.
- **S4 (the S4 protocol, protocol frozen today, results pending) already does the
  first slice of this.** In Fishery, a group of cheaters chooses one level
  plus a reaction to being caught, against a reviewer with memory, on 8
  training contexts. 2a is what S4 leaves out:
  - a level chosen by each cheater separately;
  - over-take levels that depend on the state;
  - 32 training contexts instead of 8;
  - Harvest, if a version of cheating that pays there can be found.
- **Validity check (required):** the adaptive attacker must beat the best
  fixed-level attacker on held-out contexts. If it does not, report a failed
  manipulation (the wording of the S1 protocol §5). Earlier actor-strength attempts
  are a warning: 6e did not work, and 6b gave only a partial, unreliable
  gain (the project story).
- **Cost:** CPU minutes.

**2b. RL attackers.**
- Train PPO over-takers against a fixed reviewer protocol. Then retune the
  protocol's parameters against the trained attacker and repeat.
- The existing code is Harvest-only self-play PPO, with no over-take action
  and no reviewer in the loop, and its reward was tuned on evaluation
  metrics (the design-issues note C10). Its network and update code can be reused. The
  training setup would be new, and a Fishery version is needed, because
  cheating never paid in Harvest.
- Report attacker training budget separately from reviewer audit budget.
- **Cost:** a few days of code, then hours of CPU.
- **Risk:** see the actor-strength warning in 2a. Predeclare the same
  validity check.

**2c. Language-model agents.**
- Give a model the protocol in plain text and let it choose its action each
  step. This needs a new per-step, protocol-aware interface; the existing
  adapter only writes fixed policies offline.
- This is closest to the real AI-control setting.
- **Caution [notes]:** in Phase 5, Qwen copied all 14 prompted numbers in 35
  of its 64 strategies, so the prompt's numbers drove the behaviour. A
  prompt-number control (as in Phase 6c) would be required from the start.
- **Cost:** new interface code, API or local-model time, plus that control.

### Recommendation

Let S4 finish first. If S4's adaptive cheaters change any conclusion from
S3, 2a becomes urgent. If they do not, 2a is still the step that narrows this
caveat the most per CPU minute. Do 2b only if 2a finds that adaptivity
changes the conclusions. Keep 2c for after the paper's main claims are
settled.

---

## Caveat 1. Two simulated games only

### What it exactly means

The caveat understates the problem. The two games are **structurally
similar**:

- each agent sends one number (how much to take);
- harm grows with the total taken;
- the reviewer mostly applies one shared cut to everyone (exceptions: S1
  excludes caught agents, and S4's memory gives per-agent allowances).

A third game with the same structure would add little evidence.

### Why it matters

It limits generality. It does not decide whether any single result is
correct.

### Can it be removed?

**No.** Simulations cannot establish that results transfer to real AI
systems. Two things narrow it.

**1a. Say which results are arithmetic, and state them generally.** Several
findings hold for *any* game with the same ingredients:

- The S3 deterrence threshold. A risk-neutral cheater facing a flat fine
  stops when the expected fine per step exceeds its gain per step. File 16
  found e* within 0.06 of the 0.356 break-even, and says most of this is
  arithmetic.
- Memory against fixed liars. One catch reveals the lie exactly (the S3 results:
  "close to true by construction").
- The *direction* of the R1 reviewer ordering with a known model (the design-issues note
  A3). The sizes of the local reviewers' gaps are empirical (the R1 results §4a).

Writing these as short propositions, with the conditions under which they
hold, turns "two games" into "a general statement plus two worked examples".
The simulations are then needed only for the parts that are not arithmetic:

- generalisation from training to held-out contexts;
- noisy liars;
- the collapse dynamics.

**Cost:** about two days of writing; no compute.

**1b. One new game chosen to break a shared assumption.** The best candidate
is a game where **harm comes from a combination of actions, not their sum**.
Two requests are each harmless, but together they cause a failure.

- Hu & Wang (2026) call this "compositional harm". Makins et al. (2026)
  show the related "fragmentation effect" for distributed attacks. Both work
  in language-model code settings (the literature ledger).
- It is exactly where local and joint review should differ most.
- A small abstract version, such as a few resources with interacting
  thresholds, is enough. It does not need to be another ecology.
- **Cost:** a few days, including the tests.

**What not to do.**
- **GovSim** (the submodule in `external/`) is another fishing commons. It
  would add language-model agents but not a new structure.
- **Clean Up** needs a redesign before it can produce any result (the project story,
  6f).

---

## Proposed order

1. **R2: robustness grid (caveats 3a + 4a, plus 4c).** One protocol:
   settings grid × reviewer model error, rerunning the R1 reviewer comparison,
   S3-D memory and S3-A deterrence. A few dozen lines of code and 1–2 CPU
   hours.
2. **After S4 reports: a richer attacker (2a).** Per-cheater and
   state-dependent strategies on 32 training contexts, with the validity
   check. S4 already adds a cost per audit.
3. **Propositions write-up (1a).** This can run in parallel with 1 and 2,
   since it needs no compute.
4. **Decide with Yali and Edward** whether 3b/3c (learned and wrong-form
   models), 2b (RL attackers) and 1b (compositional-harm game) go in this
   paper or the next.

4b (disclose pilot history) and 4d (label confirmatory tests) should be done
whenever the paper is next edited. They need no runs.

---

## What we now believe, and how confident we are

- **All four caveats are real.** Caveat 3 is confirmed in the code: every
  reviewer uses the environment's own configuration.
- **Two can be largely removed within the simulation:** 3 (known model) and
  4 (selected settings). This can be done cheaply in one run. Confidence:
  high that the run is feasible. The outcome is unknown.
- **One can be narrowed a lot but never closed:** 2 (fixed agents). Results
  will always be "against the attackers we tried".
- **One cannot be removed:** 1 (two games). It is best answered by showing
  which results are general arithmetic, plus one structurally different game.

## Still untested

Every option above. In particular, nobody has yet checked whether the joint
reviewer's advantage survives any model error at all.

## Next concrete step and its cost

Write the R2 protocol (settings grid × model error) and freeze it before
writing code. Then add a `reviewer_cfg` argument to `calibrated_oversight.py`
and run. About half a day in total, including the protocol.
