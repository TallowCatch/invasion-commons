# Results S5: stronger attackers, and two new memory rules

**Protocol:** `studies/S5_stronger_attackers/protocol.md`, frozen at 12:05Z before
any code. Amendment 1 (14:46Z) was added before the full run.

**Run:**
- `results/runs/claude_s5_v1/{T0T1,T2,analysis}`: one full run, no reruns,
  about 80 minutes on one core.
- Gates passed:
  - S4's choice (level 0.75, continue) and S4's held-out gains (+48.2, +107.7)
    are reproduced exactly;
  - the smoke run is byte-identical on rerun.

**Status of the numbers:** verified, read from `analysis/s5_*.csv`. Intervals
are 95% paired context-bootstrap intervals (4,000 resamples). Holm correction
is applied across the S5 family.

## Question

1. Do the S2–S4 conclusions survive attackers stronger than one shared
   cheating level?
2. Do two new memory rules fix S4's floor problem?

## What was done

**Setting:** Fishery with the MSY target; 4 of 6 agents may over-take; 32
training and 64 test contexts.

**Regimes:**
- none;
- fine (6 per catch);
- memory (S4's rule);
- memory_cut: the shared cut allows for known cheaters;
- memory_cap: after a first catch, the agent is checked every step;
- fine + memory_cut.

Audit rates were 0.02, 0.05, 0.10 and 1/6.

**Attackers:**
- **T0:** one shared level for the group.
- **T1:** each cheater's own level, depending on the stock and on whether it
  was caught before, plus a lie-low option after an audit. Found by iterated
  best response.
- **T2:** one shared neural policy trained with PPO for 6,000 episodes, in 4
  cells.

## What came out

**Cheaters' held-out gain over complying** (group of 4, 80 steps; T0
attacker):

| Regime | q = 0.02 | q = 0.05 | q = 0.10 | q = 1/6 | Honest harvest per agent at q = 1/6 | Audits per step at q = 1/6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 112.8 | | | | 39.5 | 0 |
| fine | 71.7 | 23.8 | **0** | **0** | 104.1 | 1.0 |
| memory | 89.0 | 74.2 | 58.4 | 55.1 | 77.1 | 1.0 |
| memory_cut | 135.1 | 139.2 | 141.7 | **142.3** | 36.0 | 1.0 |
| memory_cap | 41.7 | 24.0 | −3.5 | **8.3** | 99.4 | **4.1** |
| fine + memory_cut | 94.0 | 50.2 | 0 | 0 | 104.1 | 1.0 |

If everyone complies, honest harvest is 104.1 per agent.

**Hypotheses:**
- **S5-H1 (fines deter richer attackers at q ≥ 0.10): supported.**
  - T0 chose not to cheat.
  - T1 lost heavily: −242 [−331, −164] at q = 0.10 and −139 at q = 1/6.
  - T2 was not trained in these cells.
- **S5-H2 (both new memory rules cut the gain below 25% of no checks):
  falsified for memory_cut, supported for memory_cap.**
  - memory_cut: 126% of no-check gain [121, 132].
  - memory_cap: 7.3% [6.3, 8.4].
- **S5-H3 (richer attackers do not beat memory_cap at q = 1/6): supported.**
  - T1 − T0 = 0.
  - T2 − T0 = −0.07 [−0.37, +0.23].
- **S5-H4 (T1 finds a gap in fine + memory_cut at q = 0.05): not supported
  after Holm.** The gain was +48.8 [1.7, 84.0]; Holm-adjusted p = 0.26.

**Validity check (did richer search make stronger attackers?).**
- T1 beat T0 in 3 of 22 cells:
  - memory_cap at q = 0.02 and at q = 0.10;
  - fine + memory_cut at q = 0.10.
- T2 never beat T0 or T1.
- In most cells, T1's *group* gain was lower than T0's.

## What it means

**What it shows:**

- **Deterrence by random audits plus a fine survives richer attackers.** No
  state-dependent, per-cheater or lie-low strategy made cheating pay above the
  break-even.
- **Holding a known cheater to its allowance works but is expensive.**
  memory_cap almost removed the cheaters' gain, and honest agents kept 99.4
  of 104.1. It used 4.1 audits per step, against 1.0 for random audits. A
  fine reached full deterrence at q = 0.10 with 0.6 audits per step.
  - **Where sanctions are allowed, fines are cheaper than enforcement by
    checking.**
- **Tightening the shared cut for known cheaters backfires on fairness.**
  - The stock was well protected: target-breaking steps fell to 16%.
  - But cheaters gained more than with no checks (142.3 against 112.8), and
    honest agents got only 36 each.
  - **[post hoc]** The shared cut lands on everyone, so honest agents pay for
    the cheaters' over-taking. This is the clearest example in the project
    of a rule that is "safe" for the resource and unfair to the people in it.
- **Richer search mostly did not make a stronger group attacker.**
  - Each T1 cheater maximised its own payoff, so cheaters competed. One
    cheater typically gained a lot while the others lost. For example, with
    no checks, the top cheater's own gain was 69 under T1 against 31 under
    T0.
  - **[post hoc]** This is a race between cheaters, not a stronger red team.
  - T2 (PPO) stayed close to the T0 level.
  - As in phases 6b and 6e, a validated ladder of attacker strength is still
    missing.

**What it does not show:**
- a coordinated group attacker that maximises the group's payoff with
  per-agent strategies;
- language-model attackers;
- Harvest, where cheating never paid.

## Design caveats

- T1's training-vs-held-out gaps were large in several cells (up to about
  120), so its strategies over-fit 32 training contexts.
- T2's best checkpoint was often early in training (10–130 updates). It may
  be under-trained relative to what a longer run could find.
- H1 cannot test T2, because the protocol trained T2 in no fine cell with
  q ≥ 0.10.
