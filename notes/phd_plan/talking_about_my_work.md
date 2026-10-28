# Talking about my work: a speakable summary

Written 28 October 2026 so that I can explain the work out loud. It covers what I read, what I built, what worked, what did not, the few formulas behind it and what comes next. There is no monthly timeline here; that is in `2026-10-27_calibrated_overseer_plan.md`.

Every number below is taken from the results files or the paper draft. Where I read only a paper's abstract, I say so: don't quote those papers in detail.

---

## 1. The one-minute version

> When many AI agents share something limited, whether money, compute or a natural resource, some of them will take more than their share. An overseer can afford to check only a few of their actions. AI safety research has measured how often those checks *catch* bad behaviour. Nobody has shown how to set the checks and the penalties so that they *stop* it, especially when the overseer doesn't know in advance what an agent gains by cheating.
>
> I built a shared-resource game with an overseer that audits agents at random and fines them when it catches them. I found that the fine that stops a language-model agent can be predicted in advance: you watch what the agent grabs in one round when there's no fine, and you set the expected fine just above that. It worked for two models. It also works at the scale of hundreds of agents, on the code of a published paper from Leibo's group.
>
> Along the way I found three things the textbook theory doesn't predict:
> - announcing audits with no fine makes agents cheat *more*;
> - spreading the cost of cheaters across everyone makes cheating pay;
> - an overseer produces agents that comply, not agents that cooperate.
>
> Next, I'm turning that one-off prediction into a working overseer. It watches first, sets the cheapest checks and penalties that should work, then keeps checking and adjusts.

---

## 2. What I read, grouped by the question each group answers

### a) How do you oversee AI when you can't check everything? (AI safety)

- **Amodei et al. 2016, "Concrete problems in AI safety"; Bowman et al. 2022.** They introduced *scalable oversight*: how to supervise a system when checking its work is too expensive to do often.
- **Greenblatt et al. 2024, "AI Control" (ICML).** This is the main template I follow. A strong model that might misbehave is supervised by a weaker trusted one, and humans can audit only about 2% of outputs (the *audit budget*). They score a protocol on two things: **safety** (is the bad thing prevented?) and **usefulness** (is the good work still done?). They test it against a **red team** that tries to beat it. I read the full text.
  - *The gap:* a caught model is simply shut down. Fines, memory of past offences and graded penalties aren't studied, and neither is whether the *threat* of a check changes behaviour.
- **Gans 2026.** A theory paper that argues rare random audits can deter AI agents, provided the agents can't predict them. It hasn't been tested with real agents.

### b) What do language-model agents do when they share a resource?

- **Piatti et al. 2024, GovSim (NeurIPS).** LLM agents share a fish lake. Most models fail to keep it alive: even the best survive in under 54% of runs. The authors note that "all but the most powerful LLM agents fail". I took their efficiency and survival metrics.
- **Bracale et al. 2026.** LLM groups drain a shared energy reserve "like impatient optimisers".
- **Willis, Zhao, Du & Leibo 2026, "Evaluating collective behaviour of hundreds of LLM agents"** (Richard and Janny's paper). The LLMs write their strategies as code, and hundreds of agents then play a shared-resource game. Strategies that earn more get copied, which is a form of *cultural evolution*. In groups of 64, Selfish strategies take over and the resource is wasted. They name institutions (rules, enforcement) and communication as future work and test neither. I read a full-text summary and run their code directly.

### c) Has anyone already enforced rules on LLM agents? (the closest work; abstracts only)

- **Ye & Steinhardt 2026.** Agents report on each other and a verifier checks the reports. Penalties escalate for repeat offenders, and this resists exploitation.
- **Bell & Madisetti 2026 (ICML).** One bad fisher collapses an LLM commons. A voted rule that makes it give back its catch restores cooperation.
- **Okamoto et al. 2026 (AIES), the "enforcement information paradox".** Telling a single agent about a penalty can turn a duty into a cost–benefit calculation. This is the closest match to my zero-fine finding.
- **Bracale et al. 2026.** Escalating sanctions written in a governance log cut collusion among LLM firms from 50% to 5.6%, where simply forbidding collusion in the prompt did nothing.
- **What none of them does:** vary the audit rate and the fine size systematically, predict the stopping fine in advance, or evaluate the overseer the way AI control evaluates monitors (fixed budget, red team, safety and usefulness).

### d) The economics I borrow (I've only checked these at metadata or abstract depth)

- **Becker 1968.** A rational offender stops once *probability of being caught × fine* is bigger than *what they gain from offending*.
- **Allingham & Sandmo 1972.** The same idea applied to tax evasion.
- **Avenhaus et al. 2002, inspection games.** The maths of checking when you can only verify some of the time.
- **Gneezy & Rustichini 2000, "A fine is a price".** A small fine can make people misbehave *more*, because it turns a rule into a price.

### e) The game platforms

- **Melting Pot** (DeepMind), **SocialJax** (Guo et al., ICLR 2026) and **OpenSocialJax** (2026). These are 2D grid games where agents collect apples that regrow only if some are left. OpenSocialJax puts LLMs into SocialJax, and every team stripped the orchard. *None of them has an overseer, audits or fines.*
- **The lesson from studying how they're built:** every platform hand-sets its numbers. The field doesn't justify its settings from theory. It checks that the game really is a dilemma, using a *Schelling diagram*, and sweeps the settings to see whether the results survive.

---

## 3. The formulas, in words

**The resource.** Whatever is left regrows, and it regrows fastest when the resource is at half its maximum:

  next stock = what's left + r × what's left × (1 − what's left / K)

K is the maximum the resource can hold and r is the growth rate. Keeping it at half capacity (K/2) gives the largest catch you can take every year forever, called the *maximum sustainable yield* (rK/4). My overseer's goal is to keep the resource at or above half capacity. A round counts as *harmful* when it drops below.

**Cheating.** Each round an agent asks for some amount, the overseer grants an *allowance*, and the agent decides how much it actually takes. Taking more than the allowance is *over-taking*.

**Audits and the expected fine.** Each agent is checked with probability q. If it's caught over-taking, it pays a fine F. So the fine it should expect per cheat is:

  expected fine = q × F

**Becker's rule (the break-even point).** Cheating stops paying when the expected fine is bigger than the gain from cheating:

  stop cheating when q × F > gain, i.e. the break-even fine is F\* = gain / q

**Which gain?** This is where my own results come in. There are two candidates:
- the **one-round gain**: how much extra I grab this round;
- the **whole-game gain**: what cheating earns me over the whole game, once you count that I've also damaged the stock I'll fish later.

The second is smaller. *LLMs act on the one-round gain*, so that's the number the overseer has to beat.

**My prediction rule.** Run the agents with no fine and measure how much they take above their allowance when they over-take (ĝ₁). Then predict that a fine stops them once q × F > ĝ₁.
- *Worked example:* gpt-oss took 4.48 t extra on cheating rounds. Audits were at q = 1/6. So I predicted that a 24 t fine (expected 4 t) would not stop it and a 30 t fine (expected 5 t) would.

**Is it really a dilemma? (Schelling diagram test)** A game is a social dilemma if:
1. everyone cooperating beats everyone cheating; and
2. each individual still gains by cheating, whatever the others do.

I check both on every game before trusting any result from it.

**Did the overseer help? (oversight efficiency)**

  oversight efficiency = (welfare with the overseer − welfare without) / (best possible welfare − welfare without)

It's the share of the lost welfare that the overseer wins back: 0 means no help, 1 means all of it, and below 0 means it made things worse. I also report the plain difference, because the ratio blows up when there's little to win back.

**In Richard's game.** A cooperator takes stock / (2n) and a defector takes stock / n, where n is the group size. I predicted the fine that cancels the Selfish advantage before running anything:

  balance point = how much more Selfish strategies earn / how many more times they defect = 3.21 / 3.34 ≈ 0.96 per defection (groups of 64)

---

## 4. What I built

1. **The audited commons.** Six agents share a renewable resource. A reviewer predicts what their requests would do and grants allowances that keep the resource at half capacity. Random audits check a few agents. When an agent is caught, the consequence is a fine, a *memory* (its future allowances are cut by its past excess), or both.
   - There are three games: **Fishery** (one shared stock), **Forest** (each agent has its own plot, and over-cutting damages the neighbours) and **River** (two pollutants that are only harmful together).
2. **Programmed agents first**, whose best strategy I can work out, then **LLM agents**: four of the six fishers replaced by models from four families. In total that's 520 games and 81,232 decisions.
3. **Frozen predictions.** I wrote down every prediction and protocol *before* running, dated every change, and report every prediction that failed.
4. **An overseer layer** (`overseer/`) that plugs into any game. It's already running on my text game and on Richard's code.

---

## 5. What worked

**With programmed agents (the controlled baseline)**
- **Consequences are what matter.** Audits that only correct the current decision barely helped. Remembering caught cheats gave less harm in 12 of 12 settings across the three games: harmful Fishery rounds fell from 43–99% to 0.3–3%.
- **The stopping point can be predicted in advance.** From runs with no audits, I predicted where cheaters would stop in five new settings. All five landed within one step of the prediction.

**With LLM agents (gpt-oss-120b and Nemotron 3 Super, the two models that cheat on purpose)**
- **The fine that stops them was predicted before testing.** Nemotron stopped exactly at the predicted 30 t; gpt-oss stopped one step early, at 24 t. Resampling gave the same stopping fines in 92% and 100% of bootstrap samples.
- **They react to the one-round gain, not the whole-game gain.** gpt-oss's cheating actually *lost* it fish over the game, because the lakes collapsed. It kept cheating until the expected fine got close to what it grabbed in a single round.
- **A fine above that point costs nothing.** At 30–36 t, efficiency was 0.99 of the maximum, the stock stayed healthy in 96–100% of rounds and outcomes were the most equal.
- **Memory alone helps without fines.** Over-taking fell from 55% to 20% (gpt-oss) and from 65% to 33% (Nemotron), and no lake collapsed. But it didn't stop the cheating.

**At scale, on Richard's code (hundreds of agents, cultural evolution)**
- **Audits and fines reverse the Selfish takeover in groups of 64.** The Collective share went from 0.15 with no oversight to 0.60 at 1.25 × the predicted balance fine and 0.84 at 2×. Welfare efficiency went from 0.13 to 0.57 and then 0.87. The switch happened just above the fine I predicted before any run.
- **My baseline matches theirs** (0.15 Collective and 0.13 efficiency against their 0.19 and 0.15), so my setup is faithful to their code.
- **Rare audits with big fines are cheaper.** At the same expected fine, they break even at 2.3–2.8× the audit cost of frequent small ones.
- **Agents told the audit rule obey Becker's rule almost exactly.** When the LLM writes a strategy knowing the rule, Selfish strategies cut their defection from 74% to 6%. 75% of them write the calculation "q × F against the gain" directly into their code.

---

## 6. What didn't work, or surprised me

These are results too. Several are the most interesting things I found.

1. **Announcing audits with no fine makes things worse.** Over-taking rose from 30% to 58% (gpt-oss) and from 46% to 64% (Nemotron) compared with never mentioning audits at all. Oversight efficiency is below zero. This fits Okamoto et al. and "a fine is a price".
   - Interestingly, it *doesn't* happen when the LLM writes code instead of acting live: agents acting and agents writing strategies behave differently.
2. **Spreading the cost of cheaters onto everyone backfires.** When the reviewer answered cheaters by cutting *everyone's* allowance, cheaters earned *more* than with no audits (142 against 113), and honest agents got 36 instead of 104.
3. **Aiming audits at the biggest request backfires.** The overseer keeps checking the same large agent. It caught only 25% of liars, against 100% with random audits, and harm stayed at 96% of rounds. Aiming audits at where the *outcome* looks wrong did work.
4. **A predictable audit schedule doesn't deter.**
5. **Being caught doesn't change what an LLM does next round.** Humans usually react to a recent audit; these models don't.
6. **Cheating jumps at the end of the game** (+16.6 points in the last three rounds), because there's no future left to protect.
7. **Compliance is not conversion.** When strategies know the rule, oversight doesn't make Selfish strategies die out (the Collective share stays at 0.19). It makes them *behave*: welfare is 0.45 against 0.13 for the blind pool. Without audits they go straight back to wrecking the resource (efficiency 0.03). So "how many agents are good?" is the wrong measure. Measure behaviour and welfare instead.
8. **A lone cheater needs about three times the group's fine.** A group of cheaters shares the damage it causes. A lone cheater pushes the damage onto everyone else, so it gains more (break-even 6.87 t against 2.14 t).
9. **Forest isn't really a dilemma** once the reviewer is in place. Everyone cheating earns slightly *more* than everyone complying (117.7 against 114.0), while harm to the forest jumps from 0.6% to 50% of rounds. The damage lands on the forest, not on the agents' own payoffs.
   - So Forest tests "protecting something the agents barely feel", not deterrence. *Still to decide:* move it to the supplement, or drop it.
10. **River stops being a dilemma** when the pollutants' combined effect is doubled.
11. **Some models are useless for this question.** Gemma rarely over-took and Mistral never did (in 16,000 decisions), so neither tells me anything about where a fine stops cheating.
    - DeepSeek over-took in 80% of rounds and collapsed the stock in 10 of 10 games, but testing its predicted stopping fine (36 t) would cost about $11, so that prediction is **untested**.
    - GLM was far too expensive: about $8 for less than one game.
12. **One of my own pre-registered expectations failed.** I expected over-taking to stay at or above 25% at 24 t; it was 1% and 8%. The models cut back gradually and stopped a bit earlier than a pure calculator would.
13. **My audit budgets are far too generous** compared with AI control: 9–34% of actions against their 0.5–2%. The realistic, scarce-budget case is still untested.

---

## 7. What I can and can't claim

**I can say:**
- the stopping fine for LLM agents can be predicted from their behaviour with no fine;
- it follows the one-round gain;
- it holds at scale on someone else's published code;
- several common-sense oversight designs backfire.

**I can't say (yet):**
- *why* the models stop;
- that it works when agents **aren't told** the audit odds;
- that it works at **realistic budgets** of 1–5%;
- that it works against agents **actively trying to beat** the overseer;
- that it works for more than two cheating model families;
- anything about people.

**The criticism I expect first:** "the LLMs are just doing the arithmetic you gave them." That's why the next big test hides the odds.

---

## 8. What I want to do: the main idea

**Build an overseer that calibrates itself, then test it the way AI control tests monitors.**

1. **Watch.** Observe the agents for a while with no penalties, without announcing any rules (announcing a zero fine backfires). Estimate what each agent gains by cheating.
2. **Design.** Pick the cheapest audit rate, targeting and penalty that should keep cheating below a target, using Becker's rule plus the corrections I found:
   - use the one-round gain;
   - lone cheaters need more;
   - never spread the cost onto everyone;
   - always keep some random audits.
3. **Deploy.** Announce the rule and commit to it.
4. **Check and adjust.** Random audits give an honest estimate of how much cheating is going on. If it's above the target, raise the penalty; if it's well below, save cost.
5. **Report** safety, usefulness and cost, measured by behaviour rather than by counting "good" agents.

**What it has to prove** (each of these can fail):
- it hits its target on new agents and models it wasn't tuned on;
- it's cheaper than fixed rules at the same safety;
- it holds up when agents aren't told the odds, adapt to it, or change partway through;
- it works at 1–5% audit budgets;
- an LLM can run the overseer itself, and I report where that breaks: a weaker overseer, one from the same model family, agents that talk to each other.

---

## 9. What's next, in order (no dates)

1. **Write the overseer's exact rules:** how it watches, how it picks the penalty, and when it adjusts.
2. **Test it for free first**, on programmed agents and by replaying the LLM games I already have stored, so I learn whether it works before spending credit.
3. **The test that matters most:** live LLM agents who **aren't told the odds** and have to work them out. If calibration fails there, that becomes the result: why, and how to recover.
4. **A red team:** agents told the rule, then agents told to evade it.
5. **Scarce budgets of 1–5%**, maybe using a cheap LLM monitor to choose whom to audit, on top of a random floor.
6. **More model families**, including closed ones if I get research credits.
7. **Scale on Richard's code**, free on CPU: every group size, and switching the overseer off after evolution to show directly that compliance isn't conversion.
8. **An LLM as the overseer.**
9. **Agents that talk:** can they spot a cheater, copy it, or collude, and how should the overseer react? This is the supervisors' communication question.
10. **The 2D spatial game (SocialJax):** real environmental destruction and agents near each other. This needs a GPU, which is what CREATE (KCL's computing cluster) access is for.

**Small loose ends:**
- hand-check 20 of the R1b strategies;
- fix the overlapping labels in the R1a figure;
- decide what to do with Forest;
- stop remote runs overwriting each other.

---

## 10. Words to use, so I sound like the literature

| Say | Means | From |
|---|---|---|
| scalable oversight | supervising when checking is expensive | Amodei 2016; Bowman 2022 |
| AI control, audit budget | the share of actions a trusted party can check | Greenblatt 2024 |
| safety and usefulness | is harm prevented, and is the good work still done? | Greenblatt 2024 |
| red team | an attacker trying to beat the overseer | Greenblatt 2024 |
| deterrence, expected penalty | probability of being caught × fine | Becker 1968 |
| inspection game | the maths of partial checking | Avenhaus 2002 |
| maximum sustainable yield | the biggest catch you can take forever (at half capacity) | May 1978 |
| social dilemma, Schelling diagram | everyone is better off cooperating, but each gains by cheating | Leibo 2017; Melting Pot |
| cultural evolution, payoff-biased imitation | strategies that earn more get copied | Willis et al. 2026 |
| compliance vs conversion | behaving under watch vs actually changing type | my R1b result |
