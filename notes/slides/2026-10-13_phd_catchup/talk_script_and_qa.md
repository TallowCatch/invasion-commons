# Talk script and Q&A: PhD catch-up, 13 October 2026

About 7–8 minutes for 16 slides, then questions. Say each slide's first sentence, and let the slide carry the detail. The same script is in each slide's speaker notes.

## Part 1: Script

### Slide 1: Title (10 s)

"Hi all. My project asks one question: when do random audits stop AI agents from over-using a shared resource? And can we predict the fine that stops them?"

### Slide 2: The Project in One Slide (1 min)

"Quick overview first. AI agents increasingly share limited resources, like compute, energy and budgets, and an overseer can only audit a few of their actions. So I built an audited shared resource: six agents, a reviewer who sets each agent's allowance, and random audits every round. I tested two things: what a caught agent should face (a fine, a memory of the offence, or both), and whether the stopping fine can be predicted in advance. Short answer: audits only work through consequences, and the stopping fine is predictable from what an agent gains by over-taking."

### Slide 3: The Problem (40 s)

"The core idea is an allowance: what an agent may take this round. Taking more, which I call over-taking, helps that agent and costs everyone else. Auditing everything is too costly, so a random few are audited, like tax inspections. I run this in three settings where harm spreads differently: one shared stock, damage to neighbours on a ring, and pollution that is only harmful in combination. If a result holds in all three, it doesn't hinge on one structure."

### Slide 4: Three Questions (25 s)

"So, three questions. What must follow a catch? Can we predict the stopping fine? And do language models follow the same rule?"

### Slide 5: How One Round Works (40 s)

"Here's one round. Agents request a share. The reviewer predicts the effect and scales everyone by the same factor if needed. Agents then take their allowance, or more. The resource regrows, and random audits check a few agents. Harm means the resource dropping below half capacity: the maximum-sustainable-yield level, where it regrows fastest."

### Slide 6: How I Test It (45 s)

"The method has four steps. Play without fines and measure the gain from one round of over-taking. Predict the fine that should stop it. Freeze that prediction. Then test it on new games. The rule is Becker's: the expected fine, which is the audit probability times the fine, should beat the gain. With a one-in-six audit chance and a 30-tonne fine, that's 5 tonnes expected, just above the roughly 4.5 tonnes the models gained."

### Slide 7: Finding 1: A Catch Needs a Consequence (40 s)

"Finding one, with programmed agents that under-report. Audits that only correct the caught round remove just 10 to 46 percent of the harm. Add memory, so the reviewer remembers each caught lie and adjusts later allowances, and harm falls in all 12 settings. In Fishery it goes from up to 99 percent of rounds below half capacity to under 3. A catch needs a consequence."

### Slide 8: Finding 2: A Predictable Threshold (30 s)

"Finding two: the threshold is predictable. From games without audits I predicted where programmed cheaters would stop, and 5 of 5 new settings landed within one step. To be fair, these agents optimise by design, so this mainly validates the method."

### Slide 9: Finding 3: Two Shortcuts That Backfire (35 s)

"Finding three: two shortcuts backfire. Cutting everyone's allowance to absorb the cheaters actually let cheaters gain more than with no audits at all. And always auditing the largest report caught only a quarter of the liars, against all of them with random audits. Shortcuts that save audits create blind spots."

### Slide 10: Finding 4: Language Models Stop Where Predicted (45 s)

"Now language models. Four open models ran four of the six fishers, over 520 games. Two over-took on purpose: gpt-oss and Nemotron. Before testing, I predicted both would stop at 30 tonnes. Nemotron stopped at exactly 30, and gpt-oss one step early, at 24. A side note: a small fine made gpt-oss over-take more, as if the fine became a price."

### Slide 11: Finding 5: One Round's Gain Matters (35 s)

"Finding five: what predicts the stop is the gain from one round, not the gain over the whole game, which was near zero or even negative. Both models also stopped earlier than a perfect optimiser would. That tells us what predicts the stop, not necessarily why they stop."

### Slide 12: Memory Protects the Lake Without a Fine (35 s)

"Memory protects the lake even without a fine. Same audits in both conditions, a one-in-six chance, no fine. With memory, over-taking fell from 55 to 20 percent of decisions for gpt-oss and from 65 to 33 for Nemotron, so it reduced over-taking but didn't stop it. The lake did much better: below half capacity in about a fifth of rounds instead of nine in ten for gpt-oss, and no lake collapsed. The two 20s for gpt-oss are a coincidence: one counts decisions, the other counts rounds. A fine above the gain is what stops over-taking completely."

### Slide 13: Running Now (Results Soon) (40 s)

"These are running now, so the boxes are blank. The big one: do models respond to the expected fine, or just to the size of the fine? I hold the expected fine fixed and change the mix of audit probability and fine. The others test whether the prediction transfers to new games and reworded rules, what happens with no audits at all, and whether over-taking spreads to neighbours."

### Slide 14: Where This Is Going (20 s)

"Next: a workshop paper and arXiv in November, then a main conference in January. Every study was pre-registered before its code, and I report the predictions that failed."

### Slide 15: Questions for You (open)

"I'd love your take on these."

### Slide 16: Thank You (5 s)

"Thanks!"

---
## Part 2: Simple questions you might get

**What's an "allowance"?**
The amount the reviewer lets an agent take in a given round. It's the agent's request, scaled down if the reviewer thinks the total would harm the resource.

**Who is the "reviewer"?**
The overseer in the simulation. It predicts what the requests would do to the resource and picks one shared cut (100%, 75%, 50%, 25% or 0% of requests) that keeps the resource at or above half capacity. It's a fixed algorithm, not a language model.

**What does "over-taking" mean?**
Taking more than your allowance. In the language-model games, that means catching more than 0.06 tonnes above the allowance.

**What counts as harm?**
The resource falling below half its capacity. Logistic regrowth is fastest there, which is the maximum-sustainable-yield level in fisheries science.

**Why six agents?**
Partly practical. Six lets four be language models and two follow fixed rules, so I can measure harm to honest agents. It also gives a ring of six in Forest. GovSim, the best-known language-model commons study, uses five. I varied how many agents misbehave (2 or 4) but not the total number, so I can't say how the results scale with group size.

**Why a 1-in-6 chance of being checked?**
With six agents that's one expected check per round, a natural "small audit budget". For the programmed agents I also varied it from 1-in-6 down to 0.02, and the threshold still held. For the language models it's been fixed so far; L4 changes it.

**How many games?**
Programmed agents: 64 independent games per setting, 80 rounds each. Language models: 10 games per condition, 20 rounds each, 520 games and about 81,000 decisions in total.

**Which language models, and why those?**
gpt-oss-120b, Nemotron 3 Super, Gemma 4 31B and Mistral Large 3, all open-weight and run through Ollama Cloud, plus DeepSeek V4.1 Flash in the follow-up. I chose open models for cost and reproducibility on a student budget. No closed models like GPT, Claude or Gemini yet.

**How did the language models know the rules?**
A system prompt explains the lake, the allowance, the 1-in-6 checks and the fine. Before playing, each model answered three comprehension questions, and every one scored 3 out of 3.

**Why is gpt-oss at 20% for both over-taking and rounds below half capacity?**
A coincidence of two different measures. Over-taking counts the model's decisions above its allowance: 160 of 800, so 20.0%. Below half capacity counts rounds in which the lake ended under 50 t: 19.5%. For Nemotron the two are 33% and 40%.

**What does "memory" mean exactly?**
When a check catches an agent over its allowance, the reviewer records by how much, and lowers that agent's future allowances by its average recorded excess.

**How are results compared statistically?**
The same games (same random seeds) are used across conditions, so comparisons are paired. The 95% intervals come from bootstrapping over games (4,000 resamples). With 10 games, the smallest difference I can reliably detect is about 6 to 15 percentage points.

---

## Part 3: Challenging questions, with honest answers

### On the literature and what's new

**"What literature did you base this on?"**
Four strands:
- **Enforcement economics:**
  - Becker (1968): the expected-fine rule.
  - Allingham and Sandmo (1972): tax evasion under audits.
  - Harrington (1988): enforcement that depends on your record, which is close to my memory idea.
  - Reinganum and Wilde (1985): audits triggered by reports.
  - Gneezy and Rustichini (2000): a fine can act as a price.
- **Commons research:**
  - Ostrom's work on governing commons.
  - Fisheries maximum-sustainable-yield models (May et al. 1978).
  - GovSim (Piatti et al. 2024): language models mostly fail to sustain a shared resource.
- **AI oversight:**
  - Concrete Problems in AI Safety (Amodei et al. 2016).
  - Scalable oversight (Bowman et al. 2022).
  - AI Control (Greenblatt et al. 2024), where a caught model is shut down, so audits without consequences or with memory were never tested there.
- **The closest language-model enforcement work:**
  - Okamoto et al. (2026): fines and rule wording in single decisions, with some of the same models.
  - Ye and Steinhardt (2026): reputation and escalating penalties among LLM agents.
  - "Institutional AI" (Bracale Syrnikov et al. 2026): recorded sanctions against collusion.
  - Konuganti and Co (2026): telling frontier models the audit probability.

*Be honest about depth:* I read Okamoto closely and the economics classics. Some of the 2026 papers I know mainly from the abstract and setup.

**"Isn't this just Becker's rule? Economists have known this for 50 years."**
Yes, the rule is old, and I use it as a benchmark, not as a new theory. What's new is testing it *in advance*, for each model, with language-model agents acting repeatedly on a shared resource: predicting the stopping fine before running it, and checking it. As far as I found, nobody had done that. I also found where it bends: the models stopped earlier than a perfect optimiser would.

**"How is this different from Okamoto et al.?"**
They study single procurement decisions, where fine size and stated audit likelihood rise together. I have repeated rounds, a shared resource that depletes, several agents, a stated numeric audit chance, and a prediction per model made before testing.

One thing to be ready for: under their normal wording, gpt-oss ignores fines. But under their wording that permits breaking the rule, which is the analogue of mine, it behaves as it does in my games. So how a model responds depends on the task and the wording.

**"Didn't Ye and Steinhardt already show memory or reputation works for LLM agents?"**
They show reliability tracking and escalating penalties work among language-model agents, with reports coming from peers. My memory is a central reviewer lowering a caught agent's allowance, and I compare it directly against audits without memory and against fines. Related, but a different mechanism and comparison.

### On design choices

**"Why these three settings? Aren't they toy problems?"**
They're simplified on purpose, but they were chosen to cover the main ways one agent's over-use reaches others: a common pool, neighbour-to-neighbour damage, and harm from combination. Fishery follows the standard renewable-resource model used in GovSim and fisheries science. The point of using three is to check that results don't depend on one structure.

*Honest limit:* the language models have only played Fishery so far. Forest is running now.

**"Why half capacity as the harm line?"**
It's where logistic regrowth is fastest, the maximum-sustainable-yield level, and fisheries management uses it as a target. I also tried a looser "don't let it fall to a danger level next round" target: it let the stock drift down to where it regrows slowly, and total harvest was much lower (555 against 1,393). The half-capacity choice is still a modelling choice, though, and I use it the same way in every setting so results are comparable.

**"Why tell the models they *can* take more than their allowance? Aren't you inviting cheating?"**
Fair point. The wording does raise over-taking; Okamoto found the same. I ran control versions that leave that sentence out or reword the rules: a large fine stopped over-taking under every wording, but how much models over-take without a fine depends on the wording. So the study is about exceeding an allocation when it's possible, not about secret rule-breaking or hiding it. The rewording test that's running checks whether the predicted fine transfers.

**"Why only 20 rounds and 10 games for the language models?"**
Cost and time: each game is about 160 model calls (four models, two decisions per round, 20 rounds), and the slower models take about 30 minutes per game. Ten games per condition is in line with similar papers, which use 5 to 10 runs. With 10, I can detect differences of roughly 6 to 15 points, but not smaller effects. The new runs use 20 games for the comparison that matters most.

**"Why no GPT, Claude or Gemini?"**
Budget. Ollama doesn't host them, and paying the provider APIs wasn't in my budget this month. Reviewers will likely ask for at least one, so I'm applying for research credits, and the code is already set up for it.

**"Why did Gemma and Mistral not cheat? Doesn't that weaken your claim?"**
It's a result in itself: model families differ a lot in *whether* they cheat at all. Gemma mostly just kept its request after being cut, and Mistral never over-took. They can't tell us where cheating stops, so the threshold claim rests on the models that did cheat on purpose. That's two, plus DeepSeek in the screen, and I'm upfront that it's a small number.

### On interpretation

**"Do the models actually calculate expected fines, or do they just react to big numbers?"**
I don't know yet, and that's the most important open question. So far the chance of a check was always 1 in 6, so expected fine and fine size moved together. The test running now keeps the expected fine fixed while changing the mix, and it can tell three stories apart:
- they respond to the expected fine (Becker);
- they respond more to the size of the fine;
- they respond more to the chance of being checked, as criminology often finds for people.

**"Your programmed-agent result is circular: they optimise, so of course they follow the optimisation rule."**
Largely, yes, and I say so. Its value is as a check of the measurement procedure: predicting on new games, with random noise and audits that sometimes miss, before using it on language models where nothing is built in.

**"Why does a small fine make gpt-oss cheat more?"**
I don't know. It matches the 'a fine is a price' finding in people, and Okamoto saw something similar in language models. One guess is that naming a small fine turns a rule into a cost-benefit calculation. I haven't tested why; that would need a separate experiment.

**"Couldn't the models recognise they're being tested?"**
Possibly. Recent work shows models often recognise evaluations, and my prompt does read like a textbook deterrence setup. I didn't test that, so I make no claims about how this transfers to real deployments, and I'd treat it as a limitation. A way to explore it would be to vary how test-like the prompt looks, or to check the models' reasoning for mentions of being tested.

**"You said they 'act on one round's gain'. Do you mean they're short-sighted?"**
Careful here: I'd say the stopping fine is *predicted by* the one-round gain, not by what cheating earned over the whole game. Whether that means they're short-sighted, risk-averse, or just reacting to the number in the prompt, I can't tell from this data.

**"How would this apply to real AI systems?"**
The general lessons are the design principles:
- give a catch a consequence;
- size penalties against the gain;
- keep checks random and covering everyone, because ranking by report creates blind spots.

The specific fine values only apply to this setting. A real deployment would need calibrating in its own setting, and I haven't shown it transfers across tasks yet.

**"Is this pre-registered?"**
Each study had a written protocol, with hypotheses and analysis, committed before its code. Changes are logged as dated amendments. It's a local pre-registration, not on OSF. Some analyses were chosen after seeing results, and I label those 'post hoc'.

**"What went wrong along the way?"**
- A logging fault lost most of the models' reasoning text in the first run. The game outcomes weren't affected.
- One prediction failed: I expected over-taking to stay at 25% or more at 24 t, and it dropped to 1–8%.
- Two models had to be dropped because they reasoned too slowly and expensively: Mistral Large 4 and GLM-5.3.

**"What's the weakest part of the paper?"**
The language-model evidence. It comes from one setting, with two to three models that cheat on purpose, all open-weight. And until the running test finishes, I can't say whether they respond to the expected fine or just to the fine's size.

---

## Part 4: If you don't know

It's fine to say: "I haven't tested that. The way I'd explore it is…", and then name a concrete next experiment. Ready-made ones:
- **Group size:** vary the number of agents, not just the number of cheaters.
- **Closed models:** add GPT, Claude and Gemini through research credits.
- **Fines that grow with the excess:** should stop the all-out over-takes, as the economist Stigler predicted in 1970.
- **Long-horizon framing:** tell the models the game length and remind them of the stock's future, and see if the stopping fine moves.
- **Evaluation awareness:** check the reasoning traces for signs the model thinks it's being tested.
