# Talk script and Q&A: PhD catch-up, 13 October 2026

## The 10-minute version (use this tomorrow)

**Deck:** `phd_update_10min_2026-10-13.pptx`. There are 10 main slides, about 7–8 minutes plus discussion, and 6 backup slides after "Thank You" for questions. Each slide's script is also in its speaker notes.

### Slide 1: Penalties Sized to the Gain (10 s)

"Hi everyone. My project asks one question: when do random spot-checks stop AI agents from over-using something they share?"

### Slide 2: The Project in One Slide (1 min 15)

"AI agents are starting to share resources like compute, energy and budgets, and whoever oversees them can only check a sample of what they do. I built a simulated shared resource: six agents draw on it over many rounds, a reviewer sets how much each may take, and a few agents are checked at random each round. I tested what should happen to an agent that gets caught taking too much, and whether the fine that stops it can be worked out in advance. Short answer: checks only help when a catch has a consequence, and the stopping fine can be predicted from how much an agent gains by cheating. That held for programmed agents and for language models."

### Slide 3: How One Round Works (1 min)

"One round: every agent asks for a share; the reviewer predicts the effect and scales everyone's request by the same factor if needed; each agent then takes its allowance, or more if it cheats; the resource regrows; and a few agents are checked. Harm means the resource falling below half its capacity, the level where it regrows fastest. I run this in three settings where harm spreads differently, one shared pool, neighbour damage on a ring, and pollution that is harmful in combination, so results don't rest on one structure."

### Slide 4: How I Test It (1 min)

"The method: run games with no fine, measure how much an agent gains from one round of over-taking, predict the fine that should stop it, write that prediction down, then test it on new games. The one formula is Becker's rule from 1968: the expected fine is the chance of a check times the fine, and over-taking should stop once that beats the gain. With a 1 in 6 chance and a 30 tonne fine the expected fine is 5 tonnes, just above the roughly 4.5 tonnes the models gained."

### Slide 5: Finding 1: A Catch Needs a Consequence (45 s)

"First finding, with programmed agents that under-report. If the reviewer only corrects the round in which it catches a lie, only 10 to 46 percent of the harm goes away. If it remembers each caught lie and adjusts that agent's later allowances, harm drops in all 12 settings, in Fishery from up to 99 percent of rounds below half capacity to under 3 percent. A catch needs a consequence."

### Slide 6: Finding 2: Language Models Stop Where Predicted (1 min)

"Second finding, with language models: four open models controlled four of the six fishers, 520 games. Two of them, gpt-oss and Nemotron, over-took on purpose. From their no-fine behaviour I predicted both would stop at 30 tonnes, before running those fines. Nemotron stopped exactly at 30 and gpt-oss one step early at 24. One surprise: a small fine made gpt-oss cheat more, as if the fine were a price."

### Slide 7: More Findings, in Brief (1 min)

"A few more results, briefly; the backup slides have each in full. The stopping fine for programmed agents was predicted in 5 of 5 new settings. Two shortcuts backfire: always checking the biggest request misses most liars, and cutting everyone's allowance rewards the cheaters. Memory works for language models too, cutting gpt-oss's over-taking from 55 to 20 percent without any fine. And what predicts where models stop is the gain of one round, not what cheating earns over the whole game; they also stopped earlier than a perfect optimiser would."

### Slide 8: Running Now (Results Soon) (1 min)

"These are running now, so the boxes are empty. The main one tests whether models respond to the expected fine or to the fine's size, by keeping the expected fine fixed and changing the mix of check chance and fine. The others check the prediction on new games and reworded rules, what happens with no checks at all, and whether cheating spreads to neighbours in Forest. Next: a workshop paper and arXiv in November, then a main conference in January."

### Slide 9: Questions for You (1–2 min (discussion))

"I'd love your views on these."

### Slide 10: Thank You (5 s)

"Thanks."

**If you run short of time:** drop slide 7 (More findings), and say "there are a few more results in the backup slides".

**Backup slides**, if someone asks: the problem and the settings; the predictable threshold; the two shortcuts; one round's gain; memory for language models; the timeline. Their longer explanations are in the full-length script below.

---

This is about 18–20 minutes at a relaxed pace, followed by questions. Read it aloud once tonight. You don't need to say it word for word: each slide's first sentence is the point, and the rest supports it.

---

## Part 1: Full-length script (16-slide version; use it for the backup slides or a longer talk)

### Slide 1: Title
"Hi everyone. I'll catch you up on what I've been working on. The title is *Penalties Sized to the Gain*, and the question underneath is really the whole project: when do random spot-checks stop AI agents from over-using something they share?"

### Slide 2: The project in one slide
"If you only remember one slide, make it this one.

**Why it matters:** AI agents are starting to act on things they share with other agents and people, like compute, energy and budgets. Whoever oversees them can't check every action, so they check a sample.

**What I built:** a simulated shared resource. Six agents draw on it over many rounds, a reviewer decides how much each one is allowed to take, and a few agents are checked at random each round.

**What I tested:** what should happen to an agent that gets caught taking too much — a fine, a memory of the offence, or both — and whether you can work out *in advance* how big the fine needs to be to stop it.

**What I found:** checks only help when getting caught has a consequence, and the fine that stops an agent can be predicted from how much it gains by cheating. That held for programmed agents and for language models.

The rest of the talk unpacks those four boxes."

### Slide 3: The problem
"Let me make the problem concrete. Each agent gets an **allowance**, the amount it's allowed to take this round. Taking more than that helps the agent and leaves less for everyone else.

Checking every action costs too much, so the overseer spot-checks a few, the way tax inspectors or fisheries inspectors do. The open question is when a sampled check actually stops over-use, and how big the penalty has to be.

I ask this in three settings, on the right. They aren't three toy problems; I picked them because harm travels differently in each:
- in **Fishery**, everyone draws from one shared stock;
- in **Forest**, each agent has its own plot, but over-cutting damages your two neighbours;
- in **River**, two kinds of pollution are mostly harmless alone and harmful together.

If a result holds in all three, it isn't an artefact of one particular structure."

### Slide 4: Three questions
"So there are three questions.
1. When a check catches someone, what has to follow for the check to matter: a fine, a memory, or nothing?
2. Is there a fine size where cheating stops, and can I calculate it *before* I try it?
3. When most of the agents are language models making their own choices, do they stop where my prediction says?"

### Slide 5: How one round works
"This is one round.
1. Every agent asks for a share.
2. The reviewer predicts what those requests would do to the resource. It then scales everyone's request down by the same factor, if needed, to keep the resource healthy.
3. Each agent then takes its allowance, or more if it's cheating.
4. The resource regrows.
5. A few agents are checked at random.

The one definition to keep in mind is **harm**: the resource falling below half of its capacity. That's the level where it regrows fastest, the 'maximum sustainable yield' level from fisheries management. I use the same harm line in all three settings."

### Slide 6: How I test it
"Here's the method, in four steps.
1. Run the setting.
2. Run games with no fine and measure how much an agent gains from one round of taking more than its allowance.
3. Use that to predict the fine that should stop it, and write the prediction down before running any game with that fine.
4. Test the predicted fine, and the fines around it, on new games, and report the result whether or not it matches.

The card on the right has the only formula in the talk:
- *q* is the chance of being checked;
- *F* is the fine;
- so *q* times *F* is the expected fine for one round of cheating;
- *g* is what you gain from that round.

The prediction, which goes back to the economist Gary Becker in 1968, is that cheating stops once the expected fine is bigger than the gain. For example, with a 1-in-6 chance of a check and a 30-tonne fine, the expected fine is 5 tonnes. A model that gains about 4.5 tonnes per round should stop there, and not at 24 tonnes, where the expected fine is only 4."

### Slide 7: Finding 1, a catch needs a consequence
"First finding. Here some agents report less than they actually intend to take, so the reviewer underestimates their use unless a check catches them.

If the reviewer only corrects the round in which it caught a lie, that removes just 10 to 46 percent of the harm. If it **remembers** each caught lie and adjusts that agent's future allowances, harm drops a lot in all 12 settings I tested. In Fishery, the share of rounds below half capacity fell from somewhere between 43 and 99 percent to under 3 percent.

The takeaway: a catch by itself barely helps. It needs a consequence."

### Slide 8: Finding 2, a predictable threshold
"Second finding. These programmed agents choose how much to over-take so as to earn the most. Using only games without any checks, I predicted the expected fine at which they'd stop. In five out of five new settings, the stopping point landed within one step of the prediction.

To be fair about it: these agents are built to maximise their payoff, so this is mostly a check that my measurement method works before I try it on language models."

### Slide 9: Finding 3, two shortcuts that backfire
"Third finding: two shortcuts that look sensible and backfire.

On the left, the overseer knows there are cheaters and lowers *everyone's* allowance to make up for them. Cheaters actually gained more than with no checks at all, 142 against 113, and honest agents paid for it.

On the right, the overseer always checks whoever asks for the most. Over a game that caught only 25 percent of the liars, against 100 percent with random checks, because the biggest requester is often honest, so the same honest agent keeps getting checked.

The lesson is that shortcuts which save checks can open blind spots."

### Slide 10: Finding 4, language models stop where predicted
"Now language models. Four open models each controlled four of the six fishers, with two rule-following fishers left in so I can see who gets hurt. That's 520 games in total.

The left panel shows how often each model took more than its allowance, at each fine size. Two models, gpt-oss and Nemotron, over-took on purpose. The other two barely did: Gemma rarely, and Mistral never.

Before testing, I predicted from their no-fine behaviour that both would stop at 30 tonnes. Nemotron stopped exactly at 30, and gpt-oss one step early, at 24. The right panel shows the rule-following fishers getting their catch back once the fine is large enough.

One interesting side result: a *small* fine made gpt-oss cheat **more**, from about 55 to about 90 percent of rounds. It's as if the fine became a price for cheating, which has also been seen in people."

### Slide 11: Finding 5, one round's gain matters
"Fifth finding: *which* gain predicts the stop. It was how much the model took above its allowance in a single round, about 4.5 to 4.7 tonnes.

It was *not* what over-taking earned over the whole game. Over the whole game their cheating gained little or even lost fish, because it depleted the lake, and yet they kept going.

They also stopped *earlier* than a perfect payoff-maximiser would. That agent would hold out until 36 tonnes; both models stopped before that, and gradually.

I want to be careful here: this tells us what *predicts* where they stop, not *why* they stop."

### Slide 12: Memory works for language models too
"And memory works for language models too. With checks and memory, but no fine at all, gpt-oss over-took in 20 percent of rounds instead of 55. Rounds below half capacity fell from 92 to 20 percent, and no lake collapsed, against 5 out of 10 without memory.

So memory reduced over-taking but didn't stop it. A big enough fine did stop it, and cost nothing in efficiency."

### Slide 13: Running now
"These are still running. The boxes are empty because the results aren't in yet. Each one answers a question an outside review raised.
1. **Do the models respond to the expected fine, or to the size of the fine?** I keep the expected fine the same but change how it's made up: a 1-in-3 chance of a check with a small fine, 1-in-6, or 1-in-12 with a large fine.
2. **Does the prediction hold on new games and with reworded rules?** The predictions are already written down: 30 tonnes for both models.
3. **What happens if nothing ever checks?**
4. **Does cheating spread?** In Forest, I put one programmed cheater among five language-model agents and see whether its neighbours start over-cutting, with and without being able to see each other's cuts.

I also screened a fifth model, DeepSeek. It over-took in 80 percent of rounds without a fine, so it's a third model that cheats on purpose."

### Slide 14: Where this is going
"The plan:
- The programmed-agent work and the first language-model studies are done.
- The follow-ups finish in the next day or so.
- I'm aiming for an AAAI workshop on 20 November, plus arXiv.
- Then a main conference in January, IJCAI or ICML. Those dates are estimates, because the calls aren't out yet.

Every study had its plan written down before the code, and I report the predictions that failed."

### Slide 15: Questions for you
"I'd really like your views on these. Is taking more than your allowance a fair stand-in for an AI agent misbehaving? If a fine works under one wording of the rules, how far would you trust it under another? Would more models or more kinds of setting convince you more? And where in your own work does someone check only a sample?"

### Slide 16: Thank you
"Thanks. I'm happy to share the draft."

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
