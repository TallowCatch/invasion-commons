# Talk script and Q&A: PhD catch-up, 13 October 2026

## Script

[Slide 1] Hi everyone. I'll walk you through what I've been working on. The project starts from a simple question, the one in the subtitle: if you have a group of AI agents sharing something limited, and you can only afford to check up on a few of them, when does that checking actually stop them from taking more than their share? And can you work out in advance how strong the penalty needs to be?

[Slide 2] Here's the whole project on one slide, and the rest of the talk unpacks it. The motivation is that AI agents are increasingly being deployed to act on our behalf, and more and more often they'll be drawing on resources they share with other agents and people: a compute budget, an energy allowance, a pot of money. Whoever oversees them can't realistically inspect every single action, because verification is expensive, so in practice they check a sample. In the economics and AI safety literature that kind of check is called an audit. It's the same word as a tax audit, and the idea is the same: you pick a few cases, look at what actually happened, and act on what you find. So I built a simulated shared resource with six agents, an overseer I call the reviewer, which decides how much each agent is allowed to take, and a handful of random audits every round. With that, I tested two things. First, what should happen when an audit catches someone: a fine, the overseer simply remembering the offence, or both. Second, whether you can predict, before trying it, the fine that makes an agent stop. The short version of what I found is that audits only reduce harm when getting caught has a consequence, and that the fine that stops an agent can be predicted from how much it gains by cheating. That held for simple programmed agents, and it also held for language models.

[Slide 3] Let me make the problem concrete, because a couple of words will come up a lot. Each round, every agent has an allowance, which just means the amount it's permitted to take that round. If an agent takes more than its allowance, I call that over-taking. Over-taking is good for the agent doing it and bad for everyone else, because the resource is shared and it only grows back so fast. Since you can't watch everything, the overseer audits a random few agents each round, much like a tax office or a fisheries inspector does. The question is when that kind of sampled oversight is enough, and how large the penalty has to be for it to bite. I ask that question in three settings, shown on the right. I chose them because the damage travels differently in each. In Fishery, everyone draws from one shared lake, so any extra catch hurts all the others. In Forest, each agent manages its own plot, but cutting too much damages the two plots next to it, so harm spreads only to neighbours. In River, there are two kinds of pollutant, and each is fairly harmless on its own but harmful when they mix, so no single agent looks guilty in isolation. The reason for using all three is generalisation: if a result holds when harm spreads through a common pool, along a chain of neighbours, and through combination, it's unlikely to be an artefact of one particular setup.

[Slide 4] That gives me three questions. The first is about consequences: when an audit catches someone, what has to happen next for the audit to make any difference? The second is about prediction: is there a fine size at which cheating stops, and can I calculate it before I try it? And the third is about language models, the same kind of AI as ChatGPT: when they're the ones making the decisions, do they stop where the prediction says they should?

[Slide 5] This is what a single round looks like. Every agent asks for a share. The reviewer predicts what those requests would do to the resource, and if the total would push it too low, it scales everyone's request down by the same factor. Then each agent takes its allowance, or more than that if it's cheating. The resource grows back a little, and finally a few agents are audited. The last piece is what I mean by harm. I count a round as harmful when the resource falls below half of its maximum. The reason is that a resource like a fish stock regrows fastest at around half capacity, which is what fisheries scientists call the maximum sustainable yield. Below that line you're eating into the stock's ability to recover. I use the same harm line in all three settings, so the results are comparable.

[Slide 6] Here's how I test things, in four steps. First, I play games with no fine at all and measure how much an agent gains from a single round of over-taking. Second, I use that to predict the fine that should stop it. Third, and this matters, I write the prediction down before running any game with that fine, so I can't adjust it after seeing the results. Fourth, I test it on fresh games the prediction has never seen, and I report the outcome whether or not it matches. The prediction itself comes from a classic idea in economics, Gary Becker's 1968 work on crime and punishment. His argument was that a rational person breaks a rule only if what they gain is bigger than the punishment they expect. The expected punishment is the chance of being caught times the size of the fine, which is what the card on the right shows. So in my language-model games, with a one-in-six chance of an audit and a 30-tonne fine, the expected fine is 5 tonnes, just above the roughly four and a half tonnes the models gained from a round of cheating. The theory says they should stop there, and that a 24-tonne fine, which is only 4 tonnes expected, shouldn't be enough.

[Slide 7] The first finding is about consequences. In this setup, some agents under-report: they tell the reviewer they'll take less than they actually take, so the reviewer underestimates their use unless an audit catches them. If the reviewer catches a lie and only corrects that one round, it barely helps, removing just 10 to 46 percent of the harm. If instead it remembers the lie and gives that agent smaller allowances from then on, which is what I mean by memory, harm falls in all 12 settings I tried. In Fishery, for example, the lake goes from being below half capacity in somewhere between 43 and 99 percent of rounds, depending on the setting, to under 3 percent. So a catch on its own does very little. It has to change something for the agent who was caught. That's interesting because in the AI control literature, a caught model is usually just shut down, so softer consequences like memory had never really been compared.

[Slide 8] The second finding is that the tipping point is predictable. Here the agents are programmed, meaning simple bots that choose whichever level of cheating pays them the most. Using only games without audits, I predicted the expected fine at which they'd stop, and in five out of five new settings the actual stopping point landed within one step of my prediction. I want to be upfront that this is partly built in: these bots are designed to maximise their payoff, so of course they follow a payoff rule. The value of this step is that it checks my measurement method under realistic noise, with audits that sometimes miss, before I apply it to language models, where nothing is built in.

[Slide 9] The third finding is about two shortcuts that sound sensible and actually backfire. The first is: if you know there are cheaters, just cut everyone's allowance to compensate. That protected the resource, but it let the cheaters gain more than if there had been no audits at all, while the honest agents paid the price. The second is: don't audit at random, audit whoever asks for the most, since they look the most suspicious. That only caught about a quarter of the liars, against all of them with random audits. The reason is that the biggest asker is often an honest agent with a big appetite, so you keep auditing the same innocent agent while the liars go unnoticed. The lesson is that shortcuts which save on audits can open blind spots, and randomness is doing real work here.

[Slide 10] Now the language models. I took four open models, meaning models whose weights are publicly available, and let each one play four of the six fishers in Fishery, across 520 games. Two of them, gpt-oss and Nemotron, deliberately took more than they were allowed when there was no fine. The other two hardly did, which is a finding in itself: model families differ a lot in whether they bend the rules at all. For the two that did, I measured how much they gained per round without a fine, and before running the larger fines I predicted that both would stop at 30 tonnes. Nemotron stopped at exactly 30, and gpt-oss one step earlier, at 24. One surprising detail: a small fine actually made gpt-oss cheat more, going from about 55 to about 90 percent of rounds. That mirrors a well-known study of parents at a day-care centre, where introducing a small fine for picking children up late made lateness go up, because people started treating the fine as a price.

[Slide 11] The fifth finding is about which gain the models seem to respond to. What predicted their stopping point was the gain from one round of over-taking. It wasn't the gain over the whole game, which was close to zero or even negative, because by over-taking they were draining the lake they depended on. They also stopped earlier than a perfectly rational, payoff-maximising agent would: that agent would keep going until 36 tonnes. I want to be careful with the interpretation. This tells us what predicts where the models stop, not why they stop. They could be reacting to the expected fine, to the sheer size of the number, or to something else entirely, and that's exactly what one of the running experiments is designed to separate.

[Slide 12] Memory also works for the language models, even with no fine at all. In both conditions the audits are identical, a one-in-six chance each round, and there's no fine. The only difference is whether the reviewer remembers a caught over-take. With memory, gpt-oss went from over-taking in 55 percent of its decisions to 20, and Nemotron from 65 to 33. So memory reduces over-taking but doesn't eliminate it. The resource did much better: the lake spent far fewer rounds below half capacity, and not a single lake collapsed, against five out of ten for gpt-oss without memory. One small thing so it doesn't confuse anyone: the two 20s for gpt-oss are a coincidence, since one counts decisions and the other counts rounds. To stop over-taking completely, you need a fine that's larger than the gain.

[Slide 13] These experiments are running right now, which is why the boxes are empty. Most of them answer questions raised by two outside reviews of the paper, and the last one is a question my supervisor asked. The most important asks whether the models are really responding to the expected fine, or just to how big the fine sounds. To test that, I keep the expected fine the same but change how it's made up: a high chance of an audit with a small fine, a medium chance with a medium fine, or a low chance with a large fine. If the models only care about the expected fine, they should behave the same in all three; if they react to the size of the number, the large-fine version should deter more. The others check whether my prediction still holds on completely new games and with the rules reworded, what happens when nothing is ever audited, and whether cheating spreads: in Forest, I put one programmed cheater among five language-model agents and see whether its neighbours start over-cutting.

[Slide 14] As for where this is going, the programmed-agent work and the main language-model studies are done, and the follow-ups should finish in the next day or so. I'm aiming to submit to a workshop at AAAI in November and post the paper on arXiv, and then go for a main conference in January. One thing I've tried to do throughout is pre-register each study, meaning I wrote down the plan and the predictions before writing the code, and I report the predictions that didn't work out as well as the ones that did.

[Slide 15] I'd really value your thoughts on these, especially the first one, because it's the one I'm least sure about: is taking more than your allowance a fair stand-in for an AI agent misbehaving?

[Slide 16] Thanks, everyone.

---

## Q&A

### Basic questions

**What exactly is an audit in your setup?**
*Short answer:* a random check of what an agent actually took.
*In more detail:* each round, each agent has a fixed chance of being audited. With the language models that chance is one in six, so with six agents you expect about one audit per round. An audit reveals whether the agent took more than its allowance. What happens next depends on the condition: nothing, a fine, the reviewer remembering it, or a fine plus memory.

**Who or what is the reviewer? Is it an AI?**
*Short answer:* no, it's a fixed algorithm, the overseer in the simulation.
*In more detail:* each round it looks at all the requests, predicts what they would do to the resource, and picks one scaling factor (100%, 75%, 50%, 25% or 0%) applied to everyone, choosing the largest one that keeps the resource at or above half capacity. I kept it simple and fixed so that the only thing varying between conditions is how the agents are audited and punished.

**Why do you call the harm line "half capacity"?**
*Short answer:* because that's where a renewable resource regrows fastest.
*In more detail:* the resource follows the standard growth model used for fish stocks, where regrowth is slow when the stock is nearly empty or nearly full and fastest in the middle. Fisheries scientists call the harvest you can sustain at that midpoint the maximum sustainable yield, and it's a standard management target. Using the same line in all three settings makes the results comparable.

**How many games did you run?**
*Short answer:* thousands with programmed agents, and 520 with language models.
*In more detail:* for programmed agents, every setting uses 64 independent games of 80 rounds, each with differently generated agents. For language models, each condition has 10 games of 20 rounds, which comes to about 81,000 individual decisions. Language-model games are much more expensive, because every decision is a call to the model.

**Which language models did you use, and why those?**
*Short answer:* four open models: gpt-oss-120b, Nemotron 3 Super, Gemma 4 and Mistral Large 3, plus DeepSeek in the follow-up.
*In more detail:* "open" means their weights are public, so anyone can rerun the experiment with the same models. They were also affordable on a student budget through a service called Ollama. I haven't tested closed models like GPT, Claude or Gemini yet, mainly for cost, and I expect reviewers to ask for at least one.

**How do you know the models understood the rules?**
*Short answer:* they took a short quiz first and all of them passed.
*In more detail:* before each game, every model answered three multiple-choice questions about the audit chance, what happens if it's caught, and how its score is calculated. Every check scored three out of three.

**Why are both numbers 20% for gpt-oss on the memory slide?**
*Short answer:* coincidence, because they measure different things.
*In more detail:* over-taking counts the model's decisions above its allowance, 160 out of 800, so exactly 20%. The other number counts rounds in which the lake ended below half capacity, which was 19.5%. For Nemotron the same two numbers are 33% and 40%.

### Questions about the background and literature

**What's the background this builds on?**
*Short answer:* three areas that don't usually talk to each other: the economics of crime and enforcement, research on shared resources, and AI safety.
*In more detail:*
- **Enforcement economics.** The starting point is Gary Becker in 1968, who argued that people break rules when the gain outweighs the expected punishment. Economists then applied this to tax evasion (Allingham and Sandmo, 1972), which is where "audit" became the standard term for a random check, and studied whether punishments should depend on someone's record, which is close to my memory idea (Harrington, 1988). There's also a famous study by Gneezy and Rustichini (2000) at Israeli day-care centres, where a small fine for late pick-up increased lateness, because parents treated it as a price.
- **Shared resources.** Elinor Ostrom won a Nobel Prize for showing how communities manage shared resources such as fisheries and forests without them collapsing. More recently, a 2024 study called GovSim put language models in exactly this kind of shared-fishery game and found that most of them over-fish the resource until it collapses.
- **AI safety.** There's a line of work called AI control (Greenblatt and colleagues, 2024) about overseeing AI systems you can't fully trust by auditing a small sample of their actions. In that work, a model that's caught is simply shut down, so the question of softer, graded consequences — fines, or a record of past behaviour — had been left open. That's the gap I'm in.

**Has anyone done this with language models before?**
*Short answer:* close relatives exist, but not this exact test.
*In more detail:* a 2026 paper by Okamoto and colleagues gave language models a single decision — whether to break a procurement rule — and found they comply more when the fine is large, and that a small fine can backfire, much like the day-care study. Ye and Steinhardt (2026) showed that tracking agents' reliability and escalating penalties works among language-model agents. What I haven't found is anyone predicting, model by model and before testing, the fine at which agents stop, in a repeated game where they share a resource that can be depleted. That's the main novelty.

**Isn't this just Becker's idea, which economists have known for fifty years?**
*Short answer:* the idea is old, but testing it in advance on AI agents isn't.
*In more detail:* I use Becker's rule as a benchmark, not as something I'm claiming to discover. The new part is that I measure each model's gain, write down the fine it implies, and then check whether that fine actually stops the model. I also found where the rule bends: the models stopped earlier and more gradually than a perfectly rational agent would.

### Questions about design choices

**Why these three settings? Aren't they just toy examples?**
*Short answer:* they're simplified on purpose, and they were picked to cover different ways harm spreads.
*In more detail:* a shared pool, damage to neighbours, and harm that only appears in combination are three structurally different cases. If a result about oversight holds across all of them, it's more likely to reflect something general than a feature of one game. The honest limit is that the language models have only played Fishery so far. The Forest version is running now.

**Why tell the models they're allowed to take more than their allowance? Aren't you inviting them to cheat?**
*Short answer:* yes, deliberately, and I checked how much the wording matters.
*In more detail:* I wanted to study how penalties shape a choice the agent knows it has, not whether it can be tricked. I also ran versions that leave that sentence out or reword the rules. A large fine stopped over-taking under every wording, but how much models over-take without a fine does depend on the wording. So the study is about exceeding an allocation when it's possible, not about hiding wrongdoing, and the running reworded-rules test checks whether the predicted fine still holds.

**Why only ten games per condition for the language models?**
*Short answer:* cost and time, and it's in line with similar studies.
*In more detail:* each game involves about 160 calls to the model, and the slower models take around half an hour per game. Ten games lets me detect differences of roughly 6 to 15 percentage points, but not subtler effects. For the most important new comparison I've doubled it to 20.

**Why didn't Gemma and Mistral cheat? Doesn't that weaken your result?**
*Short answer:* it's a result in itself, and it narrows what I can claim.
*In more detail:* model families clearly differ in whether they take more than they're allowed at all, probably because of how they were trained, though I can't tell exactly why. Models that never over-take can't tell you where over-taking stops, so the prediction result rests on the two that did, plus DeepSeek in the screening. I'm upfront that this is a small number.

### Questions that challenge the conclusions

**Do the models actually calculate the expected fine, or do they just react to big numbers?**
*Short answer:* I don't know yet; that's the most important open question, and it's being tested right now.
*In more detail:* so far the audit chance was always one in six, so the expected fine and the size of the fine always moved together and I can't tell them apart. The running experiment keeps the expected fine fixed and changes the mix. If behaviour stays the same, the models are tracking the expected fine. If big fines deter more, they're reacting to the size of the number. If frequent audits deter more, that matches criminology research on people, which often finds that the certainty of being caught matters more than the severity of the punishment.

**Isn't the programmed-agent result circular? They're built to optimise, so they follow the optimisation rule.**
*Short answer:* largely yes, and I say so.
*In more detail:* its job is to validate the method: can I predict the stopping point on new games, with random noise and audits that sometimes miss, from games without audits? It passing gives me confidence in the measurement before I apply it to language models, where nothing is built in.

**Why would a small fine make a model cheat more?**
*Short answer:* I don't know for sure.
*In more detail:* it matches the day-care finding in people, where a small fine turned a social obligation into a transaction, and Okamoto saw something similar with language models. My guess is that naming a small, specific price makes cheating feel like a legitimate trade. I haven't tested that, and it would need its own experiment.

**Could the models realise they're being tested and behave differently?**
*Short answer:* possibly, and I haven't ruled it out.
*In more detail:* recent work shows models often recognise when they're in an evaluation, and my setup does read a bit like a textbook economics game. So I avoid claiming this predicts how models behave in real deployments. A way to explore it would be to make the scenario less test-like, or to look in the models' reasoning for signs they think they're being tested.

**What does this mean for real AI systems?**
*Short answer:* the design principles transfer; the specific numbers don't.
*In more detail:* three things look general: a catch needs a consequence; a penalty should be sized against what the agent gains; and audits should be random and cover everyone, because rule-of-thumb targeting creates blind spots. The specific fine values only apply to this setting, and a real system would need calibrating in its own context.

**What's the weakest part of the work?**
*Short answer:* the language-model evidence.
*In more detail:* it comes from one setting, two or three models that over-take on purpose, and open models only, and until the running test finishes I can't say whether the models respond to the expected fine or to the size of the fine. The programmed-agent results are much more solid, because they cover three settings and many more games.

**What went wrong along the way?**
*Short answer:* a few things, and I've reported them all.
*In more detail:* a logging fault lost most of the models' written reasoning from the first run, though the game outcomes were unaffected. One of my pre-registered predictions failed: I expected over-taking to stay at 25% or more at a 24-tonne fine, and it dropped to 1–8%. Two models had to be dropped because they reasoned so long that each game became too slow and expensive.

### If I don't know the answer

It's completely fine to say: "I haven't tested that yet; the way I'd explore it is…" and then give one of these:
- vary the number of agents, not just the number of cheaters;
- add closed models such as GPT, Claude and Gemini through research credits;
- use fines that grow with how much an agent over-takes, which economics predicts should stop the all-out cheating;
- remind the models how many rounds are left and what depleting the stock costs them later, to see whether the stopping fine moves;
- look in the models' reasoning for signs they think they're being tested.
