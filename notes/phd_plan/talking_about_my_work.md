# My work, explained simply

Rewritten 28 October 2026 so that it's easy to remember and say out loud. It's framed the way the multi-agent RL community frames it: SocialJax, Melting Pot, social dilemmas.

---

## 1. The story in one breath

> In games like **Harvest**, agents share an orchard. Apples only grow back if some are left nearby, so greedy agents strip the orchard and everyone loses. MARL has many fixes for this: agents that care about fairness, agents that punish each other, agents that pay each other, or a planner that taxes everyone. In all of these the fixer can see **everyone, all the time**. Real oversight can't. Watching costs something, so you can only check a few agents. **My question: can an overseer that can only check a few agents still stop the orchard being destroyed, and how should it decide who to check and how hard to punish?**

## 2. Three ideas to remember

1. **Watch first, then set the punishment.** If you watch agents for a while with no punishment, you can see how much they grab when they cheat. Make the punishment, on average, bigger than that grab and they stop. I predicted this in advance, and it worked.
2. **Some obvious designs backfire.** Announcing checks with no punishment, punishing everyone for a few cheaters, and always watching the biggest agent all make things worse.
3. **Watched agents behave; they don't change.** An overseer makes selfish agents act well while it's watching. Take it away and they go back to cheating.

---

## 3. What I read

### The games (where my work lives)

- **Leibo et al. 2017, sequential social dilemmas.** Social dilemmas that play out over time, like Gathering, studied with RL agents. This is the basic idea everything else builds on.
- **Perolat et al. 2017.** RL agents harvest a regrowing resource and can zap each other out of the game for a while. It gives the four standard scores: **efficiency** (total reward), **equality**, **sustainability** and **peace**.
- **Melting Pot** (Leibo et al. 2021; Agapiou et al. 2022). DeepMind's test suite. You test a group of agents against fixed "background" bots.
- **SocialJax** (Guo, Shi, Willis, Tomilin, Leibo & Du, ICLR 2026). The same kind of games rewritten in JAX so that training is fast. In Harvest, **a single defector can ruin it for everyone**, because it takes the last apple in a patch and that patch never regrows.
- **OpenSocialJax** (Guo, Fang, Leibo & Du, 2026). LLM agents play these games, and **no model keeps the orchard alive** for a full trial.
- *Note:* SocialJax and OpenSocialJax come from Yali's group, and Richard is a co-author of SocialJax. This is home ground.

### How MARL already tries to fix it (each fix changes something different)

| What it changes | Paper | Idea |
|---|---|---|
| What agents want | Hughes et al. 2018 | Agents that dislike unfairness cooperate more |
| What agents want | Jaques et al. 2019 | Agents rewarded for influencing others coordinate better |
| Agents punish each other | Vinitsky et al. 2023 | Agents learn the group's rules by seeing who gets punished |
| Agents punish each other | Köster et al. 2022 | Practising on a harmless "silly rule" makes agents better at enforcing the real ones |
| Agents pay each other | Yang et al. 2020 (LIO) | Agents learn to reward or charge each other |
| Agents pay each other | Christoffersen et al. 2023 | Binding contracts between agents fix the dilemma |
| A planner on top | Baumann et al. 2020 | A planner learns rewards and penalties; sometimes cooperation lasts after it leaves |
| A planner on top | Zheng et al. 2022 (AI Economist) | A planner learns taxes while the workers learn to respond |
| A planner on top | Gasztowtt et al. 2024 (Large Legislative Models) | **An LLM sets taxes for RL agents in Harvest** |

**The gap:** in the planner papers I read, the planner sees every agent's behaviour at every step. Nobody gives it a **limited number of checks**. Nobody asks whom to check, or whether learning agents figure out how to cheat where it isn't looking.

### LLM agents in groups

- **GovSim** (Piatti et al. 2024). LLM agents share a fish lake, and most models empty it.
- **Willis, Zhao, Du & Leibo 2026** (Richard and Janny's paper). Hundreds of LLM agents write their strategies as code, and the strategies that earn more get copied. In big groups, selfish strategies take over. They suggest rules and enforcement as the fix, but don't test it.

### Oversight when you can't check everything (AI safety)

- **Greenblatt et al. 2024, "AI Control".** Humans can check only about 2% of an AI's work.
  - A protocol is judged on two things: **safety** (does it stop the bad thing?) and **usefulness** (is the good work still done?).
  - It is also tested against a **red team** that tries to beat it.
  - I borrow this way of testing, applied to many agents instead of one.

---

## 4. What I've done so far: small games first

I built simple versions of these dilemmas, with no grid yet. Six agents share a resource that regrows, and the overseer gives each agent an allowance, checks a few at random, and punishes those it catches.

1. **Scripted agents first,** with rules I wrote. This is like Melting Pot's background bots.
2. **Then LLM agents:** four of the six replaced by language models, in 520 games.
3. **Then at scale, on Richard's code:** 64 agents per group, with strategies evolving over 2,000 generations.

---

## 5. What worked

- **Punishment has to be real.** Checks that only correct the current decision barely helped. Remembering who cheated and cutting their future share cut the damage in **12 of 12** settings.
- **I predicted the punishment that stops LLM agents, before testing it.** I watched each model with no penalty and measured how much it grabbed when it cheated. Then I set the average penalty just above that.
  - **Nemotron stopped exactly at the predicted level, 30.** gpt-oss stopped one step earlier, at 24.
- **LLMs care about this round, not the future.** One model's cheating actually lost it fish over the whole game, because the lake collapsed, and it kept cheating anyway.
- **It works with 64 agents on Richard's code.** Welfare rose from **0.13 to 0.87**, and selfish strategies stopped taking over. Rare checks with big penalties were cheaper than frequent small ones.

## 6. What didn't work, or surprised me

- **Saying "we check" with no penalty made cheating worse:** **58%** of rounds against **30%** when checks were never mentioned.
- **Punishing everyone for a few cheaters backfired.** Cheaters earned more than with no oversight at all, and honest agents lost out.
- **Always watching the biggest agent backfired.** It caught only **25%** of liars, against **100%** with random checks.
- **Watched agents behave; they don't change.** With oversight, the selfish strategies **survived** (only 19% of strategies were cooperative) but behaved well enough to triple welfare (0.45 against 0.13 without oversight). Remove the overseer and they wreck it again.
- **Getting caught didn't change what an LLM did next round.**
- **Some models never cheat,** so they can't tell me anything: Gemma barely did and Mistral never did. Some were too expensive to test properly (GLM, and DeepSeek's follow-up).
- **My Forest game turned out not to be a real dilemma.** The damage hit the forest, not the agents' own scores.

## 7. Five numbers to remember

| Number | What it means |
|---|---|
| **30** | the predicted stopping penalty; Nemotron stopped exactly there |
| **58% vs 30%** | cheating when checks are announced with no penalty, vs never mentioned |
| **25% vs 100%** | liars caught by watching the biggest agent, vs random checks |
| **0.13 → 0.87** | welfare with 64 agents, without vs with the overseer |
| **19% vs 0.45** | with oversight, only 19% of strategies are cooperative, but welfare still reaches 0.45: they behave without changing |

## 8. What I can't claim yet

- that it works in a **real grid game** like Harvest;
- that it works against **RL agents that learn** to dodge the overseer;
- that it works with **very few checks**, around 1–5% (mine were around 10–30%);
- that it works when agents **aren't told** how often they're checked.

The gameplan below targets exactly these.

---

## 9. The gameplan

**Goal:** an overseer that can only check a few agents. It decides whom to check and how hard to punish, and it keeps the orchard alive. It's tested in SocialJax against RL agents and LLM agents.

| Step | What I do | What I need | Done when |
|---|---|---|---|
| **1. Put the overseer into SocialJax Harvest** | Add it as a wrapper around the game. Each step it can check k agents; "cheating" means, for example, taking an apple from a patch that's nearly empty. Test with scripted bots, and check the game is still a dilemma. | Laptop or CPU, free | The wrapper runs, and the scripted cheaters stop at the predicted punishment |
| **2. Train RL agents against it** | Use SocialJax's own training code (IPPO). Compare: no overseer; an overseer that sees everything; an overseer with only a few checks; agents zapping each other. Score apples left, total reward, equality and time spent punished. | Small map on CPU first, then full runs on CREATE | I know how few checks still save the orchard |
| **3. Make it set itself up** | It watches with no punishment first, sets the penalty, then keeps checking and adjusts. Compare it with fixed rules. | Same | It hits its target with fewer checks than fixed rules |
| **4. Let agents try to beat it** | RL agents will naturally learn to cheat where it isn't looking, which makes them a ready-made red team. Compare random checks with targeted ones. | CREATE | I know which way of checking survives learning cheaters |
| **5. LLM agents and an LLM overseer** | Run the same overseer in OpenSocialJax with LLM players, then let an LLM run the overseer itself. | Ollama credit; keep it small | It works for LLM players too, or I know where it breaks |
| **6. Talking and copying** | Let agents see or message each other. Does cheating spread, and can the overseer react? | Credit and CREATE | This answers my supervisors' communication question |

**Where my earlier work fits:** steps 1–3 use the rules I've already learned: watch first, use random checks, punish the cheater rather than everyone, and measure behaviour rather than counting "good" agents. The small games and Richard's game are the evidence that these rules work.

**What I can start right now, for free:** step 1.
