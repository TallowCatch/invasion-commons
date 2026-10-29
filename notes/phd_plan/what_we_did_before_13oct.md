# What we did before the 13 October meeting, and a critical look at it

Written 28 October 2026. Sources:
- `PROJECT_HISTORY.md`;
- `notes/research_review/` (the September audits and feedback);
- `notes/claude_audit_20261005/` (the October studies, the literature ledger and the novelty notes);
- the talk script in `notes/slides/2026-10-13_phd_catchup/`.

The work falls into two acts:
- **Act 1 (Feb–Sep)** was governance in commons games. It was MARL-shaped: Fishery, Harvest, evolving strategies, PPO and a SocialJax adapter.
- **Act 2 (5–13 Oct)** was audits and fines, with programmed agents and then LLM agents. That is where the economics came in.

---

## Act 1: February to September, governing a commons

| When | What we did | What we were reading | Why (the basis) |
|---|---|---|---|
| Feb | Built Fishery: one fish stock that regrows, agents asking to harvest, and collapse. Strategies that did badly were replaced by new ones, with mutation, so better strategies could **invade**; hence the repo name. Added quotas, monitoring and sanctions. | Ostrom 1990 (communities governing commons); Leibo et al. 2017 (sequential social dilemmas); GovSim (Piatti et al. 2024) | Start where individual gain damages a shared stock and failure can be measured. A fixed, friendly population is too easy, so let strategies change and seek advantage. |
| Mar | Added **Harvest**: patches, neighbours affecting each other, messages and payments. Compared local, system-wide and combined interventions, and added a PPO (RL) check. | Perolat et al. 2017; Melting Pot; SocialJax | Fishery has one stock, so it can't show the gap between what one agent sees locally and what happens across the whole map. |
| Apr | Added real-world limits for the monitor: missed detections, delays, limited targeting and the cost of intervening. Built LLM strategy-bank tooling. | Scalable oversight (Bowman 2022; Engels 2025); safe RL "shielding" (Elsayed-Aly 2021) | Yali asked what the monitor can actually see and do. Edward warned against treating central control as the universal answer. |
| Jun | **Stage A:** local checks against system-wide checks, and the "each request looks fine, but together they break the system" case. Ranked architectures on an actor-versus-overseer "capability gap" axis. | Makins et al. 2026 (multi-agent monitoring); Hu & Wang 2026 | To give the oversight idea measurable actors, monitors and failures. |
| Jul–Aug | Small local LLMs (Qwen 3B, Llama 3B) wrote Harvest strategies, which were then tested in the same pipeline. | Willis, Du & Leibo 2025; Sudhir et al. 2025 | To test whether model-written strategies could enter the benchmark. |
| By 13 Sep | Audited Stage A's story. | (same) | The group's questions at the meeting: how is "capability" measured, and why do the curves fall? |
| 21–23 Sep | **A fair reviewer test:** local and joint reviewers given the same target and the same powers, differing only in what they see. A budgeted reviewer checks 0, 3 or 6 requests. Then a fresh-seed confirmation: 1,920 episodes and 101,241 reviewer decisions. | AI control's audit budget (Greenblatt et al. 2024 entered here) | Older comparisons changed what the reviewer saw *and* what it could do, so neither effect could be isolated. |
| 22–23 Sep | Built a **SocialJax Clean Up** adapter and tried to admit policies into it. | SocialJax | To get a third game with a different mechanism (cleaning a river so others can harvest). |
| 23 Sep | **Attacker-strength pilot:** more search to build stronger cheaters. | — | To link "stronger actor" to reviewer failure. |

### What Act 1 established
- Each request can look fine while the combination breaks the system. **But this isn't new:** the September closeout itself says so, citing Makins et al. and Hu & Wang.
- In the fair test, joint review spared safe requests compared with a cautious local rule. This only held in two pilot-chosen settings, and better immediate decisions didn't improve the long-run stock.
- The Clean Up adapter exists, **but Clean Up never produced a usable test.** The chosen policies didn't create both safe and risky situations for the reviewer to judge.

### Critical assessment of Act 1
- **The question kept moving.** Governance, then local vs global, then the capability gap, then the information test. Many things changed at once, so few results isolate one cause.
- **The "capability gap" axis was invalid.** It subtracted one rank from another, combining unrelated settings. It was withdrawn.
- **A hidden bug overturned the headline.** On 5 October we found the Harvest weather buffer was 3.6× too large. Corrected, joint review cut **0 of 3,687** safe requests, not 2,521, so the 23 September "confirmation" was mostly that bug.
- **"Stronger attackers" never worked.** More search or PPO didn't give a reliable ladder of attacker strength.
- **What was good:** it *was* in the MARL domain (Harvest, evolving strategies, PPO, SocialJax). Yali's two questions, "what problem are we solving?" and "who are the players?", were the right ones, and they're still the right ones now.

---

## Act 2: 5–13 October, audits, fines and LLM agents

### The reading this rested on (the 5 October literature ledger, then the novelty notes)

| Source | Read at what depth | Used for |
|---|---|---|
| Amodei et al. 2016; Bowman et al. 2022 | full text of the relevant sections | the name and definition of scalable oversight |
| Greenblatt et al. 2024 (AI Control) | full text, sections 2–3 | audit budgets, safety vs usefulness, the red team |
| Avenhaus et al. 2002 (inspection games) | abstract | the maths of partial checking |
| Becker 1968; Allingham & Sandmo 1972 | metadata only | the break-even rule: stop when the expected fine exceeds the gain |
| May et al. 1978 (maximum sustainable yield) | — | harm = the stock falling below half capacity |
| Okamoto et al. 2026; Ye & Steinhardt 2026; Bracale et al. 2026; Gneezy & Rustichini 2000 ("a fine is a price"); Reinganum 1985; Ensign et al. 2018 (runaway feedback in predictive policing) | mostly abstracts | positioning, and explaining surprises |

**The key point:** the backbone of Act 2 was **Becker plus AI control**. The talk itself says "the prediction comes from a classic idea in economics, Gary Becker's work". This is where the economics lean started.

### What we did, in order

| Date | Study | What it asked | What came out |
|---|---|---|---|
| 5 Oct | **Audit** | Re-check the September confirmation | Found the weather-buffer bug (above) |
| 5 Oct | **R1** | A fair reviewer comparison, and which safety target? | Joint review is near-exact with true inputs. Keeping the stock at half capacity gave a total harvest of **1,393 against 555** under the old one-step safety line. |
| 5 Oct | **S1, S1b** | What if agents lie about their requests? | Lying broke the reviewer: the Fishery stock collapsed in 61 of 64 games. Audits helped **only by removing the cheater**, not by improving the reviewer's decisions. |
| 5 Oct | **S2, S3** | Do checks deter? Where does deterrence start? | Cheaters stopped at an expected fine of **0.30–0.40**, against a predicted 0.356. A **known** audit schedule was worse than no audits. **Memory** (cutting a caught cheater's future share) cut harm sharply. |
| 5 Oct | **S4, S5** | Memory against smart cheaters; stronger attackers | Memory alone doesn't deter, but memory plus a fine needs half the audits. **Tightening everyone's allowance** let cheaters gain more than with no audits (**142 vs 113**). |
| 5 Oct | **R2** | Do the findings hold in new settings, and with a wrong model? | Memory was safer in 9 of 9 settings. A reviewer with a wrong model approved **26–82%** of risky requests. |
| 5 Oct | **C1** | A third game, River, where two pollutants are harmful together | The reviewer ranking held. Aiming audits at the largest report did worse than random. |
| 6 Oct | **L1 pilot** | Can an LLM play a fisher? | The interface worked (100% valid answers, about $0.41). The model **only cheated when told it was allowed to**. |
| 7 Oct | **R3** | Can deterrence thresholds be predicted in advance? | Yes, in **5 of 5** new settings, within one grid step. |
| 7 Oct | **T1** | Audit targeting | Aiming at the largest report caught only **25%** of liars, against **100%** at random, because it kept checking the same agent. |
| 7–8 Oct | **L2** | Do LLM fishers stop once the expected fine exceeds their gain? (4 models, 400 games) | gpt-oss and Nemotron cheated in 51–92% of rounds up to an expected fine of 1.33 t, and **0%** at 6 t. A **small fine raised cheating**. Gemma barely cheated and Mistral never did. Memory reduced cheating but didn't stop it. |
| 8 Oct | **L3** | Predict each model's stopping fine first, then test | Predicted **30 t**: Nemotron stopped exactly at 30, gpt-oss one step early at 24. They responded to **one round's gain**, not the whole game's. |
| 8 Oct | **Freeze** | — | Ameer stopped adding experiments, and the paper was drafted. |
| 9 Oct | **R4, R5** | Run the memory test on one shared grid of settings in all three games | **12 of 12** settings |
| 12 Oct | **Freeze lifted** after two outside reviews: **L4**, **L5** and **L6** launched | L4: does the prediction hold on new games, and is it the fine's size or the audit odds that matters? L5: more model families. L6: does one cheater spread cheating in Forest? | These were **running during the meeting**, which is why slide 13's boxes were empty. Afterwards: L4's prediction held, and **announcing audits with a zero fine made cheating worse (58% vs 30%)**. L6: cheating spread by **+25 points**, but through the reviewer's shared cut, not through imitation. |
| 13 Oct | **Meeting** | — | Feedback: the cost of monitoring; communication and contagion; welfare; a success metric; an LLM overseer |

### Critical assessment of Act 2

**What's strong**
- **The prediction was written down before each test, and it held on new games.** For two LLMs, the stopping fine was predicted from their behaviour with no fine. This is the one genuinely empirical, non-obvious result.
- **The traps are real and useful:**
  - checks with no consequence;
  - a known schedule;
  - punishing everyone for a few cheaters;
  - always checking the biggest agent;
  - a zero-fine announcement.
- **The process was disciplined:** every protocol frozen before its code, and failed predictions reported.

**What's weak, said plainly**
1. **Much of the programmed-agent work is true by construction.** The cheaters were built to maximise their payoff, so of course they stop at break-even; the paper admits this tests the measurement, not the agents. The joint reviewer is "near-exact" because it's given the true model. The targeting failure is partly mechanical: a rank rule always picks the same agent.
2. **The LLM test is a textbook test.** We told the models they *could* over-take, and we told them the odds and the fine. They may simply do the arithmetic. They didn't hide anything, so it's about exceeding an allowance, not deception.
3. **The evidence base is thin:**
   - only 2 of 4 families cheated on purpose;
   - 10 populations per condition;
   - one game;
   - 20 rounds;
   - a coarse fine grid.
4. **Checks were far too generous** (9–34% of actions) compared with AI control's 0.5–2%. The "you can only check a few" premise was never really stressed.
5. **There were no learning agents.** The October series has no RL. To a MARL reviewer, an overseer tested only against fixed scripts and prompted LLMs that don't learn is untested against the obvious threat: agents that learn to dodge it.
6. **The games are home-made text games with hand-set numbers.** Forest turned out not to be a real dilemma. None is a standard MARL environment.
7. **The framing drifted toward economics.** Act 1 was MARL. Act 2 moved to Becker and AI control because that's where the clean results were, so the work now sits between three fields and is home in none. That's the loop you're feeling.

---

## The honest one-paragraph summary

> From February to September I built commons games and compared ways of governing them. The question kept moving, and a bug undid the main September result, but it taught me to isolate one thing at a time. In October I narrowed to one question: when can a few random checks stop agents from taking more than their share? With scripted agents this mostly confirmed what the setup implied. With LLM agents I found something real: you can predict the penalty that stops a model by watching it first. Several sensible-sounding designs backfire. All of this was in simple text games, with generous checking, no learning agents and only two models that cheat. That's why the next step is a standard MARL game with learning agents and scarce checks.

---

## What to do now

**This week (decisions):**
1. **Grant: yes or no.** OpenMined closes **30 Oct**, in two days. It's a 12-week commitment.
2. **Ask Yali for CREATE access** (GPU), since the MARL direction needs it.
3. **Forest:** move it to the supplement (recommended). It doesn't matter for the new direction.

**Then, the actual work (all free):**
1. **Write one page with Yali before any code: the question and the players.** That's her own feedback from March–April, and skipping it is what made Act 1 drift. Something like: *"In Harvest, an overseer can check only k agents per step. Whom should it check and what should it do, so that learning agents keep the orchard alive at the lowest checking cost?"*
   - **The players:** the harvesters (RL learners, or LLMs).
   - **The overseer:** it sees only the agents it checks, and its actions are whom to check and what penalty to give.
   - **The scores:** apples left, total and equal reward, and checks used.
2. **Define "cheating" in Harvest, and test that definition first.** This is the question you asked at the end of the talk (slide 15), and it's where Clean Up failed in September.
   - Candidate definition: taking an apple from a nearly empty patch, which stops the patch regrowing.
   - Before building anything else, check that scripted bots produce both cheaters and non-cheaters, and that Harvest is still a dilemma with the overseer in place.
3. **Build the overseer wrapper in SocialJax Harvest**, test it with scripted bots, then run a small RL (IPPO) pilot on CPU to see what it costs.
4. **Treat the October results as the pilot study,** not the thesis. They motivate the design rules:
   - check at random;
   - punish the cheater, not everyone;
   - consequences must be real;
   - measure behaviour.

   Don't spend more credit on the text games for now.
