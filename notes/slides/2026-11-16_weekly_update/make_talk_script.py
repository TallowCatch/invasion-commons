"""Talk script for weekly_update_2026-11-16_Final.pptx (skill: supervisor-talk-script).

  <venv with python-docx>/python make_talk_script.py   ->  talk_script.docx next to this file
Numbers come from notes/socialjax/phase2_ippo/results.md, notes/socialjax/week2_bots/results.md and the papers' PDFs.
"""
from pathlib import Path

from docx import Document
from PIL import Image
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

NAVY = RGBColor(0x06, 0x27, 0x46)
HERE = Path(__file__).parent
OUT = HERE / "talk_script_with_slides.docx"
SHOTS = HERE / "script_slides"  # slide-NN.png rendered from the Final deck (render_pptx.sh); git-ignored
PX = 1600 / 13.333  # rendered pixels per slide inch
# What every symbol in a close-up means, shown in a small table under that close-up
SYMBOLS = {
    (7, 1): [
        ("p(n)", "chance an empty apple square grows back this step"),
        ("n", "apples within two squares of the spot (the 12 squares around it)"),
        ("i, t", "which agent, and which step of the game"),
        ("v(i,t)", "1 if agent i over-harvested at step t, otherwise 0"),
        ("1[ … ]", "equals 1 when the thing in the brackets is true, otherwise 0"),
        ("C(t), |C(t)| ≤ k", "the set of agents checked at step t, and at most k of them"),
        ("Pr(caught | v = 1)", "chance of being caught given a real over-harvest, which is 1 minus the miss rate"),
        ("α (alpha)", "miss rate, the chance a real over-harvest isn't caught (0 so far)"),
        ("β (beta)", "false-alarm rate, the chance an honest move is flagged (0 so far)"),
        ("r(i,t)", "the game's reward, 7 for each apple taken"),
        ("r̃(i,t) (r tilde)", "the reward after any penalty, which is what the agent learns from"),
        ("P", "penalty per catch, in apples, so 7P in reward"),
        ("caught(i,t)", "1 if agent i was caught at step t"),
        ("π(i) (pi)", "agent i's policy, its way of choosing moves"),
        ("max over π, E[ … ]", "the agent picks the policy that makes the average total largest"),
        ("Σ γ^t (gamma)", "add up rewards over the game, each step counting a bit less (γ = 0.99, about 100 steps ahead)"),
    ],
    (8, 0): [
        ("θ_O (theta O)", "the overseer's settings, which are how many to check (k), how it picks them, the penalty P"),
        ("W(θ_O, π*)", "apples collected per game with those settings, once the agents have learned"),
        ("π* (pi star)", "the agents' best reply, what they end up doing against these settings"),
        ("U(π; θ_O)", "the agents' own score, their reward after penalties"),
        ("arg max", "the choice that gives the largest value"),
        ("λ (lambda)", "how many apples one unit of cost is worth, the trade-off between the two"),
        ("Cost(θ_O), B", "what the overseer spends, and the budget it must stay within"),
        ("s.t.", "'subject to', the conditions that must hold"),
    ],
    (8, 1): [
        ("Σ over t, Σ over i in C(t)", "add up over every step and every agent checked at that step"),
        ("c_check", "cost of one check"),
        ("c_hist, w", "cost of each past step a check reads, and how many past steps it reads (w = 1 so far)"),
        ("η (eta)", "0 means no better than no overseer, 1 means as good as checking everyone"),
        ("W_none, W_all", "apples with no overseer (68) and with every agent checked every step (530 at 2 apples)"),
        ("W_worst", "the overseer's score against the toughest agents"),
        ("min over Π_adapt", "the lowest score over all agents that can see the checks and adapt to them"),
    ],
    (10, 1): [
        ("k", "agents checked per step"),
        ("N", "number of agents, 7"),
        ("k/N", "chance a given agent is checked at a step, with random checks"),
        ("α (alpha)", "miss rate (0 so far)"),
        ("P", "penalty per catch, in apples"),
        ("g", "what one over-harvest gains the agent, about 1 apple for the bots"),
        ("(k/N)(1 − α)P ≥ g", "the average penalty is at least the gain, so over-harvesting stops paying"),
    ],
    (10, 2): [
        ("β (beta)", "false-alarm rate, the chance a checked honest agent is wrongly flagged"),
        ("(k/N) β P", "average apples an honest agent loses per step (chance checked × chance wrongly flagged × penalty)"),
    ],
    (11, 2): [
        ("q(i)", "agent i's chance of being checked at a step"),
        ("q(i) = k/N", "random checks give every agent the same chance"),
        ("g(i)", "what over-harvesting gains agent i"),
        ("max over q, min over i", "pick the chances so that even the agent least deterred is still deterred"),
        ("Σ q(i) = k", "the chances add up to the number of checks"),
    ],
    (12, 1): [
        ("q(i,t)", "agent i's chance of being checked at step t"),
        ("v(i,t) = 1", "the moments when agent i actually over-harvests"),
        ("E[ A | B ]", "the average of A, counting only the moments where B is true"),
        ("catch rate", "how often over-harvests get checked; random checks give 2 in 7, about 0.29"),
    ],
}

# Close-ups placed right above the paragraph that talks about them: (slide, paragraph index) -> box in slide inches
CLOSEUPS = {
    (7, 1): (7.5, 1.4, 12.8, 6.95),    # "The model, written out" card
    (8, 0): (0.55, 1.7, 12.8, 3.75),   # the design problem
    (8, 1): (0.55, 3.9, 12.8, 6.75),   # cost, eta, worst case
    (9, 0): (6.8, 1.35, 12.8, 6.1),    # the bots chart
    (10, 0): (0.55, 1.3, 7.9, 5.95),   # the bar chart
    (10, 1): (8.1, 1.3, 12.8, 4.4),    # deterrence condition
    (10, 2): (8.1, 4.35, 12.8, 6.35),  # honest-loss prediction
    (11, 2): (0.4, 6.0, 12.9, 6.5),    # random vs mixed formula line
    (12, 0): (6.3, 1.4, 12.8, 6.05),   # the training curves
    (12, 1): (5.9, 6.05, 12.8, 6.9),   # catch-rate formula
}

TITLE = "Talk script: Weekly Update"
INTRO = ("Supervision meeting, 16 November 2026. Each slide is shown, then what I say, with a close-up of the part of the "
         "slide each paragraph is about and, under each formula, what its symbols mean. The questions I might get and the "
         "numbers to have ready are at the end.")

SLIDES = [
    ("Weekly Update", [
        "Thanks for making the time. This is what I've done since we met on the 2nd. I'll start with the three things "
        "we agreed last time, then the papers, then how I'm bounding the evaluation and how I built the setup in "
        "Harvest. After that come the first results with learning agents, what's still running, and a few questions."],
     []),
    ("What we discussed last time", [
        "Last time we left it at three things. The first was reading BazaarBench, which I hadn't read before. What I took "
        "from it is the idea of putting a cost on the overseer, so it pays for every check, for example a budget of 1 "
        "unit of tokens against 10. I also went back over two papers I'd read a while ago and have down as foundational, "
        "Engels et al. on how weak an overseer can be, and Willis et al. on populations of LLM agents.",
        "The second was how to bound the evaluation, so that if the overseer is smaller or less capable I can say "
        "whether it still keeps the agents in check, and what I'm comparing it against. The third was a more detailed "
        "plan for the experiments, especially random against targeted checks. The slides go in that order."],
     [("Why those two papers in particular?",
       "Engels et al. is the main paper that puts numbers on weak-to-strong oversight, so it's the obvious reference for "
       "a weaker overseer. Willis et al. is the group's paper on LLM populations, and I already used the code from the "
       "follow-up work for my earlier audit studies.")]),
    ("Engels et al. (2025) on weaker overseers", [
        "This is the scalable oversight paper I went back to, by Engels, Baek, Kantamneni and Tegmark at MIT. They ask how "
        "well a weaker AI can oversee a stronger one. They set up four games, Mafia, Debate, Backdoor Code and Wargames. "
        "In each one there's a guard, the overseer, and an opponent trying to get past it, and every model gets a skill "
        "rating for each role. The rating is Elo, the chess system, where a 400-point gap means the stronger side wins "
        "about ten times out of eleven.",
        "They also look at chains of overseers, where a weak model checks a slightly stronger one, which checks a stronger "
        "one again. Even then, at a 400-point gap oversight succeeds 13.5 percent of the time in Mafia, 51.7 in Debate, "
        "10 in Backdoor Code and 9.4 in Wargames, and it gets worse as the gap grows. What I take from it is a way to "
        "measure how much weaker an overseer is. What it doesn't have is many agents, repeated play, opponents that learn, "
        "or any cost for checking."],
     [("How would you use Elo in your setting?",
       "For the RL version I'd make the overseer weaker in steps, fewer checks, more mistakes, or a smaller network, and "
       "plot how well it keeps agents in check against how much weaker it is. If I get to an LLM overseer, I can do it "
       "their way and use a smaller model than the players, so the gap is a real capability gap."),
      ("Isn't 51.7 percent in Debate quite good?",
       "Not really. In Debate the judge picks between two answers, so guessing already gets 50 percent. It's barely above "
       "chance."),
      ("What do they say their limitations are?",
       "That it's single-step oversight, so errors that build up over time aren't captured, the deceptive side is "
       "prompted rather than trained, and Chatbot Arena Elo is only a rough measure of capability. Multi-step is exactly "
       "my setting.")]),
    ("Willis et al. (2025) on LLM populations", [
        "This one is by Richard, you, Joel Leibo and Michael Luck. In the Prisoner's Dilemma each player does better by "
        "betraying the other, but both do better if both cooperate. ChatGPT-4o and Claude 3.5 Sonnet each wrote 25 "
        "strategies for every attitude, aggressive, cooperative or neutral, and for each prompt style. ChatGPT-4o then "
        "turned them all into code.",
        "Then they run a Moran process. Every player plays everyone, a player is copied in proportion to how well it "
        "scored, and the copy replaces someone at random, until one attitude fills the population. Aggressive strategies "
        "usually lose out. Starting from a population that's two thirds aggressive, aggression wins 35 to 66 percent of "
        "the runs without noise, which is no better than its starting share. The exception is ChatGPT-4o with the default "
        "prompt, where aggression is the best reply to itself. Adding noise to the moves makes aggressive outcomes more "
        "likely. What's missing for me is any kind of overseer."],
     [("Why does noise help aggression?",
       "Their explanation is that noise hides it. Without noise, if someone breaks a run of cooperation you know they're "
       "exploiting you. With noise it might be an accident, so it's less clear when to retaliate."),
      ("How does this connect to what you did before?",
       "I used the code from the follow-up work, the commons version, for my earlier audit studies. There, random audits "
       "with a fine set just above the predicted level reversed the selfish takeover. But selfish strategies that knew "
       "the rule complied without changing, which is why I now test switching the overseer off."),
      ("Do cooperative and neutral strategies actually differ?",
       "Barely. Both cooperate almost all the time with each other, and the paper notes LLMs may not separate those two "
       "attitudes well in this game.")]),
    ("BazaarBench (Wang et al., 2026)", [
        "BazaarBench is by Ziyan Wang, Shuqing Shi and others, with you and Adel Bibi as corresponding authors, and it was "
        "new to me. There are three base markets, each with 100 LLM agents that list, buy and sell items for 30 simulated "
        "days. Then 20 of the agents are switched to the model being tested and the market runs for another week, under "
        "ordinary instructions, deadline pressure, or instructions to exploit other traders. That's five models, three "
        "conditions and three markets, 45 continuations.",
        "Under ordinary instructions every model tries to promise the same item to more than one buyer. Under the exploit "
        "instructions, the share of the tested sellers' deals that completed with a missing or overstated item goes from "
        "15.4 to 33.4 percent, and to 55.5 for GPT-5.4. Two things matter for me. The judging was done afterwards by GPT-5 "
        "and came to about 6,500 dollars in tokens for the 48 runs, so checking is genuinely expensive. And because the "
        "audit only rescores the records afterwards, the agents never react to being checked. A live overseer that checks "
        "some traders each day on a token budget is the open piece."],
     [("How would you put an overseer into BazaarBench?",
       "Each simulated day it would check a few of the tested traders' listings and messages with an LLM judge, under a "
       "fixed token budget, say 1 unit against 10. Anyone caught would get a consequence the agents can see, like a "
       "warning or losing listings. I'd use small markets and open models on CREATE to keep the cost down."),
      ("Why not do that now?",
       "The judge cost alone is about 6,500 dollars for 48 runs, so it needs a proper compute plan. For ICML I'm doing the "
       "RL version in Harvest first, where checks are cheap and I can run hundreds of settings, then I'd bring the same "
       "overseer design over."),
      ("What does the inspection audit actually do?",
       "It goes through the completed deals and asks whether the seller had the item and described it honestly. It assumes "
       "an inspecting buyer would have refused it. It changes the scores, not what the agents did.")]),
    ("How I'm bounding the evaluation", [
        "For bounding I borrowed the structure from AI Control, by Greenblatt and colleagues at Redwood Research. They "
        "compare every safety setup against two simple ones. Never using the risky model is completely safe but only 2 "
        "percent as useful, while auditing just the first 2 percent of its work is fully useful but only 15 percent safe. "
        "Their best setups sit in between, like trusted monitoring at 62 percent safe. Then a red team builds the "
        "strongest attack it can against each setup, which gives the worst case.",
        "My version has no overseer at the bottom, an overseer that checks every agent at every step at the top, and "
        "agents trained to get around the overseer as the worst case. To make the overseer weaker I follow Engels et al., "
        "but I change one thing at a time, fewer checks, more mistakes, or a smaller network, so I can tell which kind of "
        "weakness hurts most. The cost comes from BazaarBench, where every check and every step of history the overseer "
        "reads costs something. So far I've measured the top and bottom and the effect of checking fewer agents."],
     [("Is checking everyone really the best case?",
       "At a given average penalty, yes. With a perfect inspector it catches every over-harvest, so nothing can do better "
       "on information. The best welfare also depends on the penalty, so I compare at the same average penalty. At 2 "
       "apples, checking everyone got 530 apples per game."),
      ("What's the equivalent of usefulness here?",
       "Apples collected. If the overseer wrongly punishes honest agents or freezes them, fewer apples get collected, so a "
       "heavy-handed overseer shows up there."),
      ("What does weaker mean in practice?",
       "Fewer agents checked per step, a shorter memory for each check, a miss rate and a false-alarm rate, and later a "
       "smaller network if the overseer is learned. All except the network are already settings in my code.")]),
    ("How I built it", [
        "This is how it's built. The game and the training code are SocialJax, from your group, and Richard is one of the "
        "authors. I didn't change the game. It's Harvest Open, with 7 agents, games of 1,000 steps, and each agent seeing "
        "an 11 by 11 window around itself. I used individual rewards, so each agent keeps its own apples. SocialJax's "
        "default for Harvest training is a shared reward, which would remove the dilemma.",
        "What I wrote is the overseer, a wrapper around the game in JAX with 15 tests. At every step it looks at the map, "
        "chooses which agents to check, and changes the reward of anyone it catches. On the right, p of n is the game's "
        "own regrowth, 2.5 percent a step when 3 or more apples are within two squares, then 0.5, 0.1, and zero with "
        "none. So I set the rule at 3. v is one when an agent takes an apple with fewer than 3 left around it, counted "
        "on the map before it moves, so an honest agent can never be flagged.",
        "When the overseer checks an agent it catches a real over-harvest with chance one minus alpha and wrongly flags an "
        "honest move with chance beta. Both are zero so far. A caught agent loses 7P reward, because one apple is worth 7 "
        "in SocialJax. Each agent learns to maximise its discounted penalised reward, with gamma 0.99, which is roughly a "
        "100-step horizon. The training method is IPPO, where each agent learns by trial and error from its own reward, "
        "and here all seven share one network. Each run is 100 million steps on one GPU on CREATE, which takes 20 to 40 "
        "minutes, and every setting is run three times."],
     [("Did you use Richard's code?",
       "SocialJax, yes, which Richard co-wrote. That gives the game and the IPPO training. Richard's evolution code from "
       "the Willis paper is what I used for my earlier Fishery-style audit studies, not here. The overseer itself is my "
       "own code."),
      ("Why the threshold of 3?",
       "It's where the game regrows fastest, and I checked it. Seven rule-following bots collect 500 apples at 3, against "
       "393 at 4 and 190 at 2, and the orchard survives in all ten runs."),
      ("Can the agents see the overseer?",
       "Not yet. They only see their own window and don't know who's checked. Letting them see it is step three of the "
       "targeting plan, where targeting could be gamed."),
      ("Why one shared network?",
       "It's the SocialJax default and makes training much cheaper. The downside is that all seven agents behave alike, so "
       "there's no persistent cheater, which matters for the targeting result. Separate networks or mixed populations are "
       "a natural next step.")]),
    ("The overseer's problem, written down", [
        "This is the bounding slide written as an optimisation. The overseer chooses its settings, theta O, which are how "
        "many agents it checks, how it picks them and the penalty. It wants the most apples collected, W, minus lambda "
        "times what it spends, and it has to stay within a budget B. The catch is the constraint in the middle. The "
        "agents respond by learning their best reply to whatever the overseer does, so it's a two-level problem, like a "
        "Stackelberg game, where the leader commits first and the followers adapt.",
        "The cost counts every check, plus a bit more for every past step a check reads, which is where 1 unit against "
        "10 comes in. Eta places any overseer between no overseer at zero and checking everyone at one. For example one "
        "check per step at an average penalty of 2 apples gets 515 apples, against 530 for checking everyone and 68 with "
        "nothing, so eta is about 0.97. The worst case takes the minimum over agents that can adapt, which in practice "
        "means training agents against the overseer. So far I've searched the settings with a grid rather than solving "
        "this directly."],
     [("Can you actually solve this?",
       "Not exactly. It's a two-level problem with learning agents inside. For now it's a grid search over checks and "
       "penalties. Later the overseer itself learns, with its reward being apples collected minus the cost, trained in "
       "turns against the agents, which is the double-oracle idea from security games."),
      ("What is U?",
       "The agents' own objective, their discounted reward after penalties, the same thing as on the previous slide."),
      ("Do penalties count in W?",
       "No. W is apples collected. Penalties just move reward around, so I report net reward separately."),
      ("Is there another way to write it?",
       "Yes, the safety-first version closer to AI Control. Minimise the cost, subject to the orchard dying less than "
       "some small share of the time, or over-harvesting staying below a set rate.")]),
    ("One greedy bot wrecks the orchard", [
        "Before training anything, I checked the game with simple scripted bots that don't learn. A bot that holds back "
        "only takes apples the rule allows, and a greedy bot takes the nearest apple. The chart is a Schelling diagram. "
        "Along the bottom is how many of the seven hold back, and it shows what each kind of bot collects.",
        "If all seven hold back they each get 71.5 apples, and if all are greedy they get 9.6. But when six hold back, "
        "the greedy one gets 30 against their 9, so every agent is tempted to be greedy, which makes it a real social "
        "dilemma in Leibo's sense. It's also very fragile, because one greedy bot kills the orchard in all ten runs. With "
        "a bot that can't adapt, greed stops paying once each over-harvest costs about one apple on average, whatever the "
        "number of checks. That's where I set the penalties for the learning agents."],
     [("Is it the greed or the fear condition?",
       "Greed. When most hold back the greedy agent does better than them. Fear doesn't really hold, because with few "
       "restrained agents both kinds end up with about the same, around 10 apples."),
      ("Does greed ever pay over a whole game?",
       "Not against everyone holding back. A lone greedy bot strips the orchard and ends with 30, far below the 71.5 it "
       "would get by holding back. It only wins against the restrained bots in the same game. Learning agents look about "
       "100 steps ahead, so they see the short-term gain, which is why they learn to over-harvest."),
      ("How did you get the one-apple figure?",
       "I ran one greedy bot with six restrained ones and worked out, for each number of checks, the penalty at which the "
       "greedy one stops beating the others. Checks times penalty came out at about 7 at every k, and dividing by 7 agents "
       "gives about one apple per over-harvest.")]),
    ("It's the penalty, not the number of checks", [
        "These are the learning agents. With no overseer they strip the orchard in every game, collect only 68 apples, "
        "and almost half the apples they take break the rule. The average penalty is the chance of being checked times "
        "the penalty, so it's what an agent expects to lose each time it over-harvests. I tried half an apple, one apple "
        "and two apples, each with 1, 2, 4 or all 7 agents checked per step. With one check the penalty per catch is "
        "seven times bigger than with seven.",
        "At half an apple nothing changes, even when everyone is checked. At one apple the orchard survives in every run, "
        "and at two the agents collect 515 to 530 apples, more than the well-behaved bots, and only about 1 percent of "
        "what they take breaks the rule. The card on the right is the condition behind it. k over N, times one minus "
        "alpha, times P, has to be at least g, the gain from one over-harvest. It's Becker's condition, and it's the same "
        "thing my Fishery study found, the expected fine decides.",
        "The second formula is a prediction I haven't tested. If the overseer sometimes flags honest agents, an honest "
        "agent loses k over N times beta times P per step. At a fixed average penalty that doesn't depend on k, so one big "
        "check and seven small ones cost honest agents the same on average and differ only in how bumpy the losses are. "
        "Whether learning agents care about that is the next experiment."],
     [("Why doesn't the number of checks matter?",
       "Because the agents learn from their average reward, and the average penalty is the same across a column. The "
       "number of checks only changes how often they're caught and how big each penalty is. There's a small effect "
       "within the 1-apple column, 441 apples with one check against 477 with seven, which might be because frequent small "
       "penalties are an easier signal to learn from. With three runs that's a lead, not a result."),
      ("Why is the switch between half an apple and one, not exactly one?",
       "I only tested three levels, so all I know is it's somewhere in that range. A finer sweep, 0.6 to 0.9, would pin "
       "it down. It could sit below one because an over-harvested apple isn't a pure gain, since the agent might have got "
       "it later anyway."),
      ("How reliable is this?",
       "Three runs per setting, averaged over the last ten training updates, which is 2,560 games each. The pattern is "
       "very clean, with the orchard dead in every run at half an apple and alive in every run from one apple. Two of "
       "the 4-check bars have two runs because of a broken GPU node, and those are rerunning. The big caveat is the "
       "perfect inspector."),
      ("Isn't a perfect inspector unrealistic?",
       "Yes, which is why mistakes are next. The false-alarm formula says what to expect, and it's the first thing I'll "
       "test.")]),
    ("Testing random against targeted checks", [
        "This is the plan for random against targeted, and why it's in this order. Every version uses the same number of "
        "checks, so only who gets checked changes. Repeat offenders first means each agent has a suspicion score that "
        "goes up by one when it's caught and fades by 5 percent a step, and the overseer checks the highest scores. The "
        "location version checks the agents standing where the fewest apples are left.",
        "I start with agents that can't see who is checked, because that's the cleanest test of whether targeting finds "
        "over-harvesters better at all. With the scripted bots it did. Against one greedy bot with a 50-step freeze, "
        "repeat offenders first saved the orchard with 2 checks per step where random needed 6. Then I let the agents "
        "see who is checked, because that's where targeting can backfire. In my Fishery study, always checking the "
        "biggest reporter caught a quarter of the liars, while random checks caught all of them. The last step is to "
        "target with enough randomness mixed in that agents can't predict it, which is the security-game idea.",
        "The line underneath says the same thing. Random checks give every agent the same chance, k over N. The mixed "
        "version chooses each agent's chance q so that even the agent best placed to over-harvest still faces an "
        "average penalty above its gain, with the chances adding up to k."],
     [("How will agents see who is checked?",
       "I'll add it to what they observe, for example which agents were checked last step. That's the simplest version of "
       "an overseer whose pattern can be learned."),
      ("What does Blindfold show?",
       "It's about certifying LLM agents with random audits. The guarantee breaks if the agent can tell, before it acts, "
       "whether this action is being audited, for example from timing. So keeping the audit coin secret matters, which is "
       "the same point as targeting being predictable."),
      ("Where does a learned overseer come in?",
       "After this. It's a policy that picks k agents each step, rewarded for apples collected minus the cost, trained in "
       "turns against the agents. These hand-made choosers are the baselines it has to beat.")]),
    ("Targeting only helps with a good signal", [
        "These are the first results from that plan, with agents that can't see the checks, two checks per step and an "
        "average penalty of one apple. Every version collapses at first, because the agents learn that taking apples "
        "pays before they learn about the penalty. Repeat offenders first recovers much faster. By about 13 million steps "
        "the orchard survives in nearly every game, against 14 to 57 percent for random checks. But both end up in the "
        "same place, around 460 to 470 apples, because the seven agents share one network, so there's no single repeat "
        "offender to follow.",
        "Checking by location does worse than random. It looks at agents on empty ground, but an agent that has just "
        "over-harvested is standing right next to the last apples. The formula underneath is how I measure this, the "
        "average chance of being checked at the moments an agent actually over-harvests. Random gives 0.28, close to 2 "
        "in 7. Repeat offenders first starts at 0.44 and settles at 0.28. Location gives 0.05. It's 0.04 before any "
        "learning, so it's a bad signal rather than agents learning to hide. Freezing caught agents for 50 steps instead "
        "of taking apples never saved the orchard, at about 77 apples a game."],
     [("Did the agents learn to dodge the location checks?",
       "Mostly no, since the catch rate is already 4 percent before learning. It does rise to about 22 percent around "
       "update 20 and then falls back to 5 or 6, so there might be some learned exploitation on top, but this design "
       "can't separate the two."),
      ("Why does the location version keep the orchard alive at all?",
       "Barely. Alive just means at least one apple is left, and it ends with about 6, against 24 to 26 for random checks. "
       "Half of what the agents take still breaks the rule."),
      ("Why does freezing fail?",
       "A freeze doesn't change the reward directly, only by making the agent miss apples, so the learners don't connect "
       "it to over-harvesting. The bots showed the same thing, with two checks and a 50-step freeze not enough to save "
       "the orchard.")]),
    ("Still computing", [
        "Two sets of runs are queued on the cluster now. The first starts from the agents that learned to over-harvest "
        "with no overseer and switches it on for another 50 million steps. That's random checks at an average penalty of "
        "one apple with 1, 2, 4 or 7 checks, plus the freeze, repeat offenders first, and a no-overseer continuation as a "
        "control, 21 runs. It asks whether the overseer can fix agents that already misbehave, which is the realistic "
        "case.",
        "The second takes agents trained with the overseer, removes it, and lets them keep learning, 18 runs. It asks "
        "whether the good behaviour lasts. In my Fishery work strategies behaved while checked but didn't change. My "
        "prediction, written before the runs, is that some over-harvesting comes back."],
     [("Why not just turn the overseer off and watch, without more training?",
       "With a penalty-only overseer that tells you nothing. The penalty changes the reward, not the actions, so with "
       "learning stopped the behaviour is identical with or without it. The pilot showed exactly that, so the agents have "
       "to keep learning."),
      ("When will they be done?",
       "They're waiting for GPUs. The runs themselves are 10 to 20 minutes each, so once they start it's a few hours, but "
       "the queue has been busy.")]),
    ("Questions", [
        "A few things I'd like your view on. First, is no overseer at the bottom, checking everyone at the top and trained "
        "agents as the worst case a sensible way to bound the evaluation, or would you add something? Second, for making "
        "the overseer weaker in RL, which would convince you most, fewer checks, more mistakes, or a smaller network than "
        "the agents? And third, one check with a big penalty almost matches checking everyone. Is that worth pushing as a "
        "result, or is it only interesting once the overseer makes mistakes?"],
     []),
    ("Thank You", ["Thanks, that's everything."], []),
]

NUMBERS = [
    ("No overseer, learners", "68 apples per game, orchard dead in every run, 45% of takes break the rule",
     "phase2_ippo/results.md"),
    ("Average penalty 0.5 apples", "about 68 apples at every k, orchard dead", "phase2_ippo/results.md"),
    ("Average penalty 1 apple", "441 (k = 1) to 477 (k = 7) apples, orchard alive in every run", "phase2_ippo/results.md"),
    ("Average penalty 2 apples", "515 (k = 1) to 530 (k = 7) apples, about 1% of takes break the rule",
     "phase2_ippo/results.md"),
    ("η for one check at 2 apples", "(515 − 68) / (530 − 68) ≈ 0.97", "phase2_ippo/summary.json"),
    ("Catch rate, 2 checks", "random 0.28, repeat offenders 0.44 then 0.28, location 0.05 (0.04 at the start)",
     "phase2_ippo/results.md"),
    ("Freeze, 2 checks, 50 steps", "77 apples, orchard dead", "phase2_ippo/results.md"),
    ("Bots, no overseer", "all hold back 71.5 each, all greedy 9.6, 6 hold back: greedy 30.1 vs 9.2",
     "week2_bots/results.md"),
    ("Rule threshold m", "500 apples at m = 3, 393 at 4, 190 at 2", "week2_bots/results.md"),
    ("Bot break-even", "checks × penalty ≈ 7 at every k, about 1 apple per over-harvest", "week2_bots/results.md"),
    ("Runs", "3 seeds per setting, 100M steps, 256 games in parallel, 20–40 min per run", "phase2_ippo/protocol.md"),
    ("Engels et al.", "400-Elo gap: 13.5% Mafia, 51.7% Debate, 10.0% Backdoor Code, 9.4% Wargames", "arXiv 2504.18530"),
    ("Willis et al.", "two-thirds aggressive start: aggression wins 35–66% of runs (no noise)", "arXiv 2501.16173"),
    ("BazaarBench", "15.4% → 33.4% (GPT-5.4 55.5%); judge about $6,487 for 48 runs", "arXiv 2610.06748"),
    ("AI Control", "no untrusted model 100% safe, 2% useful; upfront audit 2% → 15% safe; trusted monitoring 62%",
     "Greenblatt et al. 2024"),
]


def main():
    d = Document()
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(11)
    for s in d.sections:
        s.left_margin = s.right_margin = Inches(1)

    def heading(text, size, space_before=12):
        p = d.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before)
        r = p.add_run(text)
        r.font.name, r.font.size, r.bold = "Georgia", Pt(size), True
        r.font.color.rgb = NAVY
        return p

    heading(TITLE, 18, 0)
    p = d.add_paragraph(INTRO)
    p.runs[0].italic = True
    crops = SHOTS / "closeups"
    crops.mkdir(exist_ok=True)

    def small_table(rows, widths):
        t = d.add_table(rows=0, cols=len(widths))
        t.style = "Table Grid"
        for row in rows:
            cells = t.add_row().cells
            for c, txt, w in zip(cells, row, widths):
                c.text = txt
                c.width = Inches(w)
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    for r in p.runs:
                        r.font.size = Pt(9.5)
            cells[0].paragraphs[0].runs[0].bold = True
        d.add_paragraph().paragraph_format.space_after = Pt(0)

    for i, (title, spoken, _) in enumerate(SLIDES, 1):
        heading(f"Slide {i}: {title}", 13)
        shot = SHOTS / f"slide-{i:02d}.png"
        if shot.exists():
            d.add_picture(str(shot), width=Inches(5.2))
            d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for j, para in enumerate(spoken):
            box = CLOSEUPS.get((i, j))
            if box and shot.exists():
                c = crops / f"slide-{i:02d}_{j}.png"
                Image.open(shot).crop(tuple(int(v * PX) for v in box)).save(c)
                w = min(4.6, (box[2] - box[0]) * 0.62)
                d.add_picture(str(c), width=Inches(w))
                d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if (i, j) in SYMBOLS:
                h = d.add_paragraph()
                r = h.add_run("What the symbols mean")
                r.bold, r.italic = True, True
                r.font.size = Pt(10)
                r.font.color.rgb = RGBColor(0x55, 0x52, 0x4E)
                h.paragraph_format.space_after = Pt(2)
                small_table(SYMBOLS[(i, j)], (1.7, 4.8))
            d.add_paragraph(para).paragraph_format.space_after = Pt(6)

    d.add_page_break()
    heading("Questions I might get", 16, 0)
    p = d.add_paragraph("Grouped by slide. Skim these, no need to memorise them.")
    p.runs[0].italic = True
    for i, (title, _, qa) in enumerate(SLIDES, 1):
        if not qa:
            continue
        heading(f"Slide {i}: {title}", 12, 8)
        for q, a in qa:
            pq = d.add_paragraph()
            pq.paragraph_format.space_after = Pt(0)
            pq.add_run(q).bold = True
            pa = d.add_paragraph(a)
            pa.paragraph_format.left_indent = Inches(0.25)
            pa.paragraph_format.space_after = Pt(6)
    heading("Numbers to have ready", 13)
    t = d.add_table(rows=1, cols=3)
    t.style = "Light Grid Accent 1"
    for c, txt in zip(t.rows[0].cells, ("What", "Number", "Source")):
        c.text = txt
    for row in NUMBERS:
        cells = t.add_row().cells
        for c, txt in zip(cells, row):
            c.text = txt
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.size = Pt(9.5)
    d.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
