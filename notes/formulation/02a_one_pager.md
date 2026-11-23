# The overseer, on one page (draft, 23 Nov 2026)

**Setting.** N agents in a sequential social dilemma: a partially observable Markov game where what is good for each agent alone is bad for the group (SocialJax Harvest: 7 agents, 1,000 steps).

**Two levels of harm.**
- **An individual rule** says whether one agent's recent behaviour is a violation (eating an apple that leaves fewer than 3 nearby).
- **A group predicate** says whether the shared system is safe (the orchard survives).
- **Local pass, global fail** is when every agent passes the rule and the commons still fails. Per-agent checks cannot see this harm.

**The overseer is a player.** Each step it acts in four stages:
1. **Look.**
   - Choose at most k of the N agents to inspect, and how far back to read (w steps).
   - Optionally read system signals (apples left).
   - Everyone else goes unseen.
2. **Judge.** An inspector turns each record into a verdict. It may miss violations (rate α) or flag clean records (rate β), and these can depend on the situation. An LLM inspector's α and β depend on its size and on what it reads.
3. **Respond.** Sanction or freeze flagged agents; optionally act on the system signal.
4. **Share** (if there are several overseers). Send messages at a cost.

**Capability.** κ^O = (k, w, channels, inspector quality, response strength, delay, cost). This is what the overseer *can* do; its policy decides how it uses it. The capability gap Δ between agents and overseer is measured by playing overseers against agent classes and fitting ratings, as in Engels et al. 2025.

**Several overseers.** M overseers, each with its own region and budget k_m, its own view, and optional messages. M = 1 is the central case. Following Du et al. 2022 and Ma et al. 2024, overseers are nodes whose links to agents are chosen each step. A measured locality number (how often agents cross regions; how much harm stays local) says when splitting works.

**Timing.** The overseer commits to its policy, and the agents learn in the game that defines (a Stackelberg-Markov game, Gasztowtt et al. 2024). It is evaluated against ordinary learners and adversarial ones that see the inspections or exploit the inspector (the honest and scheming modes of Griffin et al. 2024).

**Objectives, as a frontier:**
- safety (the commons survives);
- welfare (what agents collect);
- harm to compliant agents (wrongful sanctions);
- burden (inspections, tokens, messages).

Bounds: no overseer; inspect everyone perfectly; the worst case.

**Why this, and what is new.**
- AI Control has a budgeted, imperfect, weaker overseer, but against one model.
- MARL shapers (the AI Economist, LIO, Baumann) act on many learners, but see everything.
- Nobody studies an overseer that must *choose what to observe*, with imperfect inspection, facing many learners in a sequential commons, alone or as several regional overseers.

**Already measured:**
- With no overseer the orchard dies; with 2 random inspections per step it survives (about 68 against 460 apples).
- What the learners do is set by how often a violation is actually caught and sanctioned, not by k alone.
- An LLM inspector's misses cluster on borderline cases.
- Behaviour reverts when the overseer is switched off.
- Splitting into regions costs most when agents crowd.
