# Phase 1: how the literature models an overseer (23 Nov 2026)

**What this is.** A map of every way the papers we have read model an agent that oversees, monitors, regulates or shapes other agents. It covers what the overseer sees, what it does, how it decides, how many overseers there are, how capable they are, and how the overseen agents respond. It ends with the gaps where our formulation can sit.

**Sources:**
- 19 papers read in full and 8 skimmed. Full notes: `lit/A_oversight_and_control.md` (AI Control and scalable oversight) and `lit/B_marl_overseers_and_decentralised.md` (MARL shapers, Du's networked MARL, testbeds).
- Our own earlier drafts: `lit/C_our_prior_formulations.md`.
- Section and page citations are in the A and B notes.

## 1. The answer in one paragraph

Two literatures model overseers, and neither covers our setting.
- **AI Control and scalable oversight** (Griffin, Greenblatt, Makins, Engels, Parisi). The overseer is a real player with a budget, imperfect detection, and often less capability than the agent it watches. But the agent is one model, usually a worst-case red team, in a one-step or short task.
- **Multi-agent RL** (Sun, the AI Economist, LIO, Baumann, Vinitsky, Gasztowtt). The overseer is an agent that shapes many learners in sequential social dilemmas. But it sees everything correctly, and its only budget is on what it pays, never on what it sees.

**Nobody has an overseer that must choose which few agents to look at, sees imperfectly, and faces many learning agents in a sequential social dilemma.** That is the empty cell.

## 2. The design space: the choices every overseer model makes

Each axis below is one design decision. The tags show who takes which option.

**A. Where the overseer sits**
- A player with its own objective (Griffin, Greenblatt, Engels; Baumann, Zheng and Gasztowtt on the MARL side).
- A design committed in advance, outside the game (Casheekar, Gans & Holden, Makins, Ye).
- Part of the environment (Parisi: the monitor decides when rewards are visible; Köster: the environment labels taboos).
- An evaluator after the fact, with no effect on play (BazaarBench's judge).

**B. What it observes**
- *Scope:*
  - one agent's latest action or output (Griffin, Greenblatt, Makins, Casheekar);
  - one agent's recent history (Ye);
  - every agent's actions or the full state (Sun, LIO, Baumann, IQ-Flow);
  - aggregates only (Zheng's AI Economist, Gasztowtt's LLM legislator);
  - a local, spatial window (Wang's green-security patroller, the κ-hop view in Du's papers).
- *Quality:*
  - exact when it looks (Casheekar, Gans & Holden, all MARL shapers);
  - a noisy score whose distribution differs for safe and unsafe behaviour (Griffin, Greenblatt, Makins);
  - an LLM judge of varying strength (Engels, Ye, BazaarBench).
- *Budget on observation:*
  - a count of audits held in the overseer's state (Griffin, Greenblatt);
  - audit probabilities (Casheekar, Gans & Holden);
  - tokens or compute (Makins, Ye, BazaarBench).
  - **No MARL paper limits observation.** Baumann and LIO limit only payments.

**C. What it can do**
- Let pass, defer to a safe fallback, or audit (Griffin, Greenblatt).
- Shut down, remove or freeze (Greenblatt, Ye, Vinitsky, Perolat's tagging).
- Add penalties or rewards per agent (Baumann, LIO, IQ-Flow, Casheekar).
- Set one rate for everyone, such as a tax (Zheng, Gasztowtt).
- Move to find violations (Wang).
- Restrict what agents can do in advance (Makins).

**D. How it decides**
- A threshold on a score (Griffin, Greenblatt, Makins).
- A solver or equilibrium computation (Griffin; Wang's PSRO).
- A committed lottery (Gans & Holden, Zhu).
- Learned through the learners' next update (Baumann, LIO) or by meta-gradient (IQ-Flow).
- Two-level RL (Zheng).
- An LLM learning in context across episodes (Gasztowtt) or a prompted judge (Makins, Engels, Ye).

**E. How many overseers**
- One, in almost every paper.
- One overseer with a scarce stronger tier (Greenblatt: trusted model, then human).
- Many independent per-agent monitors that never share (Makins).
- Peers reporting to a central verifier (Ye, Casheekar).
- Overseers who talk and vote (Engels' Mafia setting, with LLMs only).
- **None has several designated overseers with their own budgets or regions acting on learners.** Du's networked template (agents as graph nodes, κ-hop neighbourhoods, a measurable locality number ξ) is the natural way to formalise it, and nobody has done so for oversight.

**F. Capability relative to the agents**
- Weaker trusted vs stronger untrusted (Greenblatt, Griffin, Makins).
- A measured, continuous gap on an Elo scale (Engels).
- Perfect when it looks (Casheekar, Gans & Holden, Parisi, every MARL shaper).
- **No MARL paper has an overseer weaker than the agents.**

**G. Timing and solution concept**
- Leader-follower with a worst-case responder (Griffin, Greenblatt).
- A Stackelberg-Markov game: the principal commits each episode, which defines a game, and the followers learn in it (Gasztowtt; the AI Economist recast).
- One-step lookahead on the learners' update (Baumann, LIO).
- Simultaneous co-learning (Vinitsky, Sun, Tang).
- Zero-sum equilibrium against one attacker (Wang).
- Commit, then judge by where learning ends up (Casheekar).

**H. The overseen agents**
- One scheming model (Griffin, Greenblatt, Engels, Gans & Holden).
- Many coordinating attackers (Makins).
- Many RL learners (Casheekar with shared parameters; all MARL shapers).
- An LLM population (Ye, BazaarBench, OpenSocialJax).

**I. Objective and evaluation**
- A safety vs usefulness frontier. Any frontier point maximises w·Safety + (1−w)·Use (Griffin, Greenblatt).
- Undetected success against the false-positive rate (Makins).
- Win rate as the capability gap changes (Engels).
- Welfare, equality, sustainability and peace: Perolat's four standard social-outcome metrics.
- "Switch the overseer off" tests (Baumann) and empirical payoff matrices (Sun, Leibo): does the overseer change what the agents learn, or only prop up behaviour while it is on?

## 3. Where our own work has sat

| | June–July drafts | Oct–Nov experiments |
|---|---|---|
| Position (A) | an oversight *architecture* (none, local, global, hybrid) | a committed rule |
| Observation (B) | **scope** explicit: local action checks vs global system signals | **budget** explicit: k of N agents, window w; quality as miss and false-alarm rates; measured LLM ladder |
| Actions (C) | restriction, sanction, delay | sanction (apples removed) |
| Decides (D) | fixed | fixed (random, repeat offenders, round robin, mixed) |
| Number (E) | one | one, or M regional overseers with optional messages |
| Capability (F) | **explicit gap Δ** between actors and overseer | inspection quality via LLM size; the gap not framed as such |
| Timing (G) | fixed populations | commit, then IPPO learners respond |
| Agents (H) | generated strategies | learners (SocialJax); bots for worst cases |
| Evaluation (I) | welfare, unsafe rate, **local-pass/global-fail**, burden | apples, survival, violations, catches; switch-off test |

The June drafts had the agent-centric pieces the literature uses: observation scope, a capability gap, and harm that no single check sees. The November work had the setting nobody else has: k-of-N inspection against learners in SocialJax. **Our formulation should combine both.**

## 4. The gaps: where a contribution can sit

These are the empty cells, merged from both readers. Ranked by how central they are to "an overseer that sees only part of a sequential commons":

1. **A budget on observation.** The overseer chooses which k of N agents to look at each step and sees only those. Not studied anywhere. The MARL shapers see everything; the AI Control monitors watch one agent.
2. **Imperfect or weaker inspection against learners in a social dilemma.** Misses, false alarms, an overseer less capable than the agents. Sun et al. name detection error as open (p. 8). Engels gives the capability scale.
3. **Learners that adapt to the inspection policy itself.** They behave well only when watched, or find the inspector's blind spots. Parisi, Zhu and Gans & Holden show this for one agent; nobody has shown it for many learners in a social dilemma.
4. **Harm that lives at the group level while inspection happens at the individual level.** Depletion from many small takes; attacks split across agents. Makins shows per-agent monitors miss split harm. Our June "local pass, global fail" names exactly this.
5. **Several overseers with their own budgets or regions, and what they communicate.** No paper has more than one designated overseer acting on learners. Du's template gives the formal structure and a measurable locality number.
6. **Two-mode evaluation in a social dilemma.** Griffin's honest and adversarial modes, used with a population whose behaviour is itself learned under the overseer.
7. **An overseer that learns alongside the agents.** Every overseer in the AI Control set is fixed. Baumann and LIO learn, but they observe everything.

## 5. Templates we can borrow, and what each gives us

| Template | What it gives us |
|---|---|
| Griffin et al. 2024, AI-Control Games | **The overseer as a player in a partially observable game**; its remaining budget in its state; a noisy per-step signal; two evaluation modes; a frontier and its scalarisation; the reduction to weighted zero-sum games (a solver route) |
| Gasztowtt et al. 2024, Stackelberg-Markov game | **The solution concept for learners**: the overseer commits to a policy, which defines a game; the agents learn in it; the overseer is judged by what they learn |
| Baumann et al. 2020 | **The overseer as an extra player** with its own observation and actions on rewards; the switch-off test |
| Vinitsky et al. 2023 | The **opportunity set**: who can sanction whom right now. A k-budget overseer is an agent whose opportunity set has size k. Local vs public observation as an ablation |
| Du et al. 2022; Ma et al. 2024 | **How to make one formulation general:** a graph of agents, local neighbourhoods, a locality assumption stated as a definition, a measurable mismatch (ξ), short bounds on what locality costs, one method, each testbed one table row with ξ measured. The overseer becomes a node whose links to agents are chosen each step (k of them); several overseers are several nodes |
| Parisi et al. 2024 | Observing as something the system controls; a prediction that learners treat "unseen" as "unpenalised" |
| Engels et al. 2025 | Overseer capability as a measured axis (Elo) |
| Makins et al. 2026 | Per-agent vs pooled monitoring when harm is split across agents |
| Casheekar & Tangirala 2026 | Judge an inspection plan by where learning ends up; shared parameters change how much inspection is needed |
| Perolat et al. 2017 | A standard panel of social outcomes: welfare, equality, sustainability, peace |

## 6. What our existing results already say, in this light

Some findings are now predicted by the literature, which is good for credibility. Some are genuinely ours.

- **Predicted:**
  - Visible, predictable inspection gets dodged (our bot preview; Zhu, Gans & Holden).
  - Per-region overseers lose to pooled ones when agents crowd (Makins' fragmentation effect).
  - Switching the overseer off brings violations straight back (Baumann's switch-off test). Ours shows the overseer props up behaviour rather than changing what is learned.
- **Ours:**
  - The learners' behaviour is set by how often a violation is actually caught and sanctioned, whatever the number of inspections. This is a regularity about how learners respond to a k-of-N inspector.
  - An LLM inspector's misses are patterned (borderline cases), not random.
  - Learning agents gain less from crossing regions than scripted bots do.

## 7. What this points to for Phase 2 (a direction, not a decision)

The literature suggests a formulation built from three borrowed pieces and one new one:
- **Borrowed:** the overseer as a player in a partially observable game (Griffin); the Stackelberg-Markov timing with learning followers (Gasztowtt); Du's graph-and-locality structure for one or several overseers.
- **New:** the overseer's action includes **choosing what to observe** (which k agents, or a system-level signal), under an observation budget, with inspection quality that can be weaker than the agents. Harm can sit at the individual or the group level.

Phase 2 would write this down properly, compare it against the main alternatives (a learned overseer; a fully observing shaper; peer enforcement), and say what it predicts that they don't.

## 8. Reading list for you (from your matrix, prioritised)

Read these yourself first. They shape the definition most.
1. **Griffin et al. 2024, Games for AI Control.** It's not in the matrix; add it. The formal template.
2. **Parisi et al. 2024, Monitored MDPs** (to read). Observation as part of the problem; what learners do when unwatched.
3. **Makins et al. 2026, Multi-Agent AI Control** (to read). Per-agent vs pooled monitors; split harm.
4. **Vinitsky et al. 2023** (to read). Sanctioning opportunities and public sanction logs in a social dilemma with learners.
5. **Wang et al. 2019, green security games** (to read). The only overseer with a local view on a map against an adaptive opponent.
6. **Greenblatt et al. 2024, AI Control** (to read). Trusted monitoring with a weaker model; the honest/scheming evaluation.

Already read and central: Baumann 2020, Sun 2019, Gasztowtt 2024, Engels 2025, Casheekar 2026, SocialJax, OpenSocialJax, BazaarBench.

## 9. Caveats

- **Ma et al. 2024 (Nature MI)** was read from the open-access web page through a summarising fetch that cut off around Theorem 4. Its ξ estimates (about 0.2 and 0.8) and some testbed details are unverified. Worth reading the PDF properly.
- **The Willis 2025 paper in the folder** is the LLM iterated Prisoner's Dilemma paper, not the reward-exchange paper SocialJax cites.
- **Other unverified items:** what BazaarBench's `report_user` action does; agent counts in Vinitsky's Clean Up variant and in Perolat. These are marked in the notes.

## 10. Questions for you at this checkpoint

1. Does the design space in §2 match how you think about an overseer, or is an axis missing?
2. Of the gaps in §4, which feel like *your* question? My read: 1 + 2 + 3 together (an observation-budgeted, imperfect overseer facing learners who adapt to it) is the core, with 4 and 5 as the multi-agent extensions.
3. Should the June ideas (the capability gap Δ; local-pass/global-fail) come back as first-class parts of the definition?
