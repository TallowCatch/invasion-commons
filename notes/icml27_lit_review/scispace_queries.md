# SciSpace prompts for filtering the ICML reading list

Written 29 October 2026. They're for the 278 papers in the Zotero collection **ICML**. The aim is to screen them, pull the same facts from each one into a table, and find the gap.

## The question all of these prompts serve

> In a sequential social dilemma, can a central overseer that can inspect only **k of N agents per step** keep learning agents (RL or LLM) cooperating? How should it choose **whom to inspect** and **what consequence to apply**, and does it hold up when agents **learn to evade it**?

Paste this at the start of each SciSpace chat as context.

---

## Step 0: get the papers into SciSpace

1. In Zotero, right-click **ICML**, choose **Find Available PDFs**, and wait. Most items were saved as metadata only.
2. If SciSpace offers a Zotero import, connect it and pick the **ICML** collection only. Never pick Hiring-LLMs.
3. Otherwise, right-click **ICML**, choose **Export Collection…**, export as **BibTeX** or **RIS**, and upload that file. Upload any PDFs you have as well.
4. Screening (step 1) works on abstracts. Extraction (step 2) needs full texts, so run it on the papers you keep.

## Step 1: screening (keep, maybe or drop)

```
For every paper in this collection, decide KEEP, MAYBE or DROP for this question:
"In a sequential social dilemma, can a central overseer that inspects only k of N agents per step keep learning
agents (RL or LLM) cooperating; how should it choose whom to inspect and what consequence to apply; and does it
hold up when agents learn to evade it?"

KEEP if the paper does at least one of these:
(a) puts learning agents (RL or LLM) in a social dilemma or shared-resource game;
(b) has an outside party (planner, principal, regulator, monitor, overseer) that changes agents' rewards, actions
    or access;
(c) limits what that party can observe, or makes observing cost something (monitoring budget, partial
    observability, costly observation, audits);
(d) tests oversight against agents that adapt to or try to evade it (red team, attack policy, best response);
(e) defines metrics for whether a multi-agent intervention worked (welfare, sustainability, safety vs usefulness,
    cost).
MAYBE if it only gives background (one agent, no social dilemma, theory with no learning agents).
DROP if it meets none of these.

Return a table: title | year | KEEP/MAYBE/DROP | which criteria (a-e) | one-line reason.
Use only the abstract and full text; write "not stated" rather than guessing.
```

## Step 2: extraction table (run on KEEP papers)

Add these as **custom columns** in SciSpace's extraction table, or paste the block into chat.

```
For each paper, fill these columns. Quote the paper (with section or page) for columns 5-8; write "not stated"
if the paper doesn't say.

1. Environment: e.g. Harvest/Commons, Cleanup, Melting Pot, GovSim, matrix game, custom gridworld, text game
2. Agents: independent RL / centralised MARL / prompted LLM / LLM-written policy / scripted / human
3. Who intervenes: nobody / peers / central planner-overseer / contracts between agents / voting
4. Intervention: reward shaping / tax or subsidy / fine or sanction / exclusion or time-out / reputation /
   communication / other
5. What the intervener observes: everything every step, OR partial; is there a cost or budget for observing?
   (state k of N, % of actions, or cost per observation)
6. Targeting: who gets checked or intervened on: everyone / random / targeted by a score / learned
7. Adaptive agents: do agents learn or adapt against the intervener? Is there a red team or attack policy?
8. Calibration: is the intervention fixed by hand, chosen by search, or learned or adapted online?
9. Metrics: efficiency, equality, sustainability, peace, safety, usefulness, intervention cost, other
10. After removal: is behaviour tested after the intervention is switched off?
11. Main result in one sentence, with a number
12. Limitations and future work the authors state themselves
13. Relevance to my question: high / medium / low, and why
```

## Step 3: finding the gap (chat with the KEEP set)

Ask these one at a time.

1. `Which papers give the intervener (planner, regulator, overseer) only partial observation of the agents, or a cost or budget for observing? Quote where each says so. Which assume full observation of every agent every step?`
2. `Which papers test an intervener against agents that LEARN to evade or exploit it (e.g. cheat where it isn't looking)? What did they find?`
3. `Which papers report how much monitoring or intervention was used (number of checks, % of agents, cost), and which compare results at different budgets?`
4. `Which papers use LLM agents in a spatial or gridworld social dilemma (Melting Pot, SocialJax, OpenSocialJax), and do any of them include an overseer or sanctions?`
5. `Which papers check whether cooperation persists after the intervention is removed? What happens?`
6. `Which papers choose whom to inspect or sanction (random vs targeted vs learned)? Do any report that targeting backfires?`
7. `Collect the future-work and limitation statements about monitoring, enforcement, sanctions or institutions across all papers. Which open problems appear most often?`
8. `Build a grid: rows = observation (full vs budgeted/partial), columns = agents (fixed or scripted vs learning RL vs LLM), and a third split = environment (matrix vs spatial). Place each paper in a cell. Which cells are empty or have only 1-2 papers?`
9. `Which paper is closest to: "a budgeted central overseer, learning agents, a spatial sequential social dilemma, evaluated against agents that adapt"? What exactly does it do differently?`

**Tip:** for each answer, ask `Give the exact sentence and section for each claim.` AI summaries miss things and invent things, so check each claimed gap by opening the 2–3 closest papers yourself.

## Step 4: searches to fill holes (SciSpace search box)

Run these to find papers missing from Zotero. Add anything useful to **ICML**.

- `multi-agent reinforcement learning central planner partial observability monitoring budget sanctions`
- `sequential social dilemma regulator limited observation learning agents`
- `incentive design learning agents costly observation reinforcement learning`
- `principal agent reinforcement learning hidden actions multiple agents`
- `monitoring a subset of agents multi-agent reinforcement learning enforcement`
- `learned evasion of monitoring multi-agent reinforcement learning adversarial agents`
- `Stackelberg security game reinforcement learning adaptive attackers limited defender resources`
- `LLM agents gridworld social dilemma governance overseer`
- `Melting Pot sanctions norms enforcement agents`
- `AI control multi-agent monitoring budget distributed attacks`
- `cooperation persists after removing incentives multi-agent learning`

## Step 5: check before you trust the gap

```
I think the gap is: "[write your gap in one sentence]". Find the 5 papers in this collection (or in SciSpace
search) that come closest to filling it. For each, say exactly what it does and what it does not do relative to
the gap, quoting the paper. Be adversarial: try to show the gap is already filled.
```
