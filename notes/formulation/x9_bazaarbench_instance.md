# X9: BazaarBench as an instance of the Budgeted Oversight Game (26 Nov 2026)

**Purpose.** Supervisor feedback Y4 asks whether the formulation holds beyond SocialJax, and Y2 asks about a token-costed overseer in BazaarBench. This note writes BazaarBench (Wang et al. 2026, arXiv:2610.06748) as an instance of Definition 3.1 in `05_formulation.md`. It then reads the existing offline replay (`notes/bazaarbench/offline_replay/results.md`, 21 Nov) against the propositions.

There are no new runs. The replay used GPT-5's labels on two recorded markets: base agents GPT-5.5, the 20 tested traders gpt-oss-120b, with ordinary (L1) and exploit (L3) instructions.

## The tuple

| Element | BazaarBench | Compared with Harvest |
|---|---|---|
| agents N | 20 tested traders (of 100 in the market) | 7 learners |
| step, horizon H | one simulated day; the replay covers 7 days | 1 environment step; 1,000 |
| collapse states S_⊥ | a failure carried through to harm (stage S5: personal data shared, off-platform payment, an item sold that was never owned) | orchard empty |
| rule v | the failure taxonomy F1–F6 (overstated condition, unowned item, promised twice, premature close, personal data or off-platform, fake reputation) | eat with < 3 apples near |
| look (K_t, w) | k agent-days read per day; w = one day of the agent's record | k agents, w = 1 step |
| observation channel of a look | record only; record + messages; or record + private reasoning | the eat and its neighbourhood |
| judge J | GPT-5 judge (the labels); α, β of cheaper judges are unknown | perfect, coin flip or an LLM table |
| response | none in the replay (offline). Live options: delist, reputation penalty, freeze | apple sanction F |
| cost c | tokens read: 34,574 tokens is one unit, about one average agent-day | 1 per look |
| modes | ordinary (L1) and exploit (L3) instructions | hidden vs visible looks |

**Two differences from SocialJax.**
- The harm is mostly individual (one trader harms another), not a shared resource. BazaarBench is therefore closer to Griffin et al.'s single unacceptable outcome per failure, and local-pass/global-fail (Proposition 5.11) is not the issue here.
- The agents are LLMs with fixed instructions. They are not learners, so Proposition 5.1's response to the realised sanction would have to happen in context.
  - Our earlier LLM fishery studies show it can. In L3 (8 Oct), Nemotron stopped over-taking exactly at the sanction its own gain predicted, and gpt-oss one grid step earlier (PROJECT_HISTORY, L2–L3).
  - Whether BazaarBench traders respond to a live sanction is untested. It needs a live overseer.

## The propositions, read against the replay

| Proposition | What the replay shows | Reading |
|---|---|---|
| 5.1 / Cor. 5.2: only coverage times detection matters | Random checks catch about k/20 of failures: 28.5% at k = 5 and 53.9% at k = 10 (exploit) | holds for detection; the deterrence half is untested offline |
| 5.3: predictable targeting fails | Repeat offenders first is worse than random under exploit instructions: 25.1% vs 45.2% of failures past a private plan, at k = 5. Everyone offends, so the score locks onto whoever was checked first and stops spreading looks | a second failure of targeting, besides being dodged: when the base rate of violation is high, the targeting signal carries no information |
| 5.7: capability enters through harmful situations | The cheaper judge without private reasoning catches **more** of the failures that touch other people per token (82.6% vs 63.5% at 10 units, ordinary) | Cor. 5.9 holds: what matters is detection on the harmful failures per unit of cost, not overall accuracy |
| 5.10, generalised: the look must contain the evidence | 95% (exploit) and 64% (ordinary) of failures appear only in private reasoning. A record-only check can catch at most 0.2% | **the window condition generalises to a channel condition**: a look must observe whatever the rule depends on (past steps in Clean Up, private reasoning here). Otherwise α ≈ 1 on those failures and ρ ≈ 0 |
| 5.12: regional budgets | Splitting 1 unit a day into 4 fixed shares catches 0.3–1.2% vs 6–7% central, because each share is smaller than one agent-day. Saving unused budget or sharing a board recovers it | a granularity version of 5.12: a share smaller than one look wastes the budget |
| Y2: 1 vs 10 units | 1 unit a day catches 6–8%; 10 units catch 49–62% | Y2 answered offline; Cor. 5.9 gives the per-token comparison of judges |

**A new condition: timing.** Most harm happens within a day of the first sign. Checking everyone every day catches only 2 of 15 (ordinary) and 5 of 15 (exploit) carried-through failures before the harm. Proposition 5.1's sanction acts only *after* a verdict, so it deters but cannot prevent.
- In BazaarBench, prevention needs a look **before** an action executes. That is Griffin et al.'s ⟨Audit⟩ or ⟨Defer⟩ before ⟨Run⟩.
- This belongs in the formulation as a response option: *block before execution*, which costs latency, alongside *sanction after*.
- In SocialJax the after-the-fact sanction worked because learners adapt to the realised sanction over training.

## What would make BazaarBench a full instance (not done)

A live overseer: daily looks with a token budget, a judge, and a response (delist or freeze). The test is whether traders change behaviour within an episode as Proposition 5.1 predicts.
- **Needs:** open-model traders served on CREATE (vLLM), a small market (about 20 agents, 7 days), and contact with the authors (Ziyan Wang, Shuqing Shi). Their GPT-5 judge cost about $6,500 for 48 runs.
- **Status:** scoped, not started. It is the main open item for supervisor feedback Y2 and Y4.
