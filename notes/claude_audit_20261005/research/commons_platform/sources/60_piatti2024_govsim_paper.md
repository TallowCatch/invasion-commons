# Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents (GovSim)

- **Authors:** Giorgio Piatti, Zhijing Jin, Max Kleiman-Weiner, Bernhard Schölkopf, Mrinmaya Sachan, Rada Mihalcea
- **Year:** 2024
- **Venue / identifier:** NeurIPS 2024; arXiv:2404.16698 (v4 read)
- **URL:** https://arxiv.org/abs/2404.16698 (HTML full text: https://arxiv.org/html/2404.16698)
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full HTML text read)

## Verbatim quotes

1. Resource model and the only stated grounding for it:
   > "The first is inspired by empirical work on understanding the norms that emerge in communities of fishermen that prevent overfishing [ 56 , 25 , 43 ] ."
   > "The lake supports up to 100 tons of fish, and the fish population doubles at the end of the month up to this capacity. For example, five fishermen can sustainably catch up to 10 tons of fish each per month, but if the total amount they catch exceeds 50 tons, the population will start to decrease."
   (Refs: [56] Ostrom 1990 *Governing the Commons*; [25] Gordon 1954 JPE; [43] Levine et al. 2020 PNAS on universalization.)
2. Round structure and collapse rule:
   > "At the end of the month, the remaining shared resources double (capped by 100). When h(t) falls below C=5 the resource collapses and nothing else can be extracted."
3. Sustainability threshold (group-level in the paper):
   > "This threshold represents the maximum resources that can be extracted at time t without diminishing the resource stock at time t+1 , considering the future resource growth multiplier g ."
   Formula: f(t) = max({x | g(h(t) − x) ≥ h(t)}).
4. Metrics borrowed from Perolat:
   > "We follow Perolat et al. [61] in defining a suite of metrics since in a mixed incentive repeated game like GovSim , no single scalar metric can track the entire state of the system."
   Metrics: survival time m, survival rate q (= share of runs with m = 12), total gain R_i, efficiency u = 1 − max(0, T·f(0) − ΣR)/(T·f(0)), equality e (1 − Gini), over-usage o (share of actions with r_i^t > f(t)).
5. Seeds and decoding:
   > "Each simulation was repeated with five random seeds."
6. Cost:
   > "the estimates are based on 680k input tokens and 124k output tokens. For each experiment, we perform 5 runs, so the total costs need to be multiplied by 5."
   Table 3 estimates per 12-month run: Claude 3 Haiku $0.3, Sonnet $4, Opus $20; GPT-3.5 $0.5, GPT-4 $30, GPT-4-turbo $11, GPT-4o $5.

## Other design facts (paraphrased from the text)
- Phases per month (Appendix B): Strategy/reflection at "home", Harvesting, Discussion at a "town hall"; harvesting is simultaneous ("concurrent harvesting") and, when demand exceeds supply, units are allocated one at a time at random.
- A moderator can disclose each agent's previous catch ("transparent harvesting reporting").
- Ablations: no communication (over-usage rises), universalization prompt (+4 months survival), greedy newcomer perturbation.
- Fifteen LLMs; temperature 0.

## Relevance
GovSim's numbers (capacity 100, doubling, 5 agents, 12 months, collapse at 5) are not derived from any fishery data or human experiment; the paper cites Gordon and Ostrom only as qualitative inspiration and chooses a doubling rule so the sustainable share is a round 10 per agent. It does not run a Schelling-diagram or equilibrium check to show the game is a dilemma; it treats "harvest more than f(t)" as defection by definition. This is the template most LLM commons papers (62, 64, 65) inherit.
