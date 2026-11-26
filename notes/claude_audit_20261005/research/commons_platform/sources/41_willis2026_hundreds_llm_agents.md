# [41] Evaluating Collective Behaviour of Hundreds of LLM Agents

- **Authors:** Richard Willis, Jianing Zhao, Joel Z. Leibo, Yali Du (author order as in the HTML; arXiv metadata lists Willis, Zhao, Du, Leibo)
- **Year:** 2026 (v1 18 Feb 2026; v2 2 Jul 2026)
- **Venue:** Strategic Engineering Workshop on LLMs and Game Theory (SE@AAMAS2026), 25 May 2026, Paphos. arXiv:2602.16662v2 [cs.MA], licence CC BY-SA 4.0
- **URL:** https://arxiv.org/abs/2602.16662 (HTML full text: https://arxiv.org/html/2602.16662)
- **Code:** https://github.com/willis-richard/emergent_llm (see [40])
- **Source type:** workshop paper, full text read (HTML)
- **Date accessed:** 14 Oct 2026

## Verbatim quotes

**Motivation and unit of oversight (Introduction).**
> "Decisions also occur too quickly and too often for users to approve individual actions, so the natural unit of human oversight is the policy: users instruct their agent at a high level and it executes autonomously."

**Why fixed strategies.**
> "Second, fixed strategies enable pre-deployment behaviour checking: developers can read, test, and reject dangerous algorithms. Finally, querying an LLM once rather than each round substantially reduces the number of API calls required, enabling scaling to much larger groups."

**Anonymity (Section 3).**
> "To represent anonymity, each player observes only the aggregate cooperator count from the previous round, $n_{c}^{t-1}$ , rather than individual actions."

**Welfare (Section 3).**
> "We measure the total reward achieved by all players in a game, which represents the social welfare under a utilitarian metric ... For each game, we identify the action profiles that minimise and maximise $U$ , which we use as reference points to normalise welfare onto a $[0,1]$ scale."

**CPR definition (Section 3).**
> "Actions: Restrained extraction ( $C$ ) or intensive extraction ( $D$ )."
> "Payoffs: $\pi_{i}^{t}(a_{i}^{t};S_{t})=\tfrac{S_{t}}{2n}\bigl(1+\mathbf{1}_{a_{i}^{t}=D}\bigr)$ ."
> "growth gives $S_{t+1}=\min(S_{t}^{\prime}+2S_{t}^{\prime}(1-S_{t}^{\prime}/k),\,k)$ ."
> "The state $S=0$ is absorbing: a single round of universal defection at full stock depletes the resource permanently."
> "Welfare bounds: $U$ is maximised when all players cooperate for the first $r-1$ rounds, preserving the stock, and then all defect in the final round, fully extracting the stock."
CPR citations given: "(Levhari and Mirman, 1980; Ostrom et al., 1994; Gordon, 1954)".

**Parameter justification (Section 3, "Experimental parameters") — the only justification given.**
> "Throughout our experiments, we use $r=20$ rounds, to be long enough for conditional strategies to express their behaviour, and to align with protocols used in literature (Fehr and Gächter, 2000). We set $k=2$ for PGG and $m=\lceil n/2\rceil$ and $k=2$ for CRD. For the CPR, $k$ enters the payoff and dynamics only as a multiplicative scale on $S_{t}$ , so it acts as a unit of reward and does not affect welfare efficiency; we set $k=4n$ to keep numerical values convenient."
No justification is given for the regrowth rate 2, the 1/(2n) vs 1/n extraction shares, or the logistic form beyond the citations above.

**Strategy generation (Section 4.1).**
> "each attitude is elicited through a set of four synonyms, with equal numbers generated per synonym: Collective $=\{$ altruistic, benevolent, collective, prosocial $\}$ and Selfish $=\{$ individualistic, opportunistic, self-interested, selfish $\}$ ."
> "Implementations failing either check are discarded and regenerated from the same natural-language description, so the filtering step removes execution errors rather than reshaping the strategy distribution."
> "We allow strategically incorrect logic provided the code runs"
> "We select three state-of-the-art LLMs: Claude Haiku 4.5 (Claude), Gemini 3.1 Flash Lite (Gemini) and GPT-5.4 Mini (GPT), and generate 512 strategies per model and attitude (128 of each synonym) for each game. This results in a total of 9216 strategies, so we use low reasoning effort to control costs."

**Fingerprinting (Section 4.2).**
> "We generate feature vectors by evaluating each algorithm’s action choices in response to all possible opponent histories in a four-player game lasting seven rounds."
> "This yields a $\sum_{r=0}^{6}4^{r}=5461$ -dimensional feature vector per algorithm."
Metrics: normalised mean pairwise distance MPD*, standardised centroid distance Δ, participation ratio PR.

**Self-play robustness (Section 4.3).**
> "For each model and game, and for each group size $n\in\{4,16,64,256\}$ , we sweep over compositions $(n_{c},n_{s})$ ... We repeat this sampling 200 times per composition."
> "We report welfare efficiency $\widetilde{U}\in[0,1]$ , defined as $U$ rescaled so that 0 corresponds to the minimum total reward achievable in the game and 1 to the maximum"

**Cultural evolution (Section 4.4).**
> "We model a population of 512 users running AI assistants under minimal supervision"
> "We set $\beta=1$ , placing the dynamics in a moderate-selection regime"
> "Each gene (either LLM or attitude) is independently replaced with probability $0.25\%$ by a uniformly random alternative."
> "We run each simulation for $G_{\max}=2000$ generations." (the repo script uses 1000; see [40])
> "We perform 100 independent runs per (game, group size) and aggregate per-genotype frequencies and the mean welfare efficiency over the final 100 generations across runs."

**Key CPR results.**
> "Surprisingly, despite the CPR’s dynamic state offering an extra conditioning variable, CPR strategies have lower participation ratios, suggesting they condition on fewer features. A qualitative inspection suggests this is because the models primarily base their decisions on the current stock levels remaining, rather than the actions of their opponents"
> "However, as the group size increases, its performance degrades significantly: with $n=64$ , even a group consisting entirely of Collective strategies can only achieve a welfare efficiency of 50%." (Gemini, CPR)
Table 2 (β=1, G=4), CPR welfare efficiency: 91(0)% at n=4, 15(0)% at n=64; Collective frequency 56(2)% at n=4, 19(1)% at n=64.
> "For all games, with $n=64$ group size, the Selfish attitude dominates, and poor social outcomes are realised."
Appendix C: results "not sensitive to $G$ , but are sensitive to the selection strength".

**Conclusion: institutions named as the next step.**
> "This indicates a need for cooperation-supporting institutions – reputational mechanisms, communication channels, or structural constraints on agent action – in deployments that resemble the larger-group cases."
> "Investigating which mechanisms transfer to LLM populations is a natural next step"

**Limitations (stated).**
> "Binary-action games, fixed strategies, and anonymous interaction were deliberate modelling choices that enabled exhaustive behavioural analysis, pre-deployment inspection, and scaling to large populations, but they constrain generality."
> "behavioural fingerprints are computed at $n=4$ , $r=7$ to keep the trajectory space enumerable, and may not fully reflect strategy behaviour at the larger group sizes and horizons used in our experiments."
> "The cultural evolution model assumes users observe peers’ payoffs and imitate successful setups; it omits brand loyalty, switching costs, and any non-payoff component of user preferences."

**Compute (Appendix D).**
> "Generating the strategies takes 1-2 days, depending on the API rate provided to the user. Computing the results provided in the paper takes a few hours for each of the three methods of analysis."

## Relevance
The paper explicitly calls for "cooperation-supporting institutions" and leaves them untested, which is the gap an audit-and-fine overseer fills. Its CPR is a binary, anonymous, 20-round logistic commons with no external enforcer; its welfare-efficiency, self-play and cultural-evolution protocols can be reused as the evaluation harness for an overseer.
