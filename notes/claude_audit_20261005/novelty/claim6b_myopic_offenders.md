# Novelty check, claim 6b: LLM agents as myopic offenders (one-round gain predicts the stopping fine)

Date of search: 8 Oct 2026. Tools: WebSearch (standard and extended), WebFetch of arXiv abstract and HTML pages, publisher and RePEc pages. The Semantic Scholar and arXiv APIs could not be reached from the shell (rate-limited or blocked), so coverage depends on the web-search index. Google Scholar was not queried directly.

## The claim under test

In a 6-fisher, 20-round fishery with overseer-set allowances, a 1/6 per-round audit chance per fisher, and a fine F when caught over-taking, LLM agents (gpt-oss-120b, Nemotron 3 Super) act like myopic offenders:
- the fine at which they stop over-taking is predicted by the one-round gain from over-taking (about 4.5 t, so predicted stopping F = 30). The prediction was exact for one model and one grid step early for the other;
- the whole-game net gain does not predict it (negative for gpt-oss, 0.7 t per over-take round for Nemotron);
- over-taking falls in a graded way, starting at about 0.4 times the one-round gain;
- implication: overseers of LLM agents should size penalties to the one-step gain.

Depth legend: **M** = metadata only; **A** = abstract page read; **F** = full text (HTML) read, at least the sections named; **S** = search-engine snippet only (page not opened, or the PDF could not be parsed). Quotes are short and exact as returned by the fetched page.

---

## 1. LLM myopia / short-termism in commons and social dilemmas

### 1.1 Piatti et al., "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents" (GovSim), arXiv:2404.16698 (v3 HTML). Depth: F (main text, first ~100k chars).
- **Shows:** most LLM agent societies fail to sustain the shared resource. The authors put the failure down to an "inability to formulate and analyze hypotheses about the long-term effects of their actions" (abstract) and an "inability to mentally simulate the long-term effects of greedy actions" (Sec. 1). Universalization prompting helps.
- **Does NOT show:** no fines, audits or overseer; the only consequence of over-use is depletion. The paper does not use the word myopia, does not model discounting, and makes no deterrence or penalty-threshold analysis.
- **Relevance:** the main prior source for "LLMs neglect the long-run effect of extraction on the stock." That covers part (a) qualitatively.

### 1.2 Bracale Syrnikov et al., "Draining the Energy Commons: Self-Defeating Over-Appropriation as a Coordination Failure in Agentic LLM Collectives", arXiv:2607.22188 (v2, Aug 2026). Depth: F (main text).
- **Shows:** LLM prosumers (GPT-5.4-mini, Gemini-3.1-flash-lite, Grok-4.3) sharing a renewable reserve over-appropriate beyond a demand threshold. The abstract says they "protect current service while undermining future service" and "behave like impatient optimisers" at the level of the public trajectory. Sec. 1 says: "We describe this as myopic behaviour in the operational sense of a horizon mismatch." Self-interested decisions "give too little weight to their future consequences."
- **Does NOT show:** no penalties or enforcement. Sec. 3.4 says: "There is no communication, reputation, entitlement balance, trading, or governance channel." The authors explicitly do not infer a discount factor ("not to infer an internal discount factor", Sec. 5), and they do not separate planning failure from discounting.
- **Relevance:** the closest prior statement of (a). It already uses "myopic" and "impatient" for LLM collectives draining a shared stock. Our paper must cite it and must not claim (a) as new.

### 1.3 Jung et al., "SODE: Analyzing Social Dynamics in LLM Agents", arXiv:2605.23949 (May 2026). Depth: F (sections located).
- **Shows:** in iterated Prisoner's Dilemma settings, reasoning models "prioritize short-horizon optimization, destabilizing long-term cooperation" (abstract). The conclusion calls this "hyper-rational myopia, prioritizing immediate rewards over long-term trust." A long-horizon framing helps.
- **Does NOT show:** no resource stock, no fines, no penalties (payoffs are the standard R/P/T/S points).
- **Relevance:** prior use of "myopia" for LLM agents in repeated games, about trust and reciprocity rather than the stock.

### 1.4 LLM time-preference studies (questionnaire-style, not commons)
- Mazyaki et al., "Temporal Preferences in Language Models for Long-Horizon Assistance", arXiv:2509.09704 (Sep 2025). Depth: A. Measures future versus present orientation on intertemporal-choice tasks and defines a "Manipulability of Time Orientation" metric. No game, no penalties.
- Rios-Sialer et al., "Temporal Preference Concepts and their Functions in a Large Language Model", arXiv:2606.05194 (v2 Jul 2026). Depth: A. Reports that "unintervened LLMs discount the future several times less steeply than humans," but "this preference is unstable across contexts." No game, no penalties.
- **Relevance:** if anything, these cut against a simple "LLMs are impatient" story when choices are stated directly. That fits a framing of horizon neglect in agentic play (failing to account for the stock) over steep discounting. The paper should not say the models "discount the future" in the time-preference sense.

### 1.5 Other commons follow-ups checked (abstract level)
- Borah, "Bosses, Kings, and the Commons" (SovSim), arXiv:2605.29062. Depth: A. Studies power asymmetry and finds "severe breakdowns in cooperation and sustainability." No fines, audits or myopia analysis in the abstract.
- Guzman Piedrahita et al., "Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games", arXiv:2506.23276 (COLM 2025). Depth: S (the PDF could not be parsed; the details come from the search summary). Peer-to-peer costly sanctioning in a public-goods game. The sanction is chosen by agents, not a fixed fine against a measured gain. Not verified beyond the snippet.
- "From Certain Doom to Survival" (GovSim-SelfGovern), arXiv:2609.22600. Depth: S. Agents write executable governance rules. Not opened.

## 2. LLM agents and deterrence, penalties, audits

### 2.1 Okamoto, Erol & Erol, "Why Do AI Agents Break Rules? How Framing, Context, and Social Signals Shape Compliance", arXiv:2608.12323 (v2 Aug 2026). Depth: F (main text plus appendix portion).
- **Shows:** twelve models acting as procurement chatbots, with four enforcement levels (Sec. 3.3: No Fine; Small $2,400 "Unlikely" audit; Medium $4,800 "Possible"; Large $7,200 "Likely"). The paper uses Becker as background and states qualitative gain-versus-expected-penalty predictions (Small fine: "Expected risk is below the compliance premium; a rational optimizing agent should violate"). The headline is a penalty paradox: stating a penalty can "turn a legal obligation into a cost-benefit calculation that favors violation." Appendix B.11 has a model reasoning "even with the $2,400 fine we still come out ahead."
- **Does NOT show:** a single-shot decision, so there is no stock, no repeated game and no one-round versus whole-game gain. Myopia is not mentioned. Audit probabilities are qualitative labels only. It does not predict the stopping fine from a measured gain and does not fit a dose-response curve; compliance is non-monotonic in fine size.
- **Relevance:** prior evidence that LLM agents weigh fines against gains (cost-benefit). Does not touch the horizon question.

### 2.2 Gans & Holden, "When Does Randomized Oversight Align AI Agents That Can Conceal?", arXiv:2609.38262 (Sep 2026, econ.TH). Depth: F (main text plus appendix).
- **Shows:** theory. "misconduct is deterred when its expected sanction exceeds its gain" (Introduction). With t = pF, deterrence needs "expected sanctions must cover the largest private gain on each channel" (Sec. 4.1). Concealment raises the required intensity.
- **Does NOT show:** static, single-agent and single-opportunity. No discounting, myopia or repeated play ("studies a single agent with correct beliefs", Conclusion). No LLM experiments, though it says its comparative statics "are testable."
- **Relevance:** supplies the pF ≥ gain benchmark our test uses, but leaves open which gain (one-period or whole-horizon) the agent uses. Our result answers that question empirically.

### 2.3 Ye & Steinhardt, "Norm Enforcement for AI Agents: Robustly Shaping Behavior in Multi-Agent Systems", arXiv:2607.09766 (Jul 2026). Depth: F (main text and Appendix A; Appendix F not reached).
- **Shows:** LLM agents (Qwen3-Next-80B; GPT-OSS-120B ablation) in three environments including a fishery commons, where aggressive harvesting "raises near-term reward at the expense of regenerative capacity" (Appendix A). Proposes reputation-based and escalating-penalty enforcement that resists exploitation.
- **Does NOT show:** the sanction is removal or zeroing rewards, or reputation, not a graded fine. In the text read there is no sweep of penalty size against violation rate, no threshold tied to the agent's gain, and no myopia analysis. The penalty-factor ablation (App. F.3) was not read: **not verified**.
- **Relevance:** closest setting (LLMs, fishery, enforcement, gpt-oss-120b). Different question.

### 2.4 Others checked
- Lazebnik & Shami, LLM+DRL tax-evasion simulation, arXiv:2501.18177. Depth: A. "Enforcement probabilities" matter, and enforcement plus public goods together are needed. Nothing on thresholds against gain, and nothing on myopia.
- Wang et al., "Law in Silico", arXiv:2510.24442. Depth: A. LLM agents reproduce macro crime trends. The search summary says perceived punishment severity is varied on a 0–5 scale; that detail is not verified from the abstract. Nothing relating the penalty to a measured gain.
- Yang et al., "Paying for Honesty Without Knowing the Truth: Reputation-Penalty Design for LLM Marketplace Agents", arXiv:2607.28330. Depth: A. LLM merchants "fabricate when lying is free but restrain themselves when fabrication costs them sales." No per-period gain sizing and no myopia.
- Bracale Syrnikov et al., "Institutional AI", arXiv:2601.11369. Depth: A. Sanctions via a governance graph in Cournot collusion. No gain-sized fines in the abstract.
- Wang, "Why LLM Agents Collapse Without Oversight", arXiv:2609.15293. Depth: A. Separates detection from enforcement probability. No fines against gain and no myopia.
- penseralabs/becker-agents (GitHub). Depth: M. Described as "varies audit probability and penalties ... and estimates deterrence curves." No results, README or paper visible. Unpublished; **not verified**. Worth monitoring, since a deterrence-curve paper could appear.

## 3. Economics: myopic offenders and deterrence

### 3.1 Lee & McCrary, "Crime, Punishment, and Myopia", NBER Working Paper w11491 (July 2005). Depth: A (NBER page).
- **Shows:** longer sentences deter "only if offenders' discount rates are relatively low." The small drop in offending at age 18 implies "potential offenders are extremely impatient, myopic, or both."
- Published version: Lee & McCrary, "The Deterrence Effect of Prison: Dynamic Theory and Evidence", in *Regression Discontinuity Designs: Theory and Applications*, Advances in Econometrics vol. 38 (Emerald, 2017), DOI 10.1108/S0731-905320170000038005. Depth: A (Emerald page). It is a "stochastic dynamic extension of Becker's (1968) model." The published abstract does not use the word "myopia."
- **Relevance:** the origin of the "myopic offender" term. Note the mechanism is different. There the gain is immediate and the punishment is delayed (prison years), so myopia weakens deterrence. In our game the fine is immediate and it is the cost of depleting the stock that is delayed and shared. So in our game myopia makes the immediate fine more effective than a forward-looking agent's whole-game calculation would imply.

### 3.2 Polinsky & Shavell, "On the Disutility and Discounting of Imprisonment and the Theory of Deterrence", J. Legal Studies 28(1):1–16 (1999), DOI 10.1086/468044 (IDEAS/RePEc). Depth: A.
- **Shows:** how "discounting of the future disutility and future public costs of imprisonment" changes deterrence theory.
- **Not relevant to** LLMs or commons. Background only.

### 3.3 Present bias and crime (background; depth S unless stated)
- McAdams, "Present Bias and Criminal Law", U. Illinois Law Review 2011(5). Depth: S. Argues deterrence can be strengthened by making deferred costs immediate. Not opened.
- Nagin & Pogarsky, "Time and Punishment: Delayed Consequences and Criminal Behavior", J. Quantitative Criminology (2004). Depth: S. Not opened.

### 3.4 Fisheries enforcement
- Sutinen & Andersen, "The Economics of Fisheries Law Enforcement", Land Economics 61:387–397 (1985). Depth: S (secondary citations only; the abstract was not opened). It merges Gordon-Schaefer with Becker. As cited, a rational operator sets catch above quota where "marginal profits equal the expected marginal penalty" (from a secondary source). That is a per-period static deterrence rule.
- Nøstbakken, "Fishermen's Compliance: A Dynamic Model", SNF Working Paper 45/06. Depth: S (the PDF could not be parsed). The search summary says fishers act period by period while the regulator takes a long view. **Not verified** from the document.
- Akpalu (2010), J. Agricultural and Resource Economics, dynamic mesh-size compliance model. Depth: S. Not opened.
- Chávez, Murphy & Stranlund, "Co-enforcement of Common Pool Resources: Experimental Evidence from TURFs in Chile", ESI Working Paper 19-18 (2019). Depth: A. Compares "expected marginal penalty" with "marginal gain from poaching," and finds "poaching levels were not sensitive to changes in monitoring levels and sanctions." This is human evidence that the gain-anchored Becker benchmark can fail. It is a useful contrast: our LLM agents track the one-round benchmark closely.
- Coelho, Filipe & Ferreira, "Sketching a Model on Fisheries Enforcement and Compliance — A Survey", arXiv:2306.16960. Depth: A. Becker plus Gordon-Schaefer survey; nothing about myopia in the abstract.

### 3.5 Standard theory (not opened; textbook knowledge)
Becker (1968) gives the expected-penalty-versus-gain condition. The one-shot deviation principle in repeated games says a deviation is deterred if it does not pay given the continuation. For a fully myopic agent, the continuation term drops out, leaving pF ≥ one-period gain. So the rule "a myopic agent stops when p·F ≥ one-period gain" is a direct consequence of standard theory. It is not a new theoretical result. What is new is the empirical test on LLMs, and the finding that the one-period version, not the whole-game version, fits.

## 4. Has anyone anchored an AI agent's compliance threshold to its own measured gain?
None found. Okamoto et al. (2.1) state qualitative gain-versus-expected-fine predictions but do not measure the agent's gain from its own play or predict a stopping fine; compliance was non-monotonic. Gans & Holden (2.2) propose that such tests are possible but run none. The becker-agents repo (2.4) claims to estimate deterrence curves but shows no results. Human CPR lab and field work (Chávez et al. 2019) compares expected marginal penalties with marginal gains, but with humans and set by design, not a measured per-agent gain.

---

## Verdicts

**(a) "LLMs in a commons are myopic about the shared stock."** *Partly shown, close to known.* GovSim (arXiv:2404.16698) attributes failure to not reasoning about "long-term effects." Draining the Energy Commons (arXiv:2607.22188) explicitly calls LLM collectives "myopic" and "impatient optimisers" draining a shared reserve. SODE (arXiv:2605.23949) uses "hyper-rational myopia" in repeated PD. None of them measure it through a deterrence threshold or under fines. Do not claim (a) as novel. Present our evidence as a new, quantitative behavioural test of it.

**(b) "The one-round gain, not the whole-game gain, predicts the fine at which LLM agents stop over-taking."** *No prior work found in this search.* Becker and Gans & Holden supply the pF ≥ gain condition, and Lee & McCrary supply the myopic-offender concept for humans. No paper found tests which horizon's gain predicts an LLM agent's stopping penalty, or measures that gain from the agents' own play. This is the novel element. Caveats to state: two models, one game, a fine grid with a step size (one model is one grid step early), and the whole-game gain is a counterfactual estimate.

**(c) "Graded deterrence starting at about 0.4 times the one-round gain."** *No prior work found for LLMs.* Graded, non-step responses to sanctions are routine in human data, and Okamoto et al. found non-monotonic LLM compliance. Nobody found reports an LLM dose-response normalised to the agent's measured gain. Novel as a measured quantity, but modest. Describe it as observed in these two models, not as a general law.

**(d) Practical implication: size penalties to the one-step gain.** *Partly shown.* Sizing expected sanctions to the gain is standard Becker and Gans & Holden. The specific advice that, for LLM agents in a dynamic commons, the relevant gain is the immediate one was not found. Note the double edge. A one-step-sized fine is enough to stop these agents, but the agents do not internalise the stock cost. Without enforcement they will deplete the stock even when over-taking does not pay over the game. Advice should be conditional ("in our setting", "for these models").

## Safest accurate wording for the paper

> "Consistent with earlier reports that LLM agents neglect the long-run effects of extraction (Piatti et al., 2024) and behave like impatient or myopic optimisers in shared-resource settings (Bracale Syrnikov et al., 2026), our agents responded to fines as myopic offenders in the sense of Lee and McCrary (2005). The fine at which they stopped over-taking matched the one-round gain from over-taking (expected fine = gain at F ≈ 30; exact for Nemotron, one grid step early for gpt-oss). It did not match the whole-game net gain, which was negative for gpt-oss and small for Nemotron. Over-taking fell gradually, starting at expected fines of about 0.4 times the one-round gain. To our knowledge, this is the first test of which horizon's gain predicts the penalty at which LLM agents comply. For the two models and the game studied here, it suggests that overseers should size expected penalties to the immediate gain from a violation. It also suggests that they should not rely on the agents to weigh the longer-run cost to the shared stock."

Avoid: "LLMs discount the future" (direct time-preference studies find the opposite, arXiv:2606.05194); "first to show LLMs are myopic in commons"; "deterrence threshold is a general property of LLMs."

Check before submission: confirm which model was exact and which was one step early (the claim text given to me says "exact for one model and one grid step early for the other"; the commit log says exact for Nemotron). Re-search for a write-up from the penseralabs/becker-agents project and for Ye & Steinhardt App. F.3 (penalty-factor ablation), which this search could not read.
