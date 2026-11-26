# Text-based LLM commons games, and where commons-game settings come from

Agent 4 of the commons-platform deep research, 14 Oct 2026. Source numbers in square brackets refer to files in `../sources/`. Numbers 60–79 are this agent's. 01, 09 and 10 are agent 1's files on Melting Pot `commons_harvest__open`, Perolat 2017 and Hughes 2018. I read Perolat and Hughes in full as well, and I agree with those files.

Depth labels. Full text was read for 60–63, 65–67, 71, 72, 75–79. Only the abstract or opening text was available for 64 (Bell & Madisetti), 68 (Alm, Jackson & McKee), 69 (Alm & Malézieux), 70 (Schelling), 73 (Ostrom, Walker & Gardner 1992) and 74 (Janssen et al. 2010, *Science*). Wherever this note gives details for those six, the details come from secondary sources that are named.

---

## 1. Short answer

- **Most commons games do not take their numbers from data.** Lab economists (Ostrom), DeepMind (Perolat, Hughes, Melting Pot) and the LLM papers (GovSim and its follow-ups) all pick their settings by hand. What separates the careful papers from the rest is not where the numbers came from. It is whether they then *show* that the chosen numbers make a real dilemma, and whether they report results under more than one setting.
- **There are three accepted ways to show that a game is a dilemma:**
  1. **By calculation.** Compute the selfish (Nash) outcome and the group optimum, and show that the gap is large. Ostrom's lab game does this: Nash is 64 tokens and earns 39% of the optimum, against an optimum of 36 tokens [71, 72]. SanctSim does it too (Nash 40, optimum 52) [62], and so does CoopEval, by proof [63].
  2. **By simulation, with an empirical payoff table or a Schelling diagram.** Train or program "restrained" and "greedy" policies. Play every mix of them. Then check the dilemma inequalities: everyone restrained beats everyone greedy, and greed or fear holds [66, 10, 09].
  3. **By pilot tests with people.** Tune the growth rate until it is "quick enough to be observed ... but sufficiently slow that the participants face a dilemma" [76].
- **GovSim does none of the three formally.** It sets capacity 100, doubling growth, 5 agents, 12 months and collapse below 5. Its only justification is "inspired by" Gordon and Ostrom, and it defines defection as taking more than the sustainable share [60, 61]. Most LLM commons papers inherit that set-up [64, 65].
- **Commons Harvest descends from a human experiment, but its numbers do not.** Melting Pot's code says the regrowth rule "was first described in Janssen et al (2010) and adapted for multi-agent reinforcement learning in Perolat et al (2017)" [01, 74]. The rule itself is inherited: an empty cell regrows faster the more full neighbours it has, and a patch with no neighbours never regrows. The probabilities are not inherited. Janssen used 0.01 × (full neighbours / 8) per second [75]. Perolat used 0 / 0.01 / 0.05 / 0.1 [09], Hughes 0 / 0.005 / 0.02 / 0.05 [10], and Melting Pot 0 / 0.0025 / 0.005 / 0.025 [01]. None of them gives a reason for its numbers.
- **The fishery maths is the one place with real data behind it.** The fisheries standard is logistic growth with maximum sustainable yield (MSY) at half the carrying capacity [78]. FishBase gives published ranges for the growth rate r, per year: high resilience 0.6–1.5, medium 0.2–0.8, low 0.05–0.5, very low 0.015–0.1. I checked these against a 2020 table that reproduces them [79].
- **Audit and fine levels have human-lab precedents:**
  - The canonical tax-evasion lab game uses a random audit with probability 0.1, a fine of twice the evaded tax and a 25% tax rate [67].
  - Quota-enforcement experiments in commons games use audit chances of about 10–20% with low fines ("weak"). They treat 50% as "theoretically deterrent" and 0% as a norms-only control [77].
  - The one field commons experiment run with an external regulator inspected 1 time in 16 per round. It found that the imposed rule *crowded out* cooperation [72].

---

## 2. LLM text commons games: how each is built and how its settings were chosen

| Game | Resource and rules | Agents, rounds, runs | How the settings were chosen | How the dilemma is shown |
|---|---|---|---|---|
| **GovSim** (Piatti et al., NeurIPS 2024) [60, 61] | Fishery, pasture or pollution, all "mathematically equivalent". Stock starts at 100, which is also the cap. The leftover stock doubles each month. Collapse below 5. Each month has three phases: plan/reflect, harvest (all at once; if demand exceeds the stock, units are handed out at random), then a discussion of up to 10 utterance steps, with a moderator who may reveal each agent's catch. | 5 agents, 12 months, 5 seeds, temperature 0. About 680k input and 124k output tokens per run; $0.3 (Haiku) to $30 (GPT-4) per run. | Qualitative only: "inspired by" fisher norms (Ostrom 1990, Gordon 1954). Doubling was chosen so the sustainable share is a round 10 per agent. The code comment reads "each day the fish double and cap at 100, so maximum 50 can be fished" [61]. | Not shown. Defection is *defined* as catching more than f(t). The metrics follow Perolat: survival time and rate, gain, efficiency, equality (1 − Gini), over-usage [60]. |
| **SanctSim / Corrupted by Reasoning** (2025) [62] | Public-goods game, not a harvest game. Each round, every agent chooses a sanctioning or a sanction-free institution. 20-token endowment; contributions multiplied by 1.6. Sanction budget of 20; a reward costs 1 to give +1, a punishment costs 1 to take −3. | 7 agents, 15 rounds, 5 runs per model. | Taken from a human paradigm: "Inspired by the experimental paradigm of Gurerk et al. (2006)". | By calculation: "full cooperation yields 52 tokens ... universal free-riding yields only 40 tokens (the Nash equilibrium)". The paper also sweeps the multiplier, the punishment cost and effect, and the endowment. |
| **CoopEval** (2026) [63] | Four normal-form games: Prisoner's Dilemma, Traveler's Dilemma, Trust Game, Public Goods. Four mechanisms: repetition, reputation, mediation, contracts. | Every model pairing in cross-play, 3 repeats each; more than 50,000 decisions. | Mechanism settings (k = 3, δ = 0.8) are set so a theorem guarantees cooperation: "As proven in Appendix D, these settings are comfortably sufficient to sustain cooperation". | By proof (Theorem 1, subgame-perfect equilibrium). Results are judged by mean cross-play payoff, by fitness under replicator dynamics, and by deviation ratings. |
| **Bell & Madisetti, "Restitution Over Deterrence"** (ICML 2026) [64] | Abstract only: "a standard common-pool fishing scenario" with a "greedy outsider". It is probably GovSim's fishery and its outsider perturbation, but that is not confirmed. | 4 model families, n = 8 seeds. | Not visible. | Not visible. Headline: restitution decided by majority vote (the sanctioned agent's catch goes back to the pool) is 2.7–4.3 times better. Ejecting the outsider, or delaying restitution, works less well. |
| **GovSim-SelfGovern** (Rehm, 2026) [65] | GovSim's doubling-to-capacity rule. Each agent needs 7 fish per round to survive. Three games: capacity held at 100, capacity falling to 65 (tight), and capacity falling to 56 (infeasible). Agents write laws as Python code, which is checked in a sandbox before a vote. | 5 agents, 12 rounds. | Arithmetic on supply and demand, so the three cases are feasible, tight or infeasible: "G3 is deliberately infeasible through harvest alone." | By feasibility arithmetic. It replaced GovSim's per-agent over-usage metric with a group-level overshoot, because the original "never exceeds 1.08% of harvest events, so it does not discriminate between arms". |

**What this means for Ameer.** In this literature the convention is to borrow GovSim's numbers or SanctSim's human-lab numbers, and then argue from feasibility or from the payoffs. Nobody uses ecological data. The stronger papers add two things: a check of the gap between the selfish and the optimal outcome [62, 63], and a sweep over the parameters [62]. A small code-versus-paper point: GovSim's over-usage threshold in the code is *per agent* (half the pool divided by the number of agents). SelfGovern describes it differently [61, 65].

---

## 3. The human-experiment roots

### 3.1 The Ostrom–Gardner–Walker CPR lab game [71, 72, 73]

- **Structure.** "The structure of Gordon's time-independent model (1954) has been used as the foundation" [71]. In Ostrom's own words, it is the simplest set-up "without losing crucial aspects of the appropriation problems facing harvesters in the field" [72].
- **Settings.** 8 subjects, each with 10 or 25 tokens per round. Each subject splits the tokens between a fixed-return outside market and the commons. The commons returns a share of a quadratic group production function, F(Σx), in proportion to what each subject put in. Subjects see only the group totals. Sessions earned $15–25 [71, 72].
- **Why these numbers.** They put the selfish outcome far from the optimum. Nash is 8 tokens each (64 in total); the optimum is 36 tokens in total; Nash earns 39% of the optimum [71, 72].
  - The coefficients usually quoted are F(X) = 23X − 0.25X² with an outside return w = $0.05. They fit the 36/64 figures exactly: the optimum is X = 18/0.5 = 36, and Nash is (8/9) × 72 = 64.
  - I did **not** see those coefficients printed in a primary text during this run.
- **Treatments (the 1992 "Covenants" design) [73, 71, 72].**
  - Communication only: one-shot or repeated, face to face.
  - Sanctions only: after 10 baseline rounds, a subject could "pay a fee in order to impose a fine on another subject".
  - Both together, and a further version where the group votes on its own sanctioning rule.
- **Results, as net yield (percent of optimum, after fees and fines) [71].**

  | Condition | Gross yield | Net yield |
  |---|---|---|
  | No sanctions | 21% | 21% |
  | Sanctions only | 37% | 9% |
  | Sanctions plus one chance to talk | 85% | 67% |
  | Talk plus a sanctioning rule the group voted for | 93% | 90% (only 4% defection) |

- **An external regulator with random inspection.** Cardenas, Stranlund & Willis (2000) imposed a cap on Colombian villagers with an inspection probability of 1/16 per round, "a low but realistic probability for monitoring rule conformance in rural areas". Withdrawals rose compared with communication alone; the imposed rule crowded out cooperation [72].

### 3.2 The spatial, real-time foraging commons (Janssen et al. 2010) [74, 75, 76]

This is the human version of Commons Harvest.

- **Science 2010 set-up (as described in [75]).**
  - 5 players on a 29 × 29 grid that does not wrap at the edges.
  - At the start, 25% of cells hold tokens (210 tokens).
  - Players move avatars with the arrow keys and press the space bar to harvest.
  - Each token pays $0.02. Decision periods last 4 minutes.
  - Each second, an empty cell grows a token with probability p = 0.01 × (number of the 8 neighbours holding a token) / 8. A cell with no full neighbours never regrows.
  - The best strategy is a checkerboard pattern.
- **Treatments (Science 2010).** No communication; costly fines; chat; and both [74]. The result: "costly punishment is again used but lacks a gross positive effect on resource harvesting unless combined with communication" [74].
- **E&S 2010 variant [76].**
  - 4 players on a 28 × 28 grid; 5 rounds of 4 minutes.
  - Punishment: paying 1 token removes 2 tokens from someone else, up to 50.
  - 5 minutes of chat before each of rounds 3–5.
  - Growth itself was a treatment: high growth starting at 50% density; low growth (half the probability) starting at 25%; growth that changes partway; growth that differs by place.
- **How the growth rate was set.** By pilot testing: "The parameter p is defined after a series of pre-tests in such a way that the renewal of the resource is quick enough to be observed by the participants, but sufficiently slow that the participants face a dilemma" [76]. This is the one explicit, citable statement of hand-tuning that I found.
- **Did Melting Pot or Perolat borrow from it? Yes for the rule, no for the numbers.**
  - Perolat: "quite similar to the dynamic game used in human behavioral experiments" [09]. Perolat also explicitly swapped the human-lab fine for a time-out beam, on the grounds that a beam "has an effect within the game itself, not just on the scores" [09].
  - The Melting Pot code docstring names Janssen et al. (2010) as the origin [01].
  - Each paper changed the regrowth probabilities (see section 1). Janssen's own data were later used as a benchmark in a competition to predict human behaviour with agent models [75]. That is a route for checking whether agent behaviour is realistic in a 2D commons.

### 3.3 What human CPR studies use across the literature (review of about 40 years) [77]

- **Group size.** "nearly always in the single digits".
- **Regeneration in dynamic games.** Examples include 1.05/1.15, 1.2/1.5 and 1.25/1.85. Doubling (2.0) also appears: Hine & Gifford 1996 used 2, varied between 1.25 and 2.5 in an uncertainty treatment. The review notes that "a higher replenishment rate corresponds to an 'easier' social dilemma". So GovSim's doubling has a human precedent, as a per-round multiplier.
- **Enforcement of quotas.**
  - Weak enforcement means "generally a 10 or 20% chance of being audited along with a rather low fine". Two of six studies found that this helped; four of six found that the benefit did not last.
  - Leibbrandt & Lynham (2018) compared 0%, 10% ("mild law, non-incentive compatible") and 50% ("theoretically deterrent") audit chances.
  - Peer fines: only one study in seven found substantial efficiency gains.
- **Advice on design.** "calibrate a baseline game with rather low cooperation, so that new effects are rather easy to detect".

---

## 4. Theory roots: Schelling diagrams and sequential social dilemmas

- **Schelling (1973).** The paper is about "binary choices with externalities". Its hockey-helmet example is the overseer argument in miniature: "It's foolish not to wear a helmet. But I don't because the other guys don't." [70] (I saw only the opening text.)
- **Leibo et al. (2017)** defined a sequential social dilemma (SSD) through empirical game-theoretic analysis [66]. The payoffs must satisfy R > P, R > S and 2R > T + S, plus greed (T > R) or fear (P > S). Here R is the reward when both cooperate, P the payoff when both defect, S the payoff to a cooperator facing a defector, and T the payoff to a defector facing a cooperator.
  - Cooperative and defecting policy sets come from "thresholding a continuous social behavior metric".
  - In Gathering, the cooperative policies were those trained in high-abundance, low-conflict-cost settings; the defecting ones came from scarce, high-cost settings. All four pairings were played to fill in the payoff matrix.
  - The parameters (apple respawn time, tag time-out) were swept, and each resulting game was classified as Prisoner's Dilemma, Chicken or Stag Hunt [66].
- **Perolat et al. (2017)** moved to Schelling diagrams because the 2-player method "cannot be applied directly to the case of N>2 players" [09]. They also introduced four group metrics: efficiency, equality, sustainability and peace. GovSim adopted the same suite [60].
- **Hughes et al. (2018)** gave the formal N-player test [10].
  - With R_c(ℓ) the average payoff of the ℓ cooperators and R_d(ℓ) that of the defectors, the game is a dilemma if all of these hold:
    - all cooperating beats all defecting: R_c(N) > R_d(0);
    - all cooperating beats being the exploited cooperator: R_c(N) > R_c(0);
    - and either fear or greed holds.
  - In Harvest, cooperation was *enforced* by stopping some agents from harvesting in low-density areas.
  - "The resulting empirical Schelling diagrams ... prove that our environments are indeed social dilemmas."
  - Fine beam: −1 to the user, −50 to the target [10].

---

## 5. Fishery bioeconomics: the one data-backed layer

- **Gordon (1954).** Under open access, effort grows until rent is gone (the "bionomic equilibrium"), because "No one owns the fish population, so no one has an incentive to protect it" [78]. I read this through Clark's 2013 essay, not the paywalled original.
- **Logistic growth and MSY.** Growth G(x) = rx(1 − x/K) is "the 'logistic' model". MSY is the peak of G, which is rK/4 at x = K/2 [78]. I did not reach Schaefer (1954) directly. The Schaefer production model is the core of today's CMSY/BSM stock-assessment tools [79].
- **Growth rate r.** FishBase's resilience-based prior ranges for r are high 0.6–1.5, medium 0.2–0.8, low 0.05–0.5 and very low 0.015–0.1, all **per year** [79]. They match the brief.
  - If a round in GovSim is a month, its doubling is far faster than any of these categories. Read as a per-round multiplier, it has precedent in human labs [77].

---

## 6. Where audit probabilities and fines come from

- **Theory.** Allingham & Sandmo (1972): compliance rises with the audit probability and with the fine. Their model predicts no change in behaviour after an audit that finds nothing [67]. I saw only the metadata of the 1972 paper.
- **The canonical lab default** [67]:
  - random audit probability 0.1, announced to subjects;
  - fine equal to twice the evaded tax;
  - tax rate 25%;
  - all "calibrated according to the considerations detailed in Alm (2019)".
  - With p = 0.1 and a 2× fine, the expected penalty is below the gain from evading, yet humans mostly comply. Around 65% of all reports in tax-evasion games are either zero or full compliance (Alm & Malézieux, quoted in [67]).
  - After an audit, people *lower* their compliance (the "bomb-crater" effect) [67].
- **Responses are real but small.** Alm, Jackson & McKee (1992): "Taxpayer reporting increases with greater audit and penalty rates; however, these responses are not large." [68]
- **Margins.** Alm & Malézieux's meta-analysis of 70 papers finds that audits and fines act differently on whether people cheat and on how much they cheat [69]. I could not read the range of audit rates across those 70 studies; check it before citing a range.
- **In commons games specifically** (sections 3.1 and 3.3):
  - 1/16 per round (field, external rule, crowding out) [72];
  - 10–20% for weak enforcement, 50% as deterrent, 0% as a norms-only control [77].

---

## 7. Hypotheses from the plan

- **H2 (settings are hand-tuned, then checked) is supported, with one refinement.**
  - MARL substrates hand-tune and check with Schelling diagrams [09, 10, 66].
  - LLM text commons hand-tune and mostly *do not* check [60, 64, 65].
  - Human labs either certify the dilemma analytically [71, 72] or tune it by pilot tests [76].
  - The refinement: Commons Harvest's *rule* does come from a human behavioural experiment [01, 74, 75], but its *numbers* do not.
  - No commons substrate I found cites ecological data for its parameters. Fisheries growth-rate ranges are the only data-based anchor [79].
- **H1 (the resource model is grounded; the evaluation protocol is not) is supported** for the logistic-plus-MSY resource [78, 79]. What a grounded study adds is set out in the checklist below.

---

## 8. Conventions a well-grounded commons study follows (checklist)

1. **Name where the resource model comes from.** Logistic (Schaefer) growth with MSY at K/2 [78], with r taken from a FishBase resilience category and the time step stated [79]. For a spatial game, cite Janssen's neighbour-dependent regrowth [75] and say which probabilities you use and why.
2. **Say openly how each free parameter was chosen.** Pilot-tuning is acceptable if stated: "quick enough to be observed ... sufficiently slow that ... they face a dilemma" [76]. Where you can, put each value inside the published ranges: group size in single digits, regeneration 1.05–2.5 per round [77], r in FishBase ranges [79].
3. **Prove the game is a dilemma under your settings.** Either:
   - (a) analytically: selfish equilibrium against group optimum, with the gap reported (as with Nash at 39% of optimum) [71, 62]; or
   - (b) empirically: run programmed "restrained" and "greedy" agents in every mix, and draw a Schelling diagram that checks R_c(N) > R_d(0), R_c(N) > R_c(0), and fear or greed [10, 66, 09].
   Do it for every game (Fishery, Forest, River) and for every setting you report.
4. **Calibrate the baseline so cooperation is low**, so an institution's effect can be detected [77].
5. **Use the classic factorial institution design.** No institution; communication only; sanctions or oversight only; both; and, where possible, an endogenously chosen institution [73, 71, 74, 62]. Expect sanctions alone to fail and sanctions plus communication to work, as in humans [71, 74].
6. **Ground the audit and fine numbers in human-lab values.**
   - Random audit p ≈ 0.1 (tax default), fine = 2× evaded [67].
   - Commons quotas: 10–20% weak, 50% deterrent, 0% norms-only [77].
   - Field realism: 1/16 [72].
   - Report the expected fine against the gain from cheating, so it is clear whether the audit is deterrent "in theory".
7. **Test the known human departures from the deterrence model.**
   - Over-compliance under non-deterrent audits [67].
   - Change in compliance after an audit (bomb-crater) [67].
   - Crowding out of cooperation by an imposed rule [72].
   - Random against targeted (endogenous) audits [67, 69].
8. **Report a suite of social-outcome metrics, not one number.**
   - Efficiency relative to the optimum, *net of enforcement costs* (net yield) [71].
   - Equality (1 − Gini), sustainability or survival time, over-usage [09, 60].
   - Check that each metric separates conditions [65].
   - Report both how many agents cheat and how much they take [69].
9. **Sweep the key parameters and show the conclusions hold** (multiplier, punishment cost and effect, endowment in SanctSim) [62]. Also treat resource growth itself as a factor (high, low, changing) [76, 77].
10. **Use enough seeds and report variance.** GovSim used 5, Bell & Madisetti 8, CoopEval 3 repeats over many pairings. State the decoding temperature [60, 62, 63, 64].
11. **Evaluate against mixed populations when agents are LLMs.** Use cross-play, focal against background agents, and, where relevant, replicator or "fitness" analysis [63]. Include an adversarial or greedy outsider as a stress test [60, 61, 64].
12. **Report cost per run in tokens and dollars** [60] and release configs and seeds [61, 65].
