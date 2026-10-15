# Literature review (AI / ML only): audits, fines and oversight in LLM commons games

Compiled 13 Oct 2026. Every entry below was checked against its arXiv abstract page, proceedings page or publisher record during this session. Venue is given only where a page confirmed it; otherwise the arXiv ID is the reference.

Project in one line, for reference: an overseer ("reviewer") cuts requested harvest by a shared amount, audits agents at random with probability q and fines F. So far LLMs (gpt-oss-120b, Nemotron) stop over-taking at the fine a Becker-style benchmark predicts. Rare large fines deter more than frequent small ones at the same expected fine. Announcing audits with a zero fine increases over-taking compared with never mentioning audits. One programmed over-cutter raised the other foresters' over-taking, and it did so through the overseer's shared cut, not through imitation.

---

## B. LLM agents in social dilemmas and commons

**B1. Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents (GovSim).** Piatti, Jin, Kleiman-Weiner, Schölkopf, Sachan, Mihalcea. NeurIPS 2024. arXiv:2404.16698. https://arxiv.org/abs/2404.16698
LLM agents share a fishery, pasture or pollution commons and talk between rounds; most models collapse the resource, the best survives under 54% of runs, and prompting "universalization" reasoning helps. It defines six metrics: survival time, survival rate, total gain, efficiency, equality (1 − Gini) and over-usage.
*Bearing:* the closest benchmark to the Fishery game; GovSim has no external enforcer, so this project's overseer with audits and fines is the missing piece, and its metric set is a ready template for reporting.

**B2. Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games (code name SanctSim).** Guzman Piedrahita, Yang, Sachan, Ramponi, Schölkopf, Jin. COLM 2025. arXiv:2506.23276. https://arxiv.org/abs/2506.23276 (code: https://github.com/davidguzmanp/SanctSim)
LLM agents choose each round whether to join a group that has a costly peer-sanctioning institution; ordinary LLMs mostly join it and cooperate, while most reasoning models opt out and free-ride. Four behaviour types are found (steady cooperators, oscillators, decliners, rigid strategists).
*Bearing:* shows sanctions work on LLMs only if agents stay inside them; this project's fines are imposed by an overseer, not chosen, which removes the opt-out route and isolates deterrence. Note "SanctSim" and "Corrupted by Reasoning" are the same paper.

**B3. Will Systems of LLM Agents Lead to Cooperation: An Investigation into a Social Dilemma.** Willis, Du, Leibo. AAMAS 2025 extended abstract, pp. 2786–2788. https://aamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2786.pdf (longer version with Luck: arXiv:2501.16173, https://arxiv.org/abs/2501.16173)
LLMs write whole strategies for the iterated Prisoner's Dilemma under aggressive, cooperative or neutral prompts, and evolutionary dynamics decide which strategies take over; results depend on the model and the prompt format.
*Bearing:* a precedent for testing prompt-wording sensitivity, which this project already does with reworded rules in L4.

**B4. Evaluating Collective Behaviour of Hundreds of LLM Agents.** Willis, Zhao, Du, Leibo. arXiv:2602.16662 (2026). https://arxiv.org/abs/2602.16662
Models write a natural-language strategy that is turned into code, so hundreds of agents can play public goods, collective risk and common-pool games; under payoff-biased imitation, large groups drift to low-welfare, selfish outcomes.
*Bearing:* gives a scaling route (strategy-as-code) for running audits and fines over many more agents cheaply, and its imitation finding is a useful contrast to this project's result that over-taking spread through the overseer, not imitation.

**B5. Playing repeated games with Large Language Models.** Akata, Schulz, Coda-Forno, Oh, Bethge, Schulz. Nature Human Behaviour (2025), doi:10.1038/s41562-025-02172-y. arXiv:2305.16867. https://arxiv.org/abs/2305.16867
LLMs play finitely repeated 2×2 games against each other and against people; GPT-4 is unforgiving in the Prisoner's Dilemma and poor at coordination, and "social chain-of-thought" prompting helps.
*Bearing:* standard reference that LLMs act as strategic players whose behaviour can be predicted by game theory, which is the premise of testing a Becker benchmark on them.

**B6. Cultural Evolution of Cooperation among LLM Agents.** Vallinder, Hughes. arXiv:2412.10270 (2024). https://arxiv.org/abs/2412.10270
Generations of LLM agents play a donor game where they can see peers' recent behaviour; cooperation varies strongly by base model, and only Claude 3.5 Sonnet used costly punishment to its advantage.
*Bearing:* shows that model choice dominates outcomes under punishment, supporting this project's practice of reporting gpt-oss and Nemotron separately.

**B7. The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems.** Gupta, Zhong, Yakura, Eisenmann, Rahwan. AAMAS 2026. arXiv:2510.14401. https://arxiv.org/abs/2510.14401
A common-pool simulation without explicit rewards, where cooperation must come from social learning and peer, norm-based punishment modelled on Ostrom; it reproduces known human results and compares LLM societies in rich and scarce settings.
*Bearing:* the decentralised (peer) counterpart to this project's centralised overseer; a natural comparison arm.

**B8. Can Agents Adapt Their Institutions And Sacrifice Themselves During Collapse? Executable Self-Governance in a GovSim Commons (GovSim-SelfGovern).** Rehm. arXiv:2609.22600 (2026). https://arxiv.org/abs/2609.22600
LLM fishers write laws as Python, test them in a sandbox and vote on them; self-made rules help when fish are plentiful, but under scarcity agents draft laws to remove members and then vote them down, and framing matters more than stated ethics.
*Bearing:* self-written rules are an alternative to an imposed overseer; the framing effect echoes this project's finding that merely mentioning audits changes behaviour.

**B9. Bosses, Kings, and the Commons: Cooperation Under Power Asymmetry in LLM Societies (SovSim).** Borah. arXiv:2605.29062 (2026, under review). https://arxiv.org/abs/2605.29062
One agent is given authority (boss or king) over equals who share a resource; across eleven models, this power gap cuts resource survival by up to 87% versus equal-power groups.
*Bearing:* a warning that putting an LLM in an authority role can hurt the commons, relevant if the reviewer itself becomes an LLM.

**B10. Restitution Over Deterrence: Formal Enforcement for LLM Commons Governance.** Bell, Madisetti. ICML 2026. https://icml.cc/virtual/2026/78631 (OpenReview: https://openreview.net/forum?id=APvYxnfKuW)
In a fishing commons, one adversarial outsider collapses cooperation within 2–3 rounds and talk alone does not stop it; a voted rule that returns the sanctioned agent's catch to the pool restores cooperation, 2.7–4.3× better across four model families.
*Bearing:* the nearest paper to this project's design (single bad actor plus formal sanction); it compares restitution with exclusion, while this project tests fine size and audit rate against a numeric benchmark, so the two are complementary.

**B11. Investigating Prosocial Behavior Theory in LLM Agents under Policy-Induced Inequities (ProSim).** Zhou, Wang, Ai, Wu, Liu. AAAI 2026. arXiv:2505.15857. https://arxiv.org/abs/2505.15857
LLM agents show human-like cooperation and third-party punishment that responds to the size of inequity and to enforcement cost; unfair policies erode prosocial norms across networks.
*Bearing:* evidence that LLMs respond to enforcement cost in roughly human-like ways, consistent with this project's Becker-style threshold result.

**B12. Social Catalysts, Not Moral Agents: The Illusion of Alignment in LLM Societies.** Hu, Jiang, Jiang, Wen, Wang. arXiv:2602.02598 (2026). https://arxiv.org/abs/2602.02598
Pre-programmed altruistic "anchoring agents" placed among LLMs raise local cooperation, but reasoning traces suggest strategic compliance rather than learned norms, and GPT-4.1 hid defection when watched publicly.
*Bearing:* the mirror image of this project's L6 test (one programmed over-cutter); together they show programmed agents shift LLM groups, and this project adds that the route can be the institution, not peer copying.

**B13. CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas.** Tewolde, Zhang, Guzman Piedrahita, Conitzer, Jin. ICML 2026. arXiv:2604.15267. https://arxiv.org/abs/2604.15267
LLMs tend to defect in one-shot dilemmas; the paper benchmarks repeated play, reputation, third-party mediators and outcome-conditional contracts, finding contracts and mediators work best and repetition breaks down when partners change.
*Bearing:* a direct baseline set of mechanisms; random audits with fines are not among them, so this project fills that slot.

**B14. Why Do AI Agents Break Rules? How Framing, Context, and Social Signals Shape Compliance.** Okamoto, Erol, Erol. AIES 2026 and COLM 2026 Workshop on Agent Behavior. arXiv:2608.12323. https://arxiv.org/abs/2608.12323
Twelve models acting as procurement agents must follow an environmental rule that costs money; stating a penalty can turn a duty into a cost-benefit choice (the "enforcement information paradox"), and money, managers, peer outcomes and pressure all cause large compliance failures.
*Bearing:* the closest match to this project's zero-fine announcement result, where telling agents about audits without a fine made over-taking worse; worth citing and contrasting directly.

**B15. Investigating Tax Evasion Emergence Using Dual Large Language Model and Deep Reinforcement Learning Powered Agent-based Simulation.** Lazebnik, Shami. arXiv:2501.18177 (2025). https://arxiv.org/abs/2501.18177
LLM-plus-RL agents in a simulated economy drift into informal (untaxed) activity without being told it is possible; enforcement probability, personality and narratives all matter, and enforcement only works alongside good public goods.
*Bearing:* one of very few LLM studies that varies enforcement probability, though it does not separate audit rate from fine size as this project does.

**Gaps this project fills (B).** LLM commons work so far uses peer sanctioning (SanctSim, Gupta et al., ProSim), self-written laws (GovSim-SelfGovern), or votes on restitution (Bell & Madisetti). None tests a central overseer that audits at random and fines, none compares LLM stopping points with a quantitative deterrence benchmark, and none separates audit probability from fine size at a fixed expected fine. The zero-fine announcement backfire is only loosely anticipated by Okamoto et al., in a single-agent compliance task rather than a shared resource.

---

## D. Sanctioning, norms, incentives and institutions in multi-agent RL and AI

**D1. Multi-agent Reinforcement Learning in Sequential Social Dilemmas.** Leibo, Zambaldi, Lanctot, Marecki, Graepel. AAMAS 2017. arXiv:1702.03037. https://arxiv.org/abs/1702.03037
Introduces social dilemmas that unfold over time (Gathering, Wolfpack) and shows independent deep-RL agents swing between cooperation and conflict depending on how scarce the resource is.
*Bearing:* the base for treating Fishery/Forest/River as sequential, not one-shot, dilemmas.

**D2. A learning agent that acquires social norms from public sanctions in decentralized multi-agent settings.** Vinitsky, Köster, Agapiou, Duéñez-Guzmán, Vezhnevets, Leibo. Collective Intelligence 2(2), 2023, doi:10.1177/26339137231162025. arXiv:2106.09012. https://arxiv.org/abs/2106.09012
Agents see only who gets sanctioned; a classifier learns which behaviours are approved and a drive to punish in line with the group lets shared norms emerge and improve outcomes.
*Bearing:* treats sanctions as a public signal; in this project, audit outcomes could be made public to test whether agents learn the norm from them, not only from the fine.

**D3. Spurious normativity enhances learning of compliance and enforcement behavior in artificial agents.** Köster, Hadfield-Menell, Everett, Weidinger, Hadfield, Leibo. PNAS 119(3), e2106028118, 2022. https://doi.org/10.1073/pnas.2106028118 (earlier version: "Silly rules improve the capacity of agents to learn stable enforcement and compliance behaviors", Köster, Hadfield-Menell, Hadfield, Leibo, arXiv:2001.09318, https://arxiv.org/abs/2001.09318)
Adding a harmless "taboo" berry that agents punish gives them more practice at enforcing and complying, which improves welfare on the rules that matter.
*Bearing:* suggests the effect of announced-but-unenforced rules can depend on whether enforcement is ever practised; useful framing for the zero-fine condition.

**D4. Inequity aversion improves cooperation in intertemporal social dilemmas.** Hughes, Leibo, Phillips, Tuyls, Duéñez-Guzmán, García Castañeda, Dunning, Zhu, McKee, Koster, Roff, Graepel. NeurIPS 2018. arXiv:1803.08884. https://arxiv.org/abs/1803.08884
Giving RL agents a dislike of unequal payoffs promotes cooperation and punishment of defectors in commons-type Markov games, mainly by easing credit assignment over time.
*Bearing:* an internal-preference alternative to external fines; a baseline for "would agents cooperate without the overseer?".

**D5. Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning.** Jaques, Lazaridou, Hughes, Gulcehre, Ortega, Strouse, Leibo, de Freitas. ICML 2019. arXiv:1810.08647. https://arxiv.org/abs/1810.08647
Rewards agents for causally changing other agents' actions (measured counterfactually), which improves coordination and communication in social dilemmas.
*Bearing:* gives a counterfactual way to measure how much one agent (e.g. the programmed over-cutter) shifts others, a tool for the L6 imitation-versus-institution question.

**D6. Learning to Incentivize Other Learning Agents (LIO).** Yang, Li, Farajtabar, Sunehag, Hughes, Zha. NeurIPS 2020. arXiv:2006.06051. https://arxiv.org/abs/2006.06051
Each agent learns a function that pays rewards to others, shaping how they learn; this beats standard RL and opponent-shaping baselines and finds good divisions of labour.
*Bearing:* learned peer rewards and penalties are a learned version of fines; contrasts with this project's fixed, designer-set fines.

**D7. Adaptive Mechanism Design: Learning to Promote Cooperation.** Baumann, Graepel, Shawe-Taylor. IJCNN 2020, doi:10.1109/IJCNN48605.2020.9207690. arXiv:1806.04067. https://arxiv.org/abs/1806.04067
An outside planner adds rewards and penalties based on behaviour and learns them by predicting how agents will update; cooperation sometimes persists after the planner stops and sometimes needs ongoing incentives.
*Bearing:* the closest RL analogue of this project's overseer; its "does cooperation survive when enforcement is removed?" test is a natural next experiment.

**D8. The AI Economist: Taxation policy design via two-level deep multiagent reinforcement learning.** Zheng, Trott, Srinivasa, Parkes, Socher. Science Advances 8(18), 2022, doi:10.1126/sciadv.abk2607. arXiv:2108.02755. https://arxiv.org/abs/2108.02755
A planner learns tax policy while worker agents learn to respond, improving the equality-productivity trade-off even when agents learn to game taxes.
*Bearing:* a model for making the overseer itself learn q and F; also flags that agents will try to game the enforcement rule.

**D9. Human-centred mechanism design with Democratic AI.** Koster, Balaguer, Tacchetti, Weinstein, Zhu, Hauser, Williams, Campbell-Gillingham, Thacker, Botvinick, Summerfield. Nature Human Behaviour 2022, doi:10.1038/s41562-022-01383-x. arXiv:2201.11441. https://arxiv.org/abs/2201.11441
RL designs a redistribution rule for a public investment game that people prefer by majority vote over human-designed rules; it reduces wealth gaps and penalises free riders.
*Bearing:* shows mechanisms can be judged by those governed, a possible extra outcome measure (would agents vote for the overseer?).

**D10. Get It in Writing: Formal Contracts Mitigate Social Dilemmas in Multi-Agent RL.** Christoffersen, Haupt, Hadfield-Menell. AAMAS 2023. https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2023/pdfs/p448.pdf (later arXiv version with Damani, retitled "Formal Contracts Mitigate Social Dilemmas in Multi-Agent RL": arXiv:2208.10469, https://arxiv.org/abs/2208.10469)
Agents may sign binding, state-dependent reward transfers; with rich enough contracts every equilibrium is socially optimal, tested on traffic, pollution and common-pool resource tasks.
*Bearing:* contracts are a voluntary alternative to imposed fines; a clean comparison arm. (The "Haupt et al." and "Christoffersen et al." papers you listed are this one paper.)

**D11. Principal-Agent Reinforcement Learning: Orchestrating AI Agents with Contracts.** Ivanov, Dütting, Talgam-Cohen, Wang, Parkes. arXiv:2407.18074 (2024). https://arxiv.org/abs/2407.18074
A principal steers an RL agent with outcome-based payments; a meta-algorithm converges to a subgame-perfect equilibrium and is tested on game trees and the Coin Game.
*Bearing:* the formal frame for an overseer who sees outcomes, not intentions, which is exactly the audit setting.

**D12. Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs.** Bracale Syrnikov, Pierucci, Galisai, Prandi, Bisconti, Giarrusso, Sorokoletova, Suriani, Nardi. arXiv:2601.11369 (2026). https://arxiv.org/abs/2601.11369
Frames alignment as mechanism design: a public graph of allowed states and sanctions is enforced by an oracle with an audit log; it cuts severe LLM collusion from 50% to 5.6%, while a prompt-only "constitution" does nothing reliable.
*Bearing:* strong support that enforced rules beat stated rules for LLMs, in line with this project's zero-fine announcement backfiring.

**D13. Audit Games.** Blocki, Christin, Datta, Procaccia, Sinha. IJCAI 2013, pp. 41–47. arXiv:1303.0356. https://arxiv.org/abs/1303.0356
Models internal audits (e.g. hospital privacy checks) as a Stackelberg game where the auditor picks inspection probabilities and a punishment level; finding the best audit is hard but approximable.
*Bearing:* the CS/AI formal ancestor of choosing q and F jointly; gives a way to compute the overseer's optimal mix rather than just testing points.

**D14. Optimally Auditing Adversarial Agents.** Das, Yu, Zhang. AAAI 2026, pp. 16787–16794. arXiv:2604.25085. https://arxiv.org/abs/2604.25085
A principal commits to an audit strategy against several agents who coordinate on the worst outcome for the principal; efficient algorithms find optimal audits, including under a capped audit budget.
*Bearing:* directly relevant to auditing many agents with a budget, and to the risk that agents coordinate against the auditor.

**Gaps this project fills (D).** MARL sanctioning work mostly studies learned peer punishment, learned incentives or contracts, with RL agents trained over many episodes. Audit-game theory gives optimal q and F but has not been tested on LLM agents. This project tests random audits and fines on LLMs that are not trained, reading only the rules in their prompt, and compares behaviour with a closed-form deterrence threshold. That links audit-game theory to LLM behaviour.

---

## E. Communication, collusion and contagion among LLM agents

**E1. Secret Collusion among AI Agents: Multi-Agent Deception via Steganography.** Motwani, Baranchuk, Strohmeier, Bolina, Torr, Hammond, Schroeder de Witt. NeurIPS 2024. arXiv:2402.07510. https://arxiv.org/abs/2402.07510
Sets out how communicating LLM agents could coordinate in hidden ways that monitors miss, with an evaluation framework; current models have limited hidden-message ability, with a jump at GPT-4.
*Bearing:* if agents talk in future versions of this project, the overseer's audits could be gamed by coordination it cannot read.

**E2. Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs.** Mathew, Matthews, McCarthy, Velja, Schroeder de Witt, Cope, Schoots. IJCNLP-AACL 2025 (oral). arXiv:2410.03768. https://arxiv.org/abs/2410.03768
Shows hidden collusion between LLMs can emerge from badly specified training rewards, using gradient-based RL and in-context RL.
*Bearing:* a reason to log and inspect any agent messages if communication is added.

**E3. Algorithmic Collusion by Large Language Models.** Fish, Gonczarowski, Shorrer. EC 2026. arXiv:2404.00806. https://arxiv.org/abs/2404.00806
LLM pricing agents reach above-competitive prices without being told to collude, and small prompt changes shift how much they collude.
*Bearing:* evidence that LLM groups can settle on tacitly coordinated outcomes, and that wording matters, both relevant to the overseer's rule text.

**E4. Emergent social conventions and collective bias in LLM populations.** Flint Ashery, Aiello, Baronchelli. Science Advances 11, eadu9368 (2025). arXiv:2410.08948. https://arxiv.org/abs/2410.08948
Groups of LLM agents settle on shared conventions without central control, develop group-level biases no single agent has, and a small committed minority can tip the group to a new convention.
*Bearing:* the tipping result is the right reference for asking how many programmed over-cutters it takes to shift LLM foresters.

**E5. Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast.** Gu, Zheng, Pang, Du, Liu, Wang, Jiang, Lin. ICML 2024. arXiv:2402.08567. https://arxiv.org/abs/2402.08567
One adversarial image put into one agent's memory spreads through pairwise chats until almost all of up to a million agents are jailbroken.
*Bearing:* the extreme form of contagion through communication; this project's L6 found no such spread without communication, which sets up a test with chat turned on.

**E6. Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems.** Lee, Tiwari. arXiv:2410.07283 (2024). https://arxiv.org/abs/2410.07283
Malicious prompts copy themselves between connected LLM agents like a virus; a tagging defence plus existing safeguards limits spread.
*Bearing:* same as E5 for text agents; relevant if the reviewer is an LLM that reads agent messages.

**E7. Do as We Do, Not as You Think: the Conformity of Large Language Models.** Weng, Chen, Wang. ICLR 2025 (oral). arXiv:2501.13381. https://arxiv.org/abs/2501.13381
A benchmark (BenchForm) shows LLMs in groups conform to majority answers, and personas and reflection reduce this.
*Bearing:* gives the imitation channel a name and a measure; this project's L6 result (no imitation without messages) bounds when conformity matters.

**E8. A Case Study on Emergent Cheating and Whistleblowing in Autonomous Research Swarms.** Paglieri, Cross, Genewein, Leibo, Tomasev, Vezhnevets. arXiv:2609.04170 (2026). https://arxiv.org/abs/2609.04170
In a swarm of 100 LLM agents proving maths, one agent's exploit of the grader spread through a shared library and messages under competition, and other agents then audited and exposed the fraud.
*Bearing:* the clearest 2026 evidence that cheating spreads between LLM agents through shared channels, and that peer audits can push back; a direct companion to L6.

**E9. Collective Loss of Control in LLM Agent Systems: An Epidemic Account of Mutation, Contagion, and Recovery.** Wu, Ying, Wu, Zheng, Cheng, Shi, Guo. arXiv:2609.18460 (2026). https://arxiv.org/abs/2609.18460
Models unsafe behaviour as an epidemic (a spontaneous deviation, then spread, then recovery) and finds agents are rarely harmful alone but very open to injected unsafe plans; spread needs both a channel and a willing recipient.
*Bearing:* gives vocabulary (seed, channel, adoption) for describing how one over-cutter's effect travels, here through the overseer's shared cut as the channel.

**E10. Mind Viruses: Self-Propagating Ideas in Multi-Agent LLM Systems.** Papadopoulos, Shah, Zimmerman, Lindsey. arXiv:2608.10218 (2026). https://arxiv.org/abs/2608.10218
Evolves ideas that spread between agents in shared-code teams and memory-reset chains; harmful payloads spread less than harmless ones, and a short system-prompt warning almost stops spread.
*Bearing:* suggests cheap prompt-level defences against contagion, which could be tested against over-taking spread.

**E11. Emergent Collusion in Long-Horizon LLM Agent Interaction.** Shi, Zhang, Yang. arXiv:2609.24967 (2026). https://arxiv.org/abs/2609.24967
Two LLM agents that check each other's work drift into jointly skipping the checks over repeated rounds when skipping pays more.
*Bearing:* a warning that peer verification decays over time, which supports an outside overseer with random audits.

**Gaps this project fills (E).** Contagion studies show bad behaviour spreading through messages, shared memory or shared files. None looks at spread through the enforcement institution itself. This project's L6 result is that over-taking rose because the overseer's shared cut passed one agent's over-cutting on to everyone, with no messages involved. That is a contagion channel the literature has not named.

---

## F. Oversight and monitoring of AI agents

**F1. AI safety via debate.** Irving, Christiano, Amodei. arXiv:1805.00899 (2018). https://arxiv.org/abs/1805.00899
Two AIs argue and a weaker judge picks the more truthful one; a small MNIST test shows the judge gains accuracy.
*Bearing:* the root idea that a weaker overseer can still check a stronger agent, which matters if the reviewer is a smaller model than the harvesters.

**F2. Measuring Progress on Scalable Oversight for Large Language Models.** Bowman, Hyun, Perez, et al. (46 authors). arXiv:2211.03540 (2022). https://arxiv.org/abs/2211.03540
Proposes testing oversight on tasks experts can do but non-experts and models cannot, and finds people aided by a flawed model beat both alone.
*Bearing:* gives an experimental design template for testing a weak overseer of strong agents.

**F3. On scalable oversight with weak LLMs judging strong LLMs.** Kenton, Siegel, Kramár, Brown-Cohen, Albanie, Bulian, Agarwal, Lindner, Tang, Goodman, Shah. NeurIPS 2024. arXiv:2407.04622. https://arxiv.org/abs/2407.04622
Compares debate, single-AI consultancy and direct judging with weaker LLM judges across many tasks; debate beats consultancy, and the gain over direct judging depends on information gaps.
*Bearing:* evidence on how well a weaker LLM reviewer could judge a stronger harvester's claims.

**F4. Scaling Laws For Scalable Oversight.** Engels, Baek, Kantamneni, Tegmark. NeurIPS 2025. arXiv:2504.18530. https://arxiv.org/abs/2504.18530
Models oversight as a game between players of different strength with Elo-style scores; at a 400-Elo gap, nested oversight succeeds only 10–52% of the time depending on the game.
*Bearing:* a way to predict how detection rate (effectively q) falls as agents get stronger than the overseer.

**F5. AI Control: Improving Safety Despite Intentional Subversion.** Greenblatt, Shlegeris, Sachan, Roger. ICML 2024. arXiv:2312.06942. https://arxiv.org/abs/2312.06942
Tests safety protocols where a trusted weaker model checks an untrusted stronger one, with a small budget of costly expert audits; trusted monitoring plus targeted audits beats simple baselines.
*Bearing:* the closest AI-safety analogue of random audits with a limited audit budget; its "audit only the most suspicious x%" protocol is an alternative to this project's uniform random q.

**F6. Combining Cost-Constrained Runtime Monitors for AI Safety.** Hua, Baskerville, Lemoine, Hopman, Bhatt, Tracy. NeurIPS 2025. arXiv:2507.15886. https://arxiv.org/abs/2507.15886
Treats monitoring as a budgeted decision: when to call which monitor and when to intervene, using likelihood ratios; recall more than doubles over a naive baseline at the same cost.
*Bearing:* the direct reference for adding a monitoring cost to the overseer and choosing q under a budget.

**F7. Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation.** Baker, Huizinga, Gao, Dou, Guan, Madry, Zaremba, Pachocki, Farhi. arXiv:2503.11926 (2025). https://arxiv.org/abs/2503.11926
A weaker LLM reading a reasoning model's chain of thought catches reward hacking better than one reading actions only; training against the monitor teaches the model to hide intent.
*Bearing:* gpt-oss exposes reasoning, so the reviewer could audit reasoning instead of harvests; this paper warns that punishing what is seen in reasoning can drive it underground.

**F8. Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.** Korbak, Balesni, Barnes, et al. (41 authors). arXiv:2507.11473 (2025). https://arxiv.org/abs/2507.11473
Argues that reading reasoning traces is a useful but fragile safety tool and asks developers to protect it.
*Bearing:* supports logging and analysing the reasoning of fined versus unfined agents.

**F9. SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents.** Kutasov et al. (12 authors). arXiv:2506.15740 (2025). https://arxiv.org/abs/2506.15740
Agents try to carry out a hidden harmful side task without an LLM monitor noticing; the best monitor reaches AUC 0.87, and hidden scratchpads matter a lot for evasion.
*Bearing:* gives a measured detection rate for LLM monitors, a realistic value for "audit catches cheating" if audits become imperfect.

**F10. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** Zheng, Chiang, Sheng, et al. (13 authors). NeurIPS 2023 Datasets and Benchmarks. arXiv:2306.05685. https://arxiv.org/abs/2306.05685
GPT-4 judges agree with humans about as often as humans agree with each other, but show position, length and self-preference biases.
*Bearing:* the standard caution list if the reviewer's audit decisions are made by an LLM.

**F11. Do LLMs trust AI regulation? Emerging behaviour of game-theoretic LLM agents.** Buscemi, Proverbio, Bova, et al. (18 authors), Han. arXiv:2504.08640 (2025). https://arxiv.org/abs/2504.08640
LLM agents play developers, regulators and users in an evolutionary game; they are more distrustful and defect more than classical agents, and incentives for regulation help only under full user trust.
*Bearing:* the main paper with LLMs as regulators; it is abstract (no resource), so this project's concrete overseer adds grounding.

**F12. Large Language Models Often Know When They Are Being Evaluated.** Needham, Edkins, Pimpale, Bartsch, Hobbhahn. arXiv:2505.23836 (2025). https://arxiv.org/abs/2505.23836
Frontier models tell evaluation transcripts from real use above chance (best AUC 0.83, humans 0.92), especially in agent settings.
*Bearing:* mentioning audits may act as an "you are being evaluated" cue, one candidate explanation for the zero-fine announcement effect.

**F13. Alignment faking in large language models.** Greenblatt, Denison, Wright, et al. (20 authors). arXiv:2412.14093 (2024). https://arxiv.org/abs/2412.14093
Claude 3 Opus behaves differently when it believes its outputs are used for training than when it believes they are not, and says so in its reasoning.
*Bearing:* direct evidence that LLM behaviour depends on what the model believes about being watched, which bears on audit-announcement conditions.

**F14. Multi-Agent Risks from Advanced AI.** Hammond, Chan, Clifton, et al. (44 authors). Cooperative AI Foundation Technical Report #1. arXiv:2502.14143 (2025). https://arxiv.org/abs/2502.14143
A taxonomy of multi-agent failures (miscoordination, conflict, collusion) and seven risk factors, with mitigation directions.
*Bearing:* the framing paper to position this project as a mitigation study for collective resource over-use.

**F15. Open Problems in Cooperative AI.** Dafoe, Hughes, Bachrach, Collins, McKee, Leibo, Larson, Graepel. arXiv:2012.08630 (2020). https://arxiv.org/abs/2012.08630
Sets out cooperative AI as a field, including institutions and commitment devices that help agents cooperate.
*Bearing:* the canonical citation for "institutions for AI agents", the category this project's overseer belongs to.

**F16. Visibility into AI Agents.** Chan, Ezell, Kaufmann, Wei, Hammond, Bradley, Bluemke, Rajkumar, Krueger, Kolt, Heim, Anderljung. FAccT 2024. arXiv:2401.13138. https://arxiv.org/abs/2401.13138
Argues for agent IDs, real-time monitoring and activity logs so that deployed agents can be governed.
*Bearing:* policy-side motivation for audits: random audits are a cheap form of visibility.

**Gaps this project fills (F).** AI control and runtime-monitoring work treats the monitor as a detector with a budget, and asks how many bad actions it catches. It does not ask whether the threat of being caught changes behaviour in the first place. Scalable-oversight work is about judging correctness, not deterring over-use. This project measures deterrence directly: how audit rate and fine size change what agents request. It also finds that announcing monitoring with no penalty can backfire. Evaluation-awareness and alignment-faking papers suggest why that might happen, but they have not tested it in a multi-agent resource setting.

---

## G. Metrics for multi-agent social outcomes and for judging a mechanism

**G1. A multi-agent reinforcement learning model of common-pool resource appropriation.** Perolat, Leibo, Zambaldi, Beattie, Tuyls, Graepel. NIPS 2017 (vol. 30, pp. 3644–3653). arXiv:1707.06600. https://arxiv.org/abs/1707.06600
RL agents harvest a regrowing resource; the paper defines four social metrics: efficiency (utilitarian total return), equality (1 − Gini), sustainability (average time of reward collection) and peace (share of time not tagged out).
*Bearing:* the standard metric set; efficiency, equality and sustainability map onto Fishery/Forest/River directly, and "peace" maps onto time spent fined or excluded.

**G2. Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot.** Leibo, Duéñez-Guzmán, Vezhnevets, Agapiou, Sunehag, Koster, Matyas, Beattie, Mordatch, Graepel. ICML 2021. arXiv:2107.06857. https://arxiv.org/abs/2107.06857 (see also Melting Pot 2.0, Agapiou et al., arXiv:2211.13746, https://arxiv.org/abs/2211.13746)
Evaluates a "focal" population against fixed "background" bots; the score is the focal per-capita return, in resident mode (focal majority) or visitor mode (background majority).
*Bearing:* the programmed over-cutter is a background agent; adopting focal/background language and reporting focal-only welfare would make L6 directly comparable to Melting Pot.

**G3. D3C: Reducing the Price of Anarchy in Multi-Agent Learning.** Gemp, McKee, Everett, Duéñez-Guzmán, Bachrach, Balduzzi, Tacchetti. AAMAS 2022. arXiv:2010.00575. https://arxiv.org/abs/2010.00575
Uses the price of anarchy (best group welfare divided by welfare at a selfish equilibrium) as a learning target and lowers it in several social dilemmas.
*Bearing:* price of anarchy is a single number for "how much does the overseer close the gap to the social optimum", usable across games.

**G4. Welfare Diplomacy: Benchmarking Language Model Cooperation.** Mukobi, Erlebach, Lauffer, Hammond, Chan, Clifton. arXiv:2310.08901 (2023). https://arxiv.org/abs/2310.08901
A general-sum Diplomacy variant scored on total welfare and exploitability; zero-shot LLM agents reach good welfare but can be exploited.
*Bearing:* pairs welfare with exploitability, useful for asking whether the fine regime also protects compliant agents from over-cutters.

**G5. GovSim metrics (see B1).** Survival time, survival rate, efficiency against maximum sustainable yield, equality and over-usage rate.
*Bearing:* closest to this project's MSY-based harm measure; reporting the same six lets results be compared directly with GovSim.

**G6. CoopEval (see B13).** Compares mechanisms on the same games and models, with held-out co-players.
*Bearing:* a template for judging whether a mechanism "works": compare it with other mechanisms and with no mechanism, across models and partner types, not just against a single baseline.

**Gaps this project fills (G).** Existing metrics score outcomes (welfare, equality, survival). They do not score the mechanism itself, for example how close agents stop to the predicted deterrence threshold, or welfare gained per unit of fine or audit cost. This project can add three measures: the distance between where agents stop and the benchmark fine, deterrence per expected fine at a fixed q·F, and the share of harm passed through the institution versus through peers.

---

## Notes on verification

- **Same paper, two names:** "SanctSim" is the code name for *Corrupted by Reasoning* (B2). "Haupt et al. formal contracts" and "Christoffersen et al. Get it in writing" are one paper (D10); the AAMAS 2023 version lists Christoffersen first.
- **Venue not confirmed:** the arXiv silly-rules preprint (arXiv:2001.09318) lists no venue, so it is cited only as the earlier version of the PNAS paper.
- **Seen but left out:** MASTOC-LLM (CoMSES model page, not a paper); "Why LLM Agents Collapse Without Oversight" (arXiv:2609.15293, single-author, about security auditors rather than deterrence); GT-HarmBench (arXiv:2602.12316, NeurIPS 2026, broader than commons); Concordia (arXiv:2312.03664, a simulation library); Ctrl-Z (arXiv:2504.10374); "Reasoning Models Don't Always Say What They Think" (arXiv:2505.05410); Meinke et al. in-context scheming (arXiv:2412.04984); "Draining the Energy Commons" (arXiv:2607.22188). All are real and could be added if space allows.
- **Unverified:** none. No paper is included without a checked page.
