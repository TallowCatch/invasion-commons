# A. How oversight and control papers model the overseer

Focused read for the overseer formulation (23 Nov 2026). Ten papers: eight read in full (main text plus the appendix parts that define the model), two skimmed for positioning. The question throughout is: what does the overseer see, what can it do, how does it decide, and how do the overseen agents respond?

Notation used below is mine unless stated. "Overseer" covers the paper's own word (protocol, Blue, monitor, Guard, auditor, mechanism, judge, defender). Page and section numbers refer to the PDFs in `notes/icml27_lit_review/papers/` (Griffin is in Downloads).

Marked **[unverified]** where the text I read did not settle the point.

---

## 1. Comparison tables

Split into two tables for width. Cells are short; detail is in Section 2.

### 1a. Game structure

| Paper | Players; is the overseer a player? | What the overseer observes | Overseer actions and budget | Overseer policy | # overseers | Capability relation | Timing / solution concept | Overseen agents |
|---|---|---|---|---|---|---|---|---|
| **Griffin+ 2024** Games for AI Control | Blue (protocol) vs Red (proxy for scheming model). Overseer is a player. | Noisy suspicion score x of current output, plus problems left m and audits left c. Partial, per step, free. | Audit (perfect, ends game if backdoor), Defer (use weak model), Run. Budget: C = 2% of M audit "coins" per episode, a state variable. | Threshold on x whose cut-offs may depend on (m, c); found by grid search, DP, or randomisation. Memoryless in results; memoryful allowed in model. | 1 (with a human auditor as a resource) | T weaker than U; capability enters only through the two score distributions. | Leader-follower: Blue commits, Red best-responds (memoryless). "Control equilibrium" = Red minimises safety, Blue Pareto-optimal. Equals maximin of a weighted zero-sum POSG. | One scheming untrusted model, worst case. |
| **Greenblatt+ 2024** AI Control | Blue team protocol vs red-team attack policy standing in for U. Overseer is the protocol. | Per item: T's suspicion score of U's code; on audit, H's exploit attempt and suspicion. | Audit with H (2% of 5,000 problems), defer to T, trusted edit, shut down (switch to T forever), validate an untrusted monitor on synthetic backdoors, redact/paraphrase inputs. | Fixed pipelines with thresholds set to hit a target false-positive rate; LLM monitors. | 1 protocol using T, U-as-monitor, H. | T = GPT-3.5 (weak, trusted), U = GPT-4, H = GPT-4 via a clean channel (strong, scarce). | Red designs attack after seeing protocol; iterated by hand. Worst-case. | One untrusted model, adversarial, stateless between calls. |
| **Makins+ 2026** Multi-Agent AI Control | K executor agents (1, 3, 9) plus optional planner vs a defender's per-commit monitor. Overseer outside the agents' game, fixed. | One commit's git diff plus the main-task description. No repo state, no other commits, no cross-agent link. | Score 0–100; flag if above threshold. One flag = attackers lose. Budget = target FPR and monitor compute (≈1–2% or 5–9% of executor compute). Design-time: read/write permission limits. | LLM (GPT-5 nano or GPT-5 mini) with a prompt; threshold from honest-run FPR. | One instance per commit, no sharing. A "joint monitor" on the squashed PR is tested but called undeployable. | Nano: weak trusted. Mini: comparable to executor (Gemini 3 Flash), maybe too strong to count as trusted. | No game solved. Attackers are prompted to coordinate and evade; empirical. | Many coordinating LLM attackers, sequential, planner with an unmonitored channel. |
| **Engels+ 2025** Scaling Laws for Scalable Oversight | Guard (overseer) vs Houdini, in four games; Narrator as environment in Wargames. Overseer is a player. | Game-specific: debate transcript (no ground truth); code with comments removed; Mafia discussion; Narrator's report of a "month". | Judge a debate; flag code; discuss and vote (Mafia); free-text plans of 5 actions (Wargames). No explicit budget. NSO: number of oversight steps n. | LLM prompted per game; in Backdoor Code, majority vote of 5 Guard calls. | 1, except Mafia (5 villager Guards who talk and vote); NSO = chain of overseers. | Measured: general Elo (Chatbot Arena) mapped to game Elo by a "double ReLU". The central variable. | Pairwise match-ups, win rates fit to Elo. No equilibrium. NSO multiplies step win probabilities. | One Houdini LLM, prompted to deceive. |
| **Parisi+ 2024** Monitored MDPs | One agent; the monitor is a separate Markov process inside the environment. Not a player. | (Inverted) The monitor decides what the agent sees of its own reward: true reward or "unobserved". | Monitor has states and actions, but the agent chooses the monitor actions (e.g. "monitor me", press a button). Monitoring has its own reward, often a cost. | Fixed environment process. | 1 | Not modelled; truthful monitors show the exact reward. | Single-agent MDP; optimal policy for reward + monitor reward. | One Q-learning agent. |
| **Ye & Steinhardt 2026** Norm Enforcement for AI Agents | n competing LLM agents; the "mechanism" is the platform. Agents double as reporters. Overseer outside the agents, fixed rule. | Peer reports (target + justification); on verification, the reported agent's recent actions (about 3 rounds). | Remove an agent (all future reward zero). Verify a report (LM call). Backfire: remove false reporters. Reputation filter decides which reports get verified. Cost tracked as verifier tokens. | Fixed rules: Naive, Checked, Backfire, RepVote, EscRepVote (Bayesian reputation with escalating penalty). Verifier = Qwen3-0.6B. | 1 central verifier fed by many peer reporters. | Verifier tiny (0.6B) vs agents (80B). Ground-truth judge gpt-5-mini used only for scoring. | Mechanism fixed; agents adapt in context over 15 rounds. Empirical. | Mixed population, 50% good / 50% bad (fine-tuned or prompted), 4–64 agents. |
| **Wang+ 2026** BazaarBench | 100 LLM traders (buyers and sellers). No online overseer. | (For a possible overseer) Full event log, inventory records (true owner, condition, cost), messages, reasoning summaries. | None online. Platform validates calls and can block some (L0 blocked failed inventory checks; L1–L3 only logged them). Offline GPT-5 judge labels failures. Judge cost ≈ $6.5k for 48 runs. | Offline LLM judge with rubric plus record checks. | 0 online | Judge GPT-5 (high reasoning); some traders are newer models (GPT-5.5). | None. Evaluation benchmark. | 100 LLM agents; 20 swapped to a tested model under ordinary, deadline or adversarial prompts. |
| **Casheekar & Tangirala 2026** Learning to Report Unsafe Tasks | Agents (witnesses) learn to report or stay silent; an auditor picks audit probabilities. Auditor outside the learning, commits once. | One audit reveals whether one agent stayed silent in an unsafe context. Perfect. Auditor knows which contexts are unsafe and the witness sets. | Audit agent i with probability a_i; fine silent audited agents. Budget Σ κ_i a_i. | Fixed allocation designed offline by a certificate (strict elimination) or baselines. | 1 auditor; the agents are themselves peer monitors. | Auditor verifies perfectly; no capability axis. | Auditor commits; agents run policy gradient (PPO) from learned silence. Outcome = where learning ends, not an equilibrium. | 64 PPO agents, grouped into shared-parameter networks, one team reward. |
| **Gans & Holden 2026** (skim) | Provider vs one agent who can conceal. Overseer is a committed design. | One evaluator per exploit channel; detection prob. falls with concealment effort. | Choose audit lottery p over K channels (Σ p_i ≤ B) and penalty size. | Committed lottery. | 1 | Implicit in detection technology. | Commitment; must work for every agent type. | One agent, best-responding, can alter records. |
| **Zhu+ 2026** Blindfold (skim) | Defender auditing a random ε of actions vs an adaptive LLM agent. | Label of audited actions only. | Audit coin per action; estimate violation rate with inverse-propensity weighting. | Random coin. | 1 | Not modelled. | Adversary picks when to act after seeing its pre-commit view. | One LLM agent, adaptive. |

### 1b. Objectives, formal core, fit

| Paper | Objective / evaluation | Formal core (short) | Borrow for an SSD overseer | Missing for our setting |
|---|---|---|---|---|
| Griffin | Worst-case safety vs usefulness frontier. | Two-mode POSG (honest / scheming) with Blue's budget in the state. | Budget as a state variable; audit/defer/run action set; mode uncertainty; frontier via linear scalarisation. | One adversary, one catastrophic event, no learning, no shared resource. |
| Greenblatt | P(no backdoor in 5,000) vs correctness. | Protocol over resources {U, T, H} with H budget. | Trusted/untrusted split; defer; honeypot validation of a monitor; redaction against collusion. | Single agent; harm is one-shot; no MARL. |
| Makins | Undetected success vs FPR. | max over commits of per-commit score vs threshold. | Per-agent local monitor vs joint view; fragmentation of harm across agents; affordance limits. | Agents are prompted, not learning; no resource; no overseer learning. |
| Engels | Guard win rate; Elo slopes; NSO success. | Elo win curve plus double-ReLU capability map. | Capability axis measured on a common scale; "does oversight gain more from capability than evasion does". | Single-step games; no budget; no many-agent dynamics. |
| Parisi | Discounted return incl. monitor cost. | MDP plus monitor MDP filtering reward. | Agent-side view: agents see sanctions only when watched; ⊥ ≠ 0. | Single agent; monitor has no goals. |
| Ye | TPR/FPR of removal over rounds; verifier cost. | Peer reports + reputation-weighted verification trigger. | Peer reporting; reputation; escalation; verifier cost; fishery commons results. | LLM in-context agents, not RL learners; removal only; no spatial or partial obs for the overseer. |
| Wang | Failure types × stages; completion; earnings. | Records vs claims; failure stage ladder. | Ground-truth records an overseer could query; stage ladder gives intervention points; token cost. | No overseer at all. |
| Casheekar | Unsafe and legitimate completion; joint recovery. | Witness hypergraph; expected reward polynomial in shared silence probs. | Audit design must follow the learning path; parameter sharing multiplies audit need by k. | One-step game; perfect auditor; auditor not learning. |
| Gans | Implementation for every type; catch rate; harm. | Lottery over evaluators vs concealment. | Concealment response; audit draws must stay unpredictable. | One agent; no dynamics. |
| Zhu | Unbiased violation-rate certificate. | Pre-commit noninterference. | Agents must not see whether they are watched now. | One agent. |

---

## 2. Paper by paper

### 2.1 Griffin, Shlegeris & Abate 2024, Games for AI Control (TiFA workshop, ICML 2024)

**Players.** Two: Blue (the protocol designer) and Red (a red team that stands in for a scheming model). The overseer is Blue, a full player with its own observations and policy (Def. 3.1, p. 4).

**Observations.** In the Trusted Monitoring game Blue sees, each step, the number of problems left m, the audit coins left c, and a suspicion score x from the weak trusted model. x is a noisy signal of whether the current code is backdoored, drawn from one of two Beta distributions (Sec. 4.1, App. A). Red sees nothing (a null observation), so it cannot time attacks.

**Actions and budget.** Audit, Defer, Run. Audit is perfect: a backdoor is caught and the game ends safely. Defer replaces U's answer with T's (safe, less useful). Run submits U's answer. The budget is a count of audit coins, C = 0.02·M, carried in the state and spent by Audit (App. A). The budget is per episode, not per step.

**Policy.** Blue's policy can depend on the full history; in the results it is memoryless: a two-threshold rule on x whose false-positive rates can vary with (m, c) (Eq. 3). Safety and usefulness of such a rule are computed exactly by a recurrence (Prop. 4.1); Red's best response is a single backdoor rate β found by search.

**Timing and solution.** Within a step both move at once, but policies are chosen leader-follower: Blue commits, Red best-responds. A control equilibrium is a pair where Red minimises safety and Blue is not Pareto-dominated in (worst-case safety, usefulness) (Def. 3.2). Theorem 5.4: each control equilibrium is a maximin point of a zero-sum POSG where reward is w·safety in scheming mode and (1−w)·usefulness in honest mode.

**Formal core (my notation).** A game G = (S, S_bad, A_O, A_R, Ω_O, Ω_R, ρ0, P, r_use, H) with a hidden mode μ ∈ {honest, scheming}. In honest mode Red's action is forced to "no interference". Overseer policy π_O: history → Δ(A_O). Safety(π_O) = min over π_R of P(s_H ∉ S_bad | scheming); Use(π_O) = E[Σ r_use | honest]. Blue does not know μ, so it must be useful when nobody attacks and safe when someone does.

**Findings relevant here.** Auditing more than the 2% "spend it evenly" rate early helps, because Red may attack several times. In the weak-monitor case, randomising the defer rate gives a large Pareto gain.

**Borrow.** (1) The two-mode trick: evaluate the same overseer policy on an honest population (usefulness / welfare cost of oversight) and an adversarial one (safety). (2) Budget as a state variable that the overseer observes and spends. (3) The audit / defer / run triple maps onto inspect / override-or-freeze / let-pass. (4) Reduction to a zero-sum POSG gives a solver route.

**Missing.** One adversary and one catastrophic event; Red does not learn and sees nothing. No shared resource, no many-agent structure, no spatial observation. This is the workshop version; a longer version may differ **[unverified]**.

### 2.2 Greenblatt, Shlegeris, Sachan & Roger 2024, AI Control

**Players.** A blue team writes a protocol; a red team writes an attack policy that replaces U during safety runs (Sec. 2). The protocol is the overseer. It is a fixed program, not a learner.

**Observations.** For each of 5,000 programming problems (1,500 generated, upsampled), the protocol sees U's code and T's suspicion score. When it spends an H point, H (GPT-4 through a clean channel) tries to build an exploit and rates suspicion (Sec. 2.2). Blue also knows summary statistics of honest GPT-4 behaviour, such as its score distribution, which is how thresholds are set (Sec. 2.3).

**Actions and budget.** Audit with H (enough H points for 2% of problems; first 10 audited free), shut down (switch to T for the rest), defer to T, ask T to edit U's code, ask a copy of U to monitor, test that copy on synthetic backdoors made with H (30 H points), redact or paraphrase what the monitor sees to block collusion signals (Secs. 3.2–3.5).

**Policy.** Hand-designed pipelines with thresholds chosen for a target false-positive rate (2% for audit; 5–50% for defer). Monitors are LLMs prompted to score.

**Capability.** Trusted means weak enough not to scheme (GPT-3.5); untrusted is GPT-4; H is strong but scarce. Untrusted monitoring uses a model as strong as the attacker, at the cost of collusion risk.

**Timing.** Red sees the protocol before choosing an attack. Iterated by hand; no equilibrium is computed.

**Results that matter.** Upfront auditing: 15% safety. Trusted monitoring: 62%. Defer-to-trusted traces a frontier (e.g. 92% safety at 83% usefulness). Untrusted monitoring with redaction: 79% at 96%. Trusted editing 20%: 92% at 94% (Table 7). Untrusted monitoring fails at 0% safety unless validated on synthetic inputs, and the red team then learns to tell synthetic from real inputs (Sec. 3.4.1).

**Formal core.** A protocol is a map from each arriving problem to a submitted answer, using {U, T, H} with |H calls| ≤ 0.02·N. Safety = P(no backdoor submitted in N problems | U = attack policy). Usefulness = correctness rate with U = real GPT-4, rescaled between T and U.

**Borrow.** The trusted / untrusted split as a capability design choice. Defer as an intervention that trades usefulness for safety. Honeypots (synthetic violations) to check that a monitor is honest. Signal jamming against collusion between monitor and monitored.

**Missing.** Single agent; failure is a single event; stateless attacker; no learning on either side; no welfare or shared resource. Sec. 5.1.1 itself lists auditing failures and heterogeneous tasks as unmodelled.

### 2.3 Makins, Angelini, Shams & Phuong 2026, Multi-Agent AI Control

**Players.** K executor agents (K ∈ {1, 3, 9}) act in sequence on a shared repo, each with a benign main task; attackers share a side task (e.g. disable monitoring, poison data). An optional planner writes one plan, broadcast to executors over a channel the monitor cannot see (Secs. 1.1, 2.1, 2.4). The overseer is a defender's monitor, outside the agents' interaction and fixed.

**Observations.** Each commit is scored on its own: the monitor sees the git diff of that commit and the main-task description (three variants: exhaustive, empirical, distractor). It does not see the repo state, other commits, or which commits belong together (Sec. 2.6, App. C.4.3).

**Actions and budget.** Output a suspicion score; flag when above a threshold. The threshold comes from a target FPR on honest runs. A single flag ends the attack. Budget appears as FPR and as monitor compute (nano ≈ 1–2% of executor compute, mini ≈ 5–9%). Write/read permission scopes are a separate design-time lever (Sec. 2.5).

**Policy.** LLM monitor with a prompt, no memory across commits.

**Capability.** GPT-5 nano is the "weak trusted" baseline and misses most attacks; GPT-5 mini is comparable in general capability to the executor and may be too strong to count as trusted (Sec. 3.3). Moving from mini to nano raises the odds of undetected success by a factor of about 53 at 5% FPR.

**Key result.** The fragmentation effect: as K grows, each commit and the most suspicious commit look less suspicious. A monitor on the squashed pull request scores attacks high (mean 0.91) regardless of K (Fig. 2e), so the evasion comes from splitting, not from milder actions. Cover-story quality explains more of the drop than dilution; about 40% stays unexplained (Table 1). Executors predict the monitor poorly (r = 0.59; 0.77 with a planner).

**Formal core.** Executors k = 1..K produce diffs δ_k. Monitor score σ_k = M(δ_k, task_k). Undetected success = side task done ∧ all main tasks done ∧ max_k σ_k ≤ τ, with τ set by honest FPR.

**Borrow.** (1) The local-vs-joint view question is our partial-observation question: a per-agent overseer misses harm that only shows in aggregate. (2) Harm split across many agents is close to commons depletion, where each agent's take looks small. (3) Affordance limits as an overseer action. (4) Monitor compute as a share of agent compute, a natural cost unit.

**Missing.** Agents are prompted, not trained; the defender never adapts; no communication between monitors; untrusted monitoring not run (App. C.4.2). Sequential, not simultaneous.

### 2.4 Engels, Baek, Kantamneni & Tegmark 2025, Scaling Laws for Scalable Oversight

**Players.** A Guard (overseer) and a Houdini (overseen) in each of four games (Sec. 2.1). In Wargames a third LLM, the Narrator, plays the environment and reports outcomes to each side (Sec. 3.4).

**Observations and actions per game.** Debate: the Guard judges a three-round transcript between two Houdini debaters without access to the source article. Backdoor Code: the Guard sees code with comments stripped and says whether it contains a backdoor; 5 independent calls, majority vote (App. A.3). Mafia: 5 villager Guards discuss and vote to eliminate 1 Mafia Houdini; they see only the conversation and night eliminations. Wargames: the Guard (lab monitor) sends five free-text actions per month and receives the Narrator's account; 5 months. No explicit budget in any game.

**Policy.** Prompted LLMs.

**Capability.** This is the paper's main variable. General capability g is Chatbot Arena Elo. Each model gets a Guard Elo and a Houdini Elo per game from pairwise win rates, p(E_G, E_H) = 1 / (1 + 10^((E_H − E_G)/400)). Game Elo is a piecewise-linear "double ReLU" in g: flat (incompetent), linear (slope α), flat (saturated) (Eq. 2).

**Findings.** Slopes differ by game. Debate: Guard Elo rises with g, Houdini flat (good for oversight). Backdoor Code: Houdini slope steeper than Guard slope (bad), though Guards start higher. Mafia: Guard slope steeper. Wargames: Houdini flat, Guard rises (Sec. 3.5).

**Nested oversight.** With n steps spanning a general-Elo gap Δ_g, step j pits a Guard at j·Δ_g/n against a Houdini one step ahead; success is the product of step win probabilities (Eqs. 3–5). At a 400-point gap, success is 9.4–51.7% depending on game.

**Formal core.** An instance is (game, context C, a_H, a_G, Win_H); win probability is E[Win_H] over contexts and sampling (App. D.1). The overseer is just the Guard seat in a two-sided game.

**Borrow.** A common capability scale for overseer and overseen, and the design target "Guard slope > Houdini slope". For a weaker-LLM overseer in an SSD, we can vary overseer model size along an Elo-like axis and fit how catch rate or welfare scales.

**Missing.** Single-step games (their own limitation, Sec. 5); no budget or cost; one Houdini (Mafia is one-vs-five); overseer and overseen are both prompted LLMs, not learners; no shared resource.

### 2.5 Parisi, Mohammedalamen, Kazemipour, Taylor & Bowling 2024, Monitored MDPs (AAMAS 2024)

**Players.** One agent. The monitor is not a player; it is a second Markov process inside the environment (Sec. 2.1, Fig. 2).

**What the monitor does.** It decides what the agent sees of the environment reward. The agent always gets judged (true reward r_e is generated), but it observes a proxy r̂ = M(r_e, s_m, a_m), which may be "unobserved" (⊥). The monitor has its own state s_m (e.g. sensor on/off, owner home), and the agent picks monitor actions a_m (e.g. "monitor me", press a button). Monitoring can carry a reward r_m, usually a cost (−0.2 per monitored step in the gridworlds, Sec. 4.1).

**Policy.** Fixed environment dynamics. The paper is about how the agent should learn, not how the monitor should act.

**Formal core.** ⟨S_e, A_e, P_e, R_e, M, S_m, A_m, P_m, R_m, γ⟩. The agent acts with joint action (a_e, a_m), maximises Σ γ^t (r_e + r_m), but sees (r̂_e, r_m). Optimal policies always exist; learning them is guaranteed if the joint process is ergodic, every reward is eventually observable, and the monitor is truthful (shows the true reward or nothing) (Props. 1–3).

**Findings that matter.** Treating "unobserved" as zero makes the agent ask to be watched only where rewards are positive and walk over penalty cells when unwatched (Sec. 4.2.1, Fig. 5). That is the textbook failure of an agent that learns from sanctions it only sometimes sees. Learning with a reward model is correct but slower than an oracle (Fig. 10).

**Borrow.** This is the agents' side of our problem. If our overseer inspects a fraction of agents per step and penalises violations, each learning agent faces a Mon-MDP-like signal: penalties appear only when inspected. Parisi's result predicts that naive learners will learn "behave well when watched". It also gives conditions (truthful, ergodic monitoring) under which agents can still learn the true objective. The "Button" monitor is a model of agents being able to switch off or avoid the monitor.

**Missing.** Single agent; the monitor has no objective, no budget choice and no capability; nothing on several agents sharing one monitor. Follow-up multi-agent work may exist **[unverified; not in this set]**.

### 2.6 Ye & Steinhardt 2026, Norm Enforcement for AI Agents

**Players.** n LLM agents in a shared environment, each with a goal and a reward (Sec. 3). Environments: social media marketing, chatbot competition, and a fishery commons with logistic regrowth adapted from GovSim (Sec. 4.1). The overseer is the "mechanism": a fixed platform rule. Agents also act as monitors, since they can file reports.

**Observations.** The mechanism sees reports (target + justification). When it verifies, its LM verifier reads the reported agent's recent actions and the reporter's justification and outputs 0, 0.5 or 1 (Sec. 6.1). It does not watch agents unprompted.

**Actions and budget.** Remove an agent (all later rewards zero). Verify a report (one LM call). Backfire also removes reporters whose reports fail. Cost is measured as verifier token use, normalised to Checked (Table 1). The paper fixes the evidence source (peer reports) and the sanction (removal), and varies only how evidence is weighed.

**Policy.** Five fixed rules. Naive: remove anyone reported. Checked: verify every report. Backfire: verify and remove false reporters. RepVote: each agent has reputation ρ_i = (α + v_i)/(α + β + v_i + f_i) (Beta prior, α=2, β=1); reports add ρ_i to the target's suspicion S_j; verify when S_j ≥ θ. EscRepVote: the j-th false report counts k·j, so mixed reporters' reputation keeps falling (Sec. 6.2, Fig. 4).

**Capability.** Verifier is Qwen3-0.6B, chosen because larger verifiers cost too much at scale. Agents are Qwen3-Next-80B-A3B. Evaluation uses a stronger gpt-5-mini judge V_N as ground truth (81–89% agreement with a Docent check).

**Overseen agents.** 50/50 good and bad. Bad agents are either fine-tuned on unrelated bad data (bad advice, insecure code) or prompted (three levels). They adapt in context over T = 15 rounds; no weight updates during play.

**Findings.** Under Naive, bad agents file false reports to remove competitors and good agents are removed about as fast as bad ones in social and chatbot (Fig. 2c). In fishing, fine-tuned bad agents mostly report accurately, because other agents' harvesting hurts them directly (Sec. 5.2). Reputation mechanisms raise TPR-vs-FPR area at equal or lower verifier cost; EscRepVote is first or second in 11 of 12 settings.

**Formal core.** Agents I, LM policies π_i(a | history). Norm oracle V_N(τ) ∈ {0,1}. Mechanism = (evidence = reports, weighting rule, sanction = removal). Evaluation = TPR and FPR of removals over rounds, plus verifier cost.

**Borrow.** (1) Peer reporting as a cheap evidence channel, and the warning that it becomes an attack surface. (2) Reputation as a prior over whom to check, which is a budget allocation rule. (3) A weak verifier with a strong evaluation judge separates overseer capability from ground truth. (4) The fishery result: in a commons, reports align with self-interest more than in zero-sum settings.

**Missing.** Agents do not learn by RL; the mechanism does not learn; the overseer has no spatial or partial view of its own; removal is the only sanction. The authors list game-theoretic grounding as future work (Sec. 7).

### 2.7 Wang et al. 2026, BazaarBench (Du is a corresponding author)

**Players.** 100 LLM agents trading second-hand goods for 30 simulated days (base market), then 7-day continuations where 20 agents switch to a tested model under ordinary, deadline-pressure or adversarial prompts (Sec. 3). There is **no online overseer**. The platform validates calls; in L0 it blocked listings that failed a title-based inventory check, in L1–L3 it only logged the mismatch (Sec. 4.1). Traders can rate and report each other (Table 2); what a report does is not described in the text I read **[unverified]**.

**What an overseer could observe.** The platform keeps an event log of every call (agent, tick, inputs, result, including blocked calls), private inventory records (true owner, condition at entry, acquisition cost), listings, offers, messages, simulated photos with privacy flags, and recorded reasoning summaries (App. E.3). So a ground-truth record exists for F1–F3 (overstated condition, unowned item, overcommitment), and claims can be checked against it. F4–F6 (premature close, PII, false reputation claims) need reading text.

**What an overseer could do.** The paper implies these levers but does not use them online: block a call; withhold a listing; show an inventory warning to the buyer (2,319 warnings were saved but not shown, App. E.3); stop a deal before completion; sanction reputation. Buyers already have an "inspect" action, but it returns the stored quality value, which is not an independent check (App. E.3).

**The judge.** Offline GPT-5 with high reasoning labels F4, F5 messages, F6 and S1 from reasoning; scripts handle F1–F3 (App. I.1). 94% agreement with humans on 200 samples. The judge is not blind to model or condition (App. I.2). Cost: about 2.3 billion input and 0.36 billion output tokens, roughly $6,487 at listed GPT-5 prices, for 48 runs (App. D). That is a hard number for what full, after-the-fact LLM oversight of a 100-agent market costs.

**Stage ladder.** S1 considered → S2 attempted → S3 reached another trader → S4 acted upon → S5 carried through (Table 3). Once an overcommitment reaches S3 it almost always reaches S5 (141 of 143 for GPT-5.5, Sec. 4.2). Intervention must happen at S2–S3 to matter.

**Formal core.** No game. State = market records; failure object = (type, object); label = furthest supported stage.

**Borrow.** A ready testbed where the overseer's observation and action sets are well defined by the platform API: observe the event log (or a sample), check against private records, act by blocking or flagging. The stage ladder gives timing for interventions. The judge cost gives a budget scale (tokens per agent-day).

**Missing.** No overseer, no budget trade-off, no adaptation by traders to oversight, no learning. Welfare is earnings only.

### 2.8 Casheekar & Tangirala 2026, Learning to Report Unsafe Tasks in a Multi-Agent Game

**Players.** n agents and a set of tasks. Each task e has a benefit b_e and a set of witnesses W_e; any witness can stop it by reporting. Agents share one team reward that pays for completed tasks. An auditor chooses audit probabilities a_i (Sec. 2). The auditor is outside the learning: it commits to an allocation and does not adapt. The agents are themselves the first line of monitoring.

**Observations.** One audit reveals whether one agent stayed silent in an unsafe context. Verification is perfect, and the auditor knows which contexts are unsafe and who witnesses what (Sec. 7 lists these as assumptions).

**Actions and budget.** Audit agent i with probability a_i at price κ_i; a silent audited agent pays fine F_i in unsafe contexts. Budget = Σ κ_i a_i.

**Policy.** A fixed allocation. The reference allocation is the cheapest one that meets their "strict elimination certificate", plus a guard against coalitions; baselines are uniform and coalition-ratio allocations (Secs. 3.2, 4).

**Overseen agents.** 64 PPO learners in one-step games with 8 contexts, starting from a policy that learned silence. Agents are grouped into shared networks; members of a group draw actions independently.

**Formal core.** With grouping G and shared silence probability p_g per group, expected team reward is J(p) = Σ_e b_e ∏_g p_g^{m_eg} − Σ_g D_g p_g, where m_eg is how many of task e's witnesses are in group g and D_g is the group's total expected penalty for silence (the paper calls it deterrence; d_i = c_i + F_i a_i) (Eq. 2). With one fully shared policy and k witnesses per task, J(p) = B p^k − D p. Silence is a local maximum whenever D < kB, even though reporting is optimal once D > B (Sec. 3.1). The certificate removes groups one at a time when D_g exceeds the remaining reward from silence; if all groups can be removed, exact gradient updates reach universal reporting (Thm. 1). Its cost is between B and k·B and equals k·B for a fully shared policy (Prop. 2).

**Findings.** Same budget, different allocation: unsafe completion differs by 42 points (Table 3). Putting co-witnesses in different networks cuts unsafe completion by 34 points. A fully shared network with a common random draw recovers in 48 of 48 runs; with independent draws, 0 of 48.

**Borrow.** This is the closest paper to our setting on the learning side: an overseer's allocation must be judged by where learning agents end up, not by whether compliance is an equilibrium. Parameter sharing, standard in SocialJax baselines, changes how much oversight is needed. The audit-the-reporters structure (oversee the peer monitors) is a two-level design.

**Missing.** One-step game; no resource dynamics; perfect, omniscient auditor with no capability limit; auditor never learns; contexts are all-unsafe or all-safe.

### 2.9 Gans & Holden 2026, When Does Randomized Oversight Align AI Agents That Can Conceal? (skim)

A provider commits to a random choice among K evaluators, each able to detect one unauthorized route, with total audit probability at most B, and to a penalty size (Sec. 2). One agent picks an authorized action or a route plus a concealment plan h; concealment lowers detection probability at a private cost. The provider must make the authorized action best for every agent type. Main points for us: stronger auditing can make agents that still violate hide better, so the catch rate can fall as auditing rises; audit requirements become infinite when evidence can be erased cheaply; if agents can learn the audit draw in advance, there is a floor on audit probability that larger penalties do not remove (Intro, Props. 1–5). Single agent, no dynamics. Useful as a warning: an overseer's observation channel is itself something agents act on.

### 2.10 Zhu, Zuo & Sun 2026, Blindfold: Keeping the Audit Coin Secret from LLM Agents (skim)

A defender audits a random fraction ε of an LLM agent's actions and estimates its violation rate from audited labels (Sec. I). The estimate is unbiased only if the coin does not leak into what the agent sees before it acts ("pre-commit noninterference"); the worst-case bias is (1 − ε) times the leak's total-variation distance, and raising ε does not fix it. On their monitor, a silent audit added latency visible in tool timestamps; a fresh coin per call closes that channel. Single agent. For us: a sampled-inspection overseer in an SSD is only meaningful if agents cannot tell, from their observation, that they are being inspected this step.

---

## 3. Design space

Each paper fixes one choice on each axis below. Listed are every distinct choice these ten papers make.

### 3.1 Axes and the choices taken

**A. Where the overseer sits**
- A1. A player with its own objective in a two-player game (Griffin, Engels, Greenblatt).
- A2. A committed design outside the agents' game, chosen once (Casheekar, Gans, Ye, Makins, Zhu).
- A3. Part of the environment, controlling what agents observe (Parisi).
- A4. Offline evaluator only, no influence on play (BazaarBench).

**B. What one observation reveals**
- B1. Noisy score of one agent's latest output, every step (Griffin, Greenblatt T, Makins).
- B2. Exact verdict on one sampled decision, at a cost (Greenblatt H, Casheekar, Gans, Zhu).
- B3. Peer reports, then a check of the reported agent's recent history (Ye).
- B4. A whole interaction transcript (Engels Debate, Mafia).
- B5. Records vs claims across a market log (BazaarBench judge).
- B6. Aggregate view across agents (Makins' joint monitor, tested only as an upper bound).
- B7. Reward visibility to the agent (Parisi, inverted direction).

**C. What the overseer can do**
- C1. Let pass / defer to a safe fallback / audit (Griffin, Greenblatt).
- C2. Shut down or remove (Greenblatt shutdown, Ye removal, Engels Mafia vote).
- C3. Edit the agent's output (Greenblatt trusted editing).
- C4. Flag (catch) only, game ends (Makins, Engels Backdoor).
- C5. Penalise reward (Casheekar fines, Gans penalties).
- C6. Restrict affordances in advance (Makins permissions).
- C7. Jam communication or redact inputs (Greenblatt).
- C8. Test a monitor with planted violations (Greenblatt).
- C9. Estimate a rate / certify (Zhu, BazaarBench).
- C10. Free-form actions narrated by an environment (Engels Wargames).

**D. Budget unit**
- D1. Count of audits per episode, in the state (Griffin, Greenblatt).
- D2. Sum of audit probabilities or prices (Casheekar, Gans, Zhu).
- D3. False-positive rate (Greenblatt, Makins).
- D4. Compute or tokens (Makins share of executor compute, Ye verifier tokens, BazaarBench judge dollars).
- D5. Monitoring cost in reward (Parisi).
- D6. None (Engels).

**E. Policy class**
- E1. Threshold on a score, possibly time- and budget-dependent (Griffin, Greenblatt, Makins).
- E2. Solver output: DP or maximin (Griffin).
- E3. Randomised mixture or committed lottery (Griffin, Gans, Zhu).
- E4. Allocation from an analytic certificate (Casheekar).
- E5. Rule with state, e.g. Bayesian reputation (Ye).
- E6. Prompted LLM judge (Makins, Engels, Ye verifier, BazaarBench).
- E7. None, environment process (Parisi).

**F. Number and coordination**
- F1. One overseer.
- F2. One overseer with a scarce stronger tier (Greenblatt T then H).
- F3. Many independent per-agent instances, no sharing (Makins).
- F4. Many peer monitors feeding a central verifier (Ye, Casheekar).
- F5. Several overseers who talk and vote (Engels Mafia).
- F6. A chain of overseers of rising strength (Engels NSO).
- F7. Ensemble or majority vote of calls (Greenblatt T+U scores, Engels 5-call vote).

**G. Capability relation**
- G1. Weaker trusted vs stronger untrusted (Greenblatt, Griffin, Makins nano).
- G2. Comparable (Greenblatt untrusted monitor, Makins mini).
- G3. Continuous Elo gap, measured (Engels).
- G4. Much weaker cheap verifier with a strong offline judge for ground truth (Ye).
- G5. Perfect when it looks (Casheekar, Gans, Parisi truthful monitor).

**H. Timing and solution concept**
- H1. Stackelberg with worst-case best response (Griffin, Greenblatt).
- H2. Commitment that must work for every agent type (Gans).
- H3. Commitment, followers run gradient learning; judge by the learning limit (Casheekar).
- H4. Fixed mechanism, in-context adaptive agents, measured empirically (Ye, Makins).
- H5. Pairwise match-ups fit to Elo (Engels).
- H6. Adaptive adversary picks when to act (Zhu).
- H7. No game (Parisi single-agent MDP; BazaarBench evaluation).

**I. Overseen agents**
- I1. One scheming model (Griffin, Greenblatt, Zhu, Gans, Engels).
- I2. Many coordinating attackers with a planner (Makins).
- I3. Mixed good/bad LLM population, competing (Ye).
- I4. Many RL learners with parameter sharing and a team reward (Casheekar).
- I5. One RL learner (Parisi).
- I6. Large LLM market under pressure prompts (BazaarBench).

**J. Objective**
- J1. Safety vs usefulness frontier (Griffin, Greenblatt).
- J2. Undetected success vs FPR (Makins).
- J3. Win rate and scaling slope (Engels).
- J4. TPR/FPR of sanctions over time plus cost (Ye).
- J5. Unsafe vs legitimate completion (Casheekar).
- J6. Failure stages and earnings (BazaarBench).
- J7. Return including monitor cost (Parisi).
- J8. Unbiased rate estimate (Zhu).

### 3.2 Combinations nobody here has studied in a sequential social dilemma with learning agents

None of the ten papers puts an overseer into a temporally extended, partially observed social dilemma with many RL learners. Specific empty cells:

1. **A1 or H1 with I4 in an SSD.** An overseer that is a player (own objective, budget in its state, Griffin-style) against many learning agents over a shared, regrowing resource. Griffin and Greenblatt have the player-overseer but one stateless adversary; Casheekar has learners but a one-step game and a committed auditor.
2. **A learned overseer policy (E learned by RL) co-trained with learning agents.** Every overseer here is a threshold, a solver output, a fixed rule or a prompted LLM. Co-learning (overseer and agents both updating) does not appear in this set.
3. **Local, spatial overseer observation (a B1/B2 variant with a view window) plus D1 budget.** No paper gives the overseer a partial view of a grid or map. Makins' per-commit view is the nearest analogue, and his joint-monitor result says aggregation is what catches split harm.
4. **F5 or F6 with learning agents.** Several overseers who share observations or vote, or a chain of overseers, only appear with prompted LLMs (Engels). Makins' monitors never share. How several budgeted overseers with local views should split coverage and communicate is open.
5. **G1/G3 (weaker LLM overseer, capability varied on a scale) acting online on RL learners.** Ye has a weak verifier but LLM agents; Engels varies capability but in one-step games; Casheekar's auditor is perfect.
6. **Diffuse, cumulative harm (resource depletion) as the safety target.** Safety here is "one backdoor", "one undetected attack" or "one unsafe task". Ye's fishery is the only commons, and its overseer only removes agents on peer reports. Harm that builds up from many small takes, and an objective that combines welfare, sustainability and oversight cost, is not studied.
7. **Agents learning to evade a sampled inspector.** Parisi (agents learn to misbehave when unwatched), Gans (concealment and learning the draw) and Zhu (coin leakage) describe this in single-agent form. No paper here measures it with many learning agents in an SSD, where agents could also learn from each other where the overseer is not.
8. **Peer reporting plus a budgeted central overseer with RL agents in a commons.** Ye (LLM agents) and Casheekar (one-step, perfect auditor) each cover half.
9. **Honest-vs-adversarial two-mode evaluation in an SSD.** Griffin's two-mode split (usefulness when nobody defects, safety when some do) has not been used with learning populations, where the "honest" population is itself the outcome of learning under the overseer.

### 3.3 Points to carry into our formulation

- Put the budget in the overseer's state (Griffin D1) and charge it per inspection; this makes the overseer's problem a POMDP or POSG with a resource, which is solvable and comparable across settings.
- Use the honest / adversarial two-mode evaluation, with welfare (or sustainable harvest) as usefulness and some depletion or violation measure as safety.
- Judge any fixed overseer rule by where learning agents end up, not by an equilibrium alone (Casheekar), and report parameter sharing as a factor.
- Expect agents to learn "comply when inspected" (Parisi); test whether inspections are hidden from agents' observations (Zhu).
- Treat overseer capability as a measured axis (Engels) and cost as tokens or compute share (Makins, Ye, BazaarBench).
- When there are several overseers, compare local per-agent views against a shared or pooled view (Makins' fragmentation effect suggests pooling matters most when harm is spread across agents).
