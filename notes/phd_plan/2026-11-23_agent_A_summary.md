# Agent A summary: decentralised overseer formulation and testbeds (23 Nov 2026)

Branch `claude/decentralised-formulation`, from `claude/r1-s1-oversight-experiments` at 55a07b6. Notes only: no code and no runs.

## What I wrote

| File | What it is |
|---|---|
| `2026-11-23_decentralised_overseer_formulation.md` | **Section 1:** how Du's group sets up decentralised problems (DMPO 2022 and NMI 2024, both read in full). **Sections 2–4:** the decentralised overseer problem written the same way. **Sections 5–6:** locality, the restated claims C1–C4 and the new claims D1–D3. **Section 7:** the scorecard. **Section 8:** the literature |
| `2026-11-23_testbed_candidates.md` | Every SocialJax env, the relevant Melting Pot substrates and the LLM commons, in tables, checked against source code and arXiv. Ends with a recommendation |
| this file | the summary |

## What the two Du papers do

- **Networked MDP:**
  - a graph G = (V, E);
  - local policies π_i(a_i | s_{N_i});
  - global reward r = (1/n) Σ r_i;
  - fully cooperative.
- **Extended value bound:** |V_i(s) − V_i(s_{N_i^κ})| ≤ r_max γ^κ / (1 − γ). This is DMPO's Theorem 1 and NMI's Theorem 3; it rests on Qu, Wierman & Li's exponential-decay result.
- **ξ-dependent system (NMI only):** sup over s, a of the TV distance between p(·|s,a) and some factorised p̄(·|s,a) is at most ξ. It adds 2 r_max γ ξ / (1 − γ)² to the return bound (Corollary 1).
- **Presentation:** theory first, then many testbeds of growing size (8 up to 199 and 436 agents), with a central reference (CPPO) and an independent one (DPPO/IL), and ablations on κ and rollout length tied to named theorems.
- **Communication:**
  - DMPO 2022 discusses it only in words;
  - NMI 2024 reports "about 5–35%" of the centralised cost (Supplementary Table 3) and a message size of O(|s_i|).

## Key formulas in the new formulation

- **Set-up:** M overseers, each with:
  - a region R_m;
  - a budget k_m, with Σ_m k_m = K;
  - a view of only the steps an agent spent inside R_m, within the last w steps;
  - optional messages to neighbouring overseers, with range κ and cost c_msg.
- **Objective:** maximise W − λ · (check cost + message cost), subject to the agents best-responding.
- **M = 1 is the central overseer.** So every Harvest result so far is the baseline.
- **Locality:**
  - χ is the crossing rate: the chance per step that an agent changes region;
  - ζ is harm containment: the share of a violation's harm that stays in its region;
  - ξ_loc = 1 − (1 − χ)^{w−1} ζ plays ξ's role.
- **Catch chance (derived under stated assumptions):**
  - central: P_C = (1 − α)[1 − (1 − q)^w];
  - independent overseers: P_0 = (1 − α) q [1 − ((1 − q)(1 − χ))^w] / [1 − (1 − q)(1 − χ)];
  - q is the per-step chance an agent is checked, and α the miss rate.
  - **With w = 1 (every Harvest run so far) the two are equal.** Boundaries only cost something when checks read history.
- **D3, sharing with range κ:** P_C − P_κ ≤ (1 − α) ((w − 1) χ)^{κ+1} / (κ + 1)!.
  - The loss falls geometrically in κ, the analogue of the r_max γ^κ / (1 − γ) bound.
  - A hand-over protocol costs at most a χ share of central collection.
  - The welfare half of D3 (overseers optimising their κ-hop neighbourhood) is **stated only as a conjecture**.
- **Scorecard:**
  - η = (W − W_0) / (W* − W_0), with W* a central, all-seeing, perfect inspector;
  - Δ_dec = η_central(K) − η_dec(K);
  - communication reported as a % of central collection cost (N · w record-steps per step);
  - harm to honest agents ≈ d_m β F, by region;
  - worst case: agents that migrate and hop boundaries (trained with a double-oracle loop, as in Wang et al. 2019), plus distributed violations (Makins et al. 2026).

## Recommended testbeds

1. **Harvest, Open and with rooms.** A custom `map_ASCII` with walls. The same mechanics with low and high χ, and the existing baseline.
2. **Clean Up.** Global harm and slow violations: the hard case for decentralising. Use `shared_rewards=False` and a w-step rule.
3. **Willis et al.'s commons.** Fixed groups give χ = 0 and ζ = 1/M, which isolates harm containment. It scales to 256 agents cheaply, and the adapter exists.
4. **BazaarBench.** LLM traders on a trade network, with one-step and many-step failures and real token costs. Offline replay first.

Reserves: Territory, OpenSocialJax and Mushrooms.

## Things found along the way

- **SocialJax defaults to `shared_rewards=True`** for Harvest, Clean Up, Coins, Coop Mining, Gift, Mushrooms and Territory. Overseer studies need `False`; the Harvest wrapper already sets it.
- **SocialJax ships only Harvest Open.** Closed is commented out in `registration.py` and Partnership is missing.
- **Zap/freeze appears disabled in SocialJax Harvest Open** (the commented-out "freeze check", around line 1024).
- **The reading list mislabels arXiv 2609.15293.** It is Yuhang Wang's enforcement-gap paper, not the Emergence World platform (2606.08367).
- **NMI reports two different New York agent counts:** 436 in the text, 432 in the Fig. 5 caption. Its ξ values in Fig. 2d are worded ambiguously.

## Open questions

1. **What is χ in the existing Harvest Open runs?** It decides whether w = 5–10 shows any decentralisation gap. It needs evaluation rollouts that log agent positions.
2. **Fixed regions, or regions that follow agents and budgets?** D2 predicts that fixed budgets are fragile when travel is cheap.
3. **Can the welfare half of D3 be proved for a two-level problem** where the agents learn? The exponential-decay property assumes local dynamics, which roaming agents break.
4. **Can M overseers be trained DMPO-style** (κ-hop extended values) against agents that adapt?
5. **Is one ξ_loc number per testbed enough to predict Δ_dec?** This is the paper's central test. It needs at least 3 testbeds with different ξ_loc.
