# 61 — Why LLM Agents Collapse Without Oversight: The Enforcement Gap

- Author: Yuhang Wang
- URL: https://arxiv.org/abs/2609.15293
- Posted: 2026-09-14 (v1); v3 2026-09-23. Comment: "27 pages, 3 figures, 8 tables. Submitted to ICLR 2027"
- Accessed: 2026-10-12
- Type: arXiv preprint (cs.AI), single author, agent-security setting
- Credibility 2/5 (single author, under review) · Recency 5/5 · Bias 3/5

## Verbatim quotes
- Abstract: "This single control-flow change exposes the \textbf{enforcement gap}: the controller receives a safety flag and executes anyway."
- Abstract: "Separating detection probability $p_d$ from enforcement probability $p_e$ establishes that $p_e \approx 0$ by default across every framework we tested, making detection quality \emph{formally irrelevant} to security when enforcement is absent"

## Threat to claim 1: low-medium — same headline idea ("detection without consequences does nothing") in a different domain (prompt-injection defence, not incentives). Reviewers at ICLR may see it. Cite in one line as concurrent support from agent security; our setting is strategic agents with payoffs.
