# 55 — Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs

- Authors: Marcantonio Bracale Syrnikov, Federico Pierucci, Marcello Galisai, Matteo Prandi, Piercosma Bisconti, Francesco Giarrusso, Olga Sorokoletova, Vincenzo Suriani, Daniele Nardi
- URL: https://arxiv.org/abs/2601.11369 (HTML: https://arxiv.org/html/2601.11369)
- Posted: 2026-01-16 (v1); v2 2026-01-20
- Accessed: 2026-10-12
- Type: arXiv preprint (cs), empirical LLM multi-agent study
- Credibility 3/5 (preprint, N=90 runs/condition, six model configs) · Recency 4/5 · Bias 3/5 (authors promote their own "Institutional AI" framework)
- Note: same first author and group as the already-cited 2607.22188 ("Draining the Energy Commons"), but this is a DIFFERENT paper. Check whether the paper cites this one; the cited-works list given to me names only 2607.22188.

## Verbatim quotes
- Abstract: "an Oracle/Controller runtime interprets this manifest, attaching enforceable consequences to evidence of coordination while recording a cryptographically keyed, append-only governance log for audit and provenance."
- Abstract: "the Institutional regime produces large reductions in collusion: mean tier falls from 3.1 to 1.8 (Cohen's d=1.28), and severe-collusion incidence drops from 50% to 5.6%. The prompt-only Constitutional baseline yields no reliable improvement, illustrating that declarative prohibitions do not bind under optimisation pressure."
- Body (Sec. on deterrence): "let S be the expected discounted sanction loss conditional on escalation (fines plus expected loss from suspension/cooldowns). A sufficient deterrence condition is pS ≥ Δπ."
- Body: "the Or-else clause determines the institutional response when a violation is detected (e.g., Active → Warning plus a recorded warning, or Warning → Fined with a specified monetary penalty and duration)."

## Threat to claim 1: high — prior LLM-agent work already shows that consequence-bearing, history-recorded escalation (warning recorded, then fine, then suspension) cuts harmful behaviour while a prohibition alone does nothing. Our "audits only work through consequences; a remembering reviewer works without fines" must be positioned against this: their escalation still uses fines/suspension; ours removes harm with allowance cuts and no fines, in a commons with a reviewer. Cite and distinguish explicitly.
## Threat to claim 2: medium — they already write the Becker condition pS ≥ Δπ for LLM agents. Our contribution must be the advance prediction of the threshold (5/5) and the q·s·F vs whole-game gain test, not the condition itself.
