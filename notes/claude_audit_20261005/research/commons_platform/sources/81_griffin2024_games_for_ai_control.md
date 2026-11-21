# Games for AI Control: Models of Safety Evaluations of AI Deployment Protocols

- **Authors:** Charlie Griffin, Louis Thomson, Buck Shlegeris, Alessandro Abate
- **Year:** 2024 (v1 12 Sep 2024; v2 7 May 2026)
- **Venue / identifier:** arXiv:2409.07985 (no peer-reviewed venue listed on the arXiv page)
- **URL:** https://arxiv.org/abs/2409.07985 (full text at https://arxiv.org/html/2409.07985)
- **Source type:** Preprint (formal / game-theoretic)
- **Date accessed:** 14 Oct 2026
- **Depth read:** Abstract plus the first ~100k characters of the HTML full text (definitions and main results).

## Verbatim quotes

1. "the protocol is suggested, and then the red team chooses the best response"
2. Audit budget: "the amount of human labour invested in overseeing the protocol"
3. "raising the audit FPR above 2% Pareto-improves upon the protocols"
4. "mixed protocols make up most of the frontier"

## Relevance to the commons overseer

Formalises a control evaluation as a multi-objective, partially observable stochastic game in which the blue team commits first (leader) and the red team best-responds. Blue policies are threshold rules parameterised by audit and defer false-positive rates; safety and usefulness are expected sums; Pareto-optimal (sometimes randomised) protocols are computed. This is the natural formal frame for an audit-rate / fine / deferral sweep in the commons, and links AI control to inspection games [96].
