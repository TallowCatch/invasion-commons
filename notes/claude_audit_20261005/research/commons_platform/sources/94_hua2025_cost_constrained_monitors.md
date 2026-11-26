# Combining Cost-Constrained Runtime Monitors for AI Safety

- **Authors:** Tim Tian Hua, James Baskerville, Henri Lemoine, Mia Hopman, Aryan Bhatt, Tyler Tracy
- **Year:** 2025
- **Venue / identifier:** arXiv:2507.15886 (v4, 21 Oct 2025). Brief says NeurIPS 2025; the HTML mentions a 'neurips_experiments.py' script but the page does not state acceptance, so treat the venue as unconfirmed here.
- **URL:** https://arxiv.org/abs/2507.15886 (HTML at https://arxiv.org/html/2507.15886)
- **Source type:** Preprint / conference paper
- **Date accessed:** 14 Oct 2026
- **Depth read:** Abstract and Section 3 (budget constraint) read from HTML full text.

## Verbatim quotes

1. "Since running monitors and applying safety interventions are costly, the protocol also needs to adhere to an average-case budget constraint"
2. "A unified budget constraint reflects the fact that developers can trade off spending on monitoring and auditing"
3. "we more than double our recall rate compared to a naive baseline in a code review setting"
4. "We also show that combining two monitors can Pareto dominate using either monitor alone"

## Relevance to the commons overseer

The protocol for monitoring cost: maximise recall (probability a misaligned output gets an intervention) subject to one average-cost budget that covers both cheap monitor calls and expensive audits; allocate audits by likelihood ratio (Neyman–Pearson). In the commons this is the design for 'cheap LLM screening of every report + expensive true audit of a few', with the audit rate q and the screening cost on one budget line.
