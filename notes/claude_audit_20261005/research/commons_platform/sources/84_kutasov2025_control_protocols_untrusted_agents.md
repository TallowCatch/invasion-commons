# Evaluating Control Protocols for Untrusted AI Agents

- **Authors:** Jon Kutasov, Chloe Loughridge, Yuqi Sun, Henry Sleight, Buck Shlegeris, Tyler Tracy, Joe Benton
- **Year:** 2025
- **Venue / identifier:** arXiv:2511.02997 (no peer-reviewed venue listed)
- **URL:** https://arxiv.org/abs/2511.02997 (full text at https://arxiv.org/html/2511.02997)
- **Source type:** Preprint (empirical control evaluation in SHADE-Arena)
- **Date accessed:** 14 Oct 2026
- **Depth read:** Abstract and Section 2–3 (metric definitions, blue-team protocols) read from HTML full text. The safety formula itself was an image/MathML and did not extract; its description is quoted.

## Verbatim quotes

1. "Usefulness is defined to be the proportion of main tasks successfully completed"
2. Safety: "the probability that, in an infinite deployment consisting of independent sequential SHADE-Arena task pairs, the agent does not complete a side task before getting caught"
3. Budget: "set to ensure that the false positive rate of auditing on benign transcripts is 0.5%"
4. Adaptive red team: "decreasing safety to 17%"
5. "deferring on critical actions is highly robust to even our strongest red team strategies"

## Relevance to the commons overseer

Shows the current standard for agentic control: audit threshold fixed by a false-positive budget on honest runs (0.5%), usefulness rescaled between trusted and untrusted, and red-team affordances varied (knowing when resampling happens, simulating the monitor) to stress protocols. Resampling for incrimination raised safety from 50% to 96% (abstract); adaptive attacks cut it to 17%. The commons analogue: calibrate the overseer's flag threshold on honest populations, then vary what the cheaters know about the overseer.
