# Literature TODO

No new citations were added in this cleanup pass. The current paper already cites commons governance, sequential social dilemmas, LLM-agent strategy populations, and scalable oversight sources. Before submission, the following claims should be checked against real sources and cited only if the sources are verified.

## Needs Citation Or Stronger Positioning

| Claim area | Source type needed | Notes |
| --- | --- | --- |
| Monitoring, enforcement, and institutional fit in commons governance | Commons/institutional governance sources | Existing Ostrom-style citations may be enough, but check whether monitoring/enforcement claims need additional support. |
| Mechanism design or incentive interventions in multi-agent systems | Multi-agent mechanism design survey or benchmark paper | Useful if the paper discusses oversight as intervention rather than only evaluation. Candidate titles from notes: "Governing multi-agent systems" and "Incentivising Monitoring in Open Normative Systems"; verify exact bibliographic details before citing. |
| Scalable oversight as capability-gap measurement | Scalable oversight benchmark/formulation papers | Current Bowman, Sudhir et al., and scaling-law citations support this direction. The HDO multi-agent oversight paper mentioned in meeting notes still needs exact title/authors/link before it can be cited. |
| LLM-generated strategies as inspectable artifacts | Willis/Du/Leibo papers and related LLM-agent evaluation work | Use to justify strategy banks rather than live action selection. |
| Sequential social dilemmas and Harvest-style commons | Original SSD/Melting Pot/SocialJax-style benchmark sources | Needed for positioning the benchmark substrate. |
| Local vs global safety or compositional safety failure | Formal safety/compositionality or multi-agent safety sources | This is central to the paper and should be literature-grounded if there is a close source. |
| Benchmark construct validity and uncertainty | Benchmark-validity papers, NIST or measurement-focused sources, safety benchmark reviews | Needed if the paper makes stronger benchmark-quality claims. Search specifically for benchmark accuracy vs generalized accuracy, uncertainty estimation, and AI-safety benchmark construct validity critiques. |
| Live LLM-agent safety benchmarks | Agent-safety benchmark papers | Only needed for future-work positioning unless live LLM agents are actually added. Candidate names from notes: Agent-SafetyBench, OpenAgentSafety, SafeArena. Verify before citing. |

## Manual Literature Search Recommendation

Search for sources in:

- scalable oversight benchmarks;
- multi-agent oversight or supervision;
- multi-agent mechanism design;
- compositional safety in multi-agent systems;
- sequential social dilemma benchmarks;
- LLM-agent evaluation and generated-policy populations.
- benchmark validity and uncertainty estimation;
- live LLM-agent safety benchmarks, only for future-work comparison.

## Verified In This Pass

- Sudhir, Kaunismaa, and Panickssery (2025), "A Benchmark for Scalable Oversight Protocols," arXiv:2504.03731. Added to `refs.bib` and cited in Related Work.

## Not Yet Verified

- HDO multi-agent oversight paper from meeting notes.
- "Governing multi-agent systems."
- "Incentivising Monitoring in Open Normative Systems."
- NIST benchmark validity / generalized accuracy source.
- Recent AI-safety benchmark review criticizing vague definitions or weak measurement practices.
- Concordia, Agent-SafetyBench, OpenAgentSafety, and SafeArena, unless the project moves toward live-agent evaluation.

Do not add references from memory unless the title, authors, venue/arXiv ID, and claim relevance are verified.
