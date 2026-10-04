# Literature Search and Reading Record

Date: 21 September 2026. Scope: sequential commons, bounded oversight,
compositional safety, and the proposed five-game expansion. No personal contact
email was supplied to a research service. This is a focused review, not a
systematic review or an exhaustive novelty search.

## Full-Text Verification

The research-papers skill's `fetch_and_parse` helper retrieved the papers below.
Every read-plan interval was covered in order, subdivided into smaller terminal
reads where needed to prevent output truncation. End markers were checked
against the helper's output. Appendices and references were included.

| Paper/version read | Coverage | Matching end marker |
| --- | --- | --- |
| GovSim, arXiv:2404.16698v4 | Parsed lines 1-4403 | `END-OF-PAPER:4615c17b27ec` |
| Safe Multi-Agent Reinforcement Learning via Shielding, arXiv:2101.11196v2 | Parsed lines 1-1072 | `END-OF-PAPER:0b44abd7a9a4` |
| A Benchmark for Scalable Oversight Mechanisms, arXiv:2504.03731v1 | Parsed lines 1-805 | `END-OF-PAPER:4e115dd1930e` |
| SocialJax, arXiv:2503.14576v3 | Parsed lines 1-1786 | `END-OF-PAPER:6bc677c77ff5` |
| Contract-Based Compositional Shielding, arXiv:2606.14130v2 | Parsed lines 1-2018 | `END-OF-PAPER:54ece2f39cc7` |
| When Local Monitors Miss Compositional Harm, arXiv:2607.11751v1 | Parsed lines 1-1845 | `END-OF-PAPER:cde3b7f979f9` |
| Scaling Laws for Scalable Oversight, arXiv:2504.18530v3 | Parsed lines 1-1934 | `END-OF-PAPER:dd1e3885584a` |
| Institutional Monitoring and Ledgers, author repository PDF | Parsed lines 1-1268, 39 PDF pages | `END-OF-PAPER:8d8a1bbcbe82` |
| RICE-N, arXiv:2208.07004v1 | Parsed lines 1-2551, both read-plan intervals | `END-OF-PAPER:a94de6df529c` |

The oversight benchmark's current arXiv landing-page title is *A Benchmark for
Scalable Oversight Protocols*; the retrieved v1 full text says *Mechanisms*.
The landing page reports acceptance at the ICLR 2025 BiAlign **workshop**.
It should not be described as an ICLR main-conference paper.

Full-text caches are under `~/.cache/research-papers/`. HTML parsing duplicates
some rendered equations and loses some figure content. The original paper,
not the parser's typography, remains authoritative.

## Access Issues

- The direct MDPI URL for *Institutional Monitoring and Ledgers for Cooperative
  Human-AI Systems* returned a short access stub through the helper and HTTP 429
  through web browsing. Further access attempts/status are recorded below when
  available; its abstract alone must not be treated as a full-text audit.
- Recovery succeeded through the author's repository, at commit
  `2f872ee8ed4d4f79328fa0c5d4208d0757bbc88a`, file `paper_artifact/main.pdf`.
  This is an author revision with editorial highlighting and placeholder
  journal headers/DOI. Do not cite the placeholder DOI or equate this file
  byte-for-byte with the publisher's version of record.
- The HDO OpenReview attachment returned a browser-verification page, despite
  the helper reporting `ok`. Its marker `3f8f5b92d484` authenticates only the
  access stub, not a paper. HDO is not counted as fully read.
  The alternative `openreview.net/pdf?id=l5Wrcgyobp` also returned an access
  stub. Its indexed abstract is available, but detailed guarantees and
  experiments have not been verified here. It is not a foundation for the
  recommendation.

## Discovery Register

Twenty-two candidates were screened. **Full** means complete parsed text was
read, including appendices and references. **Discovery** means title/metadata
and relevance were checked; it does not mean a full methods audit. The nine
full readings were selected for direct overlap with the proposed contribution
or environment decision. Citation counts were not used or estimated.

| Source, year, status | Stable source / identifier | Depth and reason |
| --- | --- | --- |
| Leibo et al., 2017, AAMAS | [Multi-agent Reinforcement Learning in Sequential Social Dilemmas](https://arxiv.org/abs/1702.03037) | Discovery; foundational policy-level social dilemmas; already cited in the manuscript. |
| Perolat et al., 2017, NeurIPS | [A Multi-Agent Reinforcement Learning Model of Common-Pool Resource Appropriation](https://papers.neurips.cc/paper_files/paper/2017/hash/2b0f658cbffd284984fb11d90254081f-Abstract.html) | Discovery; resource appropriation lineage. |
| Leibo et al., 2021, ICML | [Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot](https://arxiv.org/abs/2107.06857) | Discovery; external evaluation infrastructure and population variation. |
| Agapiou et al., 2022, technical report | [Melting Pot 2.0](https://arxiv.org/abs/2211.13746) | Discovery; broad substrate suite, not evidence for our proposed oversight protocol. |
| Guo et al., 2025 preprint / ICLR 2026 version | [SocialJax](https://arxiv.org/abs/2503.14576) | Full v3; game mechanics, dilemma validation, action and metric differences, compute caveats. |
| Piatti et al., 2024, NeurIPS | [Cooperate or Collapse / GovSim](https://arxiv.org/abs/2404.16698) | Full v4; scenario equivalence, agent architecture, capability questions, newcomer and communication tests. |
| Elsayed-Aly et al., 2021, AAMAS | [Safe Multi-Agent Reinforcement Learning via Shielding](https://arxiv.org/abs/2101.11196) | Full v2; joint and factored shielding, abstraction assumptions, coordination and intervention. |
| Adalat, Hamel-De le Court and Belardinelli, 2026, arXiv record reports EUMAS acceptance | [Contract-Based Compositional Shielding for Safe Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2606.14130) | Full v2; certified local obligations, decentralized execution, assumptions behind optimality and safety. |
| Hu and Wang, 2026, preprint | [When Local Monitors Miss Compositional Harm](https://arxiv.org/abs/2607.11751) | Full v1; directly overlapping motivation; strong local baseline caveat in Appendix D. |
| Bowman et al., 2022, arXiv paper | [Measuring Progress on Scalable Oversight for Large Language Models](https://arxiv.org/abs/2211.03540) | Discovery; original supervision framing, not re-audited in full this pass. |
| Sudhir, Kaunismaa and Panickssery, 2025, BiAlign workshop | [A Benchmark for Scalable Oversight Protocols](https://arxiv.org/abs/2504.03731) | Full v1; incentives for truthful rather than merely persuasive answers; limited GSM8K evaluation. |
| Engels, Baek, Kantamneni and Tegmark, 2025, NeurIPS | [Scaling Laws for Scalable Oversight](https://arxiv.org/abs/2504.18530) | Full v3; task-specific performance fitting, explicit dimensionality warning, sequential-risk limitations. |
| GPT-5 and Mike Bronikowski, 2025 OpenReview manuscript; venue status not established here | [Scalable Oversight in Multi-Agent Systems / HDO](https://openreview.net/forum?id=l5Wrcgyobp) | Abstract only; full-text access blocked. Do not repeat its guarantee claims as established evidence. |
| Alqithami, 2026, Mathematical and Computational Applications 31(3), 69 | [Institutional Monitoring and Ledgers for Cooperative Human-AI Systems](https://www.mdpi.com/2297-8747/31/3/69) | Full author revision; publisher listing independently located. Narrow monitored rule, probabilistic review, capacity limitations. |
| Ostrom, 1990, scholarly book | [Governing the Commons](https://doi.org/10.1017/CBO9780511807763) | Discovery/context; whole book not read in this pass. Motivation, not numerical calibration. |
| Alechina, Halpern, Kash and Logan, 2017, AAAI | [Incentivising Monitoring in Open Normative Systems](https://ojs.aaai.org/index.php/AAAI/article/view/10610) | Discovery; priority follow-up if strategic peer monitoring is added. |
| Zhang et al., 2022, RICE-N report | [AI for Global Climate Cooperation](https://arxiv.org/abs/2208.07004) | Full v1; delayed stock effects, negotiation versus binding masks, model limitations, CPU/GPU roles. |
| Radulescu, Vrancx and Nowe, 2017, arXiv record reports ALA workshop at AAMAS | [Analysing Congestion Problems in Multi-agent Reinforcement Learning](https://arxiv.org/abs/1702.08736) | Discovery; possible congestion candidate only. Persistent queues and an appropriate safety task remain unverified. |
| Pretorius et al., 2020, arXiv record reports NeurIPS | [A game-theoretic analysis of networked system control for common-pool resource management using multi-agent reinforcement learning](https://arxiv.org/abs/2010.07777) | Discovery; possible networked-resource alternative, not yet selected. |
| Agarwal et al., 2021, NeurIPS | [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) | Discovery; uncertainty reporting reference, not justification for a particular sample size here. |
| Du et al., 2023, review preprint | [A Review of Cooperation in Multi-Agent Learning](https://arxiv.org/abs/2312.05162) | Discovery; terminology and broader cooperation map. |
| Christoffersen, Haupt and Hadfield-Menell, 2023, AAMAS | [Get It in Writing: Formal Contracts Mitigate Social Dilemmas in Multi-Agent RL](https://arxiv.org/abs/2208.10469) | Discovery through IML references; alternative incentive mechanism, not reproduced. |

An earlier shorthand reference, "Governing multi-agent systems", was not resolved
to an unambiguous bibliographic record in this pass. Do not invent an entry from
that phrase. The institutional sources above provide verifiable starting points.

## Visual Checks

Viewed the extracted GovSim interaction loop, the 2021 factored-shield schematic,
SocialJax's Cleanup environment, the Scaling Laws conditional-success plot,
RICE-N's decision/dynamics schematic, and IML's institutional-layer schematic.
No figures were copied into this repository. These checks support the method
comparisons; this pass does not redesign the paper's figures.

## Search Limits

Coverage includes foundational candidates from 1990/2017 and directly related
papers available by 21 September 2026. New 2026 work matters to the novelty
assessment. This does not establish priority or an exhaustive absence of prior
work. The final paper still needs the foundational and statistical sources read
in full, pinned external code inspected, and implementation/licence checks.
No candidate repository was executed and no published experiment was reproduced.

## Search Scope

Discovery used primary arXiv, conference, publisher, and author/repository pages.
Search clusters: GovSim and common-pool appropriation; SocialJax and Melting Pot;
centralized/factored/compositional shielding; scalable oversight protocols and
capability measurement; institutional monitoring; congestion and climate games.
Candidate selection prioritizes direct methodological overlap and game dynamics,
not citation counts or the number of named scenarios.
