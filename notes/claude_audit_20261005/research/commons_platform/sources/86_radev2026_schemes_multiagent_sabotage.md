# The Best-Laid SCHEMEs: Coordinated Sabotage and Monitoring in Multi-Agent Systems

- **Authors:** Nikolay Radev, Lennart Haas, Benjamin Arnav, Pablo Bernabeu-Pérez
- **Year:** 2026 (27 May 2026)
- **Venue / identifier:** arXiv:2605.29178 (cs.CR, cross-list cs.MA)
- **URL:** https://arxiv.org/abs/2605.29178 (HTML at https://arxiv.org/html/2605.29178)
- **Source type:** Preprint / benchmark (multi-agent control)
- **Date accessed:** 14 Oct 2026
- **Depth read:** Abstract and introduction read; section headings for monitoring and topology seen; results sections not read in detail.

## Verbatim quotes

1. "no proper subset of agents can succeed alone"
2. Monitor: "a trusted monitor flags coordinated sabotage at FPR 2% in 99%/68% (Gemini/Codex) of attempts from code edits alone, rising to 100%/81% with access to agent communication"
3. "telling agents a monitor is present halves Gemini’s sabotage success without significantly breaking detection"

## Relevance to the commons overseer

The closest multi-agent control evaluation found: several agent instances must coordinate a hidden objective, communication topology is varied, and a trusted monitor is scored at a fixed 2% FPR, with and without access to agent messages. Directly maps to a commons with talk: does the overseer read the chat? Note its result (announcing a monitor halves sabotage) contrasts with Ameer's L4 finding that announcing audits with a 0 t fine nearly doubles gpt-oss over-taking, which is worth stating in the paper.
