# Paper draft (AAAI style)

`main.tex` is the full draft: about 6.5 pages of main text, then references, then the appendix. `refs.bib` contains only sources recorded in the literature ledger and the novelty notes. The figures and tables are read from `../figures` and `../tables`, which `experiments/oversight/make_paper_exhibits.py` builds.

Build:

    pdflatex main && bibtex main && pdflatex main && pdflatex main

**Before submission:**
- Move the body into the official AAAI-27 author kit. Its style and bibliography files replace this preamble, which only imitates the two-column layout.
- Check the page limit in the official call. A third-party guide says 7 pages of content plus references.
- Anonymise the code links.
- Re-check the scooping watch in `notes/claude_audit_20261005/novelty/README.md`.
