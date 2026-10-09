# AAAI-27 submission version

- `main.tex` is the anonymous submission, built with the official AAAI-27 author kit (`aaai2027.sty`, `aaai2027.bst`; the full kit is in `../aaai_kit/AuthorKit27`). It is a single source file, as AAAI requires: the figure and table sources are inlined, and there is no `\input`.
- `supplementary.tex` holds the supplementary document: the settings of every study, key terms, the reviewer figure, the audit-rules table, the LLM controls, the post hoc behaviour, and protocols.
- The draft in `../draft/main.tex` is the working copy. **Edit the text there first, then regenerate**, or edit both.

Build (needs `newtx`, `tex-gyre`, `xstring`, `placeins` and `kastrup`; these were installed in user mode on 2026-10-09):

    pdflatex main && bibtex main && pdflatex main && pdflatex main

Limits (official, aaai.org AAAI-27 submission instructions): up to 7 pages of content, with pages 8–9 for references only. The current paper is about 6 pages of content plus 1 page of references.

**Still to do:**
- the reproducibility checklist (`../aaai_kit/AuthorKit27/ReproducibilityChecklist.tex`);
- an anonymised code and data supplement.
