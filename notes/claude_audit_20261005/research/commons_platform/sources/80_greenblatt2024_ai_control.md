# AI Control: Improving Safety Despite Intentional Subversion

- **Authors:** Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, Fabien Roger
- **Year:** 2023 (arXiv v1), ICML 2024
- **Venue / identifier:** ICML 2024 (Proceedings of the 41st ICML, pp. 16295–16336, as cited by Griffin et al. [81]); arXiv:2312.06942
- **URL:** https://arxiv.org/abs/2312.06942 (full text read at https://arxiv.org/html/2312.06942v5)
- **Source type:** Peer-reviewed conference paper
- **Date accessed:** 14 Oct 2026
- **Depth read:** Full text (arXiv HTML v5) searched; Sections 2–3 and Table 2 read. Math symbols for U, T, H are stripped in the text extraction, so they appear as [U], [T] below.

## Verbatim quotes

1. Safety: "The probability of not submitting a single backdoored solution across all 5,000 APPS problems"
2. Audit budget: "enough H points to use on up to 2% of problems"
3. Trusted monitoring: "Produce code using [U], and ask [T] to review how suspicious it looks. Audit the top 2% most suspicious code."
4. Usefulness scale: "0 is the correctness rate of GPT-3.5 and 1 is the correctness rate of GPT-4"
5. Game order: "The blue team proposes a protocol" ... "With knowledge of the blue team’s protocol, a red team proposes an attack policy"
6. Untrusted monitoring check: "If the false negative rate is too high, shut down"

## Relevance to the commons overseer

The template for every later control evaluation: trusted weak model T (GPT-3.5), untrusted strong model U (GPT-4), scarce trusted labour H (2% audit budget), red team best-responds to a known protocol, and safety (no catastrophe in the whole deployment) is traded against usefulness (rescaled 0 = T alone, 1 = U alone). Table 2: trusted monitoring 62% safety / 97% usefulness; defer-to-trusted 10% 83%/93%; untrusted monitoring 79%/96%; trusted editing 20% 92%/94%. Direct template for an overseer protocol table in the commons.
