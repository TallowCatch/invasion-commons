# Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision

- **Authors:** Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeff Wu
- **Year:** 2023 (arXiv), ICML 2024
- **Venue / identifier:** arXiv:2312.09390 (ICML 2024 as commonly cited; not stated on the arXiv page)
- **URL:** https://arxiv.org/abs/2312.09390 (HTML v1 read)
- **Source type:** Preprint / conference paper
- **Date accessed:** 14 Oct 2026
- **Depth read:** Abstract and the PGR definition in Section 3 read from HTML full text.

## Verbatim quotes

1. "PGR measures the fraction of the performance gap (the difference in performance between the weak and strong ceiling models) that we can recover with weak supervision"
2. "If we achieve perfect weak-to-strong generalization, PGR is 1. If the weak-to-strong model does no better than the weak supervisor, then PGR is 0"

## Relevance to the commons overseer

Supplies a normalised 'how much of the gap did the weak overseer close' metric: PGR = (weak-to-strong − weak) / (strong ceiling − weak). The commons analogue for an overseer: (harm with no overseer − harm with overseer) / (harm with no overseer − harm with a perfect, full-information overseer), i.e. the fraction of preventable harm actually prevented. Same algebra as usefulness rescaling in AI control [80].
