# Ctrl-Z: Controlling AI Agents via Resampling

- **Authors:** Aryan Bhatt, Cody Rushing, Adam Kaufman, Tyler Tracy, Vasil Georgiev, David Matolcsi, Akbir Khan, Buck Shlegeris
- **Year:** 2025
- **Venue / identifier:** arXiv:2504.10374 (no peer-reviewed venue listed on the arXiv page)
- **URL:** https://arxiv.org/abs/2504.10374 (full text at https://arxiv.org/html/2504.10374)
- **Source type:** Preprint (empirical control evaluation, multi-step agent)
- **Date accessed:** 14 Oct 2026
- **Depth read:** Abstract plus the first ~100k characters of the HTML full text.

## Verbatim quotes

1. Setting: "a dataset of 257 challenging multi-step system administration tasks"
2. Budget: "equivalent to 1% of the actions taken in an episode"
3. "define safety to be the probability that the blue team wins by reaching the end of an episode"
4. "define usefulness to be the proportion of tasks the control protocol solves in each episode"
5. Red team: "we iteratively developed control protocols and attack policies to counter them"
6. Trusted model: "We implement our trusted model with Llama-3.1-8B"

## Relevance to the commons overseer

First multi-step (agentic) control evaluation. Audit budget is set per action (1% of actions), the untrusted model is GPT-4o mini and the trusted model Llama-3.1-8B, and protocols and attacks are iterated against each other. Resampling suspicious steps both blocks attacks and gathers evidence (incrimination): best protocol cut attack success from 58% to 7% at a 5% usefulness cost. The commons is also multi-step (rounds), so per-round audit budgets and resample/defer ideas carry over.
