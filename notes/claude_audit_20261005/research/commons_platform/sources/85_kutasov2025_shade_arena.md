# SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents

- **Authors:** Jonathan Kutasov, Yuqi Sun, Paul Colognese, Teun van der Weij, Linda Petrini, Chen Bo Calvin Zhang, John Hughes, Xiang Deng, Henry Sleight, Tyler Tracy, Buck Shlegeris, Joe Benton
- **Year:** 2025
- **Venue / identifier:** arXiv:2506.15740
- **URL:** https://arxiv.org/abs/2506.15740 (full text at https://arxiv.org/html/2506.15740)
- **Source type:** Preprint / benchmark
- **Date accessed:** 14 Oct 2026
- **Depth read:** Abstract and Section 3 (evaluation of sabotage agents, monitor threshold) read from HTML full text.

## Verbatim quotes

1. "Agents are evaluated on their ability to complete the side task without appearing suspicious to an LLM monitor"
2. "we say that the agent has evaded monitoring if the suspicion score is less than 80"
3. "success on the side task relies heavily on having access to a hidden scratchpad that is not visible to the monitor"
4. "with the top monitor (Gemini 2.5 Pro) achieving an AUC of 0.87 at distinguishing benign and malign transcripts"

## Relevance to the commons overseer

Defines the two measurements Ameer needs for an LLM overseer: (1) agent sabotage success = main task + side task + not flagged, with the flag threshold (80/100) chosen so the monitor's false-positive rate is about 1%; (2) monitor quality = AUC on benign vs malign transcripts. In the commons, 'main task' = harvest, 'side task' = over-take beyond allowance, and the LLM overseer's AUC can be measured on labelled rounds. The hidden-scratchpad finding argues for testing whether the overseer sees agent reasoning.
