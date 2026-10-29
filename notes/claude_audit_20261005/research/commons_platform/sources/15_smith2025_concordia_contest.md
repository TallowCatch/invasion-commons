# Evaluating Generalization Capabilities of LLM-Based Agents in Mixed-Motive Scenarios Using Concordia

- **Authors:** Chandler Smith, Marwa Abdulhai, Manfred Diaz, Marko Tesic, Rakshit S. Trivedi, Alexander Sasha Vezhnevets, Lewis Hammond, Jesse Clifton, Minsuk Chang, Edgar A. Duéñez-Guzmán, John P. Agapiou, Jayd Matyas, Danny Karmon, Dylan Hadfield-Menell, Natasha Jaques, Tim Baarslag, Jose Hernandez-Orallo, Joel Z. Leibo (+ contest participants)
- **Year:** 2025 (reports the NeurIPS 2024 Concordia Contest)
- **Venue / identifier:** NeurIPS 2025 Datasets and Benchmarks Track; arXiv:2512.03318
- **URL:** https://arxiv.org/abs/2512.03318
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full PDF text extracted)

## Verbatim quotes

1. Abstract:
   > "Our findings reveal significant gaps between current agent capabilities and the robust generalization required for reliable cooperation, particularly in scenarios demanding persuasion and norm enforcement."
2. Melting Pot protocol carried over:
   > "we instantiate a population consisting of a majority of agents playing π and a minority playing ρ."
3. How background agents were made:
   > "We then create a variety of background strategies ρ based intuitively on common patterns of human behavior in mixed-motive settings (such as naive altruism, stubborn strategies, and conditional cooperation)."
4. Fixed LLM to control cost:
   > "we selected the Gemma 2 model (9 billion parameters, instruction-tuned) as our LLM API provider, as it represented an effective balance between API call costs and overall model performance"
5. Ranking:
   > "The primary methodology we use to compare performance between agents is the Elo rating system (Table 3) due to its simplicity and widespread use in evaluating agent interactions"
6. Substrates:
   > "The Contest scenarios were derived from five text-based configurable substrates: REALITYSHOW, PUBCOORDINATION, HAGGLING, LABORCOLLECTIVEACTION, and STATEFORMATION"

## Relevance
The Concordia Contest moved Melting Pot's protocol (focal against held-out background, resident and visitor modes, zero-shot) into text, used one fixed cheap LLM, and ranked agents by Elo, checked against other ranking methods. None of the five substrates is a renewable-resource commons, and LLM agents were weakest at norm enforcement. Agents were built on Concordia v1.8.9. Background strategies were designed by hand "intuitively", not fitted to data.
