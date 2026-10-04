# Recent feedback and how to respond

This separates feedback supplied in the conversation from new findings in this audit. It is not a transcript of meetings and does not attribute the audit's conclusions to Yali or Ed.

| Source | Feedback/question | What it exposes | Response and concrete work |
| --- | --- | --- | --- |
| Yali, directly quoted | What problem are we trying to resolve? Write an introduction and method. | The formulation lists variables before establishing the question and why the experiment answers it. | Open with: "Can checking individual resource-use requests tell us whether the shared system is safe, and what additional information or intervention is needed when it cannot?" Then explain the implemented mechanisms and evidence. |
| Yali, directly quoted | Is this a multiplayer game between an overseer and several agents? Model stakeholders' decisions. | Actors, governance mechanisms, safety diagnostics, and the experimenter are being conflated. | Describe a partially observed resource game among users with a fixed stateful oversight policy. Specify user observations/actions/payoffs and monitor observations/signals/interventions. State that the current governor is not a trained strategic player and no equilibrium has been solved. |
| Yali, directly quoted and meeting notes | Position the work in the literature and make the formulation clear. | The paper needs a narrow contribution relative to existing benchmarks, not just a list of research areas. | Compare against sequential social dilemmas, joint-action shielding, scalable oversight protocols, and GovSim. Distinguish benchmark design assumptions from empirically validated conclusions. |
| Yali discussion, as reported by Ameer | Local decisions may appear acceptable while their combination is unsafe. | This motivates a diagnostic problem, but is not itself evidence of causal compositional failure. | Separate continuing unsafe states from safe-to-unsafe transitions. Add a real local-filter comparator and model-aware/global comparator. Specify what each can observe. |
| Group questions, as reported by Ameer | How precisely is capability measured? Is there a rubric or weighted sum? | Assigned ordinal labels have been presented as measured abilities. | State that the old gap is rank subtraction, not a weighted or calibrated measurement. Retain candidate count/horizon and monitor limits as separate experimental variables; validate generator performance on common held-out tasks. |
| Group objection, as reported by Ameer | Multiple axes cannot obviously be collapsed into one dimension. | Equal differences combine substantively different settings, sometimes with very different outcomes. | Replace the main gap-axis graph with factor-separated plots. Do not introduce arbitrary weights as a fix. A common ability scale would need additional operational justification and calibration. |
| Group questions, as reported by Ameer | Why do none/local unsafe traces decrease? Is there an overseer there? | The plotted x-axis has no active overseer interpretation for these conditions. | Both return no governor in code. Their curves mix actor settings as the rank difference changes. Explain payoff-driven selection as a possible reason for the actor trend, not as an established causal mechanism. Show the actor-only comparison. |
| Group question, as reported by Ameer | Can local failure coexist with global safety? | The presentation only shows one quadrant of diagnostic behaviour. | Yes: the selected trace has six such steps. Existing summary rates permit all four local/global categories to be recovered. Report these alongside approval coverage; local rejection does not imply system failure. |
| Ed discussion, as reported by Ameer | Avoid simplistic external control; consider internally sustained guidance and norms. | Renaming an enforced clipping mechanism "global signal" does not remove its authority. | Describe cap announcement and enforced clipping separately. Add signal-only and matched enforcement controls. Keep persistent-norm learning as later work, with explicit training/selection mechanisms. Removing prompts does not update fixed model weights. |
| Ed discussion, as reported by Ameer | Centralization can hide how constraints emerge among agents. | Local coordination, institutional enforcement, and emergent institutions are different objects. | Preserve the local coordination baseline and test what it contributes under matched targeting. Do not claim the present fixed governor models the emergence of a real institution. |
| Ameer, repeated concern | What have I actually built, and why Fishery/Harvest rather than arbitrary toy settings? | The scientific question has become obscured by project chronology and scenario names. | Fishery isolated shared-stock interventions. Harvest adds spatial externalities and local/global observation structure. These properties justify controlled mechanism tests; domain labels do not confer real-world validity. |

## New issues from this audit, not previously attributed feedback

- The 29-step illustrative mismatch is entirely persistent unsafe occupancy, not approved failure onset.
- The full threshold matrix relabels repeated trajectories; patch-health winner counts repeat each context 25 times.
- Hybrid changes communication and targeting simultaneously, with different evolved populations.
- The prompt anchors specify all fourteen model-policy fields; Qwen reproduces them exactly in 35/64 policies.
- Cost is deducted from payoff, no finite governance budget is exhausted, and capacity has discrete endpoint quirks.
- The full-ablation summarizer pools scenarios, and the stress analysis mislabels aggregate test evaluation as nominal.

## Suggested message to Yali

"I reviewed the implementation against the group's questions. The current result is that individually acceptable requests can coexist with an unsafe shared resource, and the intervention mechanisms change resource outcomes. The capability-gap plot compressed different settings too far, so I am separating strategy-generation budgets from the overseer's limits. I also found that the example trace shows continued unsafe conditions rather than the initial transition into failure. My next step is a small matched-policy test to establish what local checks miss and which part of the intervention actually helps, before expanding the ablation. Does that give us the right focus for the first benchmark paper?"

## Decisions worth asking the supervisors to make

1. Is the first contribution a benchmark of compliance-risk diagnosis and limited intervention, or must it already include a demonstrably weaker verifier supervising a stronger actor? The second target needs a more direct monitor task.
2. Should the main contrast isolate information available to the monitor, intervention authority, or both? Changing both without controls will leave the mechanism unclear.
3. Is a fixed model-aware safety controller an appropriate reference baseline, with its privileged knowledge stated explicitly?

These are scope and construct-validity decisions. They are more useful now than asking which additional language model to run.
