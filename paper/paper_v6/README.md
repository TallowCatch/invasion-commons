# paper_v6: exhibits for "What makes audits work"

Draft figures and tables for the paper in `notes/claude_audit_20261005/PAPER_PLAN.md`.

**See them all with captions:** open `exhibits_preview.pdf`, which is set as the paper would be (Times, 5.5 in text width).

**Rebuild:**
```bash
PYTHONPATH=. python -m experiments.oversight.make_paper_exhibits
cd paper/paper_v6
pdflatex exhibits_preview.tex
```

## Wording

**Every technical term is defined in plain language in Table 1 (Key terms)**, right after Figure 1. The games are in Table 2 and the audit rules in Table 3.

### Game names

Each game has one word, named after its resource, as GovSim does (Fishery, Pasture, Pollution). The names are set in small caps, so a game name is never mistaken for an ordinary word.

| Paper name | Project name | Why |
| --- | --- | --- |
| **Fishery** | Fishery | One shared stock. The same name as GovSim's fishery scenario. |
| **Forest** | Harvest | Plots in a ring, where over-cutting damages neighbours. The code already builds it from the `forest_co_management` preset. "Harvest" was dropped as a game name because harvest is also the main outcome ("total harvest"), and the two would be confused. Melting Pot's *Commons Harvest* is a different game. |
| **River** | C1, "Synergistic Pollution", "Two-Pollutant River" | Two pollutants that are harmful mainly together: *mixture toxicity* and *synergy* in ecotoxicology (Cedergreen 2014, doi:10.1371/journal.pone.0096580). It is not called "Pollution", to avoid a clash with GovSim's single-pollutant game. Related economics: non-point source pollution (Segerson 1988; seen only through other papers so far). |

### Other terms

| Name in the paper | Name in the project notes |
| --- | --- |
| Safety line | one-step target, "old line" |
| MSY target | MSY target |
| Cautious local reviewer | bounded local |
| Greedy agent | stress agent |
| Memory + extra checks | `memory_cap` |
| Memory + tighter cut | `memory_cut` |
| Audit largest report | report-targeted audit |
| Predictable audits | `periodic6` schedule |

"Harvest" is kept as the word for what agents take (the outcome).

## How the exhibits follow the field

The design follows how the closest papers present results. A survey of ten of them is summarised below.
- AI Control: Greenblatt et al. 2023/24.
- Games for AI Control: Griffin et al. 2024.
- GovSim: Piatti et al. 2024.
- Makins et al. 2026.
- Kenton et al. 2024.
- Engels et al. 2025.

| Convention in the literature | What we do |
| --- | --- |
| Hand-drawn vector protocol diagram as Figure 1 (draw.io, Illustrator or similar) | Figure 1 is TikZ, so it is vector and uses the paper's own fonts. |
| Main results table: booktabs rules, mean with 95% interval, best value in bold (GovSim Table 1; AI Control's protocol tables) | Table 3 compares the audit rules this way. |
| Safety against usefulness per protocol (AI Control Fig. 2; Griffin Fig. 2) | Figure 3a shows risky requests approved against safe requests cut, per reviewer and per model error. |
| Metric against audit budget, with a band for the interval (AI Control Fig. 3; Makins Fig. 5) | Figure 5b. |
| Small multiples or one row per setting, with intervals (Kenton; Makins) | Figure 4 shows all nine testable settings, plus effect sizes with 95% intervals. |
| Resource stock over time (GovSim Fig. 3; Perolat Fig. 3) | Figure 2a. |
| Vector PDF, (a)(b) panel labels, one colour per method throughout, long self-contained captions | All figures. |

The plots are matplotlib, which is what these papers use, styled to match the text: Times, 7–8 pt at final size, vector output.

## Exhibits, by claim

Results from several studies go into **composite multi-panel figures**, one per part of the argument. This follows GovSim (2×2 grids) and Makins et al. (2×3 and 1×4 grids):
- related results share a figure;
- each panel has a short bold title, (a)–(f);
- a colour means the same thing in every panel of a figure;
- legends are shared at the figure level where possible.

| Claim | Exhibit | Status |
| --- | --- | --- |
| Setting | **Fig. 1:** one step of the game (TikZ). **Table 1:** key terms. **Table 2:** the three games. | Ready |
| 1–2. The reviewer's target and model | **Fig. 2** (2×3 panels): (a) stock over time; (b) harvest in six settings; (c) collapse with a wrong model, all from **Fishery**; (d)–(e) risky requests let through and safe requests blocked with a wrong model, in **Forest**; (f) a reviewer that learns the model. Sources: R1, R2. | Ready |
| 3–5. What makes audits work | **Fig. 3** (5 panels): (a) memory in nine settings (R2); (b) deterrence against expected fine (S3); (c) gain against audit rate by audit rule (S5); (d) audit timing (S3 Part C); (e) audit targeting (C1). **Table 3:** audit rules (S5). | Ready. Add an R3 threshold panel; replace (e) with T1 across 3 games. |
| 6. LLM agents | **Fig. 4:** over-taking against expected fine, 2 models × 2 framings. | Waits for **L2**. |

Every number in a caption is read from the saved run tables in `notes/claude_audit_20261005/runs/` and `results/runs/`. Figure 2a needs the R1 raw file, `results/runs/claude_r1_repaired_reviewer_v1/closed_loop_decisions.jsonl.gz`, which is not in git.
