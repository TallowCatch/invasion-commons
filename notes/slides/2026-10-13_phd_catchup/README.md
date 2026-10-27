# PhD catch-up deck, 13 October 2026

- `phd_catchup_2026-10-13.pptx` is the deck. A PDF of it is in `render/`.
- `spec.json` is the slide outline. All numbers come from `paper/paper_v6/draft_aaai/body.tex` as of 12 Oct.
- `figures/` holds the paper figures, cropped to one panel per slide.

**Filling in the running studies (slide 12).** When L4–L6 finish, add `"result": "..."` to each row of the `pending` slide in `spec.json`, then rebuild:

```bash
node ~/.claude/skills/canva-academic-deck/scripts/build_deck.js spec.json phd_catchup_2026-10-13.pptx
bash ~/.claude/skills/canva-academic-deck/scripts/render_pptx.sh phd_catchup_2026-10-13.pptx render
```

Run the builder from a folder where `pptxgenjs` is installed. The style and the build steps are in the skills `canva-academic-deck` and `research-catchup-talk`.

## Which files are current (27 October 2026)

- **The deck:** `phd_catchup_2026-10-13.pptx` is the version Ameer edited by hand, including slide 13 "Follow-up Results" with L4 and L6. `spec.json` no longer matches it; do not rebuild from the spec.
- **The scripts:** the `.docx` files are the current versions (edited by Ameer). `talk_script_concise.md` is kept in sync for slide 13. `talk_script_and_qa.md` predates Ameer's docx edits.
- **Renders** (`render*/`) are scratch images and are git-ignored.
