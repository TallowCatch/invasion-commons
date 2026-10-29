# Slide style guide: Ameer's academic deck style

> **For the LLM reading this.** Make or restyle slides so they match this style exactly: the colours, fonts, sizes and layout below. `examples/contact_sheet.png` shows eight real slides in this style: title, cards, numbered, method, split, figure, timeline and closing. If slides are attached, keep their **content** and change only how they **look**, unless asked otherwise. The style comes from Ameer's Canva deck "TESSERA-FHD" (22 July 2026). `build_deck.js`, in this folder, is a working implementation in pptxgenjs. If you can run code, use it, or copy its numbers.

---

## 1. The look in one paragraph

Warm off-white slides. Bold **navy serif headings** (Georgia), often on two lines, with a **short, heavy dark bar** under the heading on the same side as the text. **Bold serif sub-heads** in near-black, and **small sans body text** (Calibri), justified in columns. Figures sit plainly on the page with **italic, centred, muted captions**. Lists use big light-grey **"01 02 03"** numbers. There is lots of empty space, no gradients, no clip art and no icons. The deck closes on a **dark slate "Thank You" slide** with a tan bar.

## 2. Page

- **Size:** 16:9 widescreen, **13.333 × 7.5 in** (PowerPoint "Widescreen").
- **Outer margin:** **0.6 in** on the left and right. Keep everything inside it.
- **Background:** `#F5F4F2` (warm off-white, not cream, not pure white) on every slide except the closing one.
- **Page number:** bottom right, at x = 12.43 in, y = 7.0 in. Calibri 10 pt, `#1C191A`, right-aligned. On every slide except the title slide; white on the closing slide.

## 3. Colours (use only these)

| Role | Hex |
|---|---|
| Background | `#F5F4F2` |
| Headings (navy) | `#062746` |
| Bar under headings | `#131B28` |
| Body text and sub-heads (near-black) | `#1C191A` |
| Title-slide text | `#2B2F33` |
| Captions, notes, secondary text (muted) | `#55524E` |
| Numbers "01 02", "before" values | `#8F8F8F` |
| Card fill | `#FFFFFF` |
| Card border, divider lines | `#E9E7E3` |
| Dashed placeholder border, timeline line | `#C9C6C1` |
| Closing-slide background (slate) | `#29343A` |
| Closing-slide bar (tan) | `#B09E88` |
| Text on dark | `#F0F0F0` |

**Emphasis is navy.** Never use red, green, bright blue or gradients for emphasis. Where a chart needs colours, use the figure's own colours.

## 4. Fonts and sizes

| Element | Font | Size | Style | Colour |
|---|---|---|---|---|
| Slide heading | Georgia | 32–36 pt (34 by default) | bold, line spacing 0.95 | navy `#062746` |
| Sub-head | Georgia | 16–19 pt (17 by default) | bold | `#1C191A` |
| Body | Calibri | 13–14 pt | regular; justified in text columns, left-aligned in cards and lists | `#1C191A` |
| Caption | Calibri | 11 pt | italic, centred under the figure | `#55524E` |
| Takeaway line under a figure | Georgia | 17 pt | bold, centred | navy |
| Italic signpost line (under cards, notes) | Calibri | 14–15 pt | italic, centred | `#55524E` |
| "01 02 03" numbers | Calibri Light | 22–26 pt | regular | `#8F8F8F` |
| Card label ("01   WHY") | Calibri Light | 13 pt | upper case, letter-spaced | `#8F8F8F` |
| Big number (stats) | Georgia | up to 54 pt | bold | navy |
| Title-slide title | Calibri Light | 54 pt | regular | `#2B2F33` |
| Title-slide subtitle | Calibri | 19 pt | italic | `#2B2F33` |
| "Thank You" | Calibri | 72 pt | regular | `#F0F0F0` |

Use no other fonts. Calibri stands in for the original Canva sans; Georgia is the serif.

## 5. The heading and its bar (the signature element)

- **Position:** the heading text box sits at about x = 0.6 in, y = 0.4–0.45 in. It is anchored to the **bottom** of its box, so one-line and two-line headings end at the same height.
- **The bar:** a solid rectangle **3.6 in wide × 0.1 in tall**, colour `#131B28`, placed **0.14 in below the heading box**.
  - It lines up with the heading's **left** edge, or its right edge when the heading is right-aligned.
  - **Centred headings get no bar.** These are figure slides.
- **Space after the bar:** about 0.34 in before the content starts.
- **Headings are short:** 2–7 words, in **Title Case**, ideally a claim ("Memory Protects the Lake Without a Fine", "Finding 1: A Catch Needs a Consequence"), not a topic label ("Results").

## 6. Slide layouts

Coordinates are in inches; W = 13.333, H = 7.5, margin M = 0.6.

1. **Title.**
   - Title: Calibri Light 54 pt at (1.05, 2.2), up to 11 in wide.
   - Italic subtitle: 19 pt at (1.05, 3.6).
   - A heavy bar 3.7 × 0.12 at (1.05, 4.6).
   - Author, event, and "affiliation, date" on three lines at the bottom right, Calibri 11 pt, right-aligned.
   - No page number.
2. **Cards** (the project on one slide).
   - Heading across the top.
   - 3–4 white rounded cards in a row, with gaps of 0.3 in, a 0.75 pt `#E9E7E3` border and a very soft shadow (8% opacity).
   - Each card has, from the top:
     - a label line, "01   WHY", in light grey;
     - a navy Georgia 17 pt bold head;
     - body text in Calibri 13.5 pt.
   - Optionally, one italic muted takeaway line centred near the bottom.
3. **Split** (text beside a figure).
   - A one-line heading across the slide (about 40 characters at most).
   - Text column about 5 in wide: 2–3 blocks, each a Georgia bold sub-head plus a short justified paragraph or 2–4 bullets.
   - The figure fills the other side, 0.45 in from the text, with an italic caption under it.
   - Alternate the figure's side from one split slide to the next. If the figure is on the left, right-align the heading and its bar.
4. **Numbered list** (questions, aims, steps).
   - Big heading on the **left half** (Georgia 36 pt, x = 0.6, y ≈ 1.2, 5.2 in wide) with its bar.
   - On the **right half**, 3–4 items, each with:
     - a grey "01" (Calibri Light 26 pt) at x = 6.55;
     - a Georgia 19 pt bold head at x = 7.45;
     - body text underneath.
   - Items are spaced evenly down the slide.
5. **Method.**
   - Left, about 6.4 in wide: the heading, then numbered steps (grey "01" 22 pt, a Georgia 16 pt bold head, Calibri 13 pt text).
   - Right: one quiet white card titled "Key quantities" (navy Georgia 16 pt bold).
   - The card lists symbol and meaning rows: the symbol in Georgia 17 pt, the meaning in Calibri 12.5 pt.
   - The most important symbol is navy and bold.
   - Formulas support this slide; they never fill it.
6. **Figure** (one wide figure).
   - **Centred** heading, 32 pt, no bar.
   - The figure is as large as fits between y ≈ 1.35 in and the bottom.
   - Italic caption under it.
   - Optionally one **bold navy takeaway line** centred at y ≈ 6.45.
7. **Two figures.** A centred heading, two equal figures side by side with a 0.4 in gap, a caption under each, and an optional navy takeaway.
8. **Compare** (before and after).
   - Heading.
   - A white card, about 7.2 in wide, holding rows. Each row has:
     - a label in Calibri 13.5 pt;
     - the **before** value in Georgia 26 pt grey;
     - an arrow →;
     - the **after** value in Georgia 26 pt **bold navy**.
   - Thin `#E9E7E3` dividers between rows, and a small italic note under the card.
   - Optionally, sub-head blocks on the right.
9. **Stats** (use rarely).
   - Heading, then 2–4 big navy Georgia numbers in equal columns, each with a Calibri 14 pt label below.
   - Every number uses the same size, about 10 characters at most.
10. **Pending** (work still running).
    - Heading, then rows. On the left of each row:
      - a Georgia 16 pt bold question;
      - a muted Calibri 12.5 pt line on what is running.
    - On the right, a white rounded result box, **dashed** `#C9C6C1`, containing the grey italic text "Result (to fill in):".
11. **Timeline.**
    - Heading, then a horizontal grey line with circles: **filled navy** means done, **white with a navy outline** means upcoming.
    - Under each circle, a navy Georgia 15 pt bold date and centred Calibri 13 pt text.
12. **Questions** (discussion, at the end).
    - The heading on the **right** (right-aligned, 36 pt), with its bar right-aligned.
    - Numbered questions down the left: grey "01" and Georgia 18 pt text.
13. **Closing.**
    - Slate `#29343A` background, with a big "Thank You" (Calibri 72 pt, `#F0F0F0`) centred.
    - A **tan bar** 3.9 × 0.13 in, centred under it at y = 4.7.
    - An optional italic subtitle.
    - The **date set vertically** (rotated 90°) near the right edge.
    - Name and email at the bottom left, 11 pt, white.

## 7. How the words go on the page

- **One message per slide.** The heading states it, and the rest supports it.
- **Short text.** Sub-heads of 2–6 words; paragraphs of 1–3 sentences; bullets of one line each; at most about 60 words of body text per slide.
- **Plain words** with real numbers: "over-took in 58% of rounds", not "significantly increased non-compliance".
- **Figures carry results.** Use the paper's own figure images, not redrawn charts. Crop panels from composite figures and remove stray panel letters like "(a)".
- **Mix the layouts.** Don't use two split slides in a row with the figure on the same side.
- **Leave white space.** Never shrink text below 12.5 pt to fit more in; cut words instead.
- **Speaker notes** hold what you'll actually say, so the slide stays sparse.

## 8. Don'ts

- No templates, themes, gradients, stock icons, emoji or clip art.
- No accent lines other than the heading bar and the closing tan bar.
- No all-caps headings (only the small card labels are in capitals), no underlines, and no coloured text boxes; the white cards described above are the only boxes.
- No pure white background; no fonts other than Georgia, Calibri and Calibri Light.
- No bar under centred headings.

---

## 9. Prompt to paste into Codex or another LLM

```
Restyle the attached slides to match my academic slide style, described in the attached
SLIDE_STYLE_GUIDE.md and shown in the attached contact_sheet.png. A working pptxgenjs implementation is in the attached build_deck.js. Keep my content and
the order of slides; change only the formatting and layout, unless I say otherwise.

Rules:
- 16:9, 13.333 x 7.5 in, 0.6 in side margins, background #F5F4F2.
- Headings: Georgia bold 32-36 pt, navy #062746, anchored to the bottom of their box, with a solid bar 3.6 x 0.1 in
  (#131B28) 0.14 in below, aligned with the heading's side. No bar under centred (figure) headings.
- Sub-heads: Georgia bold 16-19 pt, #1C191A. Body: Calibri 13-14 pt, #1C191A, justified in text columns.
- Captions: Calibri italic 11 pt, #55524E, centred under figures. Numbered items: "01 02 03" in Calibri Light
  22-26 pt, #8F8F8F.
- Cards: white, rounded, a thin #E9E7E3 border, a very soft shadow.
- Emphasis colour is navy only.
- Title slide: Calibri Light 54 pt title, italic subtitle, heavy bar, author/event/date at the bottom right.
- Closing slide: slate #29343A background, "Thank You" in Calibri 72 pt #F0F0F0, a tan bar #B09E88 3.9 x 0.13 in,
  the date rotated 90° at the right, name and email at the bottom left.
- Page number at the bottom right, Calibri 10 pt, on every slide except the title slide.
- Choose each slide's layout from the guide's section 6 (cards, split, numbered, method, figure, two figures,
  compare, stats, pending, timeline, questions, closing) according to its content.
- One message per slide, short text, plenty of white space; cut words rather than shrink fonts below 12.5 pt.

After building, render each slide to an image and check for:
- text that overflows or overlaps;
- headings wider than their box;
- figures too small;
- leftover panel letters.
Fix these and render again.
```
