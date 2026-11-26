"""Add formula cards to Ameer's hand-edited v3 deck, keeping all his edits, and save as v4.

  <venv with python-pptx>/python add_formulas.py weekly_update_2026-11-16_v3.pptx weekly_update_2026-11-16_v4.pptx
Formula images come from make_formulas.py (formulas/*.png, sizes.json). Styles are copied from shapes already in the
deck (the cards, labels and body text on the bounding slide), so the additions match the builder's look.
"""
import copy
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
FORM = HERE / "formulas"
SIZES = json.loads((FORM / "sizes.json").read_text())
EMU = 914400


def next_id(slide):
    return max(int(e.get("id")) for e in slide._element.iter() if e.tag.endswith("}cNvPr")) + 1


def clone(slide, template, x, y, w, h, text=None):
    """Copy a shape (card or text box) from the template slide onto `slide` at a new place, optionally new text."""
    el = copy.deepcopy(template._element)
    slide.shapes._spTree.append(el)
    el.find(".//{*}cNvPr").set("id", str(next_id(slide)))
    sh = slide.shapes[-1]
    sh.left, sh.top, sh.width, sh.height = (Inches(v) for v in (x, y, w, h))
    if text is not None:
        tf = sh.text_frame
        paras = tf.paragraphs
        for extra in paras[1:]:
            extra._p.getparent().remove(extra._p)
        runs = paras[0].runs
        runs[0].text = text
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
    return sh


def formula(slide, name, x, y, scale, centre_w=None):
    """Place formula `name` at one type scale (inches per point); returns (width, height) in inches."""
    w, h = SIZES[name]["w_pt"] * scale, SIZES[name]["h_pt"] * scale
    if centre_w is not None:
        x = x + (centre_w - w) / 2
    slide.shapes.add_picture(str(FORM / f"{name}.png"), Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def add_notes(slide, extra, template=None):
    ns = slide.notes_slide
    if ns.notes_text_frame is None:  # a new slide's notes page has no body box; copy one from another slide
        body = [sh for sh in template.notes_slide.placeholders if sh.placeholder_format.type == 2][0]
        ns.shapes._spTree.append(copy.deepcopy(body._element))
        ns.notes_text_frame.text = ""
    tf = ns.notes_text_frame
    tf.text = (tf.text + "\n\n" + extra).strip()


def renumber(prs):
    for i, s in enumerate(prs.slides, 1):
        for sh in s.shapes:
            if sh.has_text_frame and abs(sh.left / EMU - 12.43) < 0.05 and abs(sh.top / EMU - 7.0) < 0.05:
                sh.text_frame.paragraphs[0].runs[0].text = str(i)


def main(src, dst):
    prs = Presentation(src)
    slides = list(prs.slides)
    bound = slides[5]  # "How I'm bounding the evaluation": the style templates
    T = dict(title=bound.shapes[0], bar=bound.shapes[1], card=bound.shapes[2], label=bound.shapes[3],
             head=bound.shapes[4], body=bound.shapes[5], italic=bound.shapes[18], page=bound.shapes[19])

    # Slide 7, "How I built it": the model written out, in the space where the old card was
    s = slides[6]
    clone(s, T["card"], 7.55, 1.45, 5.18, 5.45)
    clone(s, T["head"], 7.85, 1.62, 4.6, 0.4, "The model, written out")
    rows = [("regrow", "How likely an empty square is to regrow, given n apples within two squares (SocialJax)."),
            ("rule", "Over-harvesting is taking an apple when fewer than 3 are left nearby, counted before the move."),
            ("detect", "Each step the overseer checks at most k agents (the set C). It can miss an over-harvest (α) "
                       "or wrongly flag an honest move (β). So far both are 0."),
            ("reward", "What a caught agent learns from. One apple is worth 7, so a penalty of P apples removes 7P."),
            ("agent", "Each agent learns to get as much of this as it can (IPPO, γ = 0.99).")]
    y = 2.15
    for name, cap in rows:
        _, h = formula(s, name, 7.85, y, 1.2 / 72)
        ch = 0.25 + 0.2 * (len(cap) // 70)  # about 70 characters a line at 12 pt in this width
        clone(s, T["body"], 7.85, y + h + 0.05, 4.6, ch, cap).text_frame.paragraphs[0].runs[0].font.size = Pt(12)
        y += h + 0.05 + ch + 0.1
    add_notes(s, "The card on the right writes the same thing down. p of n is the regrowth chance from SocialJax. "
                 "v is one when an agent takes an apple with fewer than three left around it. When the overseer checks "
                 "an agent it catches a real over-harvest with chance one minus alpha and wrongly flags an honest move "
                 "with chance beta. Both are zero so far, and they are how I'll make the overseer less capable. A caught "
                 "agent loses 7P reward, and each agent learns to maximise its discounted, penalised reward.")

    # New slide after the bounding slide: the overseer's problem as an optimisation
    n = prs.slides.add_slide(bound.slide_layout)
    clone(n, T["title"], 0.6, 0.4, 12.13, 0.85, "The overseer's problem, written down")
    clone(n, T["bar"], 0.6, 1.39, 3.6, 0.1)
    clone(n, T["card"], 0.6, 1.75, 12.13, 1.95)
    clone(n, T["label"], 0.85, 1.92, 6, 0.35, "01   THE DESIGN PROBLEM")
    formula(n, "design", 0.85, 2.35, 1.45 / 72, centre_w=11.63)
    clone(n, T["body"], 0.85, 2.95, 11.63, 0.65,
          "The overseer picks its settings (how many agents to check, which ones, and the penalty) to get the most "
          "apples collected (W) for the least spent, knowing the agents will learn their best reply to it. The budget B "
          "caps what it can spend.")
    cards = [("02   WHAT CHECKING COSTS", "cost",
              "Each check costs something, plus more for every past step it reads (w), so a budget of 1 unit can be "
              "compared with 10, as in BazaarBench."),
             ("03   WHERE AN OVERSEER SITS", "eta",
              "0 means no better than no overseer and 1 means as good as checking everyone. One check with an average "
              "penalty of 2 apples already reaches about 0.97."),
             ("04   THE WORST CASE", "worst",
              "The same score against agents that can see who is checked and are trained to get round the overseer, "
              "like AI Control's red team. Still to run.")]
    for j, (lab, name, cap) in enumerate(cards):
        x = 0.6 + j * 4.115
        clone(n, T["card"], x, 3.95, 3.9, 2.75)
        clone(n, T["label"], x + 0.25, 4.12, 3.4, 0.35, lab)
        _, h = formula(n, name, x + 0.25, 4.6, 1.3 / 72, centre_w=3.4)
        clone(n, T["body"], x + 0.25, 4.6 + h + 0.15, 3.4, 1.4, cap)
    clone(n, T["page"], 12.43, 7.0, 0.5, 0.3, "8")
    add_notes(n, template=bound, extra="This is the bounding slide written as an optimisation. The overseer chooses its settings, theta O, "
                 "which are how many agents to check, how it picks them and the penalty. It wants the most apples "
                 "collected minus lambda times what it spends, and the agents respond by learning their best reply, "
                 "so it is a two-level problem, like a Stackelberg game. The cost counts every check and every step of "
                 "history it reads, which is where the 1 unit against 10 units budget comes in. Eta places any overseer "
                 "between no overseer at zero and checking everyone at one. For example one check per step with an "
                 "average penalty of 2 apples gets 515 apples against 530 for checking everyone and 68 with nothing, "
                 "so eta is about 0.97. The worst case is the minimum over agents that can adapt, which in practice "
                 "means training agents against the overseer. An equivalent safety-first version, closer to AI Control, "
                 "is to minimise cost subject to the orchard dying less than some small share of the time. So far I have "
                 "searched these settings with a grid rather than solving the problem directly.")
    lst = prs.slides._sldIdLst
    new = lst[-1]
    lst.remove(new)
    lst.insert(7, new)

    slides = list(prs.slides)
    # Penalty finding (now slide 10): the deterrence condition beside the chart
    s = slides[9]
    pic = [sh for sh in s.shapes if sh.shape_type == 13][0]
    ratio = pic.height / pic.width
    pic.left, pic.top, pic.width = Inches(0.6), Inches(1.35), Inches(7.2)
    pic.height = int(pic.width * ratio)
    cap = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.startswith("Learning agents")][0]
    cap.left, cap.top, cap.width, cap.height = Inches(0.6), Inches(1.45 + pic.height / EMU), Inches(7.2), Inches(0.6)
    clone(s, T["card"], 8.15, 1.35, 4.58, 4.95)
    clone(s, T["head"], 8.4, 1.52, 4.1, 0.75, "When over-harvesting stops paying")
    formula(s, "deter", 8.4, 2.35, 1.6 / 72, centre_w=4.1)
    clone(s, T["body"], 8.4, 3.02, 4.1, 0.85,
          "k/N is the chance of being checked, P the penalty in apples, g what one over-harvest gains and α the "
          "overseer's miss rate.")
    clone(s, T["head"], 8.4, 3.85, 4.1, 0.6,
          "Bots stop at g ≈ 1 apple, and the learners switch between 0.5 and 1").text_frame.paragraphs[0].runs[0].font.size = Pt(15)
    formula(s, "honest", 8.4, 4.55, 1.45 / 72, centre_w=4.1)
    clone(s, T["body"], 8.4, 5.15, 4.1, 1.0,
          "With false alarms (β) an honest agent loses this on average. At a fixed average penalty it doesn't depend "
          "on k, which is the next thing to test.")
    add_notes(s, "The card on the right is the condition behind this chart. Over-harvesting stops paying once the chance "
                 "of being checked, k over N, times the penalty P, times one minus the miss rate, is at least the gain g "
                 "from one over-harvest. It is Becker's condition, and the same thing my Fishery study found. With the "
                 "bots the break-even was about one apple, and the learners switch between half an apple and one. The "
                 "second formula is a prediction. If the overseer sometimes flags honest agents, an honest agent loses k "
                 "over N times beta times P per step. If the average penalty is held fixed, that doesn't depend on k, so "
                 "one big check and seven small ones cost honest agents the same on average and differ only in how "
                 "bumpy the losses are. Whether learning agents care about that is the next experiment.")

    # Random vs targeted plan (now slide 11): the two ways of setting check chances, under the cards
    s = slides[10]
    formula(s, "mix", 0.6, 6.0, 1.22 / 72, centre_w=12.13)
    take = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.startswith("Each step changes")][0]
    take.top, take.height = Inches(6.52), Inches(0.38)
    add_notes(s, "The line under the cards writes the plan down. With random checks every agent has the same chance of "
                 "being checked, k over N. The mixed version is the security-game idea. Choose each agent's chance q i "
                 "so that even the agent best placed to over-harvest still faces an average penalty above its gain, "
                 "with the chances adding up to k. Agents that can see the checks will over-harvest wherever q i is "
                 "lowest, which is why pure targeting can backfire.")

    # Targeting finding (now slide 12): the catch rate under the chart
    s = slides[11]
    pic = [sh for sh in s.shapes if sh.shape_type == 13][0]
    ratio = pic.height / pic.width
    pic.width = Inches(5.75)
    pic.height = int(pic.width * ratio)
    pic.left, pic.top = Inches(6.41), Inches(1.5)
    cap = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.startswith("Apples collected")][0]
    cap.top = Inches(1.58 + pic.height / EMU)
    _, h = formula(s, "catch", 6.1, 6.22, 1.3 / 72)
    clone(s, T["body"], 8.85, 6.12, 3.88, 0.7,
          "Here q is an agent's chance of being checked at that moment. Random checks give about 0.29, location 0.05.").text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    add_notes(s, "The formula under the chart is how I measure a targeting rule. It's the average chance of being "
                 "checked at the moments an agent actually over-harvests. With two random checks out of seven that is "
                 "about 0.29, and I measured 0.28. Repeat offenders first started at 0.44 and settled at 0.28. Location "
                 "was 0.05.")

    renumber(prs)
    prs.save(dst)
    print("saved", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
