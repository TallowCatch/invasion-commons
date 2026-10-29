#!/usr/bin/env node
/*
 * Build a 16:9 PowerPoint in Ameer's academic style (from his TESSERA-FHD Canva deck, 22 July 2026)
 * from a JSON spec.  Usage:  node build_deck.js spec.json out.pptx
 *
 * Style (see ../SKILL.md): warm off-white slides, navy serif headings with a short heavy bar beneath,
 * bold serif sub-heads, small sans body text, italic centred captions under figures, grey "01/02" numbers,
 * page number bottom right, a dark slate closing slide with a tan bar.
 *
 * Slide types: title, cards, split, numbered, method, figure, figure2, stats, compare, pending, timeline, questions,
 * closing.
 * Every slide may carry "notes" (speaker notes). Image paths are resolved relative to the spec file.
 */
const fs = require("fs");
const path = require("path");
let pptxgen;
try { pptxgen = require("pptxgenjs"); }
catch (e) { pptxgen = require(path.join(process.cwd(), "node_modules", "pptxgenjs")); }

const S = {
  bg: "F5F4F2", ink: "1C191A", navy: "062746", bar: "131B28", grey: "8F8F8F", muted: "55524E",
  slate: "29343A", tan: "B09E88", white: "F0F0F0", soft: "E9E7E3", line: "C9C6C1",
  head: "Georgia", body: "Calibri", light: "Calibri Light",
};
const W = 13.333, H = 7.5, M = 0.6;

function pngSize(file) {
  const b = fs.readFileSync(file);
  if (b.toString("ascii", 1, 4) === "PNG") return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
  let i = 2; // JPEG: walk the markers to the first SOFn
  while (i < b.length) {
    const marker = b[i + 1], len = b.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xc3) return { w: b.readUInt16BE(i + 7), h: b.readUInt16BE(i + 5) };
    i += 2 + len;
  }
  throw new Error("unknown image size: " + file);
}

// Place an image inside a box, keeping its aspect ratio, centred; returns the placed rectangle.
function fit(slide, file, box) {
  const { w, h } = pngSize(file);
  const r = Math.min(box.w / w, box.h / h);
  const pw = w * r, ph = h * r;
  const x = box.x + (box.w - pw) / 2, y = box.y + (box.h - ph) / 2;
  slide.addImage({ path: file, x, y, w: pw, h: ph, altText: path.basename(file) });
  return { x, y, w: pw, h: ph };
}

// Rough line count for text of `size` pt in a box `w` inches wide (Georgia/Calibri average glyph ~0.55 em).
function linesFor(text, size, w, em = 0.55) {
  const perLine = Math.max(8, Math.floor(w / ((size * em) / 72)));
  return Math.max(1, Math.ceil((text || "").length / perLine));
}

function heading(slide, text, o = {}) {
  const align = o.align || "left";
  const x = o.x ?? M, w = o.w ?? (align === "center" ? W - 2 * M : 6.0);
  const y = o.y ?? 0.45, h = o.h ?? 1.35;
  slide.addText(text, { x, y, w, h, fontFace: S.head, fontSize: o.size || 34, bold: true, color: S.navy,
    align, valign: "bottom", margin: 0, isTextBox: true, lineSpacingMultiple: 0.95 });
  if (o.bar !== false && align !== "center") {
    const bw = o.barW || 3.6, bx = align === "right" ? x + w - bw : x;
    slide.addShape("rect", { x: bx, y: y + h + 0.14, w: bw, h: 0.1, fill: { color: S.bar }, line: { color: S.bar } });
  }
  return y + h + (o.bar === false || align === "center" ? 0.1 : 0.34);
}

function caption(slide, text, x, y, w) {
  if (!text) return;
  slide.addText(text, { x, y, w, h: 0.55, fontFace: S.body, fontSize: 11, italic: true, color: S.muted,
    align: "center", valign: "top", margin: 0, isTextBox: true });
}

// Bold serif sub-head + body paragraph blocks, stacked from y down (used by split and conclusion slides).
function blocks(slide, items, x, y, w, o = {}) {
  const gap = o.gap ?? 0.18, hs = o.headSize || 17, bs = o.bodySize || 14;
  for (const it of items) {
    if (it.head) {
      slide.addText(it.head, { x, y, w, h: 0.4, fontFace: S.head, fontSize: hs, bold: true, color: S.ink,
        margin: 0, isTextBox: true, valign: "top" });
      y += 0.42;
    }
    const lines = Math.max(1, Math.ceil(((it.text || "").length * bs * 0.0072) / w));
    const hh = (it.bullets ? it.bullets.length * 0.36 : lines * bs * 0.0175) + 0.08;
    if (it.bullets) {
      slide.addText(it.bullets.map((b, k) => ({ text: b, options: { bullet: true, breakLine: k < it.bullets.length - 1 } })),
        { x, y, w, h: hh, fontFace: S.body, fontSize: bs, color: S.ink, margin: 0, isTextBox: true, valign: "top",
          paraSpaceAfter: 3 });
    } else if (it.text) {
      slide.addText(it.text, { x, y, w, h: hh, fontFace: S.body, fontSize: bs, color: S.ink, margin: 0,
        isTextBox: true, valign: "top", align: o.justify === false ? "left" : "justify" });
    }
    y += hh + gap;
  }
  return y;
}

function pageNumber(slide, n, dark) {
  slide.addText(String(n), { x: W - 0.9, y: H - 0.5, w: 0.5, h: 0.3, fontFace: S.body, fontSize: 10,
    color: dark ? S.white : S.ink, align: "right", margin: 0, isTextBox: true });
}

// A blank, fill-in-later box for results that are not in yet.
function blank(slide, x, y, w, h, label) {
  slide.addShape("roundRect", { x, y, w, h, rectRadius: 0.06, fill: { color: "FFFFFF" },
    line: { color: S.line, width: 1, dashType: "dash" } });
  slide.addText(label || "Result:", { x: x + 0.15, y: y + 0.08, w: w - 0.3, h: 0.3, fontFace: S.body, fontSize: 11,
    italic: true, color: S.grey, margin: 0, isTextBox: true, valign: "top" });
}

const BUILD = {
  title(slide, s, meta) {
    slide.addText(s.title, { x: 1.05, y: 2.2, w: 11, h: 1.3, fontFace: S.light, fontSize: 54, color: "2B2F33",
      margin: 0, isTextBox: true, valign: "bottom" });
    slide.addText(s.subtitle || "", { x: 1.05, y: 3.6, w: 8.6, h: 0.85, fontFace: S.body, fontSize: 19, italic: true,
      color: "2B2F33", margin: 0, isTextBox: true, valign: "top" });
    slide.addShape("rect", { x: 1.05, y: 4.6, w: 3.7, h: 0.12, fill: { color: S.bar }, line: { color: S.bar } });
    const foot = s.footer || [meta.author, meta.event, meta.affiliation && meta.date ? `${meta.affiliation}, ${meta.date}` : meta.date];
    slide.addText(foot.filter(Boolean).join("\n"), { x: W - 6.2, y: H - 1.55, w: 5.4, h: 0.9, fontFace: S.body,
      fontSize: 11, color: S.ink, align: "right", margin: 0, isTextBox: true, valign: "bottom" });
  },
  split(slide, s) {   // sub-head blocks on one side, a figure (or stats) on the other
    const right = s.imageSide !== "left";
    const tw = s.textW || 5.0, tx = right ? M : W - M - tw;
    // heading across the slide (one line), so the text column and the figure share the space below it
    const y0 = heading(slide, s.title, { x: M, w: W - 2 * M, h: 0.85, y: 0.4, align: right ? "left" : "right" });
    blocks(slide, s.blocks || [], tx, y0 + 0.1, tw, s.blockStyle);
    const fw = W - 2 * M - tw - 0.45, fx = right ? M + tw + 0.45 : M;
    if (s.image) {
      const capH = s.caption ? 0.25 + 0.2 * linesFor(s.caption, 11, fw, 0.5) : 0;
      const r = fit(slide, s.image, { x: fx, y: y0 + 0.05, w: fw, h: s.imageH ?? (H - y0 - 0.75 - capH) });
      caption(slide, s.caption, fx, r.y + r.h + 0.12, fw);
    }
  },
  numbered(slide, s) {  // heading on one side, "01 … 04" items on the other
    heading(slide, s.title, { x: M, w: 5.2, y: 1.2, h: 2.0, size: 36 });
    let y = s.top ?? 1.0;
    const step = Math.min(1.45, 5.6 / Math.max(1, s.items.length));
    s.items.forEach((it, k) => {
      slide.addText(String(k + 1).padStart(2, "0"), { x: 6.55, y, w: 0.8, h: 0.5, fontFace: S.light, fontSize: 26,
        color: S.grey, margin: 0, isTextBox: true, valign: "top" });
      const hl = linesFor(it.head, 19, 5.3, 0.62);
      slide.addText(it.head, { x: 7.45, y: y - 0.02, w: 5.3, h: 0.36 * hl + 0.08, fontFace: S.head, fontSize: 19, bold: true,
        color: S.ink, margin: 0, isTextBox: true, valign: "top" });
      if (it.text) slide.addText(it.text, { x: 7.45, y: y + 0.08 + 0.36 * hl, w: 5.3, h: step - 0.5, fontFace: S.body, fontSize: 14,
        color: S.ink, margin: 0, isTextBox: true, valign: "top", align: "justify" });
      y += step;
    });
  },
  figure(slide, s) {    // centred heading, one wide figure, caption, optional one-line takeaway
    heading(slide, s.title, { align: "center", y: 0.3, h: 0.9, size: 32 });
    const top = 1.35, bottom = s.takeaway ? 1.35 : 1.0;
    const r = fit(slide, s.image, { x: M + 0.2, y: top, w: W - 2 * M - 0.4, h: H - top - bottom - 0.45 });
    caption(slide, s.caption, M + 1, r.y + r.h + 0.1, W - 2 * M - 2);
    if (s.takeaway) slide.addText(s.takeaway, { x: M + 0.5, y: H - 1.05, w: W - 2 * M - 1, h: 0.45, fontFace: S.head,
      fontSize: 17, bold: true, color: S.navy, align: "center", margin: 0, isTextBox: true });
  },
  figure2(slide, s) {   // centred heading, two figures side by side with their own captions
    heading(slide, s.title, { align: "center", y: 0.3, h: 0.9, size: 32 });
    const gap = 0.4, fw = (W - 2 * M - gap) / 2, top = 1.4, fh = H - top - (s.takeaway ? 2.1 : 1.7);
    s.images.forEach((im, k) => {
      const x = M + k * (fw + gap);
      const r = fit(slide, im.image, { x, y: top, w: fw, h: fh });
      caption(slide, im.caption, x, r.y + r.h + 0.1, fw);
    });
    if (s.takeaway) slide.addText(s.takeaway, { x: M + 0.5, y: H - 1.05, w: W - 2 * M - 1, h: 0.45, fontFace: S.head,
      fontSize: 17, bold: true, color: S.navy, align: "center", margin: 0, isTextBox: true });
  },
  stats(slide, s) {     // heading, then 2-4 big numbers with labels; optional figure/text below
    const y0 = heading(slide, s.title, { w: W - 2 * M, h: 0.85, y: 0.4 });
    const n = s.stats.length, cw = (W - 2 * M) / n;
    // one size for every number on the slide, small enough for the longest (Georgia bold ~0.78 em per glyph)
    const longest = Math.max(...s.stats.map((st) => st.value.length));
    const size = Math.min(s.statSize || 54, Math.floor(((cw - 0.4) * 72) / (longest * 0.78)));
    s.stats.forEach((st, k) => {
      const x = M + k * cw;
      slide.addText(st.value, { x, y: y0 + 0.35, w: cw - 0.3, h: 1.1, fontFace: S.head, fontSize: size, bold: true,
        color: S.navy, margin: 0, isTextBox: true, valign: "bottom" });
      slide.addText(st.label, { x, y: y0 + 1.55, w: cw - 0.4, h: 1.0, fontFace: S.body, fontSize: 14, color: S.ink,
        margin: 0, isTextBox: true, valign: "top" });
    });
    if (s.blocks) blocks(slide, s.blocks, M, y0 + 2.8, W - 2 * M, { justify: false });
  },
  pending(slide, s) {   // ongoing experiments: question, what is running, and a blank result box to fill later
    const y0 = heading(slide, s.title, { w: W - 2 * M, h: 0.85, y: 0.4 });
    const rows = s.rows, rh = Math.min(1.45, (H - y0 - (s.note ? 1.2 : 0.7)) / rows.length), lw = 5.7, bx = M + lw + 0.35;
    if (s.note) slide.addText(s.note, { x: M, y: H - 1.05, w: W - 2 * M, h: 0.45, fontFace: S.body, fontSize: 14,
      italic: true, color: S.muted, align: "center", margin: 0, isTextBox: true });
    rows.forEach((r, k) => {
      const y = y0 + 0.1 + k * rh;
      const ql = linesFor(r.question, 16, lw, 0.62);
      slide.addText(r.question, { x: M, y, w: lw, h: 0.3 * ql + 0.08, fontFace: S.head, fontSize: 16, bold: true, color: S.ink,
        margin: 0, isTextBox: true, valign: "top" });
      slide.addText(r.design || "", { x: M, y: y + 0.06 + 0.3 * ql, w: lw, h: rh - 0.2 - 0.3 * ql, fontFace: S.body, fontSize: 12.5,
        color: S.muted, margin: 0, isTextBox: true, valign: "top" });
      if (r.result) {
        slide.addShape("roundRect", { x: bx, y, w: W - bx - M, h: rh - 0.18, rectRadius: 0.06,
          fill: { color: "FFFFFF" }, line: { color: S.line, width: 1 } });
        slide.addText(r.result, { x: bx + 0.15, y: y + 0.08, w: W - bx - M - 0.3, h: rh - 0.34, fontFace: S.body, fontSize: 13,
          color: S.ink, margin: 0, isTextBox: true, valign: "top" });
      } else blank(slide, bx, y, W - bx - M, rh - 0.18, r.blankLabel || "Result (to fill in):");
    });
  },
  timeline(slide, s) {  // left-to-right steps with dates; "done" steps filled navy, upcoming ones outlined
    const y0 = heading(slide, s.title, { w: W - 2 * M, h: 0.85, y: 0.4 });
    const n = s.steps.length, cw = (W - 2 * M) / n, cy = Math.max(y0 + 1.0, 3.3);
    slide.addShape("line", { x: M + cw / 2, y: cy, w: (n - 1) * cw, h: 0, line: { color: S.line, width: 2 } });
    s.steps.forEach((st, k) => {
      const cx = M + k * cw + cw / 2;
      slide.addShape("ellipse", { x: cx - 0.17, y: cy - 0.17, w: 0.34, h: 0.34,
        fill: { color: st.done ? S.navy : "FFFFFF" }, line: { color: S.navy, width: 1.5 } });
      slide.addText(st.when, { x: cx - cw / 2 + 0.1, y: cy + 0.32, w: cw - 0.2, h: 0.4, fontFace: S.head, fontSize: 15,
        bold: true, color: S.navy, align: "center", margin: 0, isTextBox: true });
      slide.addText(st.what, { x: cx - cw / 2 + 0.12, y: cy + 0.75, w: cw - 0.24, h: 2.4, fontFace: S.body, fontSize: 13,
        color: S.ink, align: "center", margin: 0, isTextBox: true, valign: "top" });
    });
    if (s.note) slide.addText(s.note, { x: M, y: H - 1.1, w: W - 2 * M, h: 0.5, fontFace: S.body, fontSize: 12,
      italic: true, color: S.muted, align: "center", margin: 0, isTextBox: true });
  },
  questions(slide, s) { // discussion questions, numbered, heading on the right (like the conclusion slide)
    heading(slide, s.title, { x: W - M - 5.5, w: 5.5, align: "right", y: 0.4, h: 1.3, size: 36 });
    let y = 2.35;
    const step = Math.min(1.2, 4.6 / s.questions.length);
    s.questions.forEach((q, k) => {
      slide.addText(String(k + 1).padStart(2, "0"), { x: M, y, w: 0.8, h: 0.5, fontFace: S.light, fontSize: 26,
        color: S.grey, margin: 0, isTextBox: true, valign: "top" });
      slide.addText(q, { x: M + 0.95, y: y + 0.02, w: W - 2 * M - 1, h: step - 0.15, fontFace: S.head, fontSize: 18,
        color: S.ink, margin: 0, isTextBox: true, valign: "top" });
      y += step;
    });
  },
  cards(slide, s) {     // the project on one slide: 3-4 cards (label, head, text) in a row
    const y0 = heading(slide, s.title, { w: W - 2 * M, h: 0.85, y: 0.4 });
    const n = s.cards.length, gap = 0.3, cw = (W - 2 * M - gap * (n - 1)) / n;
    const top = y0 + 0.25, ch = H - top - (s.takeaway ? 1.55 : 0.9);
    s.cards.forEach((c, k) => {
      const x = M + k * (cw + gap);
      slide.addShape("roundRect", { x, y: top, w: cw, h: ch, rectRadius: 0.08, fill: { color: "FFFFFF" },
        line: { color: S.soft, width: 0.75 }, shadow: { type: "outer", color: "000000", opacity: 0.08, blur: 6, offset: 2, angle: 90 } });
      slide.addText(String(k + 1).padStart(2, "0") + (c.label ? "   " + c.label.toUpperCase() : ""), { x: x + 0.25, y: top + 0.22,
        w: cw - 0.5, h: 0.35, fontFace: S.light, fontSize: 13, color: S.grey, charSpacing: 1, margin: 0, isTextBox: true });
      const hl = linesFor(c.head, 17, cw - 0.5, 0.64);
      slide.addText(c.head, { x: x + 0.25, y: top + 0.65, w: cw - 0.5, h: 0.32 * hl + 0.1, fontFace: S.head, fontSize: 17,
        bold: true, color: S.navy, margin: 0, isTextBox: true, valign: "top" });
      slide.addText(c.text, { x: x + 0.25, y: top + 0.8 + 0.32 * hl, w: cw - 0.5, h: ch - 1.05 - 0.32 * hl, fontFace: S.body,
        fontSize: 13.5, color: S.ink, margin: 0, isTextBox: true, valign: "top" });
    });
    if (s.takeaway) slide.addText(s.takeaway, { x: M, y: H - 1.1, w: W - 2 * M, h: 0.45, fontFace: S.body, fontSize: 15,
      italic: true, color: S.muted, align: "center", margin: 0, isTextBox: true });
  },
  method(slide, s) {    // numbered steps on the left, a quiet card of key quantities (symbol + meaning) on the right
    const lw = s.stepsW || 6.4;
    const y0 = heading(slide, s.title, { w: lw, h: 0.85, y: 0.4 });
    let y = y0 + 0.05;
    const step = Math.min(1.2, (H - y0 - 0.7) / s.steps.length);
    s.steps.forEach((it, k) => {
      slide.addText(String(k + 1).padStart(2, "0"), { x: M, y, w: 0.7, h: 0.45, fontFace: S.light, fontSize: 22, color: S.grey,
        margin: 0, isTextBox: true, valign: "top" });
      slide.addText(it.head, { x: M + 0.8, y: y + 0.02, w: lw - 0.8, h: 0.38, fontFace: S.head, fontSize: 16, bold: true,
        color: S.ink, margin: 0, isTextBox: true, valign: "top" });
      slide.addText(it.text, { x: M + 0.8, y: y + 0.4, w: lw - 0.8, h: step - 0.45, fontFace: S.body, fontSize: 13, color: S.ink,
        margin: 0, isTextBox: true, valign: "top" });
      y += step;
    });
    const cx = M + lw + 0.5, cw = W - M - cx, rows = s.quantities.length;
    const ch = Math.min(H - 1.4, 0.95 + rows * 0.82), cy = Math.max(0.9, (H - ch) / 2);
    slide.addShape("roundRect", { x: cx, y: cy, w: cw, h: ch, rectRadius: 0.08, fill: { color: "FFFFFF" },
      line: { color: S.soft, width: 0.75 }, shadow: { type: "outer", color: "000000", opacity: 0.08, blur: 6, offset: 2, angle: 90 } });
    slide.addText(s.cardTitle || "Key quantities", { x: cx + 0.3, y: cy + 0.25, w: cw - 0.6, h: 0.4, fontFace: S.head,
      fontSize: 16, bold: true, color: S.navy, margin: 0, isTextBox: true });
    s.quantities.forEach((qq, k) => {
      const ry = cy + 0.85 + k * 0.82;
      slide.addText(qq.sym, { x: cx + 0.3, y: ry, w: 1.55, h: 0.6, fontFace: S.head, fontSize: 17, italic: !!qq.italic,
        color: qq.highlight ? S.navy : S.ink, bold: !!qq.highlight, margin: 0, isTextBox: true, valign: "top" });
      slide.addText(qq.meaning, { x: cx + 1.95, y: ry + 0.02, w: cw - 2.25, h: 0.72, fontFace: S.body, fontSize: 12.5,
        color: S.ink, margin: 0, isTextBox: true, valign: "top" });
    });
  },
  compare(slide, s) {   // before/after card: metric rows with two values; sub-head blocks on the right
    const y0 = heading(slide, s.title, { w: W - 2 * M, h: 0.85, y: 0.4 });
    const cw = s.cardW || 7.2, cx = M, top = y0 + 0.15, rows = s.rows.length;
    const ch = 0.95 + rows * 0.95;
    slide.addShape("roundRect", { x: cx, y: top, w: cw, h: ch, rectRadius: 0.08, fill: { color: "FFFFFF" },
      line: { color: S.soft, width: 0.75 }, shadow: { type: "outer", color: "000000", opacity: 0.08, blur: 6, offset: 2, angle: 90 } });
    const c1 = cx + 3.35, c2 = cx + 5.35;
    [[s.columns[0], c1, S.grey], [s.columns[1], c2, S.navy]].forEach(([t, x, col]) =>
      slide.addText(t, { x: x - 0.2, y: top + 0.25, w: 1.95, h: 0.5, fontFace: S.body, fontSize: 12, bold: true, color: col,
        align: "center", margin: 0, isTextBox: true, valign: "bottom" }));
    s.rows.forEach((r, k) => {
      const ry = top + 0.9 + k * 0.95;
      if (k) slide.addShape("line", { x: cx + 0.3, y: ry - 0.1, w: cw - 0.6, h: 0, line: { color: S.soft, width: 0.75 } });
      slide.addText(r.label, { x: cx + 0.3, y: ry, w: 2.9, h: 0.75, fontFace: S.body, fontSize: 13.5, color: S.ink,
        margin: 0, isTextBox: true, valign: "middle" });
      slide.addText(r.before, { x: c1 - 0.2, y: ry, w: 1.95, h: 0.75, fontFace: S.head, fontSize: 26, color: S.grey,
        align: "center", margin: 0, isTextBox: true, valign: "middle" });
      slide.addText("→", { x: c1 + 1.55, y: ry, w: 0.5, h: 0.75, fontFace: S.body, fontSize: 20, color: S.grey,
        align: "center", margin: 0, isTextBox: true, valign: "middle" });
      slide.addText(r.after, { x: c2 - 0.2, y: ry, w: 1.95, h: 0.75, fontFace: S.head, fontSize: 26, bold: true,
        color: S.navy, align: "center", margin: 0, isTextBox: true, valign: "middle" });
    });
    if (s.cardNote) slide.addText(s.cardNote, { x: cx, y: top + ch + 0.12, w: cw, h: 0.4, fontFace: S.body, fontSize: 11,
      italic: true, color: S.muted, align: "center", margin: 0, isTextBox: true });
    if (s.blocks) blocks(slide, s.blocks, cx + cw + 0.55, top + 0.1, W - M - (cx + cw + 0.55), { justify: false });
  },
  closing(slide, s, meta) {
    slide.background = { color: S.slate };
    slide.addText(s.title || "Thank You", { x: 2, y: 2.0, w: W - 4, h: 2.4, fontFace: S.body, fontSize: 72,
      color: S.white, align: "center", valign: "bottom", margin: 0, isTextBox: true, lineSpacingMultiple: 0.9 });
    slide.addShape("rect", { x: W / 2 - 1.95, y: 4.7, w: 3.9, h: 0.13, fill: { color: S.tan }, line: { color: S.tan } });
    if (s.subtitle) slide.addText(s.subtitle, { x: 2, y: 5.0, w: W - 4, h: 0.5, fontFace: S.body, fontSize: 16,
      italic: true, color: S.white, align: "center", margin: 0, isTextBox: true });
    // a rotated box wraps at its pre-rotation width, so lay it out wide and turn it about its centre
    slide.addText(s.date || meta.date || "", { x: W - 1.05 - 1.75, y: 3.5 - 0.25, w: 3.5, h: 0.5, fontFace: S.body,
      fontSize: 14, color: S.white, rotate: 90, align: "center", valign: "middle", margin: 0, isTextBox: true });
    slide.addText([meta.author, meta.email].filter(Boolean).join("\n"), { x: M, y: H - 1.2, w: 5, h: 0.6,
      fontFace: S.body, fontSize: 11, color: S.white, margin: 0, isTextBox: true, valign: "bottom" });
  },
};

async function main() {
  const [specPath, out] = process.argv.slice(2);
  if (!specPath || !out) { console.error("usage: node build_deck.js spec.json out.pptx"); process.exit(2); }
  const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
  const base = path.dirname(path.resolve(specPath));
  const abs = (p) => (p && !path.isAbsolute(p) ? path.join(base, p) : p);
  const meta = spec.meta || {};
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title = meta.title || "Deck";
  pres.author = meta.author || "";
  pres.theme = { headFontFace: S.head, bodyFontFace: S.body };
  pres.defineSlideMaster({ title: "LIGHT", background: { color: S.bg } });
  spec.slides.forEach((s, k) => {
    for (const key of ["image"]) if (s[key]) s[key] = abs(s[key]);
    if (s.images) s.images.forEach((im) => (im.image = abs(im.image)));
    for (const f of [s.image, ...(s.images || []).map((i) => i.image)].filter(Boolean))
      if (!fs.existsSync(f)) throw new Error(`slide ${k + 1}: missing image ${f}`);
    const slide = pres.addSlide({ masterName: "LIGHT" });
    if (!BUILD[s.type]) throw new Error(`slide ${k + 1}: unknown type ${s.type}`);
    BUILD[s.type](slide, s, meta);
    if (s.type !== "title") pageNumber(slide, k + 1, s.type === "closing");
    if (s.notes) slide.addNotes(s.notes);
  });
  await pres.writeFile({ fileName: out });
  console.log(`wrote ${out} (${spec.slides.length} slides)`);
}
main().catch((e) => { console.error(e.message); process.exit(1); });
