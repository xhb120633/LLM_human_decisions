import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";
import sharp from "sharp";

const WORK = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(WORK, "llm_human_decisions_tutorial.pptx");
const BUILD_DIR = path.join(WORK, "_build");
const PREVIEW = path.join(BUILD_DIR, "preview");
const LAYOUT = path.join(BUILD_DIR, "layout");
const W = 1280;
const H = 720;
const FONT = "Arial";
const MONO = "Consolas";

const C = {
  bg: "#FFFFFF",
  paper: "#FFFFFF",
  ink: "#111827",
  body: "#283643",
  muted: "#667085",
  faint: "#F3F4F6",
  line: "#CBD1D8",
  teal: "#315F63",
  blueLight: "#9CC3DD",
  orangeLight: "#E0B273",
  mutedLight: "#AEB8C4",
  blue: "#234E70",
  purple: "#5A4B6E",
  orange: "#87591F",
  coral: "#8A3B35",
  green: "#3F684B",
  white: "#FFFFFF",
};

// One teaching-deck type scale for every table and chart.
// Dense visualizations may use the compact tier, but never invent a new size locally.
const TYPE = {
  tableHeader: 18,
  tableRowLabel: 21,
  tableBody: 20,
  tableCompact: 18,
  chartTitle: 23,
  chartSubtitle: 16,
  chartAxis: 14,
  chartLegend: 14,
  chartValue: 14,
  heatmapHeader: 14,
  heatmapCell: 14,
  heatmapRowLabel: 16,
};

const STAGES = ["Foundations", "Prediction", "Representation", "Explanation", "Discovery"];

function shape(slide, geometry, x, y, w, h, fill = "none", line = "none", width = 0, radius) {
  return slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width },
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function text(slide, value, x, y, w, h, o = {}) {
  const t = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: o.fill ?? "none",
    line: { style: "solid", fill: o.line ?? "none", width: o.lineWidth ?? 0 },
  });
  t.text = value;
  t.text.style = {
    fontSize: o.size ?? 25,
    bold: o.bold ?? false,
    italic: o.italic ?? false,
    color: o.color ?? C.ink,
    alignment: o.align ?? "left",
    verticalAlignment: o.valign ?? "top",
    typeface: o.typeface ?? FONT,
    lineSpacing: o.lineSpacing ?? 1.08,
    autoFit: o.autoFit ?? "shrinkText",
    insets: o.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return t;
}

function xmlEscape(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function addStaticLineChart(slide, cfg) {
  const { x, y, w, h, categories, series, yMin, yMax, yStep, showLegend = false, labelSeries = [] } = cfg;
  const yFormat = cfg.yFormat ?? ((v) => v.toFixed(1));
  const margin = { left: 58, right: 18, top: 16, bottom: showLegend ? 66 : 42 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  const sx = (i) => margin.left + (categories.length === 1 ? plotW / 2 : (i / (categories.length - 1)) * plotW);
  const sy = (v) => margin.top + ((yMax - v) / (yMax - yMin)) * plotH;
  const ticks = [];
  for (let v = yMin; v <= yMax + yStep * 0.25; v += yStep) ticks.push(Number(v.toFixed(8)));
  const grid = ticks.map((v) => {
    const yy = sy(v);
    return `<line x1="${margin.left}" y1="${yy}" x2="${margin.left + plotW}" y2="${yy}" stroke="${C.line}" stroke-width="1"/>` +
      `<text x="${margin.left - 10}" y="${yy + 5}" text-anchor="end" font-family="Arial" font-size="${TYPE.chartAxis}" fill="${C.muted}">${xmlEscape(yFormat(v))}</text>`;
  }).join("");
  const xLabelIndices = cfg.xLabelIndices ?? categories.map((_, i) => i);
  const xTicks = xLabelIndices.map((i) => `<text x="${sx(i)}" y="${margin.top + plotH + 24}" text-anchor="middle" font-family="Arial" font-size="${TYPE.chartAxis}" fill="${C.ink}">${xmlEscape(categories[i])}</text>`).join("");
  const bands = (cfg.bands ?? []).map((band) => {
    const left = sx(band.fromIndex);
    const right = sx(band.toIndex);
    return `<rect x="${left}" y="${margin.top}" width="${right - left}" height="${plotH}" fill="${band.color}" opacity="${band.opacity ?? 0.25}"/>`;
  }).join("");
  const chartLines = series.map((entry, seriesIndex) => {
    const points = entry.values.map((v, i) => `${sx(i)},${sy(v)}`).join(" ");
    const dash = entry.dash ? ` stroke-dasharray="${entry.dash}"` : "";
    const path = `<polyline points="${points}" fill="none" stroke="${entry.color}" stroke-width="${entry.width ?? 3}" stroke-linejoin="round" stroke-linecap="round"${dash}/>`;
    const errors = entry.errors ? entry.values.map((v, i) => {
      const err = Number.isFinite(entry.errors[i]) ? entry.errors[i] : 0;
      if (err <= 0) return "";
      const xx = sx(i);
      const yTop = sy(Math.min(yMax, v + err));
      const yBottom = sy(Math.max(yMin, v - err));
      return `<line x1="${xx}" y1="${yTop}" x2="${xx}" y2="${yBottom}" stroke="${entry.color}" stroke-width="1.4"/>` +
        `<line x1="${xx - 5}" y1="${yTop}" x2="${xx + 5}" y2="${yTop}" stroke="${entry.color}" stroke-width="1.4"/>` +
        `<line x1="${xx - 5}" y1="${yBottom}" x2="${xx + 5}" y2="${yBottom}" stroke="${entry.color}" stroke-width="1.4"/>`;
    }).join("") : "";
    const markers = entry.markers === false ? "" : entry.values.map((v, i) => `<circle cx="${sx(i)}" cy="${sy(v)}" r="${entry.markerSize ?? 4}" fill="${C.white}" stroke="${entry.color}" stroke-width="2"/>`).join("");
    const labels = labelSeries.includes(seriesIndex) ? entry.values.map((v, i) => `<text x="${sx(i)}" y="${Math.max(14, sy(v) - 10)}" text-anchor="middle" font-family="Arial" font-size="${TYPE.chartValue}" font-weight="700" fill="${entry.color}">${xmlEscape(v.toFixed(3))}</text>`).join("") : "";
    return errors + path + markers + labels;
  }).join("");
  const named = series.filter((entry) => entry.name);
  const legend = showLegend ? named.map((entry, i) => {
    const itemW = plotW / named.length;
    const xx = margin.left + i * itemW;
    const yy = h - 22;
    return `<line x1="${xx}" y1="${yy - 5}" x2="${xx + 26}" y2="${yy - 5}" stroke="${entry.color}" stroke-width="3"/>` +
      `<text x="${xx + 34}" y="${yy}" font-family="Arial" font-size="${TYPE.chartLegend}" fill="${C.body}">${xmlEscape(entry.name)}</text>`;
  }).join("") : "";
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w * 2}" height="${h * 2}" viewBox="0 0 ${w} ${h}">
    <rect width="${w}" height="${h}" fill="${C.white}"/>
    ${bands}${grid}
    <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotH}" stroke="${C.ink}" stroke-width="1.2"/>
    <line x1="${margin.left}" y1="${margin.top + plotH}" x2="${margin.left + plotW}" y2="${margin.top + plotH}" stroke="${C.ink}" stroke-width="1.2"/>
    ${xTicks}${chartLines}${legend}
  </svg>`;
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  slide.images.add({
    blob: png,
    contentType: "image/png",
    alt: cfg.alt ?? "Device-independent line chart with straight line segments",
    fit: "contain",
    position: { left: x, top: y, width: w, height: h },
  });
}

function rich(slide, paragraphs, x, y, w, h, o = {}) {
  const t = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  t.text = paragraphs;
  t.text.style = {
    fontSize: o.size ?? 25,
    color: o.color ?? C.ink,
    typeface: FONT,
    lineSpacing: o.lineSpacing ?? 1.18,
    autoFit: "shrinkText",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return t;
}

function bullets(slide, items, x, y, w, h, o = {}) {
  return rich(slide, items.map((v) => ({ bulletCharacter: "-", marginLeft: 28, indent: -16, runs: [v] })), x, y, w, h, o);
}

let slideSequence = 0;
const slideNumbers = new WeakMap();

function newSlide(p, bg = C.bg) {
  const s = p.slides.add();
  s.background.fill = bg;
  slideSequence += 1;
  slideNumbers.set(s, slideSequence);
  return s;
}

function progress(slide, active) {
  const x0 = 64;
  const y = 650;
  const gap = 12;
  const w = (1152 - gap * (STAGES.length - 1)) / STAGES.length;
  STAGES.forEach((label, i) => {
    shape(slide, "rect", x0 + i * (w + gap), y, w, i === active ? 2.5 : 1.25, i === active ? C.blue : C.line);
    text(slide, label, x0 + i * (w + gap), y + 8, w, 20, { size: 13, bold: i === active, color: i === active ? C.blue : C.muted, align: "center" });
  });
}

function header(slide, title, section, num, active, source = "") {
  text(slide, title, 64, 68, 1120, 94, { size: 40, bold: true, color: C.ink, lineSpacing: 0.98 });
  text(slide, section, 64, 30, 760, 22, { size: 13, bold: true, color: C.muted });
  shape(slide, "rect", 64, 168, 1152, 1.25, C.line);
  text(slide, String(slideNumbers.get(slide) ?? num).padStart(2, "0"), 1168, 30, 48, 20, { size: 13, color: C.muted, align: "right" });
  if (active !== undefined) progress(slide, active);
  if (source) text(slide, source, 544, 686, 672, 16, { size: 10.5, color: C.muted, align: "right" });
}

function notes(slide, value) {
  slide.speakerNotes.textFrame.setText(Array.isArray(value) ? value : [value]);
  slide.speakerNotes.setVisible(true);
}

function sectionSlide(p, n, titleValue, subtitle, active) {
  const s = newSlide(p);
  text(s, `Part ${n}`, 82, 118, 220, 34, { size: 20, bold: true, color: C.blue });
  text(s, titleValue, 82, 220, 1060, 112, { size: 54, bold: true, color: C.ink, lineSpacing: 0.96 });
  shape(s, "rect", 82, 352, 116, 2.5, C.blue);
  text(s, subtitle, 82, 394, 980, 86, { size: 27, color: C.body, lineSpacing: 1.14 });
  progress(s, active);
  notes(s, `Transition into section ${n}. Keep the focus on one question and one example at a time.`);
  return s;
}

function academicPoint(slide, value, y = 548, size = 21) {
  shape(slide, "rect", 84, y + 2, 4, 46, C.blue);
  text(slide, value, 108, y, 1042, 52, { size, bold: true, color: C.ink, lineSpacing: 1.08 });
}

function arrow(slide, x, y, w = 42, h = 56) {
  const arrowH = Math.max(8, Math.min(12, Math.round(h * 0.22)));
  return shape(slide, "rightArrow", x, y + (h - arrowH) / 2, w, arrowH, C.muted);
}
function simpleListSlide(p, cfg) {
  const s = newSlide(p);
  header(s, cfg.title, cfg.section, cfg.num, cfg.active, cfg.source);
  if (cfg.lead) text(s, cfg.lead, 84, 208, 1050, 62, { size: 29, bold: true, color: cfg.leadColor ?? C.ink });
  bullets(s, cfg.items, cfg.x ?? 104, cfg.y ?? 296, cfg.w ?? 1040, cfg.h ?? 250, { size: cfg.size ?? 26, lineSpacing: cfg.lineSpacing ?? 1.24 });
  if (cfg.takeaway) academicPoint(s, cfg.takeaway, 548, 21);
  notes(s, cfg.notes ?? cfg.title);
  return s;
}
function twoColSlide(p, cfg) {
  const s = newSlide(p);
  header(s, cfg.title, cfg.section, cfg.num, cfg.active, cfg.source);
  text(s, cfg.leftTitle, 84, 220, 500, 42, { size: 27, bold: true, color: cfg.leftColor ?? C.blue });
  text(s, cfg.rightTitle, 696, 220, 500, 42, { size: 27, bold: true, color: cfg.rightColor ?? C.blue });
  if (cfg.leftQuote) text(s, cfg.leftQuote, 102, 302, 458, 170, { size: 28, italic: true, color: C.ink, align: "left", valign: "middle", lineSpacing: 1.14 });
  else bullets(s, cfg.leftItems ?? [], 96, 292, 490, 230, { size: 25, lineSpacing: 1.24 });
  if (cfg.rightQuote) text(s, cfg.rightQuote, 714, 302, 458, 170, { size: 28, italic: true, color: C.ink, align: "left", valign: "middle", lineSpacing: 1.14 });
  else bullets(s, cfg.rightItems ?? [], 708, 292, 490, 230, { size: 25, lineSpacing: 1.24 });
  if (cfg.takeaway) academicPoint(s, cfg.takeaway, 548, 21);
  notes(s, cfg.notes ?? cfg.title);
  return s;
}
function flowSlide(p, cfg) {
  const s = newSlide(p);
  header(s, cfg.title, cfg.section, cfg.num, cfg.active, cfg.source);
  const n = cfg.steps.length;
  const x0 = 72;
  const gap = cfg.gap ?? 54;
  const w = (1136 - gap * (n - 1)) / n;
  cfg.steps.forEach((st, i) => {
    const x = x0 + i * (w + gap);
    text(s, `Step ${i + 1}`, x, 246, w, 28, { size: 14, bold: true, color: C.blue });
    text(s, st[0], x, 292, w, 48, { size: cfg.stepTitleSize ?? 27, bold: true, color: C.ink });
    text(s, st[1], x, 356, w, 106, { size: cfg.stepBodySize ?? 22, color: C.body, lineSpacing: 1.16 });
    if (i < n - 1) arrow(s, x + w + (gap - 42) / 2, 330, 42, 58);
  });
  if (cfg.takeaway) academicPoint(s, cfg.takeaway, 536, 21);
  notes(s, cfg.notes ?? cfg.title);
  return s;
}
function rowsSlide(p, cfg) {
  const s = newSlide(p);
  header(s, cfg.title, cfg.section, cfg.num, cfg.active, cfg.source);
  const startY = cfg.startY ?? 206;
  const rowH = cfg.rowH ?? 78;
  cfg.rows.forEach((r, i) => {
    const y = startY + i * rowH;
    text(s, r[0], 84, y + 4, 310, rowH - 16, { size: TYPE.tableRowLabel, autoFit: "none", bold: true, color: cfg.useRowColors ? (r[2] ?? C.blue) : C.blue, valign: "middle", lineSpacing: 1.1 });
    text(s, r[1], 424, y + 4, 730, rowH - 16, { size: TYPE.tableBody, autoFit: "none", color: C.body, valign: "middle", lineSpacing: 1.1 });
    shape(s, "rect", 84, y + rowH - 8, 1070, 1, C.line);
  });
  if (cfg.takeaway) academicPoint(s, cfg.takeaway, 548, 21);
  notes(s, cfg.notes ?? cfg.title);
  return s;
}
const p = Presentation.create({ slideSize: { width: W, height: H } });

function predictionStep(p, step, titleValue, subtitle) {
  const s = newSlide(p);
  text(s, `Prediction / Step ${step} of 3`, 82, 116, 360, 30, { size: 19, bold: true, color: C.blue });
  text(s, titleValue, 82, 220, 1080, 112, { size: 52, bold: true, color: C.ink, lineSpacing: 0.98 });
  shape(s, "rect", 82, 354, 116, 2.5, C.blue);
  text(s, subtitle, 82, 398, 980, 82, { size: 27, color: C.body, lineSpacing: 1.14 });
  progress(s, 1);
  notes(s, `Prediction step ${step}: ${titleValue}`);
  return s;
}
function breakSlide(p, prompt, active) {
  const s = newSlide(p);
  text(s, "Break / 10 minutes", 210, 256, 860, 76, { size: 52, bold: true, align: "center" });
  shape(s, "rect", 570, 350, 140, 2.5, C.blue);
  text(s, prompt, 260, 396, 760, 48, { size: 23, color: C.muted, align: "center" });
  progress(s, active);
  notes(s, "Ten-minute break.");
  return s;
}

function notebookTransition(p, cfg) {
  const s = newSlide(p);
  text(s, cfg.kicker, 84, 78, 360, 30, { size: 19, bold: true, color: C.blue });
  text(s, cfg.title, 84, 152, 1088, 74, { size: 46, bold: true, color: C.ink, lineSpacing: 0.98 });
  shape(s, "rect", 84, 246, 116, 2.5, C.blue);
  text(s, "OPEN", 84, 286, 150, 26, { size: 16, bold: true, color: C.muted });
  text(s, cfg.file ?? "notebooks/01_prediction_from_zero_shot_to_icl.ipynb", 84, 322, 620, 54, { size: cfg.fileSize ?? 21, bold: true, color: C.ink, typeface: MONO, lineSpacing: 1.05 });
  text(s, "START AT", 84, 390, 150, 26, { size: 16, bold: true, color: C.muted });
  text(s, cfg.start, 84, 426, 540, 68, { size: 24, bold: true, color: C.blue, lineSpacing: 1.12 });
  shape(s, "rect", 654, 282, 1, 258, C.line);
  text(s, "DO IN THE NOTEBOOK", 704, 286, 360, 26, { size: 16, bold: true, color: C.muted });
  cfg.steps.forEach((step, i) => {
    text(s, String(i + 1), 704, 332 + i * 50, 32, 28, { size: 18, bold: true, color: C.blue });
    text(s, step, 754, 330 + i * 50, 410, 34, { size: 21, color: C.body });
  });
  text(s, "STOP", 704, 528, 72, 24, { size: 15, bold: true, color: C.muted });
  text(s, cfg.stop, 786, 526, 374, 42, { size: 18, bold: true, color: C.ink });
  academicPoint(s, "Return to PPT: " + cfg.returnTo, 584, 19);
  progress(s, cfg.active ?? 1);
  notes(s, ["Switch from the slide deck to " + (cfg.notebookLabel ?? "Notebook 1") + ".", "Resume the deck at: " + cfg.returnTo]);
  return s;
}

// 1 Cover
{
  const s = newSlide(p);
  text(s, "Decision Neuroscience Summer School", 84, 68, 700, 30, { size: 19, bold: true, color: C.blue });
  text(s, "3-hour tutorial", 960, 68, 236, 30, { size: 19, color: C.muted, align: "right" });
  text(s, "Using Large Language Models to Understand Human Decision-Making", 84, 158, 1112, 164, { size: 54, bold: true, color: C.ink, lineSpacing: 0.96 });
  shape(s, "rect", 84, 346, 132, 2.5, C.blue);
  text(s, "Prediction, representation, explanation, and model discovery", 84, 382, 1030, 48, { size: 29, color: C.body });
  text(s, "Hanbo Xie", 84, 538, 360, 34, { size: 23, bold: true, color: C.ink });
  text(s, "Georgia Institute of Technology", 84, 578, 520, 30, { size: 21, color: C.muted });
  notes(s, "Introduce one running risky-choice example. The first half builds prediction step by step; the second half asks what improved prediction can support scientifically.");
}
// 2 Research identity and tutorial relevance
{
  const s = newSlide(p);
  header(s, "My research and why this tutorial", "INTRODUCTION", 2, undefined);

  text(s, "Hanbo Xie", 84, 224, 410, 44, { size: 34, bold: true, color: C.ink });
  text(s, "Georgia Institute of Technology", 84, 278, 430, 34, { size: 22, color: C.muted });
  text(
    s,
    "I study how people and AI think, decide, and learn.",
    84,
    352,
    430,
    82,
    { size: 27, bold: true, color: C.body, lineSpacing: 1.14 },
  );
  text(s, "Cognitive modeling | behavioral experiments\nThink-aloud data | large language models", 84, 444, 430, 64, {
    size: 20,
    color: C.muted,
    lineSpacing: 1.12,
  });
  text(s, "TODAY'S FOCUS", 84, 526, 180, 20, {
    size: 14,
    bold: true,
    color: C.muted,
  });
  text(s, "Prediction | Representation | Explanation | Discovery", 84, 550, 460, 28, {
    size: 16,
    bold: true,
    color: C.blue,
  });

  shape(s, "rect", 566, 214, 1.25, 376, C.line);
  text(s, "Selected work behind this tutorial", 620, 206, 550, 38, {
    size: 26,
    bold: true,
    color: C.ink,
  });

  const selectedWork = [
    ["ICLR 2026", "Zhu*, Xie* et al.", "Using Reinforcement Learning to Train Large Language Models to Explain Human Decisions"],
    ["CCN 2026", "Xie et al.", "Think-Aloud Reshapes Automated Cognitive Model Discovery Beyond Behavior"],
    ["COLM 2024", "Xie et al.", "From Strategic Narratives to Code-Like Cognitive Models"],
    ["NeurIPS 2023 AI4Science", "Xie, Xiong, & Wilson", "Text2Decision: Decoding Latent Variables in Risky Decision Making from Think Aloud Text"],
  ];
  selectedWork.forEach((work, i) => {
    const y = 266 + i * 82;
    text(s, work[0], 620, y, 216, 22, { size: 15, bold: true, color: C.blue });
    text(s, work[1], 846, y, 310, 22, { size: 15, color: C.muted });
    text(s, work[2], 620, y + 27, 550, 48, {
      size: 18,
      bold: true,
      color: C.body,
      lineSpacing: 1.05,
    });
  });

  notes(s, [
    "Introduce the broader research program: how people and AI think, decide, and learn, studied through cognitive modeling, behavioral experiments, think-aloud data, and language models.",
    "Then narrow to today's tutorial. The four selected papers motivate the progression from prediction to representation, explanation, and model discovery.",
    "[Sources]",
    "- https://xhb120633.github.io/#about",
    "- https://openreview.net/forum?id=coJPBEZ9Te",
    "- https://arxiv.org/abs/2605.05091 (conference status: CCN 2026, confirmed by the presenter)",
    "- https://openreview.net/forum?id=1Tny4KgGO2",
    "- https://openreview.net/forum?id=fEoemPDicz",
    "[/Sources]",
  ]);
}

// 3 Participant-facing opening trial
{
  const s = newSlide(p);
  text(s, "OPENING QUESTION", 160, 92, 300, 22, { size: 14, bold: true, color: C.teal });
  shape(s, "rect", 160, 130, 960, 430, C.paper, C.line, 1);
  text(s, "Please choose one option.", 204, 174, 620, 38, { size: 27, bold: true });
  text(s, "You will receive the outcome of the option you choose.", 204, 224, 700, 30, { size: 21, color: C.body });
  text(s, "( )  Option A\n    100% chance of receiving $40", 240, 300, 760, 74, { size: 28, bold: true, color: C.blue, lineSpacing: 1.06 });
  text(s, "( )  Option B\n    50% chance of receiving $100; otherwise $0", 240, 404, 790, 74, { size: 28, bold: true, color: C.orange, lineSpacing: 1.06 });
  text(s, "Click Continue after making your choice.", 204, 508, 540, 24, { size: 18, color: C.muted });
  text(s, "Vote first. Explain later.", 160, 600, 960, 36, { size: 25, bold: true, color: C.blue, align: "center" });
  notes(s, "Run a quick vote. Keep the choice visible as the running example.");
}

// 4 Tutorial scope: two consecutive builds act as a device-stable reveal.
for (let revealStage = 1; revealStage <= 2; revealStage++) {
  const s = newSlide(p);
  header(s, "Scope of this tutorial", "Tutorial scope", 4, undefined);

  text(s, "SCOPE", 84, 208, 180, 26, {
    size: TYPE.tableHeader,
    autoFit: "none",
    bold: true,
    color: C.muted,
  });
  text(s, "LEVEL OF ANALYSIS", 286, 208, 292, 26, {
    size: TYPE.tableHeader,
    autoFit: "none",
    bold: true,
    color: C.muted,
  });
  text(s, "WHAT THAT MEANS HERE", 612, 208, 548, 26, {
    size: TYPE.tableHeader,
    autoFit: "none",
    bold: true,
    color: C.muted,
  });
  shape(s, "rect", 84, 244, 1076, 1.25, C.line);

  const rows = [
    ["Not covered", "Model implementation", "Transformer architecture, training systems, and engineering details", C.muted],
    ["Not covered", "Broad philosophy", "Consciousness, world models, or whether language models will replace scientists", C.muted],
  ];
  if (revealStage >= 2) {
    rows.push([
      "Our focus",
      "Scientific use",
      "How language models help us describe, predict, represent, explain, and model human decisions",
      C.blue,
    ]);
  }

  rows.forEach((row, i) => {
    const y = 266 + i * 104;
    text(s, row[0], 84, y, 180, 64, {
      size: TYPE.tableBody,
      autoFit: "none",
      bold: true,
      color: row[3],
      valign: "middle",
    });
    text(s, row[1], 286, y, 292, 64, {
      size: TYPE.tableBody,
      autoFit: "none",
      bold: true,
      color: C.ink,
      valign: "middle",
    });
    text(s, row[2], 612, y, 548, 64, {
      size: TYPE.tableCompact,
      autoFit: "none",
      color: C.body,
      lineSpacing: 1.1,
      valign: "middle",
    });
    shape(s, "rect", 84, y + 82, 1076, 1, C.line);
  });

  academicPoint(
    s,
    revealStage === 1
      ? "First bracket adjacent engineering and philosophical questions."
      : "Next, define testable levels of what it means to understand a human decision.",
    584,
    20,
  );
  notes(s, [
    revealStage === 1
      ? "First reveal only the two neighboring topics that the tutorial brackets."
      : "Advance to reveal the scientific target, then move directly to the understanding ladder.",
    "The foundations section gives only the minimal functional data flow needed for later prediction and representation analyses; it does not teach Transformer implementation.",
    "The tutorial also brackets broader debates about consciousness, world models, and replacement of scientists.",
  ]);
}

// 5 What counts as understanding?
{
  const s = newSlide(p);
  header(s, "What would count as understanding a human decision?", "Core question", 5, undefined);

  text(s, "Claim", 84, 202, 180, 28, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });
  text(s, "Question", 310, 202, 330, 28, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });
  text(s, "Evidence required", 660, 202, 500, 28, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });

  const levels = [
    ["Description", "What happened?", "Summarize regularities in observed choices or reports."],
    ["Prediction", "What will happen next?", "Forecast held-out behavior with calibrated uncertainty."],
    ["Representation", "What information organizes behavior?", "Identify variables or geometry that generalize beyond the fitted data."],
    ["Explanation", "Why does the pattern occur?", "Derive discriminative and counterfactual predictions."],
    ["Mechanism", "What process generates it?", "Specify an executable process that survives intervention and transfer."],
  ];
  const levelColors = [C.muted, C.blue, C.purple, C.orange, C.coral];
  levels.forEach((r, i) => {
    const y = 242 + i * 67;
    text(s, String(i + 1), 84, y + 3, 30, 40, { size: TYPE.tableCompact, autoFit: "none", bold: true, color: levelColors[i], valign: "middle" });
    text(s, r[0], 126, y, 170, 46, { size: TYPE.tableBody, autoFit: "none", bold: true, color: levelColors[i], valign: "middle" });
    text(s, r[1], 310, y, 330, 46, { size: TYPE.tableBody, autoFit: "none", bold: true, color: C.ink, valign: "middle" });
    text(s, r[2], 660, y, 500, 46, { size: TYPE.tableCompact, autoFit: "none", color: C.body, lineSpacing: 1.08, valign: "middle" });
    if (i < levels.length - 1) shape(s, "rect", 84, y + 54, 1076, 1, C.line);
  });

  academicPoint(s, "We start with prediction because it is testable, but prediction is not the endpoint of understanding.", 584, 20);
  notes(s, [
    "Use this as the conceptual map for the entire tutorial.",
    "Each level supports a stronger scientific claim and therefore needs stronger evidence.",
    "Prediction supplies a disciplined starting point; representation, explanation, and mechanism require additional tests.",
  ]);
}
// 4 Running example
{
  const s = newSlide(p);
  header(s, "One held-out choice will anchor the tutorial", "Running example", 4, undefined, "Public teaching slice; participant pseudonymized");
  shape(s, "rect", 84, 218, 620, 310, C.white, C.line, 1.25);
  text(s, "P025 / held-out Trial 21", 116, 244, 380, 30, { size: 21, bold: true, color: C.blue });
  text(s, "Please choose one option.", 116, 292, 420, 36, { size: 27, bold: true });
  text(s, "A   75%: -16; 25%: 40", 146, 356, 420, 40, { size: 30, bold: true, color: C.blue });
  text(s, "B   -3 for sure", 146, 420, 480, 66, { size: 28, bold: true, color: C.orange, lineSpacing: 1.08 });
  text(s, "Observed choice is hidden until evaluation.", 116, 492, 480, 24, { size: 18, color: C.muted });
  text(s, "Evidence added at each step", 766, 218, 410, 34, { size: 25, bold: true, color: C.blue });
  text(s, "Modeling step", 782, 268, 208, 28, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });
  text(s, "Evidence available", 1002, 268, 174, 28, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });
  const levels = [
    ["Zero-shot prediction", "Current trial only", C.blue],
    ["Participant history", "Same person's past trials", C.purple],
    ["Fine-tuned prediction", "Training participants", C.teal],
    ["Beyond prediction", "Report or diagnostic test", C.coral],
  ];
  levels.forEach((v, i) => {
    const y = 310 + i * 62;
    shape(s, "rect", 782, y + 9, 8, 8, v[2]);
    text(s, v[0], 808, y, 190, 42, { size: TYPE.tableCompact, autoFit: "none", bold: true, color: C.ink, valign: "middle" });
    text(s, v[1], 1002, y, 174, 42, { size: TYPE.tableCompact, autoFit: "none", color: C.body, lineSpacing: 1.05, valign: "middle" });
    if (i < levels.length - 1) shape(s, "rect", 782, y + 49, 394, 1, C.line);
  });
  notes(s, "P025 is an anonymized participant in the bundled public teaching slice. Trial 21 stays fixed through the worked example.");
}
sectionSlide(p, 1, "A minimal mental model of language models", "Follow the data flow once, then watch the same next-token prediction step repeat.", 0);

// 6 Generation pipeline and autoregressive loop
// Progressive reveal: first the forward pass, then the autoregressive loop.
for (let revealStage = 1; revealStage <= 2; revealStage++) {
  const s = newSlide(p);
  header(s, "Generation repeats one next-token prediction step", "Language model foundations", 6, 0, "Conceptual overview");
  const stageXs = [72, 260, 448, 636, 824, 1012];
  for (let i = 0; i < stageXs.length - 1; i++) arrow(s, stageXs[i] + 150, 267, 38, 48);
  const stages = [
    ["Risky-choice prompt", "options + task context", C.blue], ["Tokenizer", "split text into tokens", C.orange],
    ["Token identifiers", "integer labels", C.purple], ["Token embeddings", "vectors for the tokens", C.teal],
    ["Model blocks", "combine options and context", C.green], ["Logits", "raw scores for choice labels", C.coral],
  ];
  stages.forEach((st, i) => {
    const x = stageXs[i];
    shape(s, "roundRect", x, 224, 150, 112, C.white, C.line, 1.25, "rounded-lg");
    shape(s, "rect", x, 224, 150, 5, st[2]);
    text(s, st[0], x + 12, 242, 126, 42, {
      size: 17,
      autoFit: "none",
      bold: true,
      color: C.ink,
      align: "center",
      valign: "middle",
      lineSpacing: 1.02,
    });
    text(s, st[1], x + 10, 288, 130, 32, {
      size: 14.5,
      autoFit: "none",
      color: C.body,
      align: "center",
      valign: "middle",
      typeface: i === 0 || i === 2 ? MONO : FONT,
      lineSpacing: 1.05,
    });
  });
  if (revealStage >= 2) {
    const loopXs = [270, 500, 730, 960];
    for (let i = 0; i < loopXs.length - 1; i++) arrow(s, loopXs[i] + 180, 438, 50, 44);
    text(s, "Generation routine", 82, 421, 170, 32, { size: 19, bold: true, color: C.blue });
    const loopStages = [["Normalize logits", "probabilities (softmax)"], ["Choose one token", "choice label A or B"], ["Append the token", "update the context"], ["Run the model again", "new context, new logits"]];
    loopStages.forEach((st, i) => {
      const x = loopXs[i];
      shape(s, "roundRect", x, 404, 180, 94, i === 2 ? "#F8F1E8" : C.faint, C.line, 1, "rounded-lg");
      text(s, st[0], x + 12, 424, 156, 28, { size: 17, bold: true, color: i === 2 ? C.orange : C.ink, align: "center" });
      text(s, st[1], x + 12, 460, 156, 24, { size: 14.5, color: C.body, align: "center" });
    });
    text(s, "Repeat until a stop token", 842, 510, 298, 26, { size: 16, bold: true, color: C.muted, align: "center" });
    text(s, "← updated context changes the next prediction", 318, 510, 500, 26, { size: 16, bold: true, color: C.blue, align: "center", typeface: MONO });
    shape(s, "rect", 82, 552, 1098, 1.25, C.line);
    text(s, "After generation stops", 104, 570, 210, 28, { size: 18, bold: true, color: C.blue });
    text(s, "token sequence  →  tokenizer decodes  →  readable text", 332, 570, 760, 30, { size: 20, bold: true, color: C.ink, typeface: MONO, align: "center" });
  } else academicPoint(s, "One forward pass turns the current context into scores for the next token.", 494, 21);
  notes(s, [revealStage === 1 ? "Pause after the forward data flow. Advance once to reveal how generation repeats the same prediction step." : "The second build reveals the autoregressive generation routine.", "Treat the model blocks as a black box here; the teaching goal is the data flow, not Transformer internals.", "[Sources]", "- Vaswani et al. (2017), Attention Is All You Need.", "- Holtzman et al. (2020), The Curious Case of Neural Text Degeneration."]);
}

// Tokenization
{
  const s = newSlide(p);
  header(s, "A tokenizer turns text into discrete model inputs", "Language model foundations", 7, 0, "Illustrative tokenization; exact pieces depend on the model");
  text(s, "Human-readable prompt", 84, 214, 420, 34, { size: 22, bold: true, color: C.blue });
  text(s, "A: 75% chance of -16\nB: -3 for sure", 84, 270, 430, 98, { size: 27, color: C.ink, typeface: MONO, lineSpacing: 1.28 });
  arrow(s, 526, 294, 64, 54);
  text(s, "Possible token pieces", 620, 214, 500, 34, { size: 22, bold: true, color: C.orange });
  const tokenPieces = ["A", ":", " 75", "%", " chance", " of", " -", "16"];
  let tx = 620;
  tokenPieces.forEach((piece, i) => {
    const tw = Math.max(54, piece.length * 14 + 26);
    shape(s, "roundRect", tx, 270, tw, 48, i % 2 === 0 ? "#EDF3F7" : "#F8F1E8", C.line, 1, "rounded-lg");
    text(s, '"' + piece + '"', tx + 8, 282, tw - 16, 24, { size: 17, bold: true, color: i % 2 === 0 ? C.blue : C.orange, typeface: MONO, align: "center" });
    tx += tw + 8;
  });
  text(s, "Each piece is mapped to an integer token identifier.", 620, 342, 520, 34, { size: 20, color: C.body });
  shape(s, "rect", 84, 410, 1070, 1, C.line);
  text(s, "Why this matters for decision research", 84, 438, 410, 34, { size: 22, bold: true, color: C.ink });
  bullets(s, [
    "A token is not necessarily a word: numbers, signs, spaces, and labels may split differently.",
    "A, space+A, and Option A can be different tokens; this changes which log probability must be read.",
    "For a clean choice task, verify that the response labels have stable tokenization."
  ], 520, 430, 630, 142, { size: 20, lineSpacing: 1.2 });
  academicPoint(s, "The tokenizer defines the model's measurement units before any prediction is made.", 584, 20);
  notes(s, [
    "Demonstrate that tokenization is model-specific. Do not imply that the illustrative pieces are DeepSeek's exact output.",
    "Connect directly to later A/B probability extraction: a label that spans multiple tokens requires sequence probability rather than one next-token log probability.",
    "[Sources]", "- Sennrich, Haddow, & Birch (2016), Neural Machine Translation of Rare Words with Subword Units. arXiv:1508.07909.", "- Kudo & Richardson (2018), SentencePiece. EMNLP 2018.", "[/Sources]"
  ]);
}

// Token embeddings
{
  const s = newSlide(p);
  header(s, "An embedding converts each token identifier into a vector", "Language model foundations", 8, 0, "Conceptual example");
  text(s, "Token identifier", 84, 220, 240, 34, { size: 22, bold: true, color: C.blue });
  shape(s, "roundRect", 104, 280, 180, 84, C.faint, C.line, 1.25, "rounded-lg");
  text(s, "id(75)", 122, 304, 144, 32, { size: 25, bold: true, color: C.ink, typeface: MONO, align: "center" });
  arrow(s, 326, 292, 74, 58);
  text(s, "lookup", 330, 352, 66, 24, { size: 15, bold: true, color: C.muted, align: "center" });
  text(s, "Embedding table", 426, 220, 250, 34, { size: 22, bold: true, color: C.orange });
  shape(s, "rect", 446, 272, 216, 116, C.white, C.line, 1.25);
  for (let i = 0; i < 5; i++) {
    shape(s, "rect", 446, 272 + i * 23, 216, 1, C.line);
    text(s, i === 2 ? "selected row" : "vector row", 464, 278 + i * 23, 170, 18, { size: TYPE.chartAxis, autoFit: "none", bold: i === 2, color: i === 2 ? C.orange : C.muted, typeface: MONO });
  }
  arrow(s, 692, 292, 74, 58);
  text(s, "Dense vector", 792, 220, 300, 34, { size: 22, bold: true, color: C.teal });
  shape(s, "roundRect", 802, 280, 330, 84, "#EEF6F5", C.line, 1.25, "rounded-lg");
  text(s, "[0.18, -0.42, ..., 0.07]", 814, 306, 306, 30, { size: 18, bold: true, color: C.ink, typeface: MONO, align: "center" });
  shape(s, "rect", 84, 424, 1070, 1, C.line);
  text(s, "What the vector does", 84, 450, 330, 32, { size: 22, bold: true, color: C.ink });
  text(s, "It gives the model a continuous starting representation for that token. Position information is added so the same token can be interpreted differently across the sequence.", 84, 494, 470, 78, { size: 20, color: C.body, lineSpacing: 1.18 });
  text(s, "Decision-research implication", 646, 450, 430, 32, { size: 22, bold: true, color: C.ink });
  text(s, "Probabilities, payoffs, labels, and instructions all enter as learned vectors. Their geometry may support prediction, but it is not automatically a human psychological representation.", 646, 494, 490, 78, { size: 20, color: C.body, lineSpacing: 1.18 });
  academicPoint(s, "Embeddings are learned coordinates for computation, not ready-made cognitive constructs.", 588, 20);
  notes(s, ["The numbers in the vector are illustrative.", "Clarify the distinction between the input embedding table here and later use of sentence embeddings as research features.", "[Sources]", "- Vaswani et al. (2017), Attention Is All You Need. arXiv:1706.03762.", "[/Sources]"]);
}

// Residual stream
{
  const s = newSlide(p);
  header(s, "The residual stream carries an evolving representation", "Language model foundations", 9, 0, "Simplified view of a decoder-only language model");
  text(s, "For each token position", 84, 214, 300, 32, { size: 21, bold: true, color: C.blue });
  const rx = [112, 392, 672, 952];
  for (let i = 0; i < rx.length - 1; i++) arrow(s, rx[i] + 180, 306, 70, 54);
  const residualStages = [["r0", "token + position"], ["r1", "after block 1"], ["r2", "after block 2"], ["rL", "final state"]];
  residualStages.forEach((st, i) => {
    shape(s, "roundRect", rx[i], 280, 180, 104, i === 3 ? "#EEF6F5" : C.faint, C.line, 1.25, "rounded-lg");
    text(s, st[0], rx[i] + 20, 300, 140, 34, { size: 27, bold: true, color: i === 3 ? C.teal : C.ink, typeface: MONO, align: "center" });
    text(s, st[1], rx[i] + 12, 344, 156, 24, { size: 16, color: C.body, align: "center" });
  });
  text(s, "Each model block reads the current state and adds an update:", 132, 418, 640, 32, { size: 22, color: C.body });
  text(s, "r(l+1) = r(l) + delta(l)", 782, 414, 354, 38, { size: 25, bold: true, color: C.purple, typeface: MONO, align: "center" });
  shape(s, "rect", 84, 480, 1070, 1, C.line);
  text(s, "Attention mixes information across tokens; feed-forward layers transform each position. Both write updates into the same residual stream.", 84, 506, 1050, 58, { size: 21, color: C.body, align: "center", lineSpacing: 1.16 });
  academicPoint(s, "Residual means an additive information pathway, not a prediction error or unexplained variance.", 580, 20);
  notes(s, ["Keep attention and feed-forward layers at the black-box level. The instructional target is the residual stream as the evolving carrier of information.", "Use the risky-choice connection verbally: later states can combine payoff, probability, task instruction, and participant history before producing next-token scores.", "[Sources]", "- Vaswani et al. (2017), Attention Is All You Need. arXiv:1706.03762.", "- Elhage et al. (2021), A Mathematical Framework for Transformer Circuits.", "[/Sources]"]);
}

// 7 Next-token principle
{
  const s = newSlide(p);
  header(s, "A language model returns a distribution over next tokens", "Language model foundations", 7, 0, "Live DeepSeek response / 2026-07-21");
  shape(s, "rect", 80, 214, 1120, 284, C.ink);
  text(s, "A: 75% chance of -16; 25% chance of 40\nB: -3 for sure\nReply with A or B only:", 116, 256, 560, 148, { size: 24, color: C.white, typeface: MONO, lineSpacing: 1.16 });
  shape(s, "rect", 704, 238, 1, 226, "#394150");
  text(s, "top_logprobs", 748, 236, 250, 28, { size: 17, bold: true, color: C.mutedLight, typeface: MONO });
  text(s, "token", 748, 276, 150, 24, { size: 16, color: C.mutedLight, typeface: MONO });
  text(s, "logprob", 1000, 276, 120, 24, { size: 16, color: C.mutedLight, typeface: MONO, align: "right" });
  const topTokens = [
    ['"B"', "-0.289", C.orangeLight],
    ['"A"', "-1.383", C.blueLight],
    ['"Based"', "-14.395", C.white],
    ['"The"', "-14.960", C.white],
    ['"Let"', "-15.220", C.white],
  ];
  topTokens.forEach((row, i) => {
    const y = 312 + i * 34;
    text(s, row[0], 748, y, 160, 24, { size: 18, bold: i < 2, color: row[2], typeface: MONO });
    text(s, row[1], 1000, y, 120, 24, { size: 18, bold: i < 2, color: row[2], typeface: MONO, align: "right" });
  });
  academicPoint(s, "The model interface returns vocabulary log probabilities; a choice probability is constructed later.", 540);
  notes(s, "Show this as an illustrative top-five model response. The values shown are from the live zero-shot call used in Notebook 1; tokens and values vary by model and prompt. The later probability slide keeps only the A and B label tokens and re-normalizes them.");
}


// Sampling controls
{
  const s = newSlide(p);
  header(s, "Sampling controls which token is chosen from the scores", "Language model foundations", 11, 0, "Illustrative probabilities from one fixed set of logits");
  text(s, "Temperature reshapes the distribution", 84, 210, 530, 36, { size: 24, bold: true, color: C.ink });
  text(s, "p(token i) is proportional to exp(logit i / T)", 84, 256, 530, 34, { size: 20, color: C.purple, typeface: MONO });
  const tempRows = [["T = 0.5", [0.81, 0.16, 0.03], "more concentrated"], ["T = 1.0", [0.61, 0.27, 0.12], "original scale"], ["T = 2.0", [0.47, 0.32, 0.21], "flatter"]];
  tempRows.forEach((row, i) => {
    const y = 314 + i * 76;
    text(s, row[0], 84, y + 10, 100, 26, { size: 18, bold: true, color: C.blue, typeface: MONO });
    const colors = [C.blue, C.orange, C.muted];
    let bx = 200;
    row[1].forEach((v, j) => {
      const bw = v * 300;
      shape(s, "rect", bx, y + 8, bw, 30, colors[j]);
      if (j < 2) text(s, ["A", "B"][j] + " " + Math.round(v * 100) + "%", bx + 6, y + 13, Math.max(44, bw - 8), 20, { size: 14, bold: true, color: C.white });
      bx += bw;
    });
    text(s, "other " + Math.round(row[1][2] * 100) + "%", 512, y + 13, 100, 20, { size: 14, bold: true, color: C.muted });
    text(s, row[2], 84, y + 44, 420, 22, { size: 15, color: C.muted });
  });
  shape(s, "rect", 642, 212, 1, 344, C.line);
  text(s, "Other generation controls", 692, 210, 430, 36, { size: 24, bold: true, color: C.ink });
  const controls = [["Top-k", "Keep the k highest-scoring tokens."], ["Top-p", "Keep the smallest set whose cumulative mass reaches p."], ["Penalties", "Lower scores for tokens that have already appeared."], ["Random seed", "Reproduce a sampled path when the service supports seeding."]];
  controls.forEach((r, i) => {
    const y = 270 + i * 72;
    text(s, r[0], 692, y, 230, 48, { size: TYPE.tableBody, autoFit: "none", bold: true, color: i < 2 ? C.orange : C.blue, valign: "middle" });
    text(s, r[1], 922, y, 250, 48, { size: TYPE.tableCompact, autoFit: "none", color: C.body, lineSpacing: 1.12, valign: "middle" });
    if (i < controls.length - 1) shape(s, "rect", 692, y + 56, 480, 1, C.line);
  });
  academicPoint(s, "For A/B behavioral prediction, read and re-normalize the label log probabilities; report generation settings separately.", 582, 19);
  notes(s, ["Explain the distinction between the model scores and the procedure used to sample a completion.", "Temperature probabilities are illustrative and rounded; they show the qualitative effect of applying different temperatures to a fixed logit vector.", "Top-k and top-p truncate the candidate set before sampling. Penalties modify token scores based on prior output.", "[Sources]", "- Holtzman et al. (2020), The Curious Case of Neural Text Degeneration. ICLR 2020.", "- Fan, Lewis, & Dauphin (2018), Hierarchical Neural Story Generation. ACL 2018.", "[/Sources]"]);
}

sectionSlide(p, 2, "Improve choice prediction with decision-relevant evidence", "Start with option information, then test whether the person's history or task-specific training adds predictive value.", 1);

// 9 Decision-grounded prediction overview
// Progressive reveal: current-trial baseline, then personal history and training.
for (let revealStage = 1; revealStage <= 2; revealStage++) {
  const s = newSlide(p);
  header(s, "Prediction methods test different decision questions", "Prediction overview", 9, 1);
  const cols = [84, 382, 690, 920], widths = [270, 280, 202, 266];
  ["Decision question", "Evidence available", "Technical move", "What the result can support"].forEach((h, i) => text(s, h, cols[i], 208, widths[i], 42, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted, valign: "middle" }));
  shape(s, "rect", 84, 252, 1112, 1.5, C.line);
  const rows = [
    ["Can option attributes predict a new choice?", "Current trial only", "Use the base model as-is\n(zero-shot)", "A population-level baseline and sensitivity to task framing", C.blue],
    ["Do this person's earlier choices improve later predictions?", "The same participant's prior trials", "Add history to the context\n(in-context learning)", "Within-person adaptation and evidence for individual differences", C.orange],
    ["Can labeled choices teach a stable task mapping?", "Training choices from other participants", "Fine-tune parameters\non labeled choices", "Task-level population regularities that generalize to held-out people", C.coral],
  ];
  rows.slice(0, revealStage === 1 ? 1 : rows.length).forEach((r, i) => {
    const y = 268 + i * 96;
    shape(s, "rect", 84, y + 10, 4, 60, r[4]);
    text(s, r[0], cols[0] + 18, y, widths[0] - 18, 76, { size: TYPE.tableCompact, autoFit: "none", bold: true, color: C.ink, lineSpacing: 1.12, valign: "middle" });
    text(s, r[1], cols[1], y, widths[1], 76, { size: TYPE.tableCompact, autoFit: "none", color: C.body, lineSpacing: 1.12, valign: "middle" });
    text(s, r[2], cols[2], y, widths[2], 76, { size: TYPE.tableCompact, autoFit: "none", bold: true, color: r[4], lineSpacing: 1.12, valign: "middle" });
    text(s, r[3], cols[3], y, widths[3], 76, { size: TYPE.tableCompact, autoFit: "none", color: C.body, lineSpacing: 1.12, valign: "middle" });
    shape(s, "rect", 84, y + 82, 1112, 1, C.line);
  });
  academicPoint(s, revealStage === 1 ? "Start with the current trial: can option information alone predict a new choice?" : "These methods are not a ladder of understanding; they test different sources of predictive information.", 566, 20);
  notes(s, revealStage === 1 ? "Begin with the current-trial baseline. Advance once to reveal personal history and parameter updating." : "Each technique tests a different source of predictive information: option-level regularities, within-person adaptation, or task-level population regularities.");
}

// 10 Held-out target
{
  const s = newSlide(p);
  header(s, "Begin with one real held-out choice", "Prediction / one example", 10, 1, "Public teaching slice / P025 / trial 21");
  text(s, "Available before prediction", 84, 224, 470, 34, { size: 25, bold: true, color: C.blue });
  bullets(s, ["Task instruction", "Option A: 75% of -16; 25% of 40", "Option B: -3 for sure"], 104, 294, 500, 190, { size: 24, lineSpacing: 1.25 });
  shape(s, "rect", 640, 218, 1.5, 320, C.line);
  text(s, "Hidden until evaluation", 700, 224, 470, 34, { size: 25, bold: true, color: C.coral });
  text(s, "P025's observed choice", 726, 330, 410, 54, { size: 30, bold: true, color: C.ink, align: "center" });
  text(s, "No history, no rationale, no future trials.", 700, 428, 470, 42, { size: 22, color: C.muted, align: "center" });
  academicPoint(s, "First make one prediction under a clean observation contract.", 566);
  notes(s, "This exact anonymized trial is used in Notebook 1. Keep the observed human choice hidden until after probability extraction.");
}
// 11 Zero-shot prompt
{
  const s = newSlide(p);
  header(s, "The first prompt contains only the current trial", "Prediction / one example", 11, 1, "Exact structure used in Notebook 1");
  shape(s, "rect", 86, 218, 1108, 300, C.paper, C.line, 1);
  text(s, "SYSTEM", 116, 248, 130, 28, { size: 18, bold: true, color: C.blue });
  text(s, "Answer with exactly one token: A or B.", 286, 248, 760, 30, { size: 22, typeface: MONO });
  text(s, "USER", 116, 318, 130, 28, { size: 18, bold: true, color: C.orange });
  text(s, "Predict this participant's choice on a new trial.", 286, 318, 800, 30, { size: 22, typeface: MONO });
  text(s, "TRIAL", 116, 388, 130, 28, { size: 18, bold: true, color: C.teal });
  text(s, "A: 75% of -16; 25% of 40    B: -3 for sure", 286, 388, 760, 30, { size: 22, typeface: MONO });
  text(s, "TARGET", 116, 458, 130, 28, { size: 18, bold: true, color: C.purple });
  text(s, "Answer: [next token]", 286, 458, 680, 30, { size: 23, color: C.muted, typeface: MONO });
  academicPoint(s, "Zero-shot asks what the base model predicts before seeing P025's history.", 566);
  notes(s, "Show that the output interface is part of the measurement. The target is the next-token distribution, not a self-reported probability.");
}
rowsSlide(p, {
  num: 12, title: "A choice-prediction study needs a fixed interface", section: "Prediction / step 1/3", active: 1,
  rows: [
    ["Prompt", "Task wording and current trial are frozen before evaluation", C.blue],
    ["Valid labels", "The allowed answers are exactly A and B", C.orange],
    ["Probability", "Use token log probabilities, then normalize over valid labels", C.teal],
    ["Record", "Save model version, prompt, tokenization, and output", C.purple],
  ], rowH: 84, startY: 208,
  takeaway: "A prediction is reproducible only when its interface is reproducible."
});


notebookTransition(p, {
  kicker: "HANDS-ON / NOTEBOOK 1A",
  title: "Run one held-out prediction",
  start: "Section 1: Begin with one real held-out choice",
  steps: ["Call the DeepSeek API", "Inspect completion and token log probabilities", "Re-normalize probability over A and B", "Reveal the choice and compute log loss"],
  stop: "After Section 4: optional reasoning trace",
  returnTo: "Map token surface forms to labels, then normalize"
});

// 13 Token probability example
{
  const s = newSlide(p);
  header(s, "Map token surface forms to labels, then normalize", "Prediction / probability", 13, 1, "Live DeepSeek response / P025 trial 21");
  text(s, "Top-token logprobs", 84, 218, 300, 34, { size: 25, bold: true, color: C.blue });
  text(s, '"B"   -0.289\n"A"   -1.383\n"Based"  -14.395', 98, 278, 290, 116, { size: 22, color: C.ink, typeface: MONO, lineSpacing: 1.25 });
  text(s, 'also among lower-ranked tokens:\n" B"  -15.284\n" A"  -16.229', 98, 420, 300, 92, { size: 18, color: C.muted, typeface: MONO, lineSpacing: 1.18 });
  arrow(s, 424, 342, 52, 64);
  text(s, "Aggregate valid forms", 494, 218, 304, 34, { size: 25, bold: true, color: C.ink, align: "center" });
  text(s, "log P(label B) =\nlogsumexp(-0.289, -15.284)\n\nlog P(label A) =\nlogsumexp(-1.383, -16.229)", 504, 286, 284, 190, { size: 20, color: C.body, typeface: MONO, align: "center", lineSpacing: 1.17 });
  arrow(s, 814, 342, 52, 64);
  text(s, "Normalize over {A, B}", 884, 218, 310, 34, { size: 25, bold: true, color: C.blue });
  text(s, "P(A) = 0.251", 900, 310, 220, 34, { size: 25, bold: true, color: C.blue });
  shape(s, "rect", 900, 354, 92, 16, C.blue);
  text(s, "P(B) = 0.749", 900, 412, 220, 34, { size: 25, bold: true, color: C.orange });
  shape(s, "rect", 900, 456, 276, 16, C.orange);
  academicPoint(s, "Do not let token stripping overwrite probability mass; aggregate with log-sum-exp.", 552);
  notes(s, "The sampled label was B. For behavioral modeling, use argmax or the normalized probability, not one stochastic sample.");
}
// 14 Score one prediction
{
  const s = newSlide(p);
  header(s, "Reveal the human choice and compute one loss", "Prediction / one example", 14, 1, "Live DeepSeek response / human choice from public slice");
  text(s, "Model distribution", 84, 218, 350, 34, { size: 25, bold: true, color: C.blue });
  text(s, "P(A) = 0.251\nP(B) = 0.749", 108, 292, 300, 94, { size: 29, bold: true, color: C.ink, typeface: MONO, lineSpacing: 1.24 });
  shape(s, "rect", 480, 218, 1.5, 300, C.line);
  text(s, "Reveal", 536, 218, 180, 34, { size: 25, bold: true, color: C.coral });
  text(s, "P025 chose B", 536, 300, 270, 48, { size: 32, bold: true, color: C.orange });
  shape(s, "rect", 838, 218, 1.5, 300, C.line);
  text(s, "Score", 900, 218, 180, 34, { size: 25, bold: true, color: C.teal });
  text(s, "loss = -log P(B)\n     = -log 0.749\n     = 0.289", 900, 292, 270, 116, { size: 24, bold: true, color: C.ink, typeface: MONO, lineSpacing: 1.2 });
  academicPoint(s, "Accuracy says correct; log loss also records how much probability supported the observation.", 562);
  notes(s, "This is the complete measurement for one trial: prompt, label logprobs, normalized choice probability, held-out human choice, and loss.");
}
// 15 Reasoning trace
{
  const s = newSlide(p);
  header(s, "The reasoning trace is inspectable, but it is another output", "Prediction / optional audit", 15, 1, "Live DeepSeek thinking-mode excerpt / same trial");
  text(s, "Model-generated trace", 84, 218, 360, 34, { size: 25, bold: true, color: C.blue });
  text(s, '"Expected value of A = 0.75(-16) + 0.25(40) = -2.\nOption B is a sure -3. So A has the higher expected value.\nBut we need to consider risk preferences..."', 104, 286, 730, 160, { size: 24, italic: true, color: C.ink, lineSpacing: 1.18 });
  shape(s, "rect", 880, 218, 1.5, 286, C.line);
  text(s, "Useful for", 932, 218, 230, 34, { size: 25, bold: true, color: C.teal });
  bullets(s, ["Audit", "Hypothesis generation", "Ablation design"], 932, 286, 240, 150, { size: 22, lineSpacing: 1.25 });
  academicPoint(s, "A trace can suggest tests; it is not the participant's reasoning or a validated causal mechanism.", 554);
  notes(s, "The same call consumed a long reasoning budget. Treat trace length and final-answer completion as part of the model-calling protocol, not as hidden ground truth.");
}
// 16 From example to model
{
  const s = newSlide(p);
  header(s, "From one choice to a behavioral prediction experiment", "Prediction / scale-up", 16, 1);
  const items = [
    ["1 trial", "one probability and one loss"],
    ["many targets", "mean loss and calibration"],
    ["many participants", "generalization and uncertainty"],
    ["vary history length", "a learning curve, not an anecdote"],
  ];
  items.forEach((v, i) => {
    const y = 226 + i * 78;
    text(s, String(i + 1), 84, y, 48, 46, { size: TYPE.tableRowLabel, autoFit: "none", bold: true, color: C.blue, valign: "middle" });
    text(s, v[0], 160, y, 250, 46, { size: TYPE.tableRowLabel, autoFit: "none", bold: true, color: C.ink, valign: "middle" });
    text(s, v[1], 460, y, 650, 46, { size: TYPE.tableBody, autoFit: "none", color: C.body, valign: "middle" });
    shape(s, "rect", 160, y + 54, 950, 1, C.line);
  });
  academicPoint(s, "Scaling changes the claim from one anecdote to generalization across trials and people.", 570);
  notes(s, "Use this slide as the conceptual bridge from a demonstration to behavioral modeling.");
}
predictionStep(p, 2, "Test whether personal history reveals individual differences", "Add earlier choices from the same person to predict later choices, without changing model parameters.");

// 18 Concrete in-context learning comparison
{
  const s = newSlide(p);
  header(s, "Use personal history to predict later choices", "Prediction / in-context learning", 18, 1, "Public teaching slice / P025 trials 1-5");
  text(s, "Earlier trial", 84, 214, 160, 30, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });
  text(s, "Options", 282, 214, 650, 30, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });
  text(s, "Choice", 1060, 214, 100, 30, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted, align: "right" });
  const histories = [
    ["1", "A: 5% of -36; otherwise -18   |   B: 1% of 1; otherwise -8", "A"],
    ["2", "A: 10% of -7; otherwise 32   |   B: 29 for sure", "B"],
    ["3", "A: 50% of -49; otherwise 47   |   B: 11 for sure", "A"],
    ["4", "A: 40% of 1; otherwise 3   |   B: 25% of 11; otherwise -12", "B"],
    ["5", "A: 90% of 20; otherwise 36   |   B: 25 for sure", "A"],
  ];
  histories.forEach((v, i) => {
    const y = 266 + i * 54;
    text(s, v[0], 84, y, 120, 30, { size: TYPE.tableCompact, autoFit: "none", bold: true, color: C.blue });
    text(s, v[1], 282, y, 720, 34, { size: TYPE.tableCompact, autoFit: "none", color: C.body });
    text(s, v[2], 1080, y, 70, 30, { size: TYPE.tableBody, autoFit: "none", bold: true, color: C.orange, align: "right" });
    shape(s, "rect", 84, y + 40, 1066, 1, C.line);
  });
  academicPoint(s, "The target remains trial 21; history length controls how many earlier P025 trials enter context.", 566);
  notes(s, "These are different earlier trials from P025, not repeats of the target trial. The notebook varies history length across 0, 1, 2, 5, 10, and 20 earlier trials.");
}
// 19 in-context learning controls
{
  const s = newSlide(p);
  header(s, "A control matrix identifies why examples helped", "Prediction / in-context learning diagnosis", 19, 1, "Participant-level in-context learning controls");
  text(s, "Trial-choice pairing intact", 424, 212, 300, 34, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.blue, align: "center", valign: "middle" });
  text(s, "Choices shuffled across trials", 808, 212, 330, 34, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.blue, align: "center", valign: "middle" });
  text(s, "P025's other trials", 84, 278, 240, 54, { size: TYPE.tableBody, autoFit: "none", bold: true, color: C.ink, valign: "middle" });
  text(s, "Another participant's trials", 84, 388, 260, 58, { size: TYPE.tableBody, autoFit: "none", bold: true, color: C.ink, valign: "middle" });
  shape(s, "rect", 364, 252, 786, 1, C.line);
  shape(s, "rect", 364, 358, 786, 1, C.line);
  shape(s, "rect", 364, 466, 786, 1, C.line);
  shape(s, "rect", 364, 252, 1, 215, C.line);
  shape(s, "rect", 758, 252, 1, 215, C.line);
  shape(s, "rect", 1150, 252, 1, 215, C.line);
  text(s, "same person + correct links", 408, 286, 306, 54, { size: TYPE.tableCompact, autoFit: "none", color: C.body, align: "center", valign: "middle" });
  text(s, "same choices, links broken", 804, 286, 300, 54, { size: TYPE.tableCompact, autoFit: "none", color: C.body, align: "center", valign: "middle" });
  text(s, "person changed", 408, 394, 306, 54, { size: TYPE.tableCompact, autoFit: "none", color: C.body, align: "center", valign: "middle" });
  text(s, "person + links changed", 804, 394, 300, 54, { size: TYPE.tableCompact, autoFit: "none", color: C.body, align: "center", valign: "middle" });
  text(s, "Column contrast: trial-choice correspondence", 364, 482, 430, 28, { size: 18, color: C.muted });
  text(s, "Row contrast: participant specificity", 780, 482, 370, 28, { size: 18, color: C.muted, align: "right" });
  shape(s, "rect", 84, 526, 1066, 1, C.line);
  text(s, "Separate control", 84, 542, 180, 28, { size: 18, bold: true, color: C.blue });
  text(s, "Swap A/B positions → test label or position bias", 280, 538, 720, 34, { size: 22, color: C.body });
  notes(s, "This 2x2 matrix asks whether improvement depends on the participant identity and on correct trial-choice correspondence. The target trial and response remain held out in every condition.");
}
rowsSlide(p, {
  num: 21, title: "The split determines the personalization claim", section: "Prediction / evaluation", active: 1,
  rows: [
    ["Temporal split", "Future P025 choices after earlier P025 history", C.teal],
    ["Participant split", "Entirely new people never seen in training", C.orange],
    ["Task split", "New risky-choice environments or formats", C.purple],
    ["Trial split", "New trials for already observed participants", C.blue],
  ], rowH: 84, startY: 208,
  takeaway: "Participant leakage can make personalization look stronger than it is."
});

notebookTransition(p, {
  kicker: "HANDS-ON / NOTEBOOK 1B",
  title: "Turn one prediction into a learning curve",
  start: "Section 5: Move from one example to a learning curve",
  steps: ["Increase the number of same-person history trials", "Keep the failed pilot and diagnose its imbalance", "Balance histories and targets, then extend k", "Compare the three participant-level curves"],
  stop: "After: The average hides heterogeneous trajectories",
  returnTo: "A first attempt produces a suspicious learning curve"
});

// 23 Learning curve
{
  const s = newSlide(p);
  header(s, "A first attempt produces a suspicious learning curve", "Hands-on / failed pilot", 23, 1, "Unbalanced pipeline check / 2 participants x 4 targets");
  text(s, "Mean held-out log loss", 84, 214, 360, 34, { size: TYPE.chartTitle, bold: true, color: C.blue });

  await addStaticLineChart(s, {
    x: 82, y: 248, w: 966, h: 294,
    categories: ["0", "1", "2", "5", "10", "20"],
    series: [
      { name: "Mean log loss", values: [0.594, 0.772, 0.692, 1.029, 1.684, 1.281], color: C.blue, width: 3, markerSize: 4 },
      { name: "Random choice", values: [Math.log(2), Math.log(2), Math.log(2), Math.log(2), Math.log(2), Math.log(2)], color: C.mutedLight, width: 1.75, markers: false },
    ],
    yMin: 0, yMax: 1.8, yStep: 0.6,
    labelSeries: [0],
    alt: "Mean log loss by number of history examples, with random-choice reference",
  });

  shape(s, "rect", 736, 222, 34, 2, C.mutedLight);
  text(s, "Random choice: ln(2) = 0.693", 782, 208, 250, 26, { size: TYPE.chartLegend, bold: true, color: C.muted });

  text(s, "Number of earlier P025/P026 trials in context", 356, 548, 430, 28, { size: TYPE.chartSubtitle, color: C.body, align: "center" });
  text(s, "Targets:\n7 B / 1 A\n\n20 earlier trials:\n15 B / 25 A", 1072, 278, 132, 170, { size: 18, bold: true, color: C.coral, align: "center", lineSpacing: 1.15 });
  academicPoint(s, "History length is confounded with label balance; do not conclude that more context improves prediction.", 588, 20);
  notes(s, "Keep this failed pilot. The 87.5% B target slice conflicts with the A-heavy histories and motivates balancing before interpreting history length. The chart is generated directly from the six recorded mean log-loss values.");
}

// 24 Aggregate balanced learning curve
{
  const s = newSlide(p);
  header(s, "Balancing reveals an early average gain, not steady improvement", "Hands-on / balanced learning curve", 24, 1,
    "3 participants / 24 fixed balanced targets / 3 balanced history draws");
  text(s, "Mean held-out log loss", 84, 214, 360, 34, { size: TYPE.chartTitle, bold: true, color: C.blue });
  text(s, "Error bars: standard deviation across three balanced history selections", 84, 248, 650, 24, {
    size: TYPE.chartSubtitle,
    color: C.muted,
  });

  const labels = ["0", "2", "4", "6", "8", "10", "12", "14", "16", "18"];
  const meanLoss = [0.878, 0.794, 0.816, 0.818, 0.873, 0.944, 0.911, 0.959, 0.861, 0.916];
  const sdLoss = [0.000, 0.047, 0.145, 0.130, 0.138, 0.076, 0.112, 0.223, 0.119, 0.033];
  await addStaticLineChart(s, {
    x: 82, y: 278, w: 1078, h: 278,
    categories: labels,
    series: [
      { name: "Balanced-history mean", values: meanLoss, errors: sdLoss, color: C.blue, width: 3, markerSize: 4 },
      { name: "Random choice", values: labels.map(() => Math.log(2)), color: C.mutedLight, width: 1.75, markers: false, dash: "6 5" },
    ],
    bands: [{ fromIndex: 1, toIndex: 3, color: C.blueLight, opacity: 0.24 }],
    yMin: 0.5, yMax: 1.2, yStep: 0.2,
    showLegend: true,
    alt: "Aggregate held-out log loss after balancing histories and targets, with standard-deviation error bars and random-choice reference",
  });
  text(s, "Number of balanced earlier trials in context", 382, 552, 500, 24, {
    size: TYPE.chartSubtitle,
    color: C.body,
    align: "center",
  });
  academicPoint(s, "On average, 2-6 examples help relative to zero-shot; longer histories do not yield a stable additional gain.", 590, 20);
  notes(s, [
    "This is the missing aggregate result after correcting the label imbalance in the failed pilot.",
    "Zero-shot mean log loss is 0.878. Balanced histories reduce it to 0.794-0.818 at k=2-6; later points do not show monotonic improvement.",
    "The error bars are standard deviations across three balanced history selections, not participant-level confidence intervals.",
    "The next slide decomposes this average by participant.",
    "[Sources]",
    "- notebooks/results/expanded_balanced_summary.csv",
    "[/Sources]",
  ]);
}

// 25 Participant-level balanced learning curves
{
  const s = newSlide(p);
  header(s, "The same average hides participant-level differences", "Hands-on / participant-level learning curves", 25, 1,
    "3 participants / 8 fixed targets each / 3 balanced histories at each non-zero history length");

  const labels = ["0", "2", "4", "6", "8", "10", "12", "14", "16", "18"];
  const series = [
    {
      name: "P026", color: C.blue,
      accuracy: [0.500, 0.542, 0.667, 0.542, 0.583, 0.625, 0.625, 0.708, 0.667, 0.792],
      loss: [1.027, 0.718, 0.681, 0.720, 0.766, 0.744, 0.695, 0.675, 0.620, 0.638],
    },
    {
      name: "P028", color: C.coral,
      accuracy: [0.500, 0.417, 0.500, 0.333, 0.375, 0.500, 0.458, 0.417, 0.458, 0.500],
      loss: [0.815, 0.946, 1.009, 1.114, 1.208, 1.481, 1.345, 1.524, 1.277, 1.478],
    },
    {
      name: "P031", color: C.green,
      accuracy: [0.500, 0.667, 0.583, 0.708, 0.625, 0.667, 0.625, 0.625, 0.500, 0.542],
      loss: [0.793, 0.719, 0.757, 0.620, 0.644, 0.607, 0.693, 0.677, 0.688, 0.633],
    },
  ];

  async function addLearningChart(x, key, yMin, yMax, majorUnit, reference) {
    const chartSeries = series.map((entry) => ({
      name: entry.name,
      values: entry[key],
      color: entry.color,
      width: 2.5,
      markerSize: 3.5,
    }));
    chartSeries.push({
      name: "Reference",
      values: labels.map(() => reference),
      color: C.mutedLight,
      width: 1.25,
      markers: false,
    });
    await addStaticLineChart(s, {
      x, y: 260, w: 520, h: 286,
      categories: labels,
      series: chartSeries,
      yMin, yMax, yStep: majorUnit,
      xLabelIndices: [0, 3, 6, 9],
      alt: `Participant-level ${key} learning curves with reference line`,
    });
  }

  text(s, "Accuracy", 72, 208, 510, 30, { size: TYPE.chartTitle, bold: true, color: C.blue });
  text(s, "Held-out accuracy", 72, 240, 510, 22, { size: TYPE.chartSubtitle, color: C.muted });
  text(s, "Log loss", 690, 208, 510, 30, { size: TYPE.chartTitle, bold: true, color: C.blue });
  text(s, "Lower is better", 690, 240, 510, 22, { size: TYPE.chartSubtitle, color: C.muted });

  await addLearningChart(72, "accuracy", 0.2, 0.9, 0.2, 0.5);
  await addLearningChart(690, "loss", 0.5, 1.6, 0.3, Math.log(2));

  text(s, "Number of balanced earlier trials in context", 158, 552, 350, 22, { size: TYPE.chartSubtitle, color: C.body, align: "center" });
  text(s, "Number of balanced earlier trials in context", 776, 552, 350, 22, { size: TYPE.chartSubtitle, color: C.body, align: "center" });

  const legendX = 810;
  series.forEach((entry, i) => {
    const x = legendX + i * 116;
    shape(s, "rect", x, 212, 28, 3, entry.color);
    text(s, entry.name, x + 36, 202, 68, 22, { size: TYPE.chartLegend, bold: true, color: entry.color });
  });

  academicPoint(s, "Adding personal history helps P026, hurts P028, and peaks at an intermediate history length for P031.", 592, 20);
  notes(s, "The aggregate curve obscures heterogeneous trajectories. Each participant has eight fixed balanced targets. For non-zero history lengths, each point pools those targets across three independently selected balanced histories. Both panels are generated directly from the recorded data series.");
}
predictionStep(p, 3, "Test whether training participants reveal task-level regularities", "Update parameters on labeled risky choices, then test whether the learned mapping generalizes to held-out people.");

flowSlide(p, {
  num: 25, title: "Fine-tuning targets task-level choice regularities", section: "Prediction / step 3/3", active: 1,
  steps: [["Pretraining", "Learn broad statistical structure"], ["Instruction tuning", "Learn general task following"], ["Risky-choice fine-tuning", "Learn a population mapping from options to choices"]],
  colors: [C.blue, C.teal, C.coral], source: "Ouyang et al. (2022)",
  takeaway: "Its decision-science claim is task adaptation; personalization still requires within-person evidence."
});

twoColSlide(p, {
  num: 27, title: "History and task training test different claims", section: "Prediction / comparison", active: 1,
  leftTitle: "Same-person evidence", rightTitle: "Training-participant evidence", leftColor: C.orange, rightColor: C.coral,
  leftItems: ["Predict later choices for the same person", "Tests within-person adaptation", "Sensitive to history order and identity", "Easy to shuffle or ablate as a control"],
  rightItems: ["Learn from labeled choices across people", "Tests task-level population regularities", "Requires data, compute, and version control", "Must protect held-out participants"],
  takeaway: "The two methods are complementary: training adapts the task, while context can adapt to the person."
});

rowsSlide(p, {
  num: 28, title: "Evaluate every method on the same held-out data", section: "Prediction / comparison", active: 1,
  rows: [
    ["Zero-shot", "Base parameters + target trial", C.blue],
    ["Add personal history", "Base parameters + P025 history + target trial", C.orange],
    ["Fine-tune the model", "Adapted parameters + target trial", C.coral],
    ["Combine both", "Adapted parameters + P025 history + target trial", C.purple],
  ], rowH: 84, startY: 208,
  takeaway: "One test split turns four techniques into a fair scientific comparison."
});

rowsSlide(p, {
  num: 29, title: "Choose the next lever from the failure pattern", section: "Prediction / decision", active: 1,
  rows: [
    ["Prompt-sensitive", "Test robust prompts and constrained labels before training", C.blue],
    ["History helps", "Keep personal history and test shuffle, identity, and order controls", C.orange],
    ["Stable repeated task", "Test supervised fine-tuning or low-rank adaptation on training participants", C.coral],
    ["Population + person matter", "Combine fine-tuning with personal history in context", C.purple],
  ], rowH: 84, startY: 208,
  takeaway: "Method choice should follow an observed prediction failure, not a technology checklist."
});

// 30 Bridge beyond prediction
{
  const s = newSlide(p);
  header(s, "A correct prediction can still have competing explanations", "Prediction checkpoint", 30, 1);
  text(s, "Current-trial prediction, participant history, and fine-tuning can agree on the same held-out choice.", 84, 202, 1020, 54, { size: 24, bold: true, color: C.ink, valign: "middle", autoFit: "none" });
  text(s, "Candidate driver", 84, 274, 250, 30, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });
  text(s, "Decisive manipulation", 382, 274, 390, 30, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });
  text(s, "Expected diagnostic change", 824, 274, 330, 30, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });
  const tests = [
    ["Certainty cue", "Remove certainty or change the worst outcome", "Prediction should move"],
    ["History correspondence", "Shuffle choices across P025's trials or use another person", "the history-based gain should shrink"],
    ["Concave utility", "Vary probability and payoff systematically", "Choice curve should change"],
  ];
  tests.forEach((v, i) => {
    const y = 326 + i * 78;
    text(s, v[0], 84, y, 250, 48, { size: TYPE.tableBody, autoFit: "none", bold: true, color: C.blue, valign: "middle" });
    text(s, v[1], 382, y, 390, 48, { size: TYPE.tableCompact, autoFit: "none", color: C.body, valign: "middle", lineSpacing: 1.1 });
    text(s, v[2], 824, y, 330, 48, { size: TYPE.tableCompact, autoFit: "none", color: C.body, valign: "middle", lineSpacing: 1.1 });
    shape(s, "rect", 84, y + 58, 1070, 1, C.line);
  });
  academicPoint(s, "Prediction creates candidates; diagnostic manipulations separate them.", 568);
  notes(s, "This is the bridge from prediction to representation and mechanism. Every candidate driver implies a different manipulation.");
}
breakSlide(p, "Next: what information supported the improved prediction?", 1);

sectionSlide(p, 3, "What information supports a decision?", "Four questions guide the chapter: Is information present? How does it evolve? What does it mean? Does it predict behavior?", 2);

// Representation 1: motivation and definition
{
  const s = newSlide(p);
  header(s, "One choice, different internal states", "Representation / motivation", 33, 2, "Why move beyond prediction?");
  text(s, "Observed behavior", 84, 212, 280, 30, { size: 20, bold: true, color: C.muted });
  text(s, "Chooses the risky option", 84, 270, 330, 54, { size: 30, bold: true, color: C.ink });
  arrow(s, 424, 270, 56, 54);
  text(s, "Possible state 1", 516, 212, 280, 30, { size: 20, bold: true, color: C.muted });
  text(s, "Its expected value is higher.", 516, 268, 300, 78, { size: 25, italic: true, color: C.blue, lineSpacing: 1.12 });
  text(s, "Possible state 2", 872, 212, 280, 30, { size: 20, bold: true, color: C.muted });
  text(s, "The large payoff is worth the gamble.", 872, 268, 300, 78, { size: 25, italic: true, color: C.purple, lineSpacing: 1.12 });
  shape(s, "rect", 84, 388, 1080, 1.5, C.line);
  text(s, "Representation", 84, 426, 250, 34, { size: 25, bold: true, color: C.ink });
  text(s, "Information carried by a system state that a later computation can use.", 340, 426, 810, 44, { size: 25, color: C.body });
  text(s, "h(t) = f(question + reasoning up to t)", 340, 492, 810, 38, { size: 26, bold: true, color: C.purple, typeface: MONO });
  academicPoint(s, "Next question: what observations could constrain these unobserved states?", 570, 20);
  notes(s, "Use the two routes to separate behavioral equivalence from representational equivalence. Define representation functionally: information in a state that downstream computation can use. This does not yet imply interpretation or causality.\n[Sources]\n- Local tutorial framework.");
}

// Representation 2: experimental setup
{
  const s = newSlide(p);
  header(s, "What does one choice-and-reasoning record look like?", "Representation / setup", 34, 2, "One synthetic GPT-4 record from the tutorial dataset");

  text(s, "QUESTION PROMPT", 84, 202, 210, 24, { size: 14, bold: true, color: C.muted });
  text(s, "Which option do you prefer?", 84, 230, 430, 32, { size: 24, bold: true, color: C.ink });
  text(s, "Option A: 25% chance of $1,000; otherwise $0\nOption B: $240 for sure", 84, 270, 430, 72, { size: 22, color: C.body, lineSpacing: 1.18 });

  text(s, "PERSONA", 84, 374, 210, 24, { size: 14, bold: true, color: C.muted });
  text(s, "Probability-Weighted Decision-Maker", 84, 402, 430, 36, { size: 23, bold: true, color: C.purple });

  shape(s, "rect", 548, 202, 2, 330, C.line);
  text(s, "THINK-ALOUD RESPONSE", 582, 202, 280, 24, { size: 14, bold: true, color: C.muted });
  text(s, "\"To make this decision, I'll calculate the expected value of each option.\"", 582, 234, 590, 58, { size: 22, italic: true, color: C.ink, lineSpacing: 1.15 });
  text(s, "Option A: 0.25 x $1,000 = $250\nOption B: 1.00 x $240 = $240\n\"Option A has a higher expected value than Option B.\"", 582, 310, 590, 116, { size: 21, color: C.body, lineSpacing: 1.22 });
  text(s, "ACTION CLAIM", 582, 454, 190, 24, { size: 14, bold: true, color: C.muted });
  text(s, "\"I will choose Option A.\"", 582, 482, 420, 38, { size: 25, bold: true, color: C.blue });

  text(s, "For analysis, the final claim becomes \"Option X\" and the persona label is hidden.", 84, 550, 1096, 34, { size: 20, color: C.muted, italic: true, align: "center" });
  academicPoint(s, "Next question: how can this observable record constrain an unobserved representation?", 572, 20);
  notes(s, "This is one record from the local synthetic dataset, lightly reformatted for readability. Choice 0 corresponds to Option A. Be explicit that these are synthetic GPT-4 data, not human observations. For representation analysis, the action claim is masked as Option X and the persona label is withheld.\n[Sources]\n- notebooks/results/representation/text2decision_multiscale_log_trajectories/sentence_decision_states.csv, trial_row 0.");
}
// Representation 3: representation as inference
// Progressive reveal: observation → inference boundary → Qwen measurement.
for (const revealStage of [2, 4]) {
  const s = newSlide(p);
  header(s, "Representation must be inferred from observations", "Representation / measurement", 35, 2, "A three-step measurement argument");
  const inferenceBullets = [
    "Observe: We record choices, language, eye movements, physiology, or neural signals - not the representation itself.",
    "Infer: A measurement model maps those observations to a candidate latent state.",
    "Interpret cautiously: The inferred state depends on the model and its assumptions; it is not ground truth.",
  ];
  bullets(s, inferenceBullets.slice(0, Math.min(revealStage, 3)), 104, 210, 1060, 220, { size: 27, lineSpacing: 1.27 });
  if (revealStage >= 4) {
    shape(s, "rect", 84, 454, 1096, 1.5, C.line);
    text(s, "Why use Qwen here?", 84, 480, 300, 34, { size: 25, bold: true, color: C.purple });
    bullets(s, ["The observation is think-aloud language.", "An open model provides contextual hidden states that we can inspect."], 390, 470, 770, 86, { size: 22, lineSpacing: 1.18 });
    text(s, "masked think-aloud  →  frozen Qwen  →  candidate state h(t)", 190, 552, 900, 36, { size: 25, bold: true, color: C.blue, align: "center", typeface: MONO });
    academicPoint(s, "Qwen provides a model-dependent measurement of information expressed in the text - not ground truth about GPT-4 or a human.", 594, 18.5);
  } else if (revealStage === 1) academicPoint(s, "Representation research begins with observables because the latent state itself is unavailable.", 514, 20);
  else if (revealStage === 2) academicPoint(s, "A measurement model supplies the bridge from observations to a candidate latent state.", 514, 20);
  else academicPoint(s, "The candidate state inherits the assumptions and limitations of the measurement model.", 514, 20);
  notes(s, "Reveal the argument in order: observations are available, a measurement model supplies the inferential bridge, and the result remains model-dependent. The final build specializes the argument to language and Qwen.\n[Sources]\n- Local tutorial framework.\n- docs/qwen35_sentence_states.md.");
}

// Representation 3b: operationalize the measurement input
{
  const s = newSlide(p);
  header(s, "Operationalize the inference: construct the model input", "Representation / measurement input", 36, 2, "The same record is transformed without revealing the persona or original choice");
  text(s, "QWEN INPUT AT THE CURRENT SENTENCE", 84, 204, 620, 26, { size: 14, bold: true, color: C.muted });
  shape(s, "rect", 84, 244, 690, 300, C.white, C.line, 1);
  text(s, "INSTRUCTION", 108, 264, 140, 22, { size: 14, bold: true, color: C.blue });
  text(s, "Infer which option the reasoning supports.", 108, 292, 620, 28, { size: 19, color: C.body });
  text(s, "QUESTION", 108, 334, 110, 22, { size: 14, bold: true, color: C.blue });
  text(s, "A: 25% chance of $1,000; otherwise $0    B: $240 for sure", 108, 362, 620, 30, { size: 19, color: C.ink });
  text(s, "CUMULATIVE MASKED REASONING", 108, 408, 280, 22, { size: 14, bold: true, color: C.purple });
  text(s, "I will compare expected values. A gives $250; B gives $240.\nA has the higher expected value. I choose Option X.", 108, 438, 620, 74, { size: 19, color: C.body, lineSpacing: 1.15 });

  text(s, "Included", 824, 224, 320, 32, { size: 25, bold: true, color: C.blue });
  bullets(s, ["choice-recovery instruction", "risky-choice question", "reasoning observed up to sentence t"], 840, 270, 340, 126, { size: 21, lineSpacing: 1.15 });
  text(s, "Withheld", 824, 418, 320, 32, { size: 25, bold: true, color: C.coral });
  bullets(s, ["persona label", "original A/B conclusion", "GPT-4 internal activations"], 840, 464, 340, 102, { size: 21, lineSpacing: 1.15 });
  academicPoint(s, "h(t) is Qwen's state conditioned on the observed text so far - not GPT-4's or a human's latent state.", 582, 19);
  notes(s, "Continue the exact record from the previous slide. The transformation is explicit: retain the task, question, and cumulative reasoning; hide the persona and replace the final A/B claim with Option X. The resulting state belongs to Qwen and is a model-dependent measurement of information expressed in the observed reasoning.\n[Sources]\n- scripts/extract_qwen35_sentence_states.py.\n- notebooks/results/representation/text2decision_multiscale_log_trajectories/sentence_decision_states.csv, trial_row 0.");
}
// Representation 3c: exact state extraction
// Progressive reveal: cumulative token stream, then sentence-boundary sampling.
for (const revealStage of [2]) {
  const s = newSlide(p);
  header(s, "Sample Qwen states at sentence boundaries", "Representation / state extraction", 37, 2, "One causal forward pass over each question + masked reasoning trace");
  text(s, "Tokens in the Qwen prompt", 84, 206, 440, 30, { size: 24, bold: true, color: C.ink });
  const tokenBlocks = [[84, 270, 250, "choice question"], [354, 270, 210, "sentence 1"], [584, 270, 210, "sentence 2"], [814, 270, 210, "sentence 3"]];
  tokenBlocks.forEach((v, i) => {
    shape(s, "rect", v[0], v[1], v[2], 62, i === 0 ? C.faint : C.white, i === 0 ? C.line : C.purple, 1);
    text(s, v[3], v[0], v[1] + 18, v[2], 28, { size: 20, bold: i > 0, color: i === 0 ? C.body : C.purple, align: "center" });
  });
  if (revealStage >= 2) {
    text(s, "t1", 522, 344, 42, 26, { size: 18, bold: true, color: C.blue, align: "center" });
    text(s, "t2", 752, 344, 42, 26, { size: 18, bold: true, color: C.blue, align: "center" });
    text(s, "t3", 982, 344, 42, 26, { size: 18, bold: true, color: C.blue, align: "center" });
    text(s, "At each sentence-ending token ts, save", 84, 402, 430, 34, { size: 24, bold: true, color: C.ink });
    text(s, "h(l, ts) = Qwen layer l state at that token", 84, 452, 520, 40, { size: 25, bold: true, color: C.purple, typeface: MONO });
    text(s, "33 states per boundary", 704, 402, 380, 34, { size: 24, bold: true, color: C.blue });
    text(s, "layer 00: input embedding / initial residual\nlayers 01-32: successive block outputs\neach state: 4,096 numbers", 704, 452, 430, 92, { size: 21, color: C.body, lineSpacing: 1.17 });
    academicPoint(s, "This is Qwen's contextual hidden state - not GPT-4's embedding and not an isolated sentence vector.", 580, 19);
  } else academicPoint(s, "Because Qwen is causal, each position summarizes only the question and text available so far.", 470, 20);
  notes(s, "Sentence 2 is not embedded by itself: its state includes the instruction, question, sentence 1, and sentence 2. Advance once to reveal the saved boundaries and layerwise state.\n[Sources]\n- docs/qwen35_sentence_states.md.\n- scripts/extract_qwen35_sentence_states.py.");
}

// Representation 3d: first evidence gate
{
  const s = newSlide(p);
  header(s, "Can the hidden state recover the masked choice?", "Representation / choice probe", 38, 2, "A held-out linear readout anchors the state to behavior");
  text(s, "final sentence-end state h_l(T)", 84, 244, 300, 60, { size: 23, bold: true, color: C.purple, align: "center", valign: "middle", fill: C.faint, line: C.line, lineWidth: 1, autoFit: "none" });
  arrow(s, 398, 250, 64, 48);
  text(s, "standardize +\nlogistic regression", 476, 244, 270, 60, { size: 23, bold: true, color: C.blue, align: "center", valign: "middle", fill: C.faint, line: C.line, lineWidth: 1 });
  arrow(s, 760, 250, 64, 48);
  text(s, "predict A versus B", 838, 244, 300, 60, { size: 23, bold: true, color: C.ink, align: "center", valign: "middle", fill: C.faint, line: C.line, lineWidth: 1, autoFit: "none" });

  shape(s, "rect", 84, 348, 1096, 1.5, C.line);
  const probeRows = [
    ["Training unit", "one masked GPT-4 reasoning trace"],
    ["Target", "the original A/B choice"],
    ["Generalization test", "hold out entire risky-choice questions"],
    ["Leakage controls", "persona label and explicit conclusion withheld"],
  ];
  probeRows.forEach((row, i) => {
    const y = 376 + i * 44;
    text(s, row[0], 118, y, 220, 30, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted, valign: "middle" });
    text(s, row[1], 354, y, 760, 30, { size: TYPE.tableBody, autoFit: "none", color: C.body, valign: "middle" });
  });
  academicPoint(s, "Success would show linearly accessible choice information - not that the model uses a linear decision rule.", 584, 19);
  notes(s, "Introduce one evidence gate at a time. The choice probe asks whether the masked choice is linearly decodable from Qwen's final state at each layer. Standardization and logistic regression are fitted inside the training split, and complete questions are held out. A successful probe establishes accessibility, not mechanism or causal use.\n[Sources]\n- scripts/analyze_layerwise_choice_decoding.py.\n- notebooks/results/representation/layerwise_choice_decoding/layerwise_probe_report.json.");
}
// Representation 4: recoverability and layer selection
{
  const s = newSlide(p);
  header(s, "Choice signal peaks near layer 15", "Representation / first test", 36, 2, "Choice word masked / questions held out during evaluation");
  text(s, "Balanced accuracy on held-out risky-choice questions", 84, 204, 760, 34, { size: TYPE.chartTitle, bold: true, color: C.ink });
  await addStaticLineChart(s, {
    x: 82, y: 252, w: 820, h: 286,
    categories: ["0", "4", "8", "12", "15", "18", "24", "32"],
    series: [{ name: "Linear readout", values: [0.498, 0.794, 0.834, 0.874, 0.942, 0.932, 0.902, 0.894], color: C.blue, width: 3, markerSize: 4 }],
    yMin: 0.45, yMax: 1.0, yStep: 0.1,
    alt: "Balanced choice-decoding accuracy by Qwen model layer",
  });
  text(s, "Qwen output", 952, 244, 210, 28, { size: 19, bold: true, color: C.muted });
  text(s, "0.934", 952, 278, 210, 52, { size: 40, bold: true, color: C.ink });
  text(s, "balanced accuracy", 952, 334, 210, 30, { size: 19, color: C.body });
  text(s, "Layer 15 readout", 952, 398, 210, 28, { size: 19, bold: true, color: C.blue });
  text(s, "0.942", 952, 432, 210, 52, { size: 40, bold: true, color: C.blue });
  text(s, "held-out questions", 952, 488, 210, 30, { size: 19, color: C.body });
  academicPoint(s, "Layer 15 provides a behaviorally anchored state; now trace how that information forms across the answer.", 574, 20);
  notes(s, "This is the first evidence gate. The masked reasoning is informative about choice. Layer 15 is selected because its linear readout generalizes best to held-out questions; it is not a universal decision layer.\n[Sources]\n- notebooks/results/representation/qwen35_choice_recovery/qwen35_bf16_choice_eval_report.json.\n- notebooks/results/representation/layerwise_choice_decoding/layerwise_probe_report.json.");
}

// Representation 5: cumulative trajectory
{
  const s = newSlide(p);
  header(s, "Cumulative states show how information forms over time", "Representation / trajectory", 37, 2, "One causal forward pass; states sampled at sentence boundaries");
  const prefixRows = [["t = 0", "question only", "h(0)"], ["t = 1", "question + sentence 1", "h(1)"], ["t = 2", "question + sentences 1-2", "h(2)"], ["t = 3", "question + sentences 1-3", "h(3)"]];
  prefixRows.forEach((v, i) => {
    const y = 212 + i * 78;
    text(s, v[0], 84, y, 100, 46, { size: TYPE.tableCompact, autoFit: "none", bold: true, color: C.muted, valign: "middle" });
    text(s, v[1], 214, y, 560, 46, { size: TYPE.tableBody, autoFit: "none", color: C.ink, valign: "middle" });
    arrow(s, 788, y - 2, 54, 50);
    text(s, v[2], 870, y, 180, 46, { size: TYPE.tableRowLabel, autoFit: "none", bold: true, color: C.purple, valign: "middle", typeface: MONO });
    if (i < prefixRows.length - 1) shape(s, "rect", 84, y + 58, 980, 1, C.line);
  });
  text(s, "At h(2), the model has seen sentence 1 and sentence 2 - never sentence 3.", 188, 538, 904, 38, { size: 23, bold: true, color: C.ink, align: "center" });
  academicPoint(s, "Next question: do different personas follow different paths through the same question?", 588, 20);
  notes(s, "Clarify cumulative rather than isolated sentence embeddings. Because the model is causal, each sentence-end state summarizes only the question and reasoning available so far.\n[Sources]\n- docs/qwen35_sentence_states.md.");
}

// Representation 6: directly embed the analysis output to preserve the plotted geometry
{
  const s = newSlide(p);
  header(s, "Persona trajectories diverge within the same problem", "Representation / geometry", 38, 2, "Q04 / Qwen layer 15 / 72 individual traces and persona averages");
  const mdsFigureBytes = await fs.readFile(path.join(
    WORK,
    "..",
    "notebooks",
    "results",
    "representation",
    "sentence_trajectories_q04",
    "Q04_sentence_trajectories.png",
  ));
  s.images.add({
    blob: mdsFigureBytes,
    contentType: "image/png",
    alt: "MDS visualization of 72 individual cumulative reasoning trajectories and five persona-average trajectories for question Q04",
    fit: "contain",
    position: { left: 60, top: 174, width: 1160, height: 438 },
  });
  notes(s, "This slide embeds the exact PNG generated by the analysis script rather than reconstructing the MDS paths as a PowerPoint scatter chart. The left panel shows all 72 cumulative trajectories; the right panel shows the five persona averages. The orange star is the identical question-only onset. MDS is descriptive rather than a cognitive coordinate system, so the next slide validates separation in the original high-dimensional space.\n[Sources]\n- notebooks/results/representation/sentence_trajectories_q04/Q04_sentence_trajectories.png.\n- notebooks/results/representation/sentence_trajectories_q04/Q04_trajectory_report.json.");
}
// Representation 7: validate divergence and motivate interpretation
{
  const s = newSlide(p);
  header(s, "Distance reveals divergence, not meaning", "Representation / geometry to interpretation", 39, 2, "Question 4 / layer 15 / original-state distance before semantic interpretation");
  text(s, "Mean pairwise distance between persona trajectories", 84, 204, 760, 32, { size: TYPE.chartTitle, bold: true, color: C.ink });
  await addStaticLineChart(s, {
    x: 82, y: 246, w: 760, h: 268,
    categories: ["0%", "14%", "29%", "43%", "57%", "71%", "86%"],
    series: [{ name: "Persona separation", values: [0.000, 0.296, 0.522, 0.567, 0.676, 0.558, 0.394], color: C.purple, width: 3, markerSize: 4 }],
    yMin: 0, yMax: 0.75, yStep: 0.15,
    alt: "Mean pairwise distance between persona trajectories over reasoning progress",
  });
  text(s, "Validated", 902, 226, 230, 28, { size: 21, bold: true, color: C.blue });
  text(s, "Distance increases in the original 4,096D state space. The MDS pattern is not only a projection artifact.", 902, 266, 270, 104, { size: 19, color: C.body, lineSpacing: 1.14, autoFit: "none" });
  text(s, "Still missing", 902, 396, 230, 28, { size: 21, bold: true, color: C.coral });
  text(s, "Distance does not label the paths as outcome-, probability-, value-, or uncertainty-related.", 902, 436, 270, 96, { size: 19, color: C.body, lineSpacing: 1.14, autoFit: "none" });
  shape(s, "rect", 84, 548, 1096, 1.5, C.line);
  text(s, "h_t in R^4096", 160, 564, 230, 34, { size: 25, bold: true, color: C.ink, align: "center" });
  arrow(s, 416, 558, 64, 44);
  text(s, "g(h_t) in R^12", 510, 564, 240, 34, { size: 25, bold: true, color: C.purple, align: "center" });
  text(s, "Text2Decision", 830, 564, 250, 34, { size: 24, bold: true, color: C.blue, align: "center" });
  academicPoint(s, "Next: learn a task-defined coordinate system, then validate the map before interpreting trajectories.", 612, 18.5);
  notes(s, "This slide now performs two jobs that belong together. First, validate the preceding MDS visualization in the original layer-15 state space. Second, state the unresolved interpretive problem: distance reveals separation but not decision semantics. This creates the need for Text2Decision without repeating the divergence claim on a separate slide.\n[Sources]\n- notebooks/results/representation/sentence_trajectories_q04/Q04_trajectory_report.json.\n- scripts/prepare_c13k_text2decision.py.");
}
// Representation 9: Text2Decision architecture and training contract
// Progressive reveal: train the map first, then transfer it to reasoning.
for (const revealStage of [2]) {
  const s = newSlide(p);
  header(s, "Train the map on known decision variables", "Representation / method", 41, 2, "Choice13K option descriptions → frozen Qwen layer 15 → feed-forward network → 12 targets");
  text(s, "TRAINING", 84, 204, 180, 26, { size: 14, bold: true, color: C.muted });
  text(s, "Choice13K option text", 84, 260, 240, 70, { size: 24, bold: true, color: C.ink, align: "center", valign: "middle", fill: C.faint, line: C.line, lineWidth: 1 });
  arrow(s, 338, 270, 54, 50);
  text(s, "Frozen Qwen state\nlayer 15 / 4096D", 406, 260, 250, 70, { size: 23, bold: true, color: C.purple, align: "center", valign: "middle", fill: C.faint, line: C.line, lineWidth: 1 });
  arrow(s, 668, 270, 50, 50);
  text(s, "Feed-forward network", 732, 204, 286, 28, { size: 20, bold: true, color: C.blue, align: "center" });
  const layers = [["4096", 744], ["512", 820], ["128", 896], ["64", 972], ["12", 1048]];
  layers.slice(0, -1).forEach((_, i) => arrow(s, 786 + i * 76, 274, 38, 42));
  layers.forEach((v, i) => {
    shape(s, "roundRect", v[1], 262, 58, 66, i === layers.length - 1 ? "#E8E2EE" : "#E7EEF3", i === layers.length - 1 ? C.purple : C.blue, 1, "rounded-lg");
    text(s, v[0], v[1], 280, 58, 30, { size: 19, bold: true, color: i === layers.length - 1 ? C.purple : C.blue, align: "center" });
  });
  arrow(s, 1112, 270, 38, 50);
  text(s, "12 decision\nvariables", 1150, 260, 80, 70, { size: 18, bold: true, color: C.blue, align: "center", valign: "middle" });
  text(s, "Targets are computed from each lottery: five outcome summaries, five probabilities, expected value, and entropy.", 84, 364, 1110, 50, { size: 22, color: C.body, align: "center" });
  if (revealStage >= 2) {
    shape(s, "rect", 84, 432, 1112, 1.5, C.line);
    text(s, "TRANSFER AFTER TRAINING", 84, 454, 260, 26, { size: 14, bold: true, color: C.muted });
    text(s, "GPT-4 masked reasoning prefix", 84, 502, 290, 62, { size: 22, bold: true, color: C.ink, align: "center", valign: "middle" });
    arrow(s, 386, 510, 50, 44);
    text(s, "Qwen layer-15 state h_t", 448, 502, 270, 62, { size: 22, bold: true, color: C.purple, align: "center", valign: "middle" });
    arrow(s, 730, 510, 50, 44);
    text(s, "Frozen Text2Decision", 792, 502, 250, 62, { size: 22, bold: true, color: C.blue, align: "center", valign: "middle" });
    arrow(s, 1054, 510, 50, 44);
    text(s, "12D trajectory", 1110, 502, 110, 62, { size: 21, bold: true, color: C.purple, align: "center", valign: "middle" });
    academicPoint(s, "Train on stimuli with known variables; apply the frozen map to reasoning states whose content is unknown.", 590, 18.5);
  } else academicPoint(s, "First learn whether a hidden state can recover decision variables that are known for the training stimuli.", 500, 19);
  notes(s, "The first build contains only the supervised training contract. Advance once to reveal the out-of-domain transfer to cumulative reasoning states.\n[Sources]\n- scripts/train_qwen_text2decision.py.\n- artifacts/text2decision/qwen35_layer15_text2decision_multiscale_log/training_report.json.");
}

// Representation 10: held-out training performance
{
  const s = newSlide(p);
  header(s, "Validate the readout before transferring it to reasoning", "Representation / validation", 42, 2, "Choice13K test split grouped by problem ID / mean test R2 = 0.949");
  const featureLabels = ["max gain", "min gain", "max loss", "min loss", "top-2 gains", "P(max gain)", "P(min gain)", "P(max loss)", "P(min loss)", "P(top-2)", "expected value", "entropy"];
  const testR2 = [90.4, 95.2, 88.9, 89.3, 88.0, 99.4, 98.9, 99.4, 99.3, 99.4, 91.6, 99.4];
  text(s, "test R2 (%)", 260, 202, 620, 24, { size: TYPE.chartSubtitle, bold: true, color: C.muted, align: "center" });
  testR2.forEach((value, i) => {
    const y = 232 + i * 26;
    text(s, featureLabels[i], 70, y - 2, 168, 22, { size: TYPE.chartAxis, color: C.muted, align: "right" });
    shape(s, "rect", 260, y, 620, 17, C.faint, null, 0);
    shape(s, "rect", 260, y, 620 * (value / 100), 17, C.blue, null, 0);
    text(s, value.toFixed(1), 894, y - 3, 64, 22, { size: TYPE.chartValue, bold: true, color: C.ink });
  });
  text(s, "0.949", 1002, 278, 170, 58, { size: 44, bold: true, color: C.blue, align: "center", autoFit: "none" });
  text(s, "mean test R2", 1002, 344, 170, 32, { size: 20, color: C.muted, align: "center" });
  text(s, "11,656 test options\nheld out by problem ID", 986, 430, 202, 78, { size: 20, color: C.body, align: "center", lineSpacing: 1.14, autoFit: "none" });
  academicPoint(s, "Strong held-out performance validates the mapping task - not its out-of-domain interpretation of reasoning.", 598, 18.5);
  notes(s, "Report performance on the held-out problem split, not the training set. Every target reaches test R2 above 0.879 and the mean is 0.949. This establishes that the readout performs its training task. It does not yet validate applying the map to GPT-4 reasoning states.\n[Sources]\n- artifacts/text2decision/qwen35_layer15_text2decision_multiscale_log/training_report.json.");
}

// Representation 11: persona by decision-dimension heatmap
{
  const s = newSlide(p);
  header(s, "The map reveals partial persona signatures", "Representation / interpretation", 43, 2, "Endpoint change relative to other personas on the same question");
  const dimLabels = ["max\ngain", "min\ngain", "max\nloss", "min\nloss", "top-2\ngains", "P(max\ngain)", "P(min\ngain)", "P(max\nloss)", "P(min\nloss)", "P(top-2)", "expected\nvalue", "entropy"];
  const personaRows = [
    ["Outcome-focused", -0.018, -0.021, 0.050, 0.076, -0.018, 0.003, 0.007, -0.056, 0.043, -0.002, 0.040, -0.089],
    ["Probability-weighted", -0.046, -0.026, -0.065, -0.114, -0.004, 0.022, -0.007, 0.042, -0.034, 0.009, -0.055, -0.006],
    ["Rational", -0.059, -0.043, -0.019, -0.054, -0.022, -0.004, -0.012, 0.074, -0.007, -0.006, -0.033, 0.002],
    ["Risk-averse", -0.032, -0.046, 0.019, 0.069, -0.017, -0.029, -0.028, -0.034, 0.030, -0.026, 0.014, -0.019],
    ["Risk-seeking", 0.147, 0.130, 0.013, 0.017, 0.058, 0.010, 0.039, -0.022, -0.031, 0.025, 0.031, 0.105],
  ];
  const heatValues = [["persona", ...dimLabels], ...personaRows.map((r) => [r[0], ...r.slice(1).map((v) => v.toFixed(2))])];
  const heat = s.tables.add({ rows: 6, columns: 13, left: 68, top: 222, width: 1144, height: 306, columnWidths: [210, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77], values: heatValues });
  heat.borders.assign({ style: "solid", fill: C.white, width: 2 });
  for (let c = 0; c < 13; c += 1) {
    const cell = heat.getCell(0, c);
    cell.fill = C.faint;
    cell.text.style = { fontSize: TYPE.heatmapHeader, bold: true, color: C.ink, alignment: "center", verticalAlignment: "middle", typeface: FONT };
  }
  const heatFill = (v) => v >= 0.10 ? "#B84A45" : v >= 0.05 ? "#D88C84" : v >= 0.02 ? "#F1D5D1" : v <= -0.10 ? "#3E6FA6" : v <= -0.05 ? "#8FAACA" : v <= -0.02 ? "#D6E0EC" : "#F7F5F5";
  personaRows.forEach((row, r) => {
    const labelCell = heat.getCell(r + 1, 0);
    labelCell.fill = C.white;
    labelCell.text.style = { fontSize: TYPE.heatmapRowLabel, bold: true, color: C.ink, alignment: "left", verticalAlignment: "middle", typeface: FONT };
    row.slice(1).forEach((v, c) => {
      const cell = heat.getCell(r + 1, c + 1);
      cell.fill = heatFill(v);
      cell.text.style = { fontSize: TYPE.heatmapCell, bold: Math.abs(v) >= 0.05, color: C.ink, alignment: "center", verticalAlignment: "middle", typeface: FONT };
    });
  });
  text(s, "blue = lower than other personas", 250, 548, 310, 26, { size: TYPE.chartSubtitle, color: C.blue, align: "center" });
  text(s, "red = higher than other personas", 710, 548, 310, 26, { size: TYPE.chartSubtitle, color: C.coral, align: "center" });
  academicPoint(s, "Risk-seeking shows the clearest gain-oriented signature; other labels are only partly recovered.", 588, 19);
  notes(s, "Each cell is the persona's endpoint change after subtracting the question-by-progress mean, so comparisons are within the same decision problem and stage. Risk-seeking has strong positive changes in maximum gain, minimum gain, and entropy. Outcome-focused and risk-averse partly emphasize loss-related coordinates. Rational and probability-weighted do not cleanly reduce to their labels. Treat this as an exploratory semantic signature rather than a recovered mechanism.\n[Sources]\n- notebooks/results/representation/text2decision_multiscale_log_trajectories/persona_dimension_trends/persona_dimension_endpoint.csv.\n- notebooks/results/representation/text2decision_multiscale_log_trajectories/persona_dimension_trends/persona_dimension_report.json.");
}

// Representation 12: downstream behavioral validation
{
  const s = newSlide(p);
  header(s, "Does the 12D interpretation retain behavioral information?", "Representation / downstream test", 44, 2, "Held-out questions / explicit choice statement masked");
  await addStaticLineChart(s, {
    x: 82, y: 242, w: 860, h: 308,
    categories: ["0%", "25%", "50%", "75%", "100%"],
    series: [
      { name: "12D decision state", values: [0.595, 0.500, 0.591, 0.680, 0.742], color: C.blue, width: 3, markerSize: 4 },
      { name: "A-B option axis", values: [0.403, 0.369, 0.376, 0.385, 0.385], color: C.orange, width: 2.5, markerSize: 3.5 },
      { name: "Chance", values: [0.5, 0.5, 0.5, 0.5, 0.5], color: C.mutedLight, width: 1.5, markers: false },
    ],
    yMin: 0.3, yMax: 0.8, yStep: 0.1,
    showLegend: true,
    alt: "Held-out choice prediction over reasoning progress from the 12-dimensional state and A-B option axis",
  });
  text(s, "AUC", 84, 202, 90, 34, { size: TYPE.chartTitle, bold: true, color: C.blue, autoFit: "none" });
  text(s, "0.742", 1000, 292, 170, 54, { size: 41, bold: true, color: C.blue, autoFit: "none" });
  text(s, "12D state at the\nend of reasoning", 1000, 358, 180, 62, { size: 20, color: C.body, lineSpacing: 1.14 });
  text(s, "The scalar A-to-B option axis remains near chance.", 980, 458, 220, 76, { size: 18, bold: true, color: C.ink, lineSpacing: 1.12, autoFit: "none" });
  academicPoint(s, "The interpretable projection retains cross-question choice signal; unlike the raw-state probe, this tests information preserved after mapping.", 582, 19);
  notes(s, "This is the external test for the interpreted representation. The 12-dimensional state contains cross-question choice information. The simple distance to isolated option anchors is weak, so do not interpret the trajectory as moving directly toward A or B.\n[Sources]\n- notebooks/results/representation/text2decision_multiscale_log_trajectories/choice_prediction_by_progress.csv.\n- notebooks/results/representation/text2decision_multiscale_log_trajectories/analysis_report.json.");
}

notebookTransition(p, {
  kicker: "HANDS-ON / NOTEBOOK 2",
  file: "notebooks/02_representation_from_\nhidden_states_to_reasoning_trajectories.ipynb",
  fileSize: 15.5,
  active: 2,
  notebookLabel: "Notebook 2",
  title: "What you should now be able to do",
  start: "Use Notebook 2 to reproduce the same four questions",
  steps: ["Test whether choice information is decodable", "Trace how states evolve through reasoning", "Map state movement into decision variables", "Validate the mapped state against behavior"],
  stop: "Stop before calling a predictive representation a causal mechanism",
  returnTo: "Next: turn representational hypotheses into discriminative explanations"
});
sectionSlide(
  p,
  4,
  "Turn verbal explanations into testable process evidence",
  "Define an annotation codebook, validate it against held-out evidence, and only then decide whether it should constrain a model search.",
  3,
);

twoColSlide(p, {
  num: 40,
  title: "An explanation is another observation channel",
  section: "Explanation / evidence",
  active: 3,
  leftTitle: "What the text can provide",
  rightTitle: "What it cannot provide",
  leftColor: C.blue,
  rightColor: C.coral,
  leftItems: [
    "Attended attributes and comparisons",
    "Candidate computations and conflicts",
    "Hypotheses for new diagnostic trials",
  ],
  rightItems: [
    "Direct access to a true latent state",
    "Proof that the stated reason caused the choice",
    "A substitute for held-out behavioral tests",
  ],
  takeaway: "Treat explanations as process data to be coded and tested, not as verdicts about mechanism.",
});

// 41 One trace, sentence-level annotation
{
  const s = newSlide(p);
  header(s, "Annotate one reasoning trace sentence by sentence", "Explanation / annotation", 41, 3, "Synthetic GPT-4 trace; action claim masked");
  const rows = [
    ["1", "I'll calculate the expected value of each option.", "expected value · compute", C.blue],
    ["2", "A averages $250; B gives $240 for sure.", "EV + certainty · compare", C.purple],
    ["3", "I would choose Option A.", "explicit action claim · exclude", C.coral],
  ];
  rows.forEach((row, i) => {
    const y = 216 + i * 104;
    text(s, row[0], 84, y + 8, 42, 46, { size: TYPE.tableBody, autoFit: "none", bold: true, color: row[3], align: "center", valign: "middle" });
    text(s, row[1], 150, y + 8, 610, 46, { size: TYPE.tableBody, autoFit: "none", color: C.ink, valign: "middle" });
    text(s, row[2], 806, y + 8, 350, 46, { size: TYPE.tableCompact, autoFit: "none", bold: true, color: row[3], align: "right", valign: "middle" });
    shape(s, "rect", 150, y + 72, 1006, 1, C.line);
  });
  academicPoint(s, "The unit, codebook, and leakage rule must be fixed before scaling annotation.", 548);
  notes(s, "Walk through one record. The final preference statement is excluded before downstream tests.");
}

rowsSlide(p, {
  num: 42,
  title: "A codebook separates information, operations, and roles",
  section: "Explanation / annotation",
  active: 3,
  rows: [
    ["Information", "Expected value, probability, downside, upside, certainty, and risk", C.blue],
    ["Operation", "Describe, compute, compare, threshold, or consider a counterfactual", C.purple],
    ["Role", "Supports A, supports B, neither, or an explicit preference claim", C.orange],
  ],
  rowH: 96,
  startY: 220,
  takeaway: "A restricted codebook makes human and language-model annotations comparable and auditable.",
});

// 43 Construct-validity result
{
  const s = newSlide(p);
  header(s, "Can preference-free annotations identify the synthetic persona?", "Explanation / validation", 43, 3, "Notebook 3 · 1,368 trials · 5 personas · 19 question groups · grouped cross-validation");

  text(s, "1  INPUT + LEAKAGE CONTROL", 84, 205, 326, 28, { size: 16, bold: true, color: C.blue });
  text(s, "1,368 synthetic reasoning trials\n5 prompted personas; 19 questions\nKeep only text before the first explicit\nA/B preference claim.", 84, 244, 326, 132, { size: 18, color: C.body, lineSpacing: 1.18 });

  text(s, "2  BUILD ONE 7D PROFILE PER TRIAL", 458, 205, 340, 28, { size: 16, bold: true, color: C.blue });
  text(s, "Mark whether each retained sentence\nmentions:\nEV, probability, weighting, downside,\nupside, certainty, or risk.\nAverage the 7 indicators across sentences.", 458, 244, 340, 142, { size: 18, color: C.body, lineSpacing: 1.14 });

  text(s, "3  PREDICT PERSONA ON NEW QUESTIONS", 846, 205, 350, 28, { size: 16, bold: true, color: C.blue });
  text(s, "Standardize features + logistic regression\n5-fold cross-validation grouped by question\nAll versions of a question remain in one\nfold.", 846, 244, 350, 132, { size: 18, color: C.body, lineSpacing: 1.18 });

  shape(s, "rect", 432, 205, 1.25, 177, C.line);
  shape(s, "rect", 820, 205, 1.25, 177, C.line);
  shape(s, "rect", 84, 416, 1112, 1.25, C.line);

  text(s, "Balanced accuracy = mean recall across the five persona classes", 84, 456, 676, 30, { size: 19, color: C.body });
  text(s, "Target: prompted persona - not behavioral choice", 84, 494, 676, 30, { size: 19, bold: true, color: C.body });
  text(s, "0.519", 842, 449, 142, 62, { size: 48, bold: true, color: C.blue, align: "center" });
  text(s, "cross-validated", 806, 511, 214, 32, { size: 17, color: C.body, align: "center" });
  shape(s, "rect", 1028, 456, 1.5, 78, C.line);
  text(s, "0.200", 1050, 449, 142, 62, { size: 48, bold: true, color: C.muted, align: "center" });
  text(s, "chance = 1 / 5", 1014, 511, 214, 32, { size: 17, color: C.body, align: "center" });

  academicPoint(s, "The annotation profile carries persona information across held-out questions; this is not choice accuracy or evidence of a human mechanism.", 548);
  notes(s, "Method: remove explicit A/B preference claims, average seven transparent lexical annotation indicators within each trial, and predict the five prompted personas with standardized logistic regression. Evaluation uses five-fold grouped cross-validation with question_id as the group. Balanced accuracy is the macro-average recall across persona classes. The 0.519 score is persona classification, not choice prediction.");
}

twoColSlide(p, {
  num: 44,
  title: "Continuous states and annotations test each other",
  section: "Representation → Explanation",
  active: 3,
  leftTitle: "Hidden-state readout",
  rightTitle: "Text annotation",
  leftColor: C.purple,
  rightColor: C.blue,
  leftItems: [
    "Continuous trajectory in a learned 12D space",
    "Sensitive to information not explicitly verbalized",
    "Interpretation depends on the trained readout",
  ],
  rightItems: [
    "Discrete claims about attended variables and operations",
    "Auditable against the original sentence",
    "Incomplete and vulnerable to rationalization",
  ],
  takeaway: "Convergence strengthens a hypothesis; disagreement identifies the next control.",
});

// 45 Annotation to restricted candidate specification
{
  const s = newSlide(p);
  header(s, "Annotations can nominate computational ingredients", "Explanation / optional bridge", 45, 3, "Restricted candidate specification");
  text(s, "Observed annotations", 84, 220, 360, 34, { size: 25, bold: true, color: C.blue });
  bullets(s, [
    "expected value",
    "downside / minimum outcome",
    "certainty threshold",
  ], 100, 284, 360, 190, { size: 24, lineSpacing: 1.28 });
  arrow(s, 494, 330, 62, 58);
  text(s, "Approved specification", 606, 220, 550, 34, { size: 25, bold: true, color: C.orange });
  text(
    s,
    '{\n  "rule": "downside_threshold_then_EV",\n  "features": ["worst_diff", "ev_diff"],\n  "parameters": ["threshold", "sensitivity"]\n}',
    606,
    280,
    540,
    190,
    { size: 21, typeface: MONO, color: C.ink, lineSpacing: 1.18 },
  );
  academicPoint(s, "This optional constraint did not win: balanced accuracy 0.555 versus 0.609 for EV-only.", 548);
  notes(s, "This slide shows one optional bridge from validated annotation to a bounded model specification. Discovery can instead begin from behavior or prior theory. In the offline result, the annotation-guided option model did not outperform EV-only.");
}

// 46 Held-question behavior result
{
  const s = newSlide(p);
  header(s, "Annotations predict choice; simple fusion does not help", "Explanation / evaluation", 46, 3, "Notebook 3 transparent lexical baseline; API annotation not used");
  text(s, "Input to the readout", 84, 218, 430, 28, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted });
  text(s, "Balanced accuracy", 674, 218, 220, 28, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted, align: "center" });
  text(s, "Log loss", 956, 218, 180, 28, { size: TYPE.tableHeader, autoFit: "none", bold: true, color: C.muted, align: "center" });
  const rows = [
    ["Option features only", "0.529", "0.738", C.muted],
    ["Preference-free annotation profile", "0.619", "0.673", C.blue],
    ["Options + annotation", "0.587", "0.720", C.purple],
  ];
  rows.forEach((row, i) => {
    const y = 276 + i * 86;
    text(s, row[0], 84, y + 4, 500, 52, { size: TYPE.tableBody, autoFit: "none", bold: i === 1, color: row[3], valign: "middle" });
    text(s, row[1], 674, y + 4, 220, 52, { size: 24, bold: true, color: row[3], align: "center", valign: "middle" });
    text(s, row[2], 956, y + 4, 180, 52, { size: 24, bold: true, color: row[3], align: "center", valign: "middle" });
    shape(s, "rect", 84, y + 66, 1052, 1, C.line);
  });
  academicPoint(s, "The text contains behavioral information; concatenating two channels is not automatically better.", 548);
  notes(s, "All three readouts hold out entire questions. Explicit preference claims were removed before annotation. This is predictive evidence, not causal recovery.");
}

sectionSlide(
  p,
  5,
  "Search for executable models",
  "Generate candidate processes, fit them to behavior, and design diagnostic trials that separate the surviving accounts.",
  4,
);

{
  const s = newSlide(p);
  header(s, "Model discovery is a separate search problem", "Model discovery / boundary", 47, 4, "Annotation is one optional input");

  text(s, "Explanation / annotation", 84, 206, 450, 34, { size: 25, bold: true, color: C.blue });
  text(s, "Question", 84, 260, 120, 26, { size: 17, bold: true, color: C.muted });
  text(s, "What information or operation appears in a report?", 218, 256, 330, 62, { size: 23, bold: true, color: C.ink, lineSpacing: 1.08 });
  text(s, "Output", 84, 346, 120, 26, { size: 17, bold: true, color: C.muted });
  text(s, "Validated observation variables", 218, 342, 330, 40, { size: 22, color: C.body });

  shape(s, "rect", 598, 206, 1.5, 190, C.line);
  text(s, "Model discovery", 650, 206, 500, 34, { size: 25, bold: true, color: C.coral });
  text(s, "Question", 650, 260, 120, 26, { size: 17, bold: true, color: C.muted });
  text(s, "Which executable account survives new evidence?", 784, 256, 352, 62, { size: 23, bold: true, color: C.ink, lineSpacing: 1.08 });
  text(s, "Can start from", 650, 346, 120, 26, { size: 17, bold: true, color: C.muted });
  text(s, "Behavior · theory · process data · intervention", 784, 342, 372, 52, { size: 21, color: C.body, lineSpacing: 1.05 });

  const discoverySteps = [
    ["Evidence + candidate space", C.purple],
    ["Propose executable models", C.blue],
    ["Fit and evaluate", C.teal],
    ["Design diagnostic tests", C.coral],
  ];
  discoverySteps.forEach((step, i) => {
    const x = 84 + i * 280;
    text(s, step[0], x, 450, 244, 66, { size: 20, bold: true, color: step[1], align: "center", valign: "middle", fill: C.faint, line: C.line, lineWidth: 1 });
    if (i < discoverySteps.length - 1) arrow(s, x + 246, 459, 32, 48);
  });
  academicPoint(s, "Annotation can constrain discovery, but discovery neither requires annotation nor inherits its validity.", 566, 20);
  notes(s, "Draw the boundary explicitly. Annotation converts reports into testable variables. Model discovery searches over executable accounts and may begin from behavior, theory, process data, or interventions. Notebook 3 demonstrates an annotation-guided route, not a required pipeline.");
}

// 48 Diagnostic trial
{
  const s = newSlide(p);
  header(s, "A diagnostic trial separates the surviving candidates", "Model discovery / active test", 48, 4, "Toy active-design example");
  text(s, "A  /  $35 for sure", 84, 230, 430, 38, { size: 27, bold: true, color: C.blue });
  text(s, "B  /  90% chance of $50; otherwise $0", 548, 230, 610, 38, { size: 27, bold: true, color: C.orange });
  shape(s, "rect", 84, 294, 1074, 1, C.line);
  const rows = [
    ["Expected value", "Choose B", "EV(A) = $35; EV(B) = $45", C.blue],
    ["Safety-first", "Choose A", "Minimum(A) = $35; Minimum(B) = $0", C.orange],
    ["Threshold then EV", "Depends on threshold", "Vary the downside while holding EV fixed", C.purple],
  ];
  rows.forEach((row, i) => {
    const y = 330 + i * 72;
    text(s, row[0], 84, y, 290, 42, { size: TYPE.tableBody, autoFit: "none", bold: true, color: row[3], valign: "middle" });
    text(s, row[1], 410, y, 260, 42, { size: TYPE.tableBody, autoFit: "none", bold: true, color: C.ink, valign: "middle" });
    text(s, row[2], 700, y, 458, 42, { size: TYPE.tableCompact, autoFit: "none", color: C.body, valign: "middle" });
  });
  academicPoint(s, "Choose trials where candidate predictions diverge, then update both behavioral and process evidence.", 548);
  notes(s, "The hybrid is only one candidate. The scientific value comes from choosing a trial on which surviving executable accounts disagree.");
}

// 49 Surrogate mechanism
{
  const s = newSlide(p);
  header(s, "A good fit is still not mechanism recovery", "Model discovery / limit", 49, 4, "Conceptual illustration");
  text(s, "Observed task region", 96, 220, 450, 30, { size: 22, bold: true, color: C.muted, align: "center" });
  text(s, "New diagnostic region", 696, 220, 450, 30, { size: 22, bold: true, color: C.muted, align: "center" });
  shape(s, "rect", 96, 286, 450, 170, C.faint, C.line, 1);
  shape(s, "rect", 696, 286, 450, 170, C.faint, C.line, 1);
  text(s, "Ground-truth and surrogate\nmake nearly identical predictions", 134, 334, 374, 82, { size: 24, bold: true, color: C.ink, align: "center", valign: "middle" });
  text(s, "The same models diverge\nunder a targeted manipulation", 734, 334, 374, 82, { size: 24, bold: true, color: C.ink, align: "center", valign: "middle" });
  arrow(s, 594, 338, 54, 56);
  academicPoint(s, "No discovery entry point guarantees mechanism recovery; intervention and transfer are still needed for a causal claim.", 548);
  notes(s, "Keep the surrogate lesson, but express it with two task regions rather than a decorative hand-drawn curve.");
}

rowsSlide(p, {
  num: 50,
  title: "Define the level of success before running discovery",
  section: "Model discovery / claims",
  active: 4,
  rows: [
    ["Predictive success", "Held-out likelihood improves", C.blue],
    ["Process consistency", "Model operations agree with independent process evidence", C.green],
    ["Primitive recovery", "The generating computational components are recovered", C.coral],
    ["Mechanistic recovery", "The process survives intervention and transfer", C.purple],
  ],
  rowH: 80,
  startY: 204,
  takeaway: "A system can succeed at one level and fail at every stronger level.",
});

flowSlide(p, {
  num: 51,
  title: "Notebook 3 demonstrates one annotation-guided route",
  section: "Hands-on",
  active: 4,
  steps: [
    ["Mask", "Remove explicit preference claims"],
    ["Annotate", "Code information and operations"],
    ["Validate", "Held-question construct and choice tests"],
    ["Compile", "Build restricted candidates"],
    ["Diagnose", "Find questions where models disagree"],
  ],
  colors: [C.coral, C.purple, C.blue, C.teal, C.orange],
  gap: 70,
  stepTitleSize: 22,
  stepBodySize: 18,
  takeaway: "This notebook is one route: discovery can also begin from behavior, prior theory, or interventions without annotation.",
});

breakSlide(p, "After the break: return to the opening choice with stronger scientific claims.", 4);

// 53 Final synthesis
{
  const s = newSlide(p);
  header(s, "Return to the original choice with stronger questions", "Final synthesis", 53, undefined);
  const qs = [
    ["Prediction", "Which option will P025 choose?", C.blue],
    ["Representation", "What information improves the prediction?", C.purple],
    ["Explanation", "Which counterfactuals follow from that information?", C.orange],
    ["Discovery", "Which executable process survives diagnostic trials?", C.coral],
  ];
  qs.forEach((q, i) => {
    const y = 208 + i * 88;
    text(s, q[0], 84, y + 4, 300, 58, { size: TYPE.tableRowLabel, autoFit: "none", bold: true, color: q[2], valign: "middle" });
    text(s, q[1], 430, y + 4, 720, 58, { size: TYPE.tableBody, autoFit: "none", color: C.body, valign: "middle" });
    shape(s, "rect", 84, y + 78, 1066, 1, C.line);
  });
  academicPoint(s, "The technical ladder comes first; stronger scientific claims require stronger evidence.", 566);
  notes(s, "Return explicitly to the opening vote and show how each later question was motivated.");
}

// 54 Take-home messages
{
  const s = newSlide(p);
  header(s, "Four ideas to take back to your research", "Take-home messages", 54, undefined);
  const takeaways = [
    "Define prediction by its observation set, output labels, and held-out split.",
    "Improve prediction in a controlled order: current trial, participant history, then fine-tuning.",
    "Use representations and explanations only when they support external tests.",
    "Use model discovery to separate mechanisms, not to bypass identifiability.",
  ];
  takeaways.forEach((v, i) => {
    const y = 222 + i * 90;
    text(s, String(i + 1), 84, y, 62, 48, { size: 34, bold: true, color: C.blue });
    text(s, v, 166, y + 2, 976, 58, { size: 26, color: C.ink, lineSpacing: 1.1 });
    if (i < 3) shape(s, "rect", 166, y + 70, 976, 1, C.line);
  });
  notes(s, "End with four practical rules that map directly onto the tutorial sequence.");
}
// 55 Discussion
{
  const s = newSlide(p);
  text(s, "What additional evidence would convince you", 160, 210, 960, 54, { size: 38, bold: true, align: "center" });
  text(s, "that a language model had discovered a real human\ndecision mechanism?", 190, 314, 900, 104, { size: 39, bold: true, color: C.teal, align: "center", lineSpacing: 1.02 });
  shape(s, "rect", 540, 488, 200, 4, C.orange);
  text(s, "Discussion / Questions / Applications", 350, 540, 580, 30, { size: 21, color: C.muted, align: "center" });
  notes(s, "Use the final minutes for discussion and application to participants' own tasks.");
}

simpleListSlide(p, {
  num: 56, title: "Selected references", section: "Appendix", active: undefined,
  items: [
    "Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762",
    "Brown et al. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165",
    "Ouyang et al. (2022). Training language models to follow instructions with human feedback. arXiv:2203.02155",
    "Aher, Arriaga, & Kalai (2023). Using LLMs to Simulate Multiple Humans. arXiv:2208.10264",
    "Argyle et al. (2023). Out of One, Many. Political Analysis, 31(3), 337-351",
    "Binz & Schulz (2023). Using cognitive psychology to understand GPT-3. PNAS, 120(6)",
    "Li et al. (2024). Automated Statistical Model Discovery with Language Models. PMLR 235",
    "Xie, Xiong, & Wilson (2023). Text2Decision. NeurIPS 2023 AI for Science Workshop"
  ],
  x: 84, y: 202, w: 1100, h: 382, size: 19, lineSpacing: 1.15
});await fs.mkdir(path.dirname(OUT), { recursive: true });
await fs.rm(PREVIEW, { recursive: true, force: true });
await fs.rm(LAYOUT, { recursive: true, force: true });
await fs.mkdir(PREVIEW, { recursive: true });
await fs.mkdir(LAYOUT, { recursive: true });

for (const [i, slide] of p.slides.items.entries()) {
  const stem = `slide-${String(i + 1).padStart(2, "0")}`;
  const png = await p.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(PREVIEW, `${stem}.png`), Buffer.from(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(LAYOUT, `${stem}.json`), await layout.text(), "utf8");
}

const pptx = await PresentationFile.exportPptx(p);
await pptx.save(OUT);
console.log(`Saved ${p.slides.items.length} slides to ${OUT}`);
