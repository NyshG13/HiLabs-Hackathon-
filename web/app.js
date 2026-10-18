/* TermMap-X console. No dependencies.
   Motion model (Apple, "Designing Fluid Interfaces"): springs described by damping ratio + response, always animate
   from the live value, hand release velocity to the spring, project momentum to decide the outcome. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const RM = matchMedia("(prefers-reduced-motion: reduce)");
const sleep = (ms) => new Promise((r) => setTimeout(r, RM.matches ? Math.min(ms, 60) : ms));

/* ───────────── physics ───────────── */
class Spring {
  // damping 1 = critically damped (no overshoot); <1 bounces. response = seconds to reach target (not a duration).
  constructor(value = 0, { damping = 1, response = 0.4 } = {}) { this.x = this.t = value; this.v = 0; this.tune(damping, response); }
  tune(d, r) { const w = (2 * Math.PI) / r; this.k = w * w; this.c = 2 * d * w; return this; }
  step(dt) {
    if (RM.matches) { this.x = this.t; this.v = 0; return; }
    const n = Math.ceil(dt * 240), h = dt / n;                   // fixed substeps: stable for stiff springs
    for (let i = 0; i < n; i++) { this.v += (this.k * (this.t - this.x) - this.c * this.v) * h; this.x += this.v * h; }
  }
  get settled() { return Math.abs(this.v) < 0.02 && Math.abs(this.t - this.x) < 0.01; }
}
// Where a flick comes to rest, using the same exponential decay as scrolling (Apple's sample code).
const project = (v, d = 0.998) => ((v / 1000) * d) / (1 - d);
// Progressive resistance past a boundary instead of a hard stop.
const rubber = (over, dim, c = 0.55) => (over * dim * c) / (dim + c * Math.abs(over));

// One shared display-synced clock. A job returns true to keep running; idle pages cost nothing.
const jobs = new Set();
let last = 0, running = false;
function frame(now) {
  const dt = Math.min(0.033, (now - last) / 1000 || 0.016); last = now;
  for (const j of jobs) if (!j(dt, now)) jobs.delete(j);
  if (jobs.size) requestAnimationFrame(frame); else running = false;
}
function run(job) { jobs.add(job); if (!running) { running = true; last = performance.now(); requestAnimationFrame(frame); } }

// Convenience: spring a set of numbers and call write() every frame until settled.
function springTo(springs, write) {
  run((dt) => { springs.forEach((s) => s.step(dt)); write(); return !springs.every((s) => s.settled); });
}

/* ───────────── ambient field (hero background) ───────────── */
const Field = (() => {
  const cv = $("#field"), ctx = cv.getContext("2d");
  let w = 0, h = 0, dpr = 1, pts = [], scrollV = 0, lastY = scrollY, t = 0, visible = true;
  const mouse = { x: -1e4, y: -1e4, vx: 0, vy: 0 };
  const C1 = [100, 210, 255], C2 = [191, 90, 242];
  const size = () => {
    dpr = Math.min(devicePixelRatio || 1, 2); w = innerWidth; h = innerHeight;
    cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(clamp((w * h) / 2600, 260, 900));
    pts = Array.from({ length: n }, () => ({ x: Math.random() * w, y: Math.random() * h, vx: 0, vy: 0, z: 0.35 + Math.random() * 0.65, g: 0, gc: C1 }));
    ctx.fillStyle = "#050507"; ctx.fillRect(0, 0, w, h);
  };
  const flow = (x, y) => Math.sin(x * 0.0021 + t * 0.11) + Math.cos(y * 0.0027 - t * 0.09) + Math.sin((x + y) * 0.0013 + t * 0.07);
  function draw(dt) {
    t += dt;
    ctx.fillStyle = "rgba(5,5,7,0.2)"; ctx.fillRect(0, 0, w, h);
    scrollV *= 0.9;
    for (const p of pts) {
      const a = flow(p.x, p.y) * Math.PI * 0.9;
      p.vx += (Math.cos(a) * 26 * p.z - p.vx) * 0.03; p.vy += (Math.sin(a) * 26 * p.z - p.vy) * 0.03 + scrollV;
      const dx = p.x - mouse.x, dy = p.y - mouse.y, d2 = dx * dx + dy * dy;
      if (d2 < 24000) {                                   // pointer: soft repel + swirl, falls off smoothly
        const d = Math.sqrt(d2) + 0.01, f = (1 - d / 155) ** 2;
        p.vx += (dx / d) * f * 160 * dt * 60 * 0.12 + (-dy / d) * f * 22 + mouse.vx * f * 0.06;
        p.vy += (dy / d) * f * 160 * dt * 60 * 0.12 + (dx / d) * f * 22 + mouse.vy * f * 0.06;
        p.g = Math.max(p.g, f * 0.7); p.gc = lerpC(C1, C2, p.x / w);
      }
      p.x += p.vx * dt; p.y += p.vy * dt;
      if (p.x < -10) p.x = w + 10; else if (p.x > w + 10) p.x = -10;
      if (p.y < -10) p.y = h + 10; else if (p.y > h + 10) p.y = -10;
      p.g *= 0.955;
      const base = lerpC(C1, C2, p.x / w), glow = p.g;
      const c = glow > 0.02 ? lerpC(base, p.gc, Math.min(1, glow * 1.6)) : base;
      ctx.fillStyle = `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${0.12 + 0.3 * p.z + glow * 0.7})`;
      const s = (0.8 + p.z * 1.1) * (1 + glow * 2.2);
      ctx.fillRect(p.x - s / 2, p.y - s / 2, s, s);
    }
    mouse.vx *= 0.85; mouse.vy *= 0.85;
  }
  const lerpC = (a, b, k) => { k = clamp(k, 0, 1); return [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)]; };
  // A radial shockwave of light — used when a mapping lands.
  function burst(x, y, rgb) {
    for (const p of pts) {
      const dx = p.x - x, dy = p.y - y, d = Math.hypot(dx, dy) + 0.01;
      if (d < 420) { const f = (1 - d / 420) ** 1.5; p.vx += (dx / d) * f * 520; p.vy += (dy / d) * f * 520; p.g = 1; p.gc = rgb; }
    }
  }
  function start() {
    size();
    addEventListener("resize", size);
    addEventListener("pointermove", (e) => { mouse.vx = e.clientX - mouse.x; mouse.vy = e.clientY - mouse.y; if (Math.abs(mouse.vx) > 300) mouse.vx = mouse.vy = 0; mouse.x = e.clientX; mouse.y = e.clientY; }, { passive: true });
    addEventListener("pointerleave", () => { mouse.x = mouse.y = -1e4; });
    addEventListener("scroll", () => { scrollV = clamp((scrollY - lastY) * 0.05, -3, 3) * -1; lastY = scrollY; }, { passive: true });
    document.addEventListener("visibilitychange", () => { visible = !document.hidden; });
    if (RM.matches) { for (let i = 0; i < 60; i++) draw(0.016); return; }       // one static, still frame
    run((dt) => { if (visible) draw(dt); return true; });
  }
  return { start, burst };
})();

/* ───────────── small UI utilities ───────────── */
function toast(msg) {
  const el = $("#toast"); el.textContent = msg;
  const y = new Spring(24, { damping: 0.8, response: 0.4 }), o = new Spring(0, { damping: 1, response: 0.25 });
  y.t = 0; o.t = 1;
  const write = () => { el.style.transform = `translateY(${y.x}px)`; el.style.opacity = clamp(o.x, 0, 1); };
  springTo([y, o], write);
  clearTimeout(toast.id);
  toast.id = setTimeout(() => { y.t = 16; o.t = 0; springTo([y, o], write); }, 1900);
}
function haptic(ms = 10) { try { navigator.vibrate && navigator.vibrate(ms); } catch (_) { /* unsupported */ } }

const GLYPHS = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";
// Characters resolve left to right, like a decoder locking on.
function decode(el, text, dur = 760) {
  const id = (el._dec = (el._dec || 0) + 1);
  if (RM.matches) { el.textContent = text; return; }
  const t0 = performance.now();
  run(() => {
    if (el._dec !== id) return false;
    const k = clamp((performance.now() - t0) / dur, 0, 1), lock = Math.floor(k * text.length * 1.15);
    el.textContent = [...text].map((ch, i) => (i < lock || ch === "." ? ch : GLYPHS[(Math.random() * GLYPHS.length) | 0])).join("");
    if (k >= 1) { el.textContent = text; return false; }
    return true;
  });
}

/* reveal on scroll */
function initReveal() {
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  $$(".reveal").forEach((el) => io.observe(el));
}

/* pointer-tracked specular light on glass */
function initSpot() {
  addEventListener("pointermove", (e) => {
    for (const g of $$(".glass, .code-card")) {
      const r = g.getBoundingClientRect();
      if (e.clientX < r.left - 40 || e.clientX > r.right + 40 || e.clientY < r.top - 40 || e.clientY > r.bottom + 40) continue;
      g.style.setProperty("--mx", e.clientX - r.left + "px"); g.style.setProperty("--my", e.clientY - r.top + "px");
    }
  }, { passive: true });
}

/* magnetic buttons: they lean toward the pointer, and spring home */
function initMagnet() {
  $$("[data-magnet]").forEach((el) => {
    const sx = new Spring(0, { response: 0.35 }), sy = new Spring(0, { response: 0.35 });
    const write = () => { el.style.transform = `translate(${sx.x}px,${sy.x}px)`; };
    el.addEventListener("pointermove", (e) => {
      if (e.pointerType === "touch") return;
      const r = el.getBoundingClientRect();
      sx.t = (e.clientX - (r.left + r.width / 2)) * 0.22; sy.t = (e.clientY - (r.top + r.height / 2)) * 0.3; springTo([sx, sy], write);
    });
    el.addEventListener("pointerleave", () => { sx.t = sy.t = 0; springTo([sx, sy], write); });
  });
}

/* hero headline: words rise out of their lines, staggered, critically damped */
function initHero() {
  const words = $$(".display .word"); if (!words.length) return;
  const ys = words.map(() => new Spring(118, { damping: 1, response: 0.9 }));
  words.forEach((w, i) => { w.style.transform = "translateY(118%)"; setTimeout(() => { ys[i].t = 0; springTo([ys[i]], () => { w.style.transform = `translateY(${ys[i].x}%)`; }); }, 140 + i * 130); });
  const kids = $$(".eyebrow, .lede, .hero-actions");
  kids.forEach((k, i) => { k.style.opacity = 0; k.style.transform = "translateY(18px)"; setTimeout(() => { k.style.transition = "opacity 800ms ease-out, transform 900ms cubic-bezier(.2,.9,.2,1)"; k.style.opacity = 1; k.style.transform = ""; }, 420 + i * 120); });
}

/* ───────────── lab ───────────── */
const STAGES = [
  ["Guard + normalize", "entity + chunk only · subconcept never read"],
  ["Retrieve", "BM25 · char TF-IDF · SapBERT → RRF · memory"],
  ["Score", "cross-encoders ± context · 43 features"],
  ["Stack", "LightGBM lambdarank · 20 seeds · per system"],
  ["LLM gate", "Qwen2.5-1.5B re-checks SNOMED picks p < 0.6"],
  ["Guard + output", "legal code · calibrated confidence"],
];
const GATE = 4;                                                  // stage index of the LLM gate
// Illustrative answers for the built-in hard cases, written by hand from public code sets.
// They are NOT engine output; the console says so wherever they appear and never shows a confidence for them.
const SAMPLES = [
  { label: "HTN", term: "HTN", ctx: "PMH: HTN, HLD. Continues lisinopril 10 mg daily.", icd: ["I10", "Essential (primary) hypertension"], sn: ["59621000", "Essential hypertension"] },
  { label: "Near-neighbour", term: "pulmonary hypertension", ctx: "Echo: RVSP 55 mmHg consistent with pulmonary hypertension.", icd: ["I27.20", "Pulmonary hypertension, unspecified"], sn: ["70995007", "Pulmonary hypertension"] },
  { label: "MS · neurology", term: "MS", ctx: "Neurology f/u for MS, on ocrelizumab, new T2 lesions on MRI.", icd: ["G35", "Multiple sclerosis"], sn: ["24700007", "Multiple sclerosis"] },
  { label: "OCR noise", term: "chronickidney disease stag 3", ctx: "Assessment: chronickidney disease stag 3, Cr 1.6", icd: ["N18.30", "Chronic kidney disease, stage 3 unspecified"], sn: ["433144002", "Chronic kidney disease stage 3"] },
  { label: "Unspecified is right", term: "diabetes", ctx: "Diabetes - continue metformin.", icd: ["E11.9", "Type 2 diabetes mellitus without complications"], sn: ["44054006", "Diabetes mellitus type 2"] },
  { label: "Specific complication", term: "DM2 with diabetic neuropathy", ctx: "Feet: decreased monofilament sensation, DM2 with diabetic neuropathy.", icd: ["E11.40", "Type 2 diabetes mellitus with diabetic neuropathy, unspecified"], sn: ["368581000119106", "Neuropathy due to type 2 diabetes mellitus"] },
];
const norm = (s) => s.trim().toLowerCase().replace(/\s+/g, " ");

const Lab = window.Lab_ = (() => {
  if (!$("#term")) return {};                                   // this page has no lab
  const state = { mode: "sample", busy: false, llm: false, name: "TermMap-X+" };
  const el = { term: $("#term"), ctx: $("#ctx"), run: $("#run"), note: $("#runNote"), rail: $("#rail"), chips: $("#chips"), flag: $("#srcFlag"), badge: $("#modeBadge"), badgeText: $("#modeText") };
  const cards = { icd: $("#cardIcd"), sn: $("#cardSn") };

  function setMode(m, why = "") {
    state.mode = m;
    if (el.badge) { el.badge.dataset.mode = m === "live" ? "live" : "sample"; el.badgeText.textContent = m === "live" ? `Live engine · ${state.name} · running on this machine` : (why || "Illustrative samples · engine not connected"); }
    el.flag.hidden = m === "live"; el.flag.textContent = "ILLUSTRATIVE SAMPLE";
    el.note.textContent = m === "live" ? (state.llm ? "TermMap-X+ · stacker + Qwen2.5 SNOMED gate." : "TermMap-X+ · LLM gate off (about 2 pt lower SNOMED).") : "Answers shown are hand-written samples. Run python web/server.py for the live engine.";
  }
  async function pollStatus() {
    try {
      const s = await (await fetch("/api/status", { cache: "no-store" })).json();
      if (s.engine === "ready") { state.llm = !!s.llm; state.name = s.name || state.name; return setMode("live"); }
      if (s.engine === "loading") { setMode("sample", "Engine loading · models warming up…"); return setTimeout(pollStatus, 2500); }
      setMode("sample");
    } catch (_) { setMode("sample"); }
  }

  SAMPLES.forEach((s, i) => {
    const b = document.createElement("button"); b.className = "chip"; b.textContent = s.label; b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => { $$(".chip", el.chips).forEach((c) => c.setAttribute("aria-pressed", String(c === b))); el.term.value = s.term; el.ctx.value = s.ctx; go(); });
    el.chips.append(b);
  });
  el.rail.innerHTML = STAGES.map((s, i) => `<li data-i="${i}"><span class="n">${i}</span><span><div class="t">${s[0]}</div><div class="d">${s[1]}</div></span><span></span></li>`).join("");
  const lis = $$("li", el.rail);

  const shake = () => {
    const s = new Spring(0, { damping: 0.35, response: 0.25 }); s.x = 14;
    springTo([s], () => { el.term.style.transform = `translateX(${s.x}px)`; if (s.settled) el.term.style.transform = ""; });
  };
  function setRing(card, p) {
    const ring = $(".ring", card), b = $("b", ring);
    if (p == null) { ring.classList.add("na"); b.textContent = "—"; ring.style.setProperty("--p", 0); return; }
    ring.classList.remove("na"); const s = new Spring(0, { response: 0.9 }); s.t = p;
    springTo([s], () => { ring.style.setProperty("--p", s.x); b.textContent = Math.round(clamp(s.x, 0, 1) * 100); });
  }
  function fill(card, sysData, sample) {
    card.classList.remove("ghosted");
    card.dataset.code = sysData.code;
    decode($(".code", card), sysData.code);
    $(".name", card).textContent = sysData.name || "";
    setRing(card, sample ? null : sysData.confidence);
    $(".alts", card).innerHTML = sample ? `<div><span>top-3</span><span>live engine only</span></div>`
      : (sysData.top3 || []).slice(1, 3).map((a) => `<div><span>${a.code}</span><span>${a.name || ""}</span></div>`).join("");
    const r = card.getBoundingClientRect();
    Field.burst(r.left + r.width / 2, r.top + r.height / 2, card.dataset.sys === "icd" ? [10, 132, 255] : [191, 90, 242]);
  }

  async function live(term, ctx) {
    const r = await fetch("/api/map", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ term, context: ctx }) });
    if (!r.ok) throw new Error("engine " + r.status);
    return r.json();
  }
  // A sample answers only the exact case it was written for: "MS" in an echo note is a different answer.
  const sampleFor = (term, ctx = "") => SAMPLES.find((s) => norm(s.term) === norm(term) && (!ctx || norm(ctx) === norm(s.ctx)));

  async function go() {
    const term = el.term.value.trim(), ctx = el.ctx.value.trim();
    if (state.busy) return;
    if (!term) { shake(); el.term.focus(); return; }
    if (state.mode === "sample" && !sampleFor(term, ctx)) {
      shake(); el.note.textContent = "That term isn’t one of the illustrative samples. Start the engine (python web/server.py) to map anything.";
      return;
    }
    state.busy = true; el.run.disabled = true; haptic(6);
    lis.forEach((l) => l.classList.remove("on", "done", "skip"));
    Object.values(cards).forEach((c) => c.classList.add("ghosted"));
    const t0 = performance.now();
    let req = state.mode === "live" ? live(term, ctx) : Promise.resolve(sampleFor(term, ctx));
    req = req.then((r) => ({ r, sample: state.mode !== "live" })).catch(() => {
      setMode("sample", "Engine not reachable · showing illustrative samples");
      const s = sampleFor(term, ctx); return s ? { r: s, sample: true } : null;
    });
    const last = STAGES.length - 1, gateOff = state.mode === "live" && !state.llm;
    for (let i = 0; i < last; i++) {
      lis[i].classList.add("on"); if (gateOff && i === GATE) lis[i].classList.add("skip");
      await sleep(gateOff && i === GATE ? 140 : 330);
      lis[i].classList.remove("on"); lis[i].classList.add("done");
    }
    lis[last].classList.add("on");
    const out = await req;
    lis[last].classList.remove("on"); lis[last].classList.add("done");
    if (out) {
      const { r, sample } = out;
      const I = sample ? { code: r.icd[0], name: r.icd[1] } : r.systems.ICD10CM;
      const S = sample ? { code: r.sn[0], name: r.sn[1] } : r.systems.SNOMEDCT_US;
      fill(cards.icd, I, sample); setTimeout(() => fill(cards.sn, S, sample), 160); haptic(14);
      if (!sample) el.note.textContent = `Mapped in ${((performance.now() - t0) / 1000).toFixed(1)} s${state.llm ? "" : " · LLM gate off"}`;
    } else el.note.textContent = "That term isn’t one of the illustrative samples. Start the engine (python web/server.py).";
    state.busy = false; el.run.disabled = false;
  }
  el.run.addEventListener("click", go);
  addEventListener("keydown", (e) => {
    if ($("#paneSingle").hidden) return;
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); go(); }
    if (e.key === "/" && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); el.term.focus({ preventScroll: false }); }
  });
  setMode("sample"); pollStatus();
  return { state, sampleFor, setMode, pollStatus };
})();

/* ───────────── lab: mode switch ───────────── */
const Tabs = (() => {
  const tabs = $("#tabSingle") && { single: $("#tabSingle"), batch: $("#tabBatch") };
  if (!tabs) return {};
  const thumb = $(".seg-thumb"), panes = { single: $("#paneSingle"), batch: $("#paneBatch") };
  const tx = new Spring(4, { damping: 0.85, response: 0.4 }), tw = new Spring(0, { damping: 0.85, response: 0.4 });
  const write = () => { thumb.style.width = tw.x + "px"; thumb.style.transform = `translateX(${tx.x - 4}px)`; };
  let cur = "single", init = false;
  function set(name, instant) {
    cur = name;
    for (const k in tabs) { tabs[k].setAttribute("aria-selected", String(k === name)); panes[k].hidden = k !== name; }
    const b = tabs[name]; tx.t = b.offsetLeft; tw.t = b.offsetWidth;
    if (instant || !init) { tx.x = tx.t; tw.x = tw.t; write(); init = true; } else springTo([tx, tw], write);
    if (name === "batch") $$(".reveal", panes.batch).forEach((e) => e.classList.add("in"));
    try { history.replaceState(null, "", name === "batch" ? "#upload" : location.pathname); } catch (_) { /* file:// */ }
  }
  tabs.single.addEventListener("click", () => set("single")); tabs.batch.addEventListener("click", () => set("batch"));
  addEventListener("resize", () => set(cur, true));
  document.fonts && document.fonts.ready.then(() => set(cur, true));
  set(location.hash === "#upload" ? "batch" : "single", true);
  return { set };
})();

/* ───────────── lab: CSV in, CSV out ─────────────
   Parsing and export happen in the browser; only rows that are being mapped go to the local engine, 20 at a time. */
const Batch = (() => {
  const drop = $("#drop");
  if (!drop || !window.Lab_) return {};
  const Lab = window.Lab_, MAX_ROWS = 5000, MAX_BYTES = 5 * 1024 * 1024, CHUNK = 20, LOW = 0.5;   // LOW: same abstain cut as X+ (.env.example)
  const el = { file: $("#file"), panel: $("#batchPanel"), name: $("#fileName"), meta: $("#fileMeta"), warns: $("#warns"), run: $("#runBatch"), stop: $("#stopBatch"),
    note: $("#batchNote"), prog: $("#progress"), bar: $("#progress i"), kpis: $("#kpis"), tbody: $("#tbody"), filter: $("#filter"), low: $("#lowOnly"),
    dlS: $("#dlSubmit"), dlD: $("#dlDetail"), flag: $("#batchFlag") };
  let rows = [], fileName = "predictions", stopFlag = false, busy = false, trs = [];

  /* RFC-4180 style parser: quotes, escaped quotes, CRLF, embedded newlines; delimiter sniffed from the header line */
  function parseCSV(text) {
    text = text.replace(/^﻿/, "");
    const head = text.split(/\r?\n/, 1)[0], count = (c) => head.split(c).length - 1;
    const delim = [",", ";", "\t"].sort((a, b) => count(b) - count(a))[0];
    const out = []; let row = [], f = "", q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
      else if (c === '"' && f === "") q = true;
      else if (c === delim) { row.push(f); f = ""; }
      else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(f); f = ""; if (row.some((x) => x !== "")) out.push(row); row = []; }
      else f += c;
    }
    row.push(f); if (row.some((x) => x !== "")) out.push(row);
    return out;
  }
  const csvCell = (v) => { v = v == null ? "" : String(v); return /[",\r\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  const toCSV = (header, data) => [header.join(","), ...data.map((r) => header.map((h) => csvCell(r[h])).join(","))].join("\n") + "\n";
  function save(name, text) {
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" })); a.download = name;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function warn(msg, err) { const d = document.createElement("div"); if (err) d.className = "err"; d.innerHTML = msg; el.warns.append(d); }

  function load(text, name) {
    el.warns.innerHTML = ""; rows = []; trs = []; el.tbody.innerHTML = ""; el.kpis.hidden = el.prog.hidden = true; el.bar.style.width = "0";
    el.dlS.disabled = el.dlD.disabled = true; el.panel.hidden = false; el.name.textContent = name; fileName = name.replace(/\.[^.]+$/, "");
    const t = parseCSV(text);
    if (t.length < 2) { warn("That file has no data rows.", true); el.run.disabled = true; el.meta.textContent = ""; return; }
    const hdr = t[0].map((h) => h.trim().toLowerCase());
    const col = (names) => hdr.findIndex((h) => names.includes(h));
    const ie = col(["clinical_entity", "entity"]), ic = col(["chunk_text", "chunk", "context"]), ii = col(["chunk_id"]);
    if (ie < 0) { warn(`No <code>clinical_entity</code> (or <code>entity</code>) column. Found: ${hdr.map((h) => `<code>${esc(h || "(blank)")}</code>`).join(" ")}`, true); el.run.disabled = true; el.meta.textContent = `${t.length - 1} rows`; return; }
    let skipped = 0;
    for (const r of t.slice(1)) {
      const term = (r[ie] || "").trim(); if (!term) { skipped++; continue; }
      if (rows.length >= MAX_ROWS) { skipped++; continue; }
      rows.push({ chunk_id: ii >= 0 ? (r[ii] || "").trim() : "", clinical_entity: term, chunk_text: ic >= 0 ? (r[ic] || "") : "", res: null });
    }
    if (ic < 0) warn("No <code>chunk_text</code> column. The engine reads the chart chunk to disambiguate abbreviations like MS or RA — mapping will work, but weaker.");
    if (skipped) warn(`${skipped} row${skipped > 1 ? "s" : ""} skipped (empty term${t.length - 1 > MAX_ROWS ? ` or over the ${MAX_ROWS}-row limit` : ""}).`);
    el.meta.textContent = `${rows.length} rows · columns: ${[ii >= 0 && "chunk_id", "clinical_entity", ic >= 0 && "chunk_text"].filter(Boolean).join(", ")}`;
    el.flag.hidden = Lab.state.mode === "live"; el.flag.textContent = "ILLUSTRATIVE SAMPLES ONLY";
    el.note.textContent = Lab.state.mode === "live" ? `${(rows.length * 5 / 60).toFixed(1)}–${(rows.length * 7 / 60).toFixed(1)} min expected at ~5–7 s/term (GPU, LLM gate on)` : "Engine not connected: only the built-in hard cases can be mapped.";
    el.run.disabled = rows.length === 0;
    el.tbody.innerHTML = rows.map((r, i) => `<tr data-i="${i}"><td class="n">${i + 1}</td><td class="cid">${esc(r.chunk_id)}</td><td class="term">${esc(r.clinical_entity)}</td><td class="cd icd pending">—</td><td class="cd sn pending">—</td><td><span class="pill na">pending</span></td></tr>`).join("");
    trs = [...el.tbody.children]; applyFilter();
    el.panel.scrollIntoView({ behavior: RM.matches ? "auto" : "smooth", block: "start" });
  }
  function readFile(f) {
    if (!f) return;
    if (f.size > MAX_BYTES) { el.panel.hidden = false; el.warns.innerHTML = ""; warn(`That file is ${(f.size / 1048576).toFixed(1)} MB; the limit is 5 MB.`, true); return; }
    const r = new FileReader(); r.onload = () => load(String(r.result), f.name); r.readAsText(f);
  }

  /* drop zone */
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { readFile(e.dataTransfer.files[0]); haptic(8); });
  drop.addEventListener("click", () => el.file.click());
  drop.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target === drop) { e.preventDefault(); el.file.click(); } });
  el.file.addEventListener("change", () => { readFile(el.file.files[0]); el.file.value = ""; });
  $$(".drop-actions .btn").forEach((b) => b.addEventListener("click", (e) => e.stopPropagation()));
  $("#pick").addEventListener("click", () => el.file.click());
  $("#example").addEventListener("click", () => {
    const ex = [...SAMPLES.map((s, i) => ({ chunk_id: "hard" + String(i + 1).padStart(2, "0"), term: s.term, ctx: s.ctx })),
      { chunk_id: "hard08", term: "MS", ctx: "Echo shows thickened mitral leaflets; MS with mean gradient 8 mmHg." }];
    load(toCSV(["chunk_id", "clinical_entity", "chunk_text"], ex.map((r) => ({ chunk_id: r.chunk_id, clinical_entity: r.term, chunk_text: r.ctx }))), "hard-cases.csv");
  });
  $("#template").addEventListener("click", () => save("termmap_template.csv", toCSV(["chunk_id", "clinical_entity", "chunk_text"],
    [{ chunk_id: "c1", clinical_entity: "HTN", chunk_text: "PMH: HTN, HLD. Continues lisinopril 10 mg daily." }, { chunk_id: "c2", clinical_entity: "chronickidney disease stag 3", chunk_text: "Assessment: chronickidney disease stag 3, Cr 1.6" }])));

  /* table */
  function cell(td, side) { td.classList.remove("pending"); td.innerHTML = `<b>${esc(side.code)}</b>${side.name ? `<small>${esc(side.name)}</small>` : ""}`; }
  function show(i, res, sample) {
    rows[i].res = res; const tr = trs[i], tds = tr.children;
    cell(tds[3], res.icd); cell(tds[4], res.snomed);
    const c = sample ? null : Math.min(res.icd.confidence, res.snomed.confidence);
    tds[5].innerHTML = c == null ? '<span class="pill na">sample</span>' : `<span class="pill${c < LOW ? " low" : ""}">${Math.round(c * 100)}%</span>`;
    tr.dataset.low = c != null && c < LOW ? "1" : ""; tr.classList.add("new"); setTimeout(() => tr.classList.remove("new"), 800);
  }
  function applyFilter() {
    const q = el.filter.value.trim().toLowerCase(), lowOnly = el.low.checked;
    rows.forEach((r, i) => {
      const hay = (r.clinical_entity + " " + r.chunk_id + " " + (r.res ? r.res.icd.code + " " + r.res.icd.name + " " + r.res.snomed.code + " " + r.res.snomed.name : "")).toLowerCase();
      trs[i].style.display = (!q || hay.includes(q)) && (!lowOnly || trs[i].dataset.low === "1") ? "" : "none";
    });
  }
  el.filter.addEventListener("input", applyFilter); el.low.addEventListener("change", applyFilter);

  /* run */
  const kp = { mapped: $("#kMapped"), low: $("#kLow"), rate: $("#kRate"), el: $("#kElapsed") };
  async function runAll() {
    if (busy || !rows.length) return;
    busy = true; stopFlag = false; el.run.disabled = true; el.stop.hidden = false; el.kpis.hidden = el.prog.hidden = false; el.dlS.disabled = el.dlD.disabled = true;
    const live = Lab.state.mode === "live", t0 = performance.now(); let done = 0, low = 0; const prog = new Spring(0, { response: 0.5 });
    const upd = () => {
      prog.t = done / rows.length * 100; springTo([prog], () => { el.bar.style.width = prog.x + "%"; });
      kp.mapped.textContent = done + " / " + rows.length; kp.low.textContent = live ? low : "—";
      const s = (performance.now() - t0) / 1000; kp.el.textContent = s < 90 ? s.toFixed(0) + "s" : Math.floor(s / 60) + "m " + String(Math.round(s % 60)).padStart(2, "0") + "s";
      kp.rate.textContent = done && live ? (s / done).toFixed(1) + "s" : "—";
    };
    const tick = setInterval(upd, 500); rows.forEach((r) => { r.res = null; }); upd();
    try {
      if (!live) {
        for (let i = 0; i < rows.length; i++) {
          const s = Lab.sampleFor(rows[i].clinical_entity, rows[i].chunk_text);
          if (s) show(i, { icd: { code: s.icd[0], name: s.icd[1] }, snomed: { code: s.sn[0], name: s.sn[1] } }, true);
          else { trs[i].children[5].innerHTML = '<span class="pill na">needs engine</span>'; }
          done++; if (i % 4 === 3) await sleep(40);
        }
        el.note.textContent = "Engine not connected: only the built-in hard cases were mapped, with illustrative answers. Downloads are disabled.";
      } else {
        for (let i = 0; i < rows.length && !stopFlag; i += CHUNK) {
          const part = rows.slice(i, i + CHUNK);
          const r = await fetch("/api/batch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rows: part.map(({ chunk_id, clinical_entity, chunk_text }) => ({ chunk_id, clinical_entity, chunk_text })) }) });
          if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.error || "engine " + r.status); }
          const { results } = await r.json();
          results.forEach((res, k) => { show(i + k, res, false); if (Math.min(res.icd.confidence, res.snomed.confidence) < LOW) low++; });
          done += part.length; upd(); applyFilter();
        }
        const all = done === rows.length;
        el.note.textContent = all ? `Done: ${rows.length} rows in ${kp.el.textContent}. ${low} flagged low confidence for review.` : `Stopped after ${done} of ${rows.length} rows.`;
        el.dlS.disabled = !all; el.dlD.disabled = done === 0; haptic(16);
        if (all) toast(`Mapped ${rows.length} rows`);
      }
    } catch (e) { el.note.textContent = "Stopped: " + e.message; warn("Mapping stopped: " + esc(e.message), true); if (done) el.dlD.disabled = false; }
    clearInterval(tick); upd(); busy = false; el.run.disabled = false; el.stop.hidden = true;
  }
  el.run.addEventListener("click", runAll);
  el.stop.addEventListener("click", () => { stopFlag = true; el.note.textContent = "Stopping after this batch…"; });

  /* downloads: predictions.csv is exactly the schema the task specifies */
  el.dlS.addEventListener("click", () => save("predictions.csv", toCSV(["chunk_id", "clinical_entity", "icd10cm_code", "snomed_code"],
    rows.map((r) => ({ chunk_id: r.chunk_id, clinical_entity: r.clinical_entity, icd10cm_code: r.res.icd.code, snomed_code: r.res.snomed.code })))));
  el.dlD.addEventListener("click", () => save("predictions_detailed.csv", toCSV(["chunk_id", "clinical_entity", "icd10cm_code", "icd10cm_name", "snomed_code", "snomed_name", "confidence_icd", "confidence_snomed", "confidence", "low_confidence"],
    rows.map((r) => r.res ? { chunk_id: r.chunk_id, clinical_entity: r.clinical_entity, icd10cm_code: r.res.icd.code, icd10cm_name: r.res.icd.name, snomed_code: r.res.snomed.code, snomed_name: r.res.snomed.name,
      confidence_icd: r.res.icd.confidence.toFixed(4), confidence_snomed: r.res.snomed.confidence.toFixed(4), confidence: Math.min(r.res.icd.confidence, r.res.snomed.confidence).toFixed(4),
      low_confidence: Math.min(r.res.icd.confidence, r.res.snomed.confidence) < LOW ? 1 : 0 } : { chunk_id: r.chunk_id, clinical_entity: r.clinical_entity }))));
  return { parseCSV, load };
})();

/* ───────────── physical result cards ─────────────
   1:1 tracking that respects the grab offset, rubber-band past the soft edge, release velocity handed to the spring,
   and momentum projection decides the outcome: fling far enough up and the code is copied. */
function physical(card) {
  const sx = new Spring(0, { response: 0.4 }), sy = new Spring(0, { response: 0.4 });
  const rz = new Spring(0, { response: 0.3 }), rx = new Spring(0, { response: 0.35 }), ry = new Spring(0, { response: 0.35 });
  const all = [sx, sy, rz, rx, ry];
  let drag = null, hist = [], moved = 0;
  const write = () => { card.style.transform = `translate3d(${sx.x}px,${sy.x}px,0) rotateX(${rx.x}deg) rotateY(${ry.x}deg) rotateZ(${rz.x}deg)`; };
  const wake = () => springTo(all, write);
  const SOFT = 90;
  const band = (v) => { const a = Math.abs(v); return a <= SOFT ? v : Math.sign(v) * (SOFT + rubber(a - SOFT, 260)); };
  const copy = () => {
    const code = card.dataset.code; if (!code || card.classList.contains("ghosted")) return;
    (navigator.clipboard ? navigator.clipboard.writeText(code) : Promise.reject()).then(() => toast(`Copied ${code}`), () => toast(code));
    haptic(16);
  };

  card.addEventListener("pointerdown", (e) => {
    card.setPointerCapture(e.pointerId);
    all.forEach((s) => { s.t = s.x; s.v = 0; });                                  // grab the live value, never the target
    drag = { ox: e.clientX - sx.x, oy: e.clientY - sy.x }; hist = [{ t: e.timeStamp, x: sx.x, y: sy.x }]; moved = 0;
    card.classList.add("drag");
  });
  card.addEventListener("pointermove", (e) => {
    if (!drag) {                                                                    // hover tilt toward the pointer
      if (e.pointerType === "touch") return;
      const r = card.getBoundingClientRect();
      ry.t = ((e.clientX - r.left) / r.width - 0.5) * 9; rx.t = -((e.clientY - r.top) / r.height - 0.5) * 9; wake(); return;
    }
    const rx_ = e.clientX - drag.ox, ry_ = e.clientY - drag.oy;
    moved = Math.max(moved, Math.hypot(rx_, ry_));
    hist.push({ t: e.timeStamp, x: rx_, y: ry_ }); if (hist.length > 8) hist.shift();
    sx.x = sx.t = band(rx_); sy.x = sy.t = band(ry_); sx.v = sy.v = 0;
    const v = velocity(); rz.t = clamp(v.x / 140, -10, 10); ry.t = clamp(v.x / 160, -12, 12); rx.t = clamp(-v.y / 160, -12, 12);
    wake();
  });
  const velocity = () => {
    if (hist.length < 2) return { x: 0, y: 0 };
    const b = hist[hist.length - 1]; let a = hist[0];
    for (const h of hist) if (b.t - h.t <= 90) { a = h; break; }
    const dt = Math.max(1, b.t - a.t) / 1000; return { x: (b.x - a.x) / dt, y: (b.y - a.y) / dt };
  };
  const end = () => {
    if (!drag) return; const v = velocity(); drag = null; card.classList.remove("drag");
    const projY = sy.x + project(v.y, 0.99);
    if (moved < 6) copy();                                                         // a tap commits
    else if (projY < -170 && v.y < -250) { copy(); }                                // a flick up commits
    // hand the finger's velocity to the spring; bounce only because a gesture carried momentum
    sx.tune(0.78, 0.45); sy.tune(0.78, 0.45); sx.v = v.x; sy.v = v.y; sx.t = sy.t = 0;
    rz.t = rx.t = ry.t = 0; wake();
    setTimeout(() => { sx.tune(1, 0.4); sy.tune(1, 0.4); }, 900);
  };
  card.addEventListener("pointerup", end); card.addEventListener("pointercancel", end);
  card.addEventListener("pointerleave", () => { if (!drag) { rx.t = ry.t = 0; wake(); } });
  card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); copy(); } });
}
$$(".code-card").forEach(physical);

/* ───────────── pipeline: sticky, scroll-driven, particle morph ───────────── */
const PIPE = [
  ["Guard", "Read only what’s allowed.", "The input guard keeps just <code>entity</code>, <code>chunk_text</code> and <code>chunk_id</code>; every other column is dropped unread. The term is normalised — shorthand, OCR noise, abbreviations — before anything is retrieved.", "3 columns kept"],
  ["Retrieve", "Cast a wide net.", "BM25, character TF-IDF and a fine-tuned SapBERT, fused by reciprocal rank, plus a memory of training terms and hierarchy expansion. For ICD, tree siblings (unspecified, laterality, 7th character) and convention-kNN codes join the pool. The right code is in the pool for <b>94.3%</b> of ICD rows and <b>89.6%</b> of SNOMED rows.", "≤ 64 per system"],
  ["Score", "Read it in context.", "Cross-encoders score every candidate with and without the chart chunk, next to 43 features, lexical-trap flags and kNN votes. For ICD, a zero-shot Qwen3-Reranker-0.6B re-reads the top 10. No single scorer passes 58% top-1.", "term + chunk × candidates"],
  ["Stack", "Let the stacker decide.", "A LightGBM lambdarank stacker per system, averaged over 20 seeds — an ICD-only model and a shared one for SNOMED. It lifts the best single scorer to <b>74.9%</b> ICD and <b>67.5%</b> SNOMED out-of-fold.", "20 seeds averaged"],
  ["LLM gate", "Ask only when unsure.", "A local Qwen2.5-1.5B re-checks only the SNOMED picks with probability under 0.6, choosing among the top-5 letters <code>A–E</code>, blended as <code>log p + 0.25 · LLM</code>. ICD has no gate.", "p < 0.6 → top-5"],
  ["Output", "Guarantee the answer.", "Every code must exist in the supplied dictionary, or the row falls back to the most common training code at confidence 0. Each row ships a calibrated confidence per system, and the lower of the two.", "2 legal codes"],
];
const ORDER = [0, 1, 3, 2, 5, 6];                                  // which particle layout draws each stage
const Pipe = (() => {
  const track = $("#pipeTrack");
  if (!track) return { start() {} };                              // this page has no walkthrough
  const cv = $("#viz"), ctx = cv.getContext("2d");
  let w = 0, h = 0, dpr = 1, stage = -1, onScreen = false, time = 0;
  const N = 150, P = [];
  const seed = (() => { let a = 7; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();
  const rnd = Array.from({ length: N }, () => [seed(), seed(), seed()]);
  const BLUE = [10, 132, 255], CYAN = [100, 210, 255], VIO = [191, 90, 242], PINK = [255, 122, 217], WHITE = [235, 235, 245];
  // each stage: i → {x,y (0..1), r (px), a, c}
  const L = [
    (i) => { const row = Math.floor(i / 15), col = i % 15, len = 6 + ((row * 7) % 9); const on = col < len; return { x: 0.12 + 0.76 * (col / 14), y: 0.14 + 0.72 * (row / 9), r: on ? 3.2 : 1.6, a: on ? 0.95 : 0.14, c: WHITE }; },
    (i) => { const lane = i % 6, k = Math.floor(i / 6); return { x: 0.12 + 0.76 * (lane / 5), y: 0.12 + 0.76 * (k / 24), r: 2.2 + (1 - k / 24) * 2.2, a: 0.35 + 0.65 * (1 - k / 24), c: lane % 3 === 0 ? CYAN : lane % 3 === 1 ? BLUE : VIO }; },
    (i) => i < 30 ? { x: 0.1 + 0.8 * (i / 29), y: 0.78 - 0.62 * (1 - Math.exp(-(29 - i) / 9)) * (1 - i / 60), r: 6.5 - i * 0.12, a: 1, c: lerpC(CYAN, VIO, i / 29) } : { x: 0.06 + 0.88 * rnd[i][0], y: 0.08 + 0.84 * rnd[i][1], r: 1.5, a: 0.1, c: WHITE },
    (i) => { if (i === 0) return { x: 0.18, y: 0.5, r: 13, a: 1, c: WHITE }; if (i <= 30) { const a = ((i - 1) / 29 - 0.5) * 2.2; return { x: 0.82 - 0.1 * Math.cos(a), y: 0.5 + 0.4 * Math.sin(a * 0.9) * 1.05, r: 4.5, a: 1, c: lerpC(CYAN, VIO, (i - 1) / 29) }; } return { x: 0.45 + 0.1 * rnd[i][0], y: 0.5 + 0.1 * (rnd[i][1] - 0.5), r: 1.2, a: 0.06, c: WHITE }; },
    (i) => { if (i < 25) { const a = (i / 25) * Math.PI * 2 - Math.PI / 2; return { x: 0.5 + 0.3 * Math.cos(a), y: 0.5 + 0.34 * Math.sin(a), r: 8 - i * 0.18, a: 1, c: lerpC(CYAN, PINK, i / 25) }; } return { x: 0.5 + 0.04 * (rnd[i][0] - 0.5), y: 0.5 + 0.04 * (rnd[i][1] - 0.5), r: 1.2, a: 0.04, c: WHITE }; },
    (i) => i < 5 ? { x: 0.18 + 0.16 * i, y: 0.5, r: i === 1 ? 22 : 13, a: i === 1 ? 1 : 0.6, c: i === 1 ? PINK : VIO } : { x: 0.5, y: 0.5, r: 1, a: 0, c: WHITE },
    (i) => i === 0 ? { x: 0.3, y: 0.5, r: 26, a: 1, c: BLUE } : i === 1 ? { x: 0.7, y: 0.5, r: 26, a: 1, c: VIO } : { x: i % 2 ? 0.7 + 0.12 * (rnd[i][0] - 0.5) : 0.3 + 0.12 * (rnd[i][0] - 0.5), y: 0.5 + 0.3 * (rnd[i][1] - 0.5), r: 1.8, a: 0.55, c: i % 2 ? PINK : CYAN },
  ];
  function lerpC(a, b, k) { return [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)]; }
  for (let i = 0; i < N; i++) {
    const o = { x: new Spring(rnd[i][0], { response: 0.7 }), y: new Spring(rnd[i][1], { response: 0.7 }), r: new Spring(2, { response: 0.5 }), a: new Spring(0.5, { response: 0.5 }), c: [...WHITE] };
    o.x.v = o.y.v = 0; P.push(o);
  }
  let lineA = 0, lineT = 0, letterA = 0, letterT = 0, colorT = [];
  function setStage(s) {
    if (s === stage) return; stage = s; const f = L[ORDER[s]];
    for (let i = 0; i < N; i++) {                                    // per-particle response jitter: a swarm, not a slideshow
      const t = f(i), p = P[i], j = 0.55 + rnd[i][2] * 0.5;
      p.x.tune(0.86, j).t = t.x; p.y.tune(0.86, j).t = t.y; p.r.t = t.r; p.a.t = t.a; colorT[i] = t.c;
    }
    lineT = ORDER[s] === 3 ? 1 : 0; letterT = ORDER[s] === 5 ? 1 : 0;
    $("#stageKicker").textContent = `Stage ${s} · ${PIPE[s][0]}`; $("#stageTitle").textContent = PIPE[s][1]; $("#stageBody").innerHTML = PIPE[s][2];
    $("#vizCount").textContent = PIPE[s][3]; $("#vizLabel").textContent = `${String(s + 1).padStart(2, "0")} / ${String(PIPE.length).padStart(2, "0")}`;
    $$("#pips i").forEach((p, i) => { p.className = i === s ? "on" : i < s ? "past" : ""; });
  }
  const size = () => { const r = cv.getBoundingClientRect(); dpr = Math.min(devicePixelRatio || 1, 2); w = r.width; h = r.height; cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
  function draw(dt) {
    time += dt; ctx.clearRect(0, 0, w, h);
    const pad = 28, X = (u) => pad + u * (w - pad * 2), Y = (u) => pad + u * (h - pad * 2);
    lineA += (lineT - lineA) * Math.min(1, dt * 5); letterA += (letterT - letterA) * Math.min(1, dt * 6);
    for (const p of P) { p.x.step(dt); p.y.step(dt); p.r.step(dt); p.a.step(dt); }
    for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) P[i].c[k] += (colorT[i][k] - P[i].c[k]) * Math.min(1, dt * 6);
    if (lineA > 0.02) {                                              // stage 3: attention from the term to each candidate
      ctx.lineWidth = 1;
      for (let i = 1; i <= 30; i++) { const s = 0.2 + 0.8 * Math.abs(Math.sin(time * 1.6 + i * 0.7)); ctx.strokeStyle = `rgba(${P[i].c.map((v) => v | 0)},${lineA * s * 0.7})`; ctx.beginPath(); ctx.moveTo(X(P[0].x.x), Y(P[0].y.x)); ctx.lineTo(X(P[i].x.x), Y(P[i].y.x)); ctx.stroke(); }
    }
    for (let i = N - 1; i >= 0; i--) {
      const p = P[i], x = X(p.x.x), y = Y(p.y.x), r = Math.max(0.4, p.r.x), a = clamp(p.a.x, 0, 1), c = p.c.map((v) => v | 0);
      if (r > 9) { const g = ctx.createRadialGradient(x, y, 0, x, y, r * 2.8); g.addColorStop(0, `rgba(${c},${0.55 * a})`); g.addColorStop(1, `rgba(${c},0)`); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 2.8, 0, 6.2832); ctx.fill(); }
      ctx.fillStyle = `rgba(${c},${a})`; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
      if (letterA > 0.05 && i < 5) { ctx.fillStyle = `rgba(255,255,255,${letterA * (i === 1 ? 1 : 0.75)})`; ctx.font = `600 ${i === 1 ? 17 : 13}px ui-monospace, Menlo, monospace`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String.fromCharCode(65 + i), x, y + 0.5); }
    }
  }
  function onScroll() {
    const r = track.getBoundingClientRect(), p = clamp(-r.top / (r.height - innerHeight), 0, 0.9999);
    onScreen = r.top < innerHeight && r.bottom > 0; if (onScreen) setStage(Math.floor(p * PIPE.length));
  }
  function start() {
    $("#pips").innerHTML = PIPE.map(() => "<i></i>").join("");
    for (let i = 0; i < N; i++) colorT[i] = [...WHITE];
    size(); new ResizeObserver(size).observe(cv); setStage(0);
    addEventListener("scroll", onScroll, { passive: true }); onScroll();
    run((dt) => { if (onScreen && !document.hidden) draw(dt); return true; });
  }
  return { start };
})();

/* ───────────── numbers ───────────── */
function initNumbers() {
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return; io.unobserve(e.target);
    $$("[data-count]", e.target).concat(e.target.matches("[data-count]") ? [e.target] : []).forEach((el) => {
      const v = parseFloat(el.dataset.count), dec = +el.dataset.dec || 0, s = new Spring(0, { damping: 1, response: 1.1 }); s.t = v;
      const node = el.firstChild; springTo([s], () => { node.nodeValue = Math.max(0, s.x).toFixed(dec); });
    });
    $$(".fill", e.target).forEach((f, i) => { const s = new Spring(0, { damping: 0.9, response: 0.9 }); setTimeout(() => { s.t = parseFloat(f.dataset.w); springTo([s], () => { f.style.width = clamp(s.x, 0, 100) + "%"; }); }, 120 + i * 90); });
  }), { threshold: 0.35 });
  $$(".stat").forEach((s) => io.observe(s));
}

/* Status badge on pages without the lab */
async function initBadge() {
  const b = $("#modeBadge"); if (!b || $("#term")) return;
  const set = (m, t) => { b.dataset.mode = m; $("#modeText").textContent = t; };
  try { const s = await (await fetch("/api/status", { cache: "no-store" })).json(); s.engine === "ready" ? set("live", `Live engine · ${s.name || "TermMap-X+"} · running on this machine`) : set("sample", s.engine === "loading" ? "Engine loading · models warming up…" : "Illustrative samples · engine not connected"); if (s.engine === "loading") setTimeout(initBadge, 2500); }
  catch (_) { set("sample", "Illustrative samples · engine not connected"); }
}

/* ───────────── boot ───────────── */
Field.start(); initHero(); initReveal(); initSpot(); initMagnet(); Pipe.start(); initNumbers(); initBadge();
addEventListener("keydown", (e) => { if (e.key === "/" && !$("#term") && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); location.href = "lab.html"; } });
$$('a[href^="#"]').forEach((a) => a.addEventListener("click", (e) => { const t = $(a.getAttribute("href")); if (!t) return; e.preventDefault(); t.scrollIntoView({ behavior: RM.matches ? "auto" : "smooth" }); }));
