"use strict";

/* CompEx player.
 *
 * The engine is the real Python package running under Pyodide, not a port.
 * Spans are rendered one at a time and scheduled into Web Audio ahead of the
 * playhead — audio already scheduled keeps playing on the audio thread even
 * while Python is blocking the main thread, which is what makes rendering and
 * playing at the same time possible without a worker.
 */

const SAMPLE_RATE = 44100;
const LOOKAHEAD_S = 24;      // how far ahead of the playhead to stay
const PRIME_S = 6;           // buffer this much before starting playback
const KEEP_LIMIT_S = 480;    // stop keeping audio for saving past 8 minutes

const $ = (id) => document.getElementById(id);

const state = {
  py: null, web: null, cat: null,
  mood: {}, theme: null,
  ctx: null, nextTime: 0, startedAt: 0,
  playing: false, info: null,
  kept: [], keptSeconds: 0, keptOverflow: false,
  bufferedSeconds: 0, sources: [],
};

/* ---------- boot ---------- */

async function boot() {
  try {
    say("loading python…");
    state.py = await loadPyodide({ stdout: () => {}, stderr: (m) => console.warn(m) });

    say("loading numpy…");
    await state.py.loadPackage("numpy");

    say("loading the engine…");
    const zip = await fetch("compex-src.zip", { cache: "no-cache" }).then((r) => {
      if (!r.ok) throw new Error(`engine archive missing (HTTP ${r.status})`);
      return r.arrayBuffer();
    });
    state.py.unpackArchive(zip, "zip");
    state.py.runPython("import sys\nif '.' not in sys.path: sys.path.insert(0, '.')");
    state.web = state.py.pyimport("compex.web");

    state.cat = JSON.parse(state.py.runPython(
      "import json, compex.web as _w; json.dumps(_w.catalogue())"));

    buildThemes();
    buildAxes();
    wireDuration();
    selectTheme(state.cat.themes.find((t) => t.name === "hypnotic") || state.cat.themes[0]);

    $("boot").hidden = true;
    $("app").hidden = false;
    $("dock").hidden = false;
    setStatus(`engine ready · v${state.cat.version}`, "ok");
  } catch (err) {
    console.error(err);
    say(`could not start: ${err.message || err}`);
    $("boot").classList.add("failed");
  }
}

const say = (msg) => { $("boot-msg").textContent = msg; };

/* ---------- controls ---------- */

function buildThemes() {
  $("themes").innerHTML = state.cat.themes
    .map((t) => `<button class="chip" data-theme="${t.name}">${t.name}</button>`).join("");
  document.querySelectorAll(".chip").forEach((b) =>
    b.addEventListener("click", () =>
      selectTheme(state.cat.themes.find((t) => t.name === b.dataset.theme))));
}

const AXIS_HINT = {
  valence: "dark → bright", energy: "still → frantic", tension: "resolved → unresolved",
  density: "sparse → crowded", grit: "clean → destroyed",
};

function buildAxes() {
  $("axes").innerHTML = state.cat.axes.map((a) => `
    <div class="knob">
      <div class="knob-top"><span>${a}</span><span class="val" id="v-${a}"></span></div>
      <input type="range" id="a-${a}" min="0" max="1" step="0.01" value="0.5">
      <span class="hint">${AXIS_HINT[a] || ""}</span>
    </div>`).join("");
  state.cat.axes.forEach((a) => $(`a-${a}`).addEventListener("input", (e) => {
    state.mood[a] = Number(e.target.value);
    $(`v-${a}`).textContent = Number(e.target.value).toFixed(2);
    state.theme = null;
    document.querySelectorAll(".chip").forEach((b) => b.classList.remove("on"));
  }));
}

function selectTheme(theme) {
  if (!theme) return;
  state.theme = theme.name;
  state.cat.axes.forEach((a) => {
    state.mood[a] = theme[a];
    const s = $(`a-${a}`);
    if (s) { s.value = theme[a]; $(`v-${a}`).textContent = Number(theme[a]).toFixed(2); }
  });
  document.querySelectorAll(".chip").forEach((b) =>
    b.classList.toggle("on", b.dataset.theme === theme.name));
}

function wireDuration() {
  const input = $("duration");
  const show = () => { $("v-duration").textContent = clock(Number(input.value)); };
  input.addEventListener("input", show);
  show();
}

const clock = (s) => `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s`;
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

/* ---------- playing ---------- */

async function play() {
  if (state.playing) return;
  stop(true);

  state.ctx = state.ctx || new (window.AudioContext || window.webkitAudioContext)();
  await state.ctx.resume();          // must happen inside the tap, for iOS

  const seconds = Number($("duration").value);
  const seed = Number($("seed").value);
  setStatus("composing…");
  $("play").disabled = true;

  await pause(30);                   // let the UI paint before Python blocks it

  let info;
  try {
    info = JSON.parse(state.web.start(seed, seconds, JSON.stringify(state.mood), 8.0));
  } catch (err) {
    console.error(err);
    setStatus(`could not compose: ${err.message || err}`, "err");
    $("play").disabled = false;
    return;
  }

  state.info = info;
  state.playing = true;
  state.kept = []; state.keptSeconds = 0; state.keptOverflow = false;
  state.bufferedSeconds = 0; state.sources = [];
  state.nextTime = 0; state.startedAt = 0;

  showInfo(info);
  $("save").classList.add("disabled");
  $("save").removeAttribute("href");
  $("stop").disabled = false;

  pump();
  requestAnimationFrame(tick);
}

async function pump() {
  while (state.playing) {
    const ahead = state.nextTime - (state.ctx.currentTime || 0);
    if (state.startedAt && ahead > LOOKAHEAD_S) { await pause(250); continue; }

    let meta;
    try {
      meta = state.web.step();
    } catch (err) {
      console.error(err);
      setStatus(`render failed: ${err.message || err}`, "err");
      state.playing = false;
      return;
    }
    if (!meta) {
      setStatus(`done — ${mmss(state.bufferedSeconds)} rendered`, "ok");
      finishSaving();
      return;
    }

    const chunk = JSON.parse(meta);
    const proxy = state.web.samples();
    const view = proxy.getBuffer("f32");
    const samples = new Float32Array(view.data);   // copy before releasing
    view.release();
    proxy.destroy();

    schedule(samples);
    keep(samples);
    state.bufferedSeconds += samples.length / SAMPLE_RATE;
    setStatus(`${chunk.movement} · buffered ${mmss(state.bufferedSeconds)}`
              + (chunk.remaining ? ` · ${chunk.remaining} spans left` : ""));

    await pause(0);                  // hand the frame back so the UI can breathe
  }
}

function schedule(samples) {
  const ctx = state.ctx;
  const buffer = ctx.createBuffer(1, samples.length, SAMPLE_RATE);
  buffer.copyToChannel(samples, 0);

  if (!state.startedAt) {
    // Wait until a little is buffered before starting, so a slow first render
    // does not leave a gap in the middle of the opening bar.
    state.nextTime = ctx.currentTime + Math.max(0.25, PRIME_S - state.bufferedSeconds);
    state.startedAt = state.nextTime;
  }
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start(state.nextTime);
  state.sources.push(source);
  state.nextTime += buffer.duration;
}

function keep(samples) {
  if (state.keptOverflow) return;
  if (state.keptSeconds > KEEP_LIMIT_S) {
    state.keptOverflow = true;
    state.kept = [];                 // let the memory go; this track is play-only
    return;
  }
  state.kept.push(samples);
  state.keptSeconds += samples.length / SAMPLE_RATE;
}

function stop(quiet) {
  state.playing = false;
  state.sources.forEach((s) => { try { s.stop(); } catch (_) {} });
  state.sources = [];
  state.nextTime = state.ctx ? state.ctx.currentTime : 0;
  state.startedAt = 0;
  $("stop").disabled = true;
  $("play").disabled = false;
  if (!quiet) setStatus("stopped");
}

function tick() {
  if (!state.playing && !state.startedAt) return;
  const ctx = state.ctx;
  const total = state.info ? state.info.seconds : 1;
  const played = Math.max(0, Math.min(total, (ctx.currentTime || 0) - state.startedAt));

  $("played").style.width = `${(played / total) * 100}%`;
  $("buffered").style.width = `${Math.min(1, state.bufferedSeconds / total) * 100}%`;
  $("t-now").textContent = mmss(played);
  $("t-total").textContent = mmss(total);

  const movement = (state.info.movements || []).find((m) => played >= m.start && played < m.end);
  $("t-move").textContent = movement ? movement.name : "";
  highlightEvolution(played);

  if (state.playing || played < total) requestAnimationFrame(tick);
  else { $("play").disabled = false; $("stop").disabled = true; }
}

/* ---------- what it is doing ---------- */

function showInfo(info) {
  $("now").hidden = false;
  $("detail-card").hidden = false;
  $("m-theme").textContent = info.theme;
  $("m-bpm").textContent = `${info.bpm} bpm ${info.meter}/4`;
  $("m-scale").textContent = info.scale;
  $("m-form").textContent = info.archetype;
  $("m-summary").textContent = info.summary;
  $("m-formula").textContent = state.web.formula_text();
  drawEvolution(info.evolution || []);
  drawChoices(info);
}

/* ---------- what it chose ---------- */

function drawChoices(info) {
  const patterns = info.patterns || [];
  const taste = info.taste;
  $("choice-card").hidden = patterns.length === 0;
  if (!patterns.length) return;

  $("patterns").innerHTML = patterns.map((p) => {
    const cells = p.slots.map((velocity, i) => {
      const weight = p.weights[i] || 0;
      const level = velocity > 0.66 ? " hard" : velocity > 0 ? " soft" : "";
      return `<i class="cell${level}${weight > 1 ? " learned" : ""}"></i>`;
    }).join("");
    return `<div class="figure">
      <div class="figure-head"><b>${p.voice}</b>
        <span class="hint">${p.hits} hits · ${p.subdivision} beat/slot</span></div>
      <div class="cells">${cells}</div>
    </div>`;
  }).join("");

  if (taste) {
    $("taste").innerHTML = taste.criteria.map((c) => `
      <div class="weight${c.moved ? " moved" : ""}">
        <span class="wname">${c.name}</span>
        <span class="wbar"><i style="width:${Math.min(100, (c.now / 3.2) * 100)}%"></i></span>
        <span class="wval">${c.now}${c.moved ? " moved" : ""}</span>
      </div>`).join("");
    $("melody-summary").textContent =
      `the ${Math.max(0, taste.auditioned - taste.chosen)} melodies it turned down`;
  }

  $("melody").innerHTML = (info.melody || []).map((choice) => `
    <div class="audition"><b>${choice.origin}</b>
      <span class="hint">${mmss(choice.at)} · gen ${choice.generation} ·
        beat ${choice.considered - 1} by ${choice.margin}</span></div>`).join("");
}

function drawEvolution(steps) {
  const host = $("evolution");
  $("evo-card").hidden = steps.length === 0;
  host.innerHTML = steps.map((s, i) => {
    const heard = s.heard.length
      ? s.heard.map((h) => `<span class="pill" title="${h.attribution}">${h.principle} ${h.measured}</span>`).join("")
      : `<span class="pill ok">nothing to correct</span>`;
    return `<div class="evo" id="evo-${i}" data-at="${s.at}">
      <div class="evo-top"><b>${s.name}</b>
        <span class="hint">${mmss(s.at)} · plasticity ${s.plasticity}</span></div>
      <div>${heard}</div>
      <div class="did">${s.note}</div>
    </div>`;
  }).join("");
}

function highlightEvolution(played) {
  const steps = (state.info && state.info.evolution) || [];
  let current = -1;
  steps.forEach((s, i) => { if (played >= s.at) current = i; });
  steps.forEach((_, i) => {
    const el = $(`evo-${i}`);
    if (el) el.classList.toggle("live", i === current);
  });
}

/* ---------- saving what you heard ---------- */

function finishSaving() {
  const save = $("save");
  if (state.keptOverflow || !state.kept.length) {
    setStatus(`done — too long to save on a phone, playback only`, "ok");
    return;
  }
  const blob = new Blob([toWav(state.kept)], { type: "audio/wav" });
  save.href = URL.createObjectURL(blob);
  save.download = `compex_${state.info.theme}_seed${state.info.seed}.wav`;
  save.classList.remove("disabled");
}

function toWav(chunks) {
  const length = chunks.reduce((n, c) => n + c.length, 0);
  const bytes = new ArrayBuffer(44 + length * 2);
  const view = new DataView(bytes);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF"); view.setUint32(4, 36 + length * 2, true); ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true); view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  ascii(36, "data"); view.setUint32(40, length * 2, true);

  let offset = 44;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++, offset += 2) {
      const clamped = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(offset, clamped * 32767, true);
    }
  }
  return bytes;
}

const pause = (ms) => new Promise((r) => setTimeout(r, ms));

/* ---------- go ---------- */

$("play").addEventListener("click", play);
$("stop").addEventListener("click", () => stop(false));
$("reseed").addEventListener("click", () => {
  $("seed").value = Math.floor(Math.random() * 2147483647);
});
boot();
