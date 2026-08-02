"use strict";

const $ = (id) => document.getElementById(id);
const state = { themes: [], axes: [], mood: {}, theme: null, track: null, format: "wav" };

/* ---------- setup ---------- */

async function boot() {
  try {
    const data = await (await fetch("/api/themes")).json();
    state.themes = data.themes || [];
    state.axes = data.axes || [];
    buildThemes();
    buildAxes();
    buildFormats(data.formats || ["wav"]);
    if (state.themes.length) selectTheme(state.themes.find((t) => t.name === "melancholy")
                                        || state.themes[0]);
  } catch (err) {
    setStatus("could not reach the engine — is the server still running?", "err");
  }
  wireRuntime();
}

function buildThemes() {
  $("themes").innerHTML = state.themes
    .map((theme) => `<button class="chip" data-theme="${theme.name}">${theme.name}</button>`)
    .join("");
  for (const button of document.querySelectorAll(".chip")) {
    button.addEventListener("click", () => {
      selectTheme(state.themes.find((t) => t.name === button.dataset.theme));
    });
  }
}

function buildAxes() {
  $("axes").innerHTML = state.axes.map((axis) => `
    <div class="knob">
      <div class="knob-top"><span class="name">${axis}</span><span class="val" id="v-${axis}"></span></div>
      <input type="range" id="a-${axis}" min="0" max="1" step="0.01" value="0.5">
      <span class="hint">${AXIS_HINTS[axis] || ""}</span>
    </div>`).join("");

  for (const axis of state.axes) {
    $(`a-${axis}`).addEventListener("input", (event) => {
      state.mood[axis] = Number(event.target.value);
      $(`v-${axis}`).textContent = Number(event.target.value).toFixed(2);
      markCustom();
    });
  }
}

const AXIS_HINTS = {
  valence: "dark → bright",
  energy: "still → frantic",
  tension: "resolved → unresolved",
  density: "sparse → crowded",
  grit: "clean → destroyed",
};

function buildFormats(formats) {
  $("formats").innerHTML = formats.map((name, index) => `
    <label class="radio">
      <input type="radio" name="format" value="${name}" ${index === 0 ? "checked" : ""}>
      <span>${name.toUpperCase()}</span>
    </label>`).join("");
  for (const input of document.querySelectorAll('input[name="format"]')) {
    input.addEventListener("change", () => { state.format = input.value; });
  }
  state.format = formats[0];
}

function wireRuntime() {
  const input = $("k-duration");
  const show = () => {
    const seconds = Number(input.value);
    const minutes = Math.floor(seconds / 60);
    $("v-duration").textContent = minutes
      ? `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`
      : `${seconds}s`;
  };
  input.addEventListener("input", show);
  show();
}

function selectTheme(theme) {
  if (!theme) return;
  state.theme = theme.name;
  state.mood = {};
  for (const axis of state.axes) {
    state.mood[axis] = theme[axis];
    const slider = $(`a-${axis}`);
    if (slider) {
      slider.value = theme[axis];
      $(`v-${axis}`).textContent = Number(theme[axis]).toFixed(2);
    }
  }
  for (const button of document.querySelectorAll(".chip")) {
    button.classList.toggle("on", button.dataset.theme === theme.name);
  }
  $("theme-note").textContent = "";
}

function markCustom() {
  state.theme = null;
  for (const button of document.querySelectorAll(".chip")) button.classList.remove("on");
  $("theme-note").textContent = "custom";
}

/* ---------- waveform ---------- */

function drawWave(peaks, movements, duration) {
  const canvas = $("wave");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height || !peaks || !peaks.length) return;
  canvas.width = width * ratio;
  canvas.height = height * ratio;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  for (const movement of movements || []) {
    const x0 = (movement.start / duration) * width;
    const x1 = (movement.end / duration) * width;
    ctx.fillStyle = `rgba(242, 177, 52, ${0.04 + movement.energy * 0.14})`;
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), height);
  }

  const mid = height / 2;
  const step = width / peaks.length;
  ctx.fillStyle = "#f2b134";
  peaks.forEach((peak, i) => {
    const h = Math.max(0.6, peak * (height * 0.94));
    ctx.fillRect(i * step, mid - h / 2, Math.max(0.7, step * 0.85), h);
  });
}

function drawMarks(movements, duration) {
  $("marks").innerHTML = (movements || []).map((movement) => {
    const span = ((movement.end - movement.start) / duration) * 100;
    return `<span style="flex:0 1 ${span}%" title="${movement.name} — energy ${movement.energy}"
             >${movement.name}</span>`;
  }).join("");
}

/* ---------- evolution ---------- */

function drawEvolution(steps) {
  const host = $("evolution");
  if (!steps || !steps.length) {
    $("evo-panel").hidden = true;
    return;
  }
  $("evo-panel").hidden = false;
  const corrections = steps.reduce((n, s) => n + s.did.length, 0);
  $("evo-note").textContent =
    `${steps.length} listen-backs, ${corrections} corrections`;

  host.innerHTML = steps.map((step) => {
    const heard = step.heard.length
      ? step.heard.map((h) =>
          `<span class="chip" title="${h.attribution}">${h.principle} ${h.measured}</span>`).join("")
      : `<span class="chip ok">nothing to correct</span>`;
    const did = step.did.length
      ? step.did.map((d) =>
          `<div class="did"><code>${d.drive}</code> ${d.before} &rarr; ${d.after}
           <span class="note">${d.because}</span></div>`).join("")
      : `<div class="did note">held its ground</div>`;
    return `<div class="evo-step">
      <div class="evo-head"><b>${step.name}</b>
        <span class="note">beat ${step.at} · plasticity ${step.plasticity} · unrest ${step.unrest}</span>
      </div>
      <div class="evo-heard">${heard}</div>
      ${did}
    </div>`;
  }).join("");
}

/* ---------- actions ---------- */

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

function setDelivery(enabled, data) {
  const link = $("download");
  $("send").disabled = !enabled;
  link.classList.toggle("disabled", !enabled);
  if (enabled && data) {
    link.href = data.url;
    link.setAttribute("download", data.file);
  } else {
    link.removeAttribute("href");
  }
}

async function makeTrack() {
  const button = $("make");
  button.disabled = true;
  const seconds = Number($("k-duration").value);
  setStatus(`composing ${seconds}s… longer tracks take longer to synthesise`);

  try {
    const response = await fetch("/api/make", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        knobs: { seed: Number($("seed").value), duration_s: seconds },
        mood: state.mood,
        format: state.format,
      }),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);

    state.track = data;
    $("summary").textContent = data.summary;
    $("summary").classList.remove("muted");
    $("formula").textContent = data.formula;
    $("formula-panel").hidden = false;
    $("delivery").hidden = false;
    $("dl-formula").href = data.formula_url;
    $("dl-formula").setAttribute("download", data.formula_file);

    drawEvolution(data.evolution);
    setDelivery(true, data);
    drawWave(data.peaks, data.movements, data.duration);
    drawMarks(data.movements, data.duration);
    setStatus(`${data.theme} · ${data.duration}s · ${(data.bytes / 1e6).toFixed(1)} MB · ${data.fingerprint}`, "ok");
  } catch (err) {
    setDelivery(false);
    setStatus(String(err.message || err), "err");
  } finally {
    button.disabled = false;
  }
}

async function emailTrack() {
  if (!state.track) { setStatus("make something first", "err"); return; }
  const button = $("send");
  button.disabled = true;
  setStatus("sending…");
  try {
    const response = await fetch("/api/email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file: state.track.file,
        to: $("to").value,
        note: ($("note").value ? $("note").value + "\n\n" : "") + state.track.formula,
        subject: `CompEx — ${state.track.theme} (${state.track.fingerprint})`,
      }),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
    setStatus(data.message, "ok");
  } catch (err) {
    setStatus(String(err.message || err), "err");
  } finally {
    button.disabled = false;
  }
}

/* ---------- boot ---------- */

boot();
$("make").addEventListener("click", makeTrack);
$("send").addEventListener("click", emailTrack);
$("reseed").addEventListener("click", () => {
  $("seed").value = Math.floor(Math.random() * 2147483647);
});
$("copy").addEventListener("click", async () => {
  if (!state.track) return;
  try {
    await navigator.clipboard.writeText(state.track.formula);
    setStatus("formula copied", "ok");
  } catch {
    setStatus("clipboard blocked — use save .tex instead", "err");
  }
});
window.addEventListener("resize", () => {
  if (state.track) drawWave(state.track.peaks, state.track.movements, state.track.duration);
});
