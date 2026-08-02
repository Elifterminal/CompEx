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
  wireGhosts();
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

function wireGhosts() {
  const input = $("k-ghost");
  const show = () => {
    const value = Number(input.value);
    $("v-ghost").textContent = value === 0 ? "off" : value.toFixed(2);
  };
  input.addEventListener("input", show);
  show();
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

/* ---------- what it chose ---------- */

function drawChoices(data) {
  const patterns = data.patterns || [];
  const taste = data.taste;
  if (!patterns.length && !taste) {
    $("choice-panel").hidden = true;
    return;
  }
  $("choice-panel").hidden = false;
  $("choice-note").textContent = taste
    ? `${taste.chosen} phrases kept out of ${taste.auditioned} imagined · ` +
      `${taste.generations} generations deep`
    : "";

  $("patterns").innerHTML = patterns.map(drawFigure).join("");
  drawGhosts(data.ghosts);
  drawMix(data.mix);
  drawHindsight(data.hindsight);
  drawLedger(data.ledger);
  if (taste) $("taste").innerHTML = taste.criteria.map(drawWeight).join("");
  drawMelody(data.melody || []);
}

function drawGhosts(ghosts) {
  const host = $("ghosts");
  if (!ghosts || !ghosts.notes) {
    host.innerHTML = `<p class="note">no ghosts — either it was never torn, or they are switched off</p>`;
    return;
  }
  const rows = ghosts.loudest.map((g) => `
    <div class="ghostrow">
      <code>${g.origin}</code>
      <span class="gbar"><i style="width:${Math.min(100, g.heard_at * 100)}%"></i></span>
      <span class="note">${g.at}s · lost to ${g.beaten_by} by ${g.margin}</span>
    </div>`).join("");
  host.innerHTML = `<div class="note">${ghosts.notes} notes it decided against, from
    ${ghosts.turned_down} lines turned down across ${ghosts.auditions} auditions ·
    average closeness ${ghosts.torn} · played at ${ghosts.gain}</div>${rows}`;
}

function drawMix(mix) {
  const host = $("mix");
  if (!mix || !mix.voices.length) { host.innerHTML = `<p class="note">not measured</p>`; return; }
  const rows = mix.voices.map((v) => {
    const state = v.trim > 1.02 ? " lifted" : v.trim < 0.98 ? " held" : "";
    const floor = Math.min(100, v.floor * 100);
    return `<div class="mixrow${state}">
      <code>${v.voice}</code>
      <span class="mbar" title="floor for a ${v.role} that plays this much: ${(v.floor * 100).toFixed(0)}%">
        <i style="width:${Math.min(100, v.share * 100)}%"></i>
        <b style="left:${floor}%"></b>
      </span>
      <span class="note">${v.home} · ${(v.share * 100).toFixed(0)}%
        ${v.trim !== 1 ? `&rarr; ${(v.after * 100).toFixed(0)}% at ${v.trim}&times;` : ""}</span>
    </div>`;
  }).join("");
  const stuck = mix.still_buried.length
    ? `<p class="note">gain could not rescue: <b>${mix.still_buried.join(", ")}</b> —
       too many voices in one band, which is an arrangement problem rather than a mix one</p>`
    : "";
  host.innerHTML = `<div class="note">${mix.summary}</div>${rows}${stuck}`;
}

function drawHindsight(reveal) {
  const host = $("hindsight");
  if (!reveal) { host.innerHTML = `<p class="note">not measured — too short</p>`; return; }
  const bar = Math.min(100, reveal.share * 300);   // 33% would be a total re-explanation
  const recalls = reveal.recalls.length
    ? reveal.recalls.map((r) => `<span class="chip">${r.origin} at ${r.at}s</span>`).join("")
    : `<span class="note">it never reached back for a specific earlier phrase</span>`;
  return void (host.innerHTML = `
    <div class="reveal">
      <span class="rname">then</span>
      <span class="rbar"><i style="width:100%"></i></span>
      <span class="rval">${reveal.then} symbols</span>
    </div>
    <div class="reveal saved">
      <span class="rname">in hindsight</span>
      <span class="rbar"><i style="width:${(reveal.now / reveal.then) * 100}%"></i></span>
      <span class="rval">${reveal.now} symbols</span>
    </div>
    <p class="note">${reveal.note} — measured over ${reveal.phrases} opening phrases,
       against ${reveal.literal} to spell them all out.</p>
    <div>${recalls}</div>`);
}

function drawLedger(ledger) {
  const host = $("ledger");
  if (!ledger || (!ledger.open.length && !ledger.paid.length)) {
    host.innerHTML = `<p class="note">nothing was left hanging in this one</p>`;
    return;
  }
  const waits = ledger.paid.map((p) => p.waited).sort((a, b) => a - b);
  const median = waits.length ? waits[Math.floor(waits.length / 2)] : 0;

  const carrying = ledger.carrying
    ? `<div class="carrying"><b>carrying a ${ledger.carrying.kind}</b>
        <span class="note">for ${ledger.carrying.seconds}s — the deepest thing still unanswered</span></div>`
    : "";

  // Open first, and biggest first: what the piece has not done is the part you
  // cannot hear by listening to what it did.
  const open = [...ledger.open].sort((a, b) => b.pressure - a.pressure).map((owed) => `
    <div class="owed">
      <code>${owed.kind}</code>
      <span class="obar"><i style="width:${Math.min(100, owed.pressure * 100)}%"></i></span>
      <span class="note">${owed.domain} · opened ${owed.opened}s · carried ${owed.carried}s</span>
    </div>`).join("");

  const paid = ledger.paid.slice(-6).reverse().map((settled) => `
    <div class="owed paid">
      <code>${settled.kind}</code>
      <span class="note">${settled.how}, ${settled.waited}s later</span>
    </div>`).join("");

  host.innerHTML = `${carrying}
    <div class="note">${ledger.open.length} open · ${ledger.paid.length} settled ·
      median wait ${median}s · total pressure ${ledger.pressure}</div>
    ${open}${paid}`;
}

function drawFigure(pattern) {
  // Two rows on the same grid: what it plays, and what it now believes about
  // where hits belong. Seeing them together is the point — the second row is
  // what the first row taught it.
  const cells = pattern.slots.map((velocity, index) => {
    const weight = pattern.weights[index] || 0;
    const learned = weight > 1 ? " learned" : "";
    const level = velocity > 0.66 ? " hard" : velocity > 0 ? " soft" : "";
    return `<i class="cell${level}${learned}"
              title="slot ${index} · weight ${weight} (was ${pattern.start_weights[index]})"></i>`;
  }).join("");
  const moved = pattern.past_start
    ? `<span class="chip learned">${pattern.past_start} slot${pattern.past_start > 1 ? "s" : ""}
        heavier than the meter made them</span>`
    : "";
  return `<div class="figure">
    <div class="figure-head"><code>${pattern.voice}</code>
      <span class="note">${pattern.hits} hits · ${pattern.subdivision} beat per slot ·
        revised ${pattern.revisions}&times;</span></div>
    <div class="cells">${cells}</div>
    <div class="figure-foot">${moved}</div>
  </div>`;
}

function drawWeight(criterion) {
  const width = Math.min(100, (criterion.now / 3.2) * 100);
  const from = Math.min(100, (criterion.start / 3.2) * 100);
  return `<div class="weight${criterion.moved ? " moved" : ""}">
    <span class="wname">${criterion.name}</span>
    <span class="wbar"><i style="width:${width}%"></i><b style="left:${from}%"></b></span>
    <span class="wval">${criterion.now}${criterion.moved ? " <em>moved</em>" : ""}</span>
  </div>`;
}

function drawMelody(choices) {
  $("melody-details").hidden = choices.length === 0;
  $("melody").innerHTML = choices.map((choice) => {
    const beaten = choice.beat.map((other) => `${other.origin} ${other.score}`).join(", ");
    return `<div class="audition">
      <code>${choice.origin}</code>
      <span class="note">${choice.at}s · gen ${choice.generation} ·
        won by ${choice.margin} over ${choice.considered - 1}</span>
      <div class="note">${beaten || "nothing else on the table"}</div>
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
        knobs: { seed: Number($("seed").value), duration_s: seconds,
                 ghost_gain: Number($("k-ghost").value) },
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
    drawChoices(data);
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
