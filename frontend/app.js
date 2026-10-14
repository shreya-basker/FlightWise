const $ = (id) => document.getElementById(id);
const result = $("result");
const esc = (value) => String(value ?? "–").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const pct = (value) => `${Math.round(value * 1000) / 10}%`;
const list = (items) => items && items.length ? `<ul>${items.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : "";
const has = (value) => value !== null && value !== undefined && value !== "";

// Aviationstack sends airport-local wall-clock times labelled +00:00, so read the digits rather than converting zones.
const clock = (iso) => has(iso) ? iso.slice(11, 16) : "–";
const day = (iso) => has(iso) ? new Date(`${iso.slice(0, 10)}T12:00:00`).toLocaleDateString(undefined, {month: "short", day: "numeric"}) : "";
const delayChip = (minutes) => !has(minutes) ? "" : minutes >= 15 ? `<span class="chip chip--late">+${esc(minutes)} min</span>` : minutes > 0 ? `<span class="chip">+${esc(minutes)} min</span>` : `<span class="chip chip--ok">on time</span>`;

const PHASES = {
  scheduled: ["Scheduled", "neutral"], departed: ["Departed", "go"], landed: ["Departed · Landed", "go"],
  cancelled: ["Cancelled", "bad"], diverted: ["Diverted", "bad"], incident: ["Incident", "bad"], unknown: ["Status unavailable", "neutral"],
};

function setResult(html) { result.className = "result"; result.innerHTML = html; result.scrollIntoView({behavior: "smooth", block: "nearest"}); }
function loading(text) { setResult(`<div class="panel loading"><span class="spinner" aria-hidden="true"></span>${esc(text)}</div>`); }

function errorHtml(status, detail) {
  const text = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ") : JSON.stringify(detail);
  return `<div class="panel panel--error"><p class="eyebrow">Request unavailable · HTTP ${esc(status)}</p><p class="error-text">${esc(text)}</p></div>`;
}

function flightHtml(f) {
  const [label, tone] = PHASES[f.phase] || PHASES.unknown;
  const departed = f.phase === "departed" || f.phase === "landed";
  const requested = f.requested_flight_number && f.requested_flight_number !== f.flight_number ? `<span class="muted"> · requested as ${esc(f.requested_flight_number)}</span>` : "";
  const cell = (title, iso, extra = "") => `<div><dt>${title}</dt><dd><b>${clock(iso)}</b> <small>${esc(day(iso))}</small>${extra}</dd></div>`;
  return `<article class="panel flight">
    <header class="flight-head">
      <div><p class="eyebrow">Live flight · ${esc(f.provider)}</p><h2>${esc(f.flight_number)} <span>${esc(f.airline_name || f.airline)}</span>${requested}</h2></div>
      <span class="pill pill--${tone}"><i></i>${label}</span>
    </header>
    <div class="route">
      <div class="airport"><b>${esc(f.origin)}</b><small>${esc(f.origin_name || "")}</small></div>
      <div class="route-line${departed ? " route-line--flown" : ""}" aria-hidden="true"><span>✈</span></div>
      <div class="airport airport--end"><b>${esc(f.destination)}</b><small>${esc(f.destination_name || "")}</small></div>
    </div>
    <dl class="times">
      ${cell("Scheduled departure", f.scheduled_departure)}
      ${departed ? cell("Actual departure", f.actual_departure, delayChip(f.departure_delay_minutes)) : cell("Estimated departure", f.estimated_departure, has(f.departure_delay_minutes) ? `<span class="chip">provider expects +${esc(f.departure_delay_minutes)} min</span>` : "")}
      ${cell("Scheduled arrival", f.scheduled_arrival)}
      ${has(f.actual_arrival) ? cell("Actual arrival", f.actual_arrival, delayChip(f.arrival_delay_minutes)) : cell("Estimated arrival", f.estimated_arrival)}
      <div><dt>Terminal · Gate</dt><dd><b>${esc(f.terminal)} · ${esc(f.gate)}</b></dd></div>
    </dl>
    <p class="fineprint">Times are local to each airport. Status is reported by ${esc(f.provider)}, fetched ${esc(new Date(f.retrieved_at).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}))}.</p>
  </article>`;
}

function outcomeHtml(o, risk) {
  if (!o) return "";
  const delayText = has(o.departure_delay_minutes) ? `${esc(o.departure_delay_minutes)} min` : "Not reported";
  const verdict = o.delayed_15 === null ? "Unknown" : o.delayed_15 ? "Delayed (15+ min)" : "On time (under 15 min)";
  const match = o.prediction_matched === null ? ["–", "No comparison available", ""] : o.prediction_matched ? ["✓", "Estimate matched the outcome", "match"] : ["✗", "Estimate did not match the outcome", "miss"];
  return `<article class="panel outcome outcome--${match[2] || "none"}">
    <p class="eyebrow">What actually happened · this flight has departed</p>
    <div class="outcome-grid">
      <div><span>Departure delay</span><b>${delayText}</b><small>${verdict}</small></div>
      <div><span>Pre-departure estimate</span><b>${risk ? (risk.prediction === "likely_delayed" ? "Likely delayed" : "Likely on time") : "–"}</b><small>${risk ? `${pct(risk.delay_probability)} risk` : "Not available"}</small></div>
      <div class="verdict"><span>Comparison</span><b><em>${match[0]}</em> ${match[1]}</b></div>
    </div>
    <p class="fineprint">The estimate uses only information available before departure (schedule, route and weather for the scheduled hour). The actual delay is shown for comparison and is never fed into the model.</p>
  </article>`;
}

function riskHtml(data, title) {
  const delayed = data.prediction === "likely_delayed";
  const p = Math.min(100, data.delay_probability * 100), t = data.threshold * 100, b = (data.base_rate ?? 0) * 100;
  const versus = data.base_rate == null ? "" : `${data.delay_probability >= data.base_rate ? "Above" : "Below"} the ${pct(data.base_rate)} average in the training data`;
  return `<article class="panel risk">
    <p class="eyebrow">${esc(title)}</p>
    <div class="risk-top">
      <div><p class="risk-number">${pct(data.delay_probability)}</p><p class="risk-caption">estimated chance of departing 15+ minutes late</p></div>
      <span class="pill pill--${delayed ? "late" : "go"}"><i></i>${delayed ? "Likely delayed" : "Likely on time"}</span>
    </div>
    <div class="meter" role="img" aria-label="Delay probability ${pct(data.delay_probability)}; flag threshold ${pct(data.threshold)}">
      <div class="meter-fill${delayed ? " meter-fill--late" : ""}" style="width:${p}%"></div>
      <span class="tick tick--threshold" style="left:${t}%"><em>flag ≥ ${pct(data.threshold)}</em></span>
      ${data.base_rate == null ? "" : `<span class="tick tick--avg" style="left:${b}%"><em>avg ${pct(data.base_rate)}</em></span>`}
    </div>
    <div class="risk-meta"><span>${esc(versus)}</span><span>${esc(data.distance_km)} km · ${esc(data.distance_source)}</span><span>Model: ${esc(String(data.model).replace("weather_", "").replaceAll("_", " "))}</span></div>
  </article>`;
}

function weatherHtml(w) {
  const tile = (label, value, unit) => `<div class="tile"><span>${label}</span><b>${has(value) ? esc(value) : "–"}<small>${has(value) ? unit : ""}</small></b></div>`;
  const vis = has(w.VisibilityM) ? (w.VisibilityM / 1000).toFixed(1) : null;
  return `<article class="panel weather">
    <p class="eyebrow">Weather at ${esc(w.airport)} · ${esc(clock(w.observed_at))} ${esc(day(w.observed_at))} local</p>
    <div class="tiles">${tile("Temperature", w.TemperatureC, "°C")}${tile("Wind", w.WindSpeedKPH, " km/h")}${tile("Precipitation", w.PrecipitationMM, " mm")}${tile("Cloud cover", w.CloudCoverPct, "%")}${tile("Visibility", vis, " km")}</div>
    <p class="fineprint">${esc(w.source)}</p>
  </article>`;
}

function estimateHtml(data, title) {
  return `${riskHtml(data, title)}
    ${data.warnings && data.warnings.length ? `<div class="panel panel--warn">${list(data.warnings)}</div>` : ""}
    ${weatherHtml(data.weather_context || {})}
    <details class="panel limits"><summary>Data limitations</summary>${list(data.limitations)}</details>
    <p class="disclaimer">${esc(data.disclaimer)}</p>`;
}

async function call(url, options) {
  try {
    const response = await fetch(url, options);
    const body = await response.json().catch(() => ({detail: "The server returned a non-JSON response."}));
    return {ok: response.ok, status: response.status, body};
  } catch (error) {
    return {ok: false, status: "network", body: {detail: "Could not reach the FlightWise server. Is uvicorn running?"}};
  }
}

$("flight_date").value = new Date().toLocaleDateString("en-CA");
$("prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  loading("Fetching departure-airport weather and estimating risk…");
  const distance = $("distance").value;
  const payload = {airline: $("airline").value.trim().toUpperCase(), origin: $("origin").value.trim().toUpperCase(), destination: $("destination").value.trim().toUpperCase(), flight_date: $("flight_date").value, scheduled_departure_hour: Number($("hour").value), distance_km: distance ? Number(distance) : null};
  const {ok, status, body} = await call("/api/us/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const title = `Route estimate · ${payload.airline} ${payload.origin} → ${payload.destination} · ${payload.flight_date} ${String(payload.scheduled_departure_hour).padStart(2, "0")}:00`;
  setResult(ok ? estimateHtml(body, title) : errorHtml(status, body.detail));
});

$("lookup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const flight = $("flight-number").value.trim();
  if (!flight) return;
  loading(`Looking up ${flight.toUpperCase()} and departure weather…`);
  const {ok, status, body} = await call(`/api/live/flight/${encodeURIComponent(flight)}`);
  if (!ok) { setResult(errorHtml(status, body.detail)); return; }
  const departed = body.outcome && body.outcome.departed;
  const estimate = body.risk
    ? estimateHtml(body.risk, departed ? "Pre-departure delay estimate" : `Delay estimate · ${body.flight.flight_number}`)
    : `<div class="panel panel--warn"><p>${esc(body.risk_unavailable_reason)}</p></div>`;
  setResult(flightHtml(body.flight) + outcomeHtml(body.outcome, body.risk) + estimate);
});

(async () => {
  const {ok, body} = await call("/api/live/config");
  const configured = ok && body.flight_lookup_configured;
  $("live-status").textContent = !ok ? "STATUS UNKNOWN" : configured ? "PROVIDER CONFIGURED" : "API KEY REQUIRED";
  $("lookup-button").disabled = !configured;
  $("lookup-hint").textContent = configured
    ? "Uses your local Aviationstack key (kept on the server). Results are cached briefly to protect your plan quota."
    : "API key required: add AVIATIONSTACK_API_KEY to .env to enable flight-number lookup. Route estimates below work without it.";
})();

(async () => {
  const {ok, status, body} = await call("/api/us/model-info");
  const box = $("model-info");
  if (!ok) { box.innerHTML = errorHtml(status, body.detail); return; }
  const ev = body.evaluation;
  const rows = Object.entries(ev.models).map(([name, m]) => `<tr${name === body.selected_model ? ' class="selected"' : ""}><td>${esc(name.replace("weather_", "").replaceAll("_", " "))}${name === body.selected_model ? " ✓" : ""}</td><td>${m.f1}</td><td>${m.roc_auc}</td><td>${m.brier_score}</td><td>${m.precision}</td><td>${m.recall}</td></tr>`).join("");
  box.innerHTML = `<p>Trained on ${esc(body.training_data.rows.toLocaleString())} BTS flights (${esc(body.training_data.months.join(", "))}, ${body.training_data.origins.length} origin airports). Test period ${esc(ev.test.start)} → ${esc(ev.test.end)}: ${esc(ev.test.rows.toLocaleString())} flights, observed delay rate ${pct(ev.test.delay_rate)}.</p>
    <div class="table-wrap"><table><thead><tr><th>Model</th><th>F1</th><th>ROC-AUC</th><th>Brier</th><th>Precision</th><th>Recall</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="small">Actual held-out results from the last training run (${esc(body.created_at.slice(0, 10))}). ROC-AUC of 0.5 is chance; these are modest scores.</p>`;
})();
