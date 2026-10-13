const $ = (id) => document.getElementById(id);
const result = $("result");
const esc = (value) => String(value ?? "–").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const pct = (value) => `${Math.round(value * 1000) / 10}%`;
const list = (items) => items && items.length ? `<ul>${items.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : "";

function setResult(html) { result.className = "result"; result.innerHTML = html; }

function errorHtml(status, detail) {
  const text = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ") : JSON.stringify(detail);
  return `<p class="status">REQUEST UNAVAILABLE · HTTP ${esc(status)}</p><p>${esc(text)}</p>`;
}

function weatherHtml(w) {
  return `<div class="facts">
    <p class="status">WEATHER AT ${esc(w.airport)} · ${esc(w.observed_at)} (${esc(w.timezone)})</p>
    <p>${esc(w.TemperatureC)}°C · wind ${esc(w.WindSpeedKPH)} km/h · precipitation ${esc(w.PrecipitationMM)} mm · cloud ${esc(w.CloudCoverPct)}%${w.VisibilityM != null ? ` · visibility ${esc(Math.round(w.VisibilityM))} m` : ""}</p>
    <p class="small">Source: ${esc(w.source)}</p></div>`;
}

function riskHtml(data, heading) {
  const label = data.prediction === "likely_delayed" ? "Label: likely delayed" : "Label: likely on time";
  const versus = data.base_rate == null ? "" : `${data.delay_probability >= data.base_rate ? "above" : "below"} the ${pct(data.base_rate)} training-data average`;
  return `<p class="status">${esc(heading || "HISTORICAL-PATTERN ESTIMATE")} · ${esc(data.model)}</p>
    <p class="risk">${pct(data.delay_probability)} <span>estimated chance of departing ≥15 min late</span></p>
    <p><strong>${label}</strong> (flag threshold ${pct(data.threshold)}; ${versus}) · ${esc(data.distance_km)} km (${esc(data.distance_source)})</p>
    ${data.warnings && data.warnings.length ? `<div class="warn">${list(data.warnings)}</div>` : ""}
    ${weatherHtml(data.weather_context || {})}
    <details><summary>Data limitations</summary>${list(data.limitations)}</details>
    <p class="small">${esc(data.disclaimer)}</p>`;
}

function flightHtml(f) {
  return `<div class="facts"><p class="status">LIVE FLIGHT RECORD · ${esc(f.provider)} · fetched ${esc(new Date(f.retrieved_at).toLocaleString())}</p>
    <p><strong>${esc(f.flight_number)}</strong> ${esc(f.airline_name || f.airline)} · ${esc(f.origin)} → ${esc(f.destination)} · status: <strong>${esc(f.status || "unavailable")}</strong></p>
    <p class="small">Scheduled ${esc(f.scheduled_departure)} · estimated ${esc(f.estimated_departure)} · provider-reported delay ${esc(f.departure_delay_minutes)} min · terminal ${esc(f.terminal)} · gate ${esc(f.gate)}</p>
    <p class="small">The status above is the provider's; the percentage below is FlightWise's separate historical-pattern estimate.</p></div>`;
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
  setResult("<p>Fetching departure-airport weather and estimating risk…</p>");
  const distance = $("distance").value;
  const payload = {airline: $("airline").value.trim(), origin: $("origin").value.trim(), destination: $("destination").value.trim(), flight_date: $("flight_date").value, scheduled_departure_hour: Number($("hour").value), distance_km: distance ? Number(distance) : null};
  const {ok, status, body} = await call("/api/us/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  setResult(ok ? riskHtml(body) : errorHtml(status, body.detail));
});

$("lookup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const flight = $("flight-number").value.trim();
  if (!flight) return;
  setResult("<p>Looking up flight and weather…</p>");
  const {ok, status, body} = await call(`/api/live/flight/${encodeURIComponent(flight)}`);
  if (!ok) { setResult(errorHtml(status, body.detail)); return; }
  setResult(flightHtml(body.flight) + (body.risk ? riskHtml(body.risk, `FLIGHT ${body.flight.flight_number}`) : `<p>${esc(body.risk_unavailable_reason)}</p>`));
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
