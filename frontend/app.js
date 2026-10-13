const $ = (id) => document.getElementById(id);
const result = $("result");

function show(data, prefix = "") {
  result.className = "result";
  if (data.risk) { show(data.risk, `Flight ${data.flight.flight_number} · ${data.flight.status || "status unavailable"}`); return; }
  if (data.detail) { result.innerHTML = `<p class="status">REQUEST UNAVAILABLE</p><p>${typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)}</p>`; return; }
  const weather = data.weather_context || {};
  result.innerHTML = `<p class="status">${prefix || data.model || "WEATHER-AWARE ESTIMATE"}</p><p class="risk">${Math.round(data.delay_probability * 100)}% delay risk</p><p>${data.prediction.replaceAll("_", " ")} · ${data.distance_km} km</p><p>Weather at ${weather.airport || "origin"}: ${weather.TemperatureC ?? "–"}°C, wind ${weather.WindSpeedKPH ?? "–"} km/h, precipitation ${weather.PrecipitationMM ?? "–"} mm, visibility ${weather.VisibilityM ?? "–"} m.</p><p>${data.disclaimer}</p>`;
}
async function request(url, options) { const response = await fetch(url, options); const body = await response.json(); show(body); return body; }

$("flight_date").value = new Date().toISOString().slice(0, 10);
$("prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault(); result.className = "result"; result.textContent = "Fetching weather and estimating risk…";
  const distance = $("distance").value;
  await request("/api/us/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({airline: $("airline").value, origin: $("origin").value, destination: $("destination").value, flight_date: $("flight_date").value, scheduled_departure_hour: Number($("hour").value), distance_km: distance ? Number(distance) : null})});
});
$("lookup-form").addEventListener("submit", async (event) => { event.preventDefault(); const flight = $("flight-number").value.trim(); if (!flight) return; result.className = "result"; result.textContent = "Looking up flight and weather…"; await request(`/api/live/flight/${encodeURIComponent(flight)}`); });
(async () => { const response = await fetch("/api/live/config"); const state = await response.json(); $("live-status").textContent = state.flight_lookup_configured ? "LIVE CONNECTED" : "API KEY REQUIRED"; })();
