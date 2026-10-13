"""Weather enrichment shared by historical training and live prediction."""

from __future__ import annotations

import time
from datetime import date, datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import airportsdata
import httpx
import pandas as pd

from backend.app.config import DATA_DIR

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Visibility is not consistently available from the historical archive. Keep it
# as optional live display context, but do not make it a required model feature.
WEATHER_FIELDS = ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"]
WEATHER_FEATURES = ["TemperatureC", "PrecipitationMM", "WindSpeedKPH", "CloudCoverPct"]
RENAMES = {
    "temperature_2m": "TemperatureC", "precipitation": "PrecipitationMM",
    "wind_speed_10m": "WindSpeedKPH", "cloud_cover": "CloudCoverPct", "visibility": "VisibilityM",
}
CACHE_DIR = DATA_DIR / "weather_cache"


@lru_cache(maxsize=1)
def airport_catalog() -> dict:
    return airportsdata.load("IATA")


def airport_coordinates(iata: str) -> tuple[float, float, str]:
    airport = airport_catalog().get(iata.upper())
    if not airport or airport.get("country") != "US":
        raise ValueError(f"{iata.upper()} is not a recognised US airport IATA code.")
    return float(airport["lat"]), float(airport["lon"]), str(airport["tz"])


def _get_json(session: httpx.Client, url: str, params: dict, attempts: int = 3) -> dict:
    """GET with a short backoff on Open-Meteo rate limiting (HTTP 429) or transient 5xx."""
    for attempt in range(attempts):
        response = session.get(url, params=params)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
            time.sleep(2 * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json()
    raise httpx.HTTPError("unreachable")


def _normalise_hourly(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("Weather provider returned no hourly data.")
    frame = pd.DataFrame(hourly)
    times = pd.to_datetime(frame.pop("time"))
    frame["FlightDate"] = times.dt.normalize()
    frame["ScheduledDepartureHour"] = times.dt.hour
    frame = frame.rename(columns=RENAMES)
    # Local-time series repeat one hour when DST ends; keep the first so the join cannot duplicate flights.
    frame = frame.drop_duplicates(["FlightDate", "ScheduledDepartureHour"])
    return frame[["FlightDate", "ScheduledDepartureHour", *WEATHER_FEATURES]]


def historical_weather(airport: str, start: date, end: date, client: httpx.Client | None = None) -> pd.DataFrame:
    """Fetch local-time historical hourly weather for one US airport (cached on disk)."""
    cache = CACHE_DIR / f"{airport.upper()}_{start.isoformat()}_{end.isoformat()}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["FlightDate"])
    latitude, longitude, timezone = airport_coordinates(airport)
    params = {"latitude": latitude, "longitude": longitude, "start_date": start.isoformat(), "end_date": end.isoformat(), "hourly": ",".join(WEATHER_FIELDS), "timezone": timezone}
    session = client or httpx.Client(timeout=30)
    try:
        frame = _normalise_hourly(_get_json(session, ARCHIVE_URL, params))
    finally:
        if client is None:
            session.close()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def enrich_historical_weather(flights: pd.DataFrame, max_airports: int = 40) -> pd.DataFrame:
    """Add weather features, limiting to high-volume origins for repeatable API usage."""
    selected = flights["Origin"].value_counts().head(max_airports).index.tolist()
    work = flights[flights["Origin"].isin(selected)].copy()
    start, end = work["FlightDate"].min().date(), work["FlightDate"].max().date()
    parts: list[pd.DataFrame] = []
    with httpx.Client(timeout=30) as client:
        for airport in selected:
            try:
                weather = historical_weather(airport, start, end, client)
                weather["Origin"] = airport
                parts.append(weather)
            except (httpx.HTTPError, ValueError) as error:
                print(f"Weather unavailable for {airport}: {error}")
    if not parts:
        raise RuntimeError("No historical weather was retrieved. Check internet access to archive-api.open-meteo.com.")
    print(f"Historical weather joined for {len(parts)} of {len(selected)} busiest origin airports ({start} to {end}).")
    joined = work.merge(pd.concat(parts, ignore_index=True), on=["Origin", "FlightDate", "ScheduledDepartureHour"], how="inner", validate="many_to_one")
    return joined.dropna(subset=WEATHER_FEATURES).reset_index(drop=True)


def live_weather(airport: str) -> dict:
    """Return current conditions using the same semantic feature names as training."""
    latitude, longitude, timezone = airport_coordinates(airport)
    params = {"latitude": latitude, "longitude": longitude, "current": ",".join([*WEATHER_FIELDS, "visibility"]), "timezone": timezone}
    with httpx.Client(timeout=15) as client:
        current = _get_json(client, FORECAST_URL, params, attempts=2).get("current", {})
    return {
        "airport": airport.upper(), "observed_at": current.get("time"), "timezone": timezone,
        "kind": "current", "source": "Open-Meteo current conditions",
        **{RENAMES[field]: current.get(field) for field in [*WEATHER_FIELDS, "visibility"]},
    }


def departure_weather(airport: str, flight_date: date, hour: int) -> dict:
    """Weather for the scheduled departure hour when Open-Meteo covers that date, else current conditions.

    Open-Meteo's forecast endpoint serves roughly the past ~3 months and the next 16 days; outside
    that window the response falls back to current conditions and says so in ``kind``/``source``.
    """
    latitude, longitude, timezone = airport_coordinates(airport)
    today = datetime.now(ZoneInfo(timezone)).date()
    if -90 <= (flight_date - today).days <= 15:
        params = {
            "latitude": latitude, "longitude": longitude, "timezone": timezone,
            "hourly": ",".join([*WEATHER_FIELDS, "visibility"]),
            "start_date": flight_date.isoformat(), "end_date": flight_date.isoformat(),
        }
        with httpx.Client(timeout=15) as client:
            response = client.get(FORECAST_URL, params=params)
        if response.status_code != 400:  # 400 = date outside the provider's window; anything else is a real failure
            response.raise_for_status()
            hourly = response.json().get("hourly") or {}
            times = hourly.get("time") or []
            target = f"{flight_date.isoformat()}T{hour:02}:00"
            if target in times:
                index = times.index(target)
                kind = "forecast" if flight_date >= today else "recent_model_data"
                return {
                    "airport": airport.upper(), "observed_at": target, "timezone": timezone, "kind": kind,
                    "source": "Open-Meteo hourly forecast for the scheduled departure hour" if kind == "forecast"
                    else "Open-Meteo hourly modelled conditions for the scheduled departure hour",
                    **{RENAMES[field]: (hourly.get(field) or [None] * len(times))[index] for field in [*WEATHER_FIELDS, "visibility"]},
                }
    weather = live_weather(airport)
    weather["source"] += " (flight date is outside the hourly forecast window, so current weather is used as a proxy)"
    weather["kind"] = "current_proxy"
    return weather
