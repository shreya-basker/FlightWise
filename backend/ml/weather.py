"""Weather enrichment shared by historical training and live prediction."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Iterable

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
CACHE_DIR = DATA_DIR / "weather_cache"


@lru_cache(maxsize=1)
def airport_catalog() -> dict:
    return airportsdata.load("IATA")


def airport_coordinates(iata: str) -> tuple[float, float, str]:
    airport = airport_catalog().get(iata.upper())
    if not airport or airport.get("country") != "US":
        raise ValueError(f"No supported US airport coordinates for {iata}.")
    return float(airport["lat"]), float(airport["lon"]), str(airport["tz"])


def _normalise_hourly(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("Weather provider returned no hourly data.")
    frame = pd.DataFrame(hourly)
    frame["FlightDate"] = pd.to_datetime(frame.pop("time")).dt.normalize()
    frame["ScheduledDepartureHour"] = pd.to_datetime(hourly["time"]).hour
    frame = frame.rename(columns={
        "temperature_2m": "TemperatureC", "precipitation": "PrecipitationMM",
        "wind_speed_10m": "WindSpeedKPH", "cloud_cover": "CloudCoverPct", "visibility": "VisibilityM",
    })
    return frame[["FlightDate", "ScheduledDepartureHour", *WEATHER_FEATURES]]


def historical_weather(airport: str, start: date, end: date, client: httpx.Client | None = None) -> pd.DataFrame:
    """Fetch local-time historical hourly weather for one US airport."""
    latitude, longitude, timezone = airport_coordinates(airport)
    params = {"latitude": latitude, "longitude": longitude, "start_date": start.isoformat(), "end_date": end.isoformat(), "hourly": ",".join(WEATHER_FIELDS), "timezone": timezone}
    session = client or httpx.Client(timeout=30)
    try:
        response = session.get(ARCHIVE_URL, params=params)
        response.raise_for_status()
        return _normalise_hourly(response.json())
    finally:
        if client is None:
            session.close()


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
        raise RuntimeError("No historical weather was retrieved.")
    joined = work.merge(pd.concat(parts, ignore_index=True), on=["Origin", "FlightDate", "ScheduledDepartureHour"], how="inner")
    return joined.dropna(subset=WEATHER_FEATURES).reset_index(drop=True)


def live_weather(airport: str) -> dict:
    """Return current/next-hour context using the same semantic feature names."""
    latitude, longitude, timezone = airport_coordinates(airport)
    params = {"latitude": latitude, "longitude": longitude, "current": ",".join(WEATHER_FIELDS), "timezone": timezone}
    response = httpx.get(FORECAST_URL, params=params, timeout=15)
    response.raise_for_status()
    current = response.json().get("current", {})
    return {
        "airport": airport.upper(), "observed_at": current.get("time"), "source": "Open-Meteo current weather",
        "TemperatureC": current.get("temperature_2m"), "PrecipitationMM": current.get("precipitation"),
        "WindSpeedKPH": current.get("wind_speed_10m"), "CloudCoverPct": current.get("cloud_cover"),
        "VisibilityM": current.get("visibility"),
    }
