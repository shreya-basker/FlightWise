"""Optional Aviationstack integration; credentials remain local environment state."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx

from backend.ml.weather import airport_coordinates

AVIATIONSTACK_URL = "https://api.aviationstack.com/v1/flights"


def _load_dotenv() -> None:
    """Minimal .env support without exposing secrets through the frontend."""
    path = os.getenv("FLIGHTWISE_ENV_FILE", ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.strip().split("=", 1)
            os.environ.setdefault(key, value)


def configured() -> bool:
    _load_dotenv()
    return bool(os.getenv("AVIATIONSTACK_API_KEY"))


def lookup_flight(flight_number: str) -> dict:
    _load_dotenv()
    key = os.getenv("AVIATIONSTACK_API_KEY")
    if not key:
        raise RuntimeError("Live flight lookup is not configured. Add AVIATIONSTACK_API_KEY to .env.")
    response = httpx.get(AVIATIONSTACK_URL, params={"access_key": key, "flight_iata": flight_number.upper()}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"].get("message", "Flight provider request failed."))
    flights = payload.get("data", [])
    if not flights:
        raise LookupError("No current flight was returned for that flight number.")
    record = flights[0]
    departure, arrival = record.get("departure") or {}, record.get("arrival") or {}
    return {
        "provider": "Aviationstack", "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "flight_number": (record.get("flight") or {}).get("iata") or flight_number.upper(),
        "status": record.get("flight_status"), "airline": (record.get("airline") or {}).get("iata"),
        "origin": departure.get("iata"), "destination": arrival.get("iata"),
        "scheduled_departure": departure.get("scheduled"), "estimated_departure": departure.get("estimated"),
        "departure_delay_minutes": departure.get("delay"), "terminal": departure.get("terminal"), "gate": departure.get("gate"),
    }


def route_distance_km(origin: str, destination: str) -> float:
    """Great-circle route distance, used only when the provider does not supply it."""
    from math import asin, cos, radians, sin, sqrt
    lat1, lon1, _ = airport_coordinates(origin)
    lat2, lon2, _ = airport_coordinates(destination)
    a = sin(radians(lat2 - lat1) / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(radians(lon2 - lon1) / 2) ** 2
    return round(6371.0088 * 2 * asin(sqrt(a)), 1)
