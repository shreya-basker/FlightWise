"""Optional Aviationstack integration; credentials remain local environment state."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import httpx

from backend.app.config import PROJECT_DIR
from backend.ml.weather import airport_coordinates

# If your Aviationstack plan rejects HTTPS (error "https_access_restricted"), set
# AVIATIONSTACK_BASE_URL=http://api.aviationstack.com/v1/flights in .env.
DEFAULT_AVIATIONSTACK_URL = "https://api.aviationstack.com/v1/flights"
_CACHE: dict[str, tuple[float, dict]] = {}


def _load_dotenv() -> None:
    """Minimal .env support without exposing secrets through the frontend."""
    path = os.getenv("FLIGHTWISE_ENV_FILE") or str(PROJECT_DIR / ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def configured() -> bool:
    _load_dotenv()
    return bool(os.getenv("AVIATIONSTACK_API_KEY"))


def lookup_flight(flight_number: str) -> dict:
    """Return the provider's current record, reusing it for LIVE_FLIGHT_CACHE_SECONDS to protect the plan quota."""
    _load_dotenv()
    ttl = float(os.getenv("LIVE_FLIGHT_CACHE_SECONDS") or 300)
    cached = _CACHE.get(flight_number.upper())
    if cached and time.monotonic() - cached[0] < ttl:
        return cached[1]
    flight = _fetch_flight(flight_number)
    _CACHE[flight_number.upper()] = (time.monotonic(), flight)
    return flight


def _fetch_flight(flight_number: str) -> dict:
    key = os.getenv("AVIATIONSTACK_API_KEY")
    if not key:
        raise RuntimeError("Live flight lookup is not configured: API key required. Add AVIATIONSTACK_API_KEY to .env (see .env.example).")
    url = os.getenv("AVIATIONSTACK_BASE_URL") or DEFAULT_AVIATIONSTACK_URL
    try:
        response = httpx.get(url, params={"access_key": key, "flight_iata": flight_number.upper()}, timeout=20)
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        # Re-raise without chaining: the original exception text contains the request URL and therefore the key.
        raise httpx.HTTPError("Aviationstack request failed") from None
    if isinstance(payload, dict) and payload.get("error"):
        error = payload["error"]
        raise RuntimeError(f"Aviationstack error ({error.get('code', response.status_code)}): {error.get('message', 'request failed')}")
    if response.status_code >= 400:
        raise httpx.HTTPError(f"Aviationstack returned HTTP {response.status_code}")
    flights = payload.get("data", [])
    if not flights:
        raise LookupError(f"No current flight was returned for {flight_number.upper()}.")
    record = select_record(flights, flight_number)
    departure, arrival = record.get("departure") or {}, record.get("arrival") or {}
    flight, airline = record.get("flight") or {}, record.get("airline") or {}
    codeshared = flight.get("codeshared") or {}
    # A codeshare record is a partner's marketing number; the model needs the operating carrier.
    operating_number = (codeshared.get("flight_iata") or flight.get("iata") or flight_number).upper()
    operating_airline = (codeshared.get("airline_iata") or airline.get("iata") or "").upper() or None
    return {
        "provider": "Aviationstack", "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "requested_flight_number": flight_number.upper(), "flight_number": operating_number,
        "flight_date": record.get("flight_date"), "status": record.get("flight_status"),
        "airline": operating_airline,
        "airline_name": (codeshared.get("airline_name") or airline.get("name") or "").title() or None,
        "origin": departure.get("iata"), "destination": arrival.get("iata"),
        "scheduled_departure": departure.get("scheduled"), "estimated_departure": departure.get("estimated"),
        "departure_delay_minutes": departure.get("delay"), "terminal": departure.get("terminal"), "gate": departure.get("gate"),
    }


UPCOMING_STATUSES = {"scheduled", "active"}


def select_record(flights: list[dict], flight_number: str) -> dict:
    """Pick the operating flight's next/current departure from Aviationstack's mixed result list.

    Results mix several dates and partner codeshare records (e.g. BA5615 "codeshared: aa1").
    Prefer records whose own number is the requested one, then scheduled/active over landed,
    then the latest flight date.
    """
    wanted = flight_number.upper()
    exact = [r for r in flights if ((r.get("flight") or {}).get("iata") or "").upper() == wanted]
    candidates = exact or flights
    return max(candidates, key=lambda r: (r.get("flight_status") in UPCOMING_STATUSES, r.get("flight_date") or ""))


def route_distance_km(origin: str, destination: str) -> float:
    """Great-circle route distance, used only when the caller does not supply it."""
    from math import asin, cos, radians, sin, sqrt
    lat1, lon1, _ = airport_coordinates(origin)
    lat2, lon2, _ = airport_coordinates(destination)
    a = sin(radians(lat2 - lat1) / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(radians(lon2 - lon1) / 2) ** 2
    return round(6371.0088 * 2 * asin(sqrt(a)), 1)
