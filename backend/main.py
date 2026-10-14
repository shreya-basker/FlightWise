"""FastAPI service for FlightWise's local delay-risk prototype."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from functools import lru_cache

import joblib
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend.app.config import FRONTEND_DIR, METRICS_PATH, MODEL_PATH, US_METRICS_PATH, US_MODEL_PATH
from backend.app.live_flights import configured as live_configured, departure_outcome, lookup_flight, route_distance_km
from backend.ml.train_us import US_FEATURES
from backend.ml.weather import WEATHER_FEATURES, departure_weather
from backend.ml.schema import FEATURE_COLUMNS

app = FastAPI(title="FlightWise API", version="1.0.0", description="Educational flight-delay risk prototype.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class PredictionRequest(BaseModel):
    airline: str = Field(min_length=2, max_length=80)
    origin: str = Field(pattern=r"^[A-Za-z]{3}$")
    destination: str = Field(pattern=r"^[A-Za-z]{3}$")
    flight_date: date
    scheduled_departure_hour: int = Field(ge=0, le=23)
    distance_km: float = Field(gt=0, le=20000)

    @field_validator("airline")
    @classmethod
    def normalise_airline(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("origin", "destination")
    @classmethod
    def normalise_airport(cls, value: str) -> str:
        return value.upper()


def _not_ready() -> HTTPException:
    return HTTPException(status_code=503, detail="Model artifacts are missing. Run the preprocessing and training commands first.")


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[object, dict]:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise _not_ready()
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_us_artifacts() -> tuple[object, dict]:
    if not US_MODEL_PATH.exists() or not US_METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="US weather-aware model is not ready. Run the BTS preprocessing and training commands first.")
    return joblib.load(US_MODEL_PATH), json.loads(US_METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "us_model_ready": US_MODEL_PATH.exists() and US_METRICS_PATH.exists(),
        "model_ready": MODEL_PATH.exists() and METRICS_PATH.exists(),  # legacy India prototype
    }


@app.get("/api/options")
def options() -> dict:
    _, metadata = load_artifacts()
    data = metadata["training_data"]
    return {key: data[key] for key in ("airlines", "origins", "destinations", "route_counts", "distance_by_route")}


@app.get("/api/model-info")
def model_info() -> dict:
    _, metadata = load_artifacts()
    return metadata


class USPredictionRequest(BaseModel):
    airline: str = Field(min_length=2, max_length=8)
    origin: str = Field(pattern=r"^[A-Za-z]{3}$")
    destination: str = Field(pattern=r"^[A-Za-z]{3}$")
    flight_date: date
    scheduled_departure_hour: int = Field(ge=0, le=23)
    distance_km: float | None = Field(default=None, gt=0, le=20000)

    @field_validator("airline")
    @classmethod
    def normalise_code(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("origin", "destination")
    @classmethod
    def upper_airports(cls, value: str) -> str:
        return value.upper()


DISCLAIMER = (
    "Historical-pattern estimate from US BTS flights and departure-airport weather. It is not a flight status, "
    "a guarantee, or a travel recommendation; air-traffic control, aircraft rotation and crew issues are not modelled."
)


def us_prediction(request: USPredictionRequest, weather: dict) -> dict:
    model, metadata = load_us_artifacts()
    if any(weather.get(field) is None for field in WEATHER_FEATURES):
        raise HTTPException(status_code=502, detail="Weather provider did not return all required conditions (temperature, precipitation, wind, cloud cover).")
    distance = request.distance_km or route_distance_km(request.origin, request.destination)
    row = pd.DataFrame([{
        "Airline": request.airline, "Origin": request.origin, "Dest": request.destination,
        "Month": request.flight_date.month, "DayOfWeek": request.flight_date.weekday(),
        "ScheduledDepartureHour": request.scheduled_departure_hour, "Distance_KM": distance,
        **{field: weather[field] for field in WEATHER_FEATURES},
    }])[US_FEATURES]
    probability = float(model.predict_proba(row)[0][1])
    threshold = float(metadata.get("decision_threshold", 0.5))
    support = metadata["training_data"]
    warnings = [message for present, message in (
        (request.airline in support["airlines"], f"Airline {request.airline} is not in the training data; it is treated as unknown."),
        (request.origin in support["origins"], f"Origin {request.origin} is not one of the trained departure airports; it is treated as unknown."),
        (request.destination in support["destinations"], f"Destination {request.destination} is not in the training data; it is treated as unknown."),
        (f"{request.flight_date.month:02}" in {m[-2:] for m in support.get("months", [])}, "The flight month is outside the trained months, so seasonal effects are not learned."),
    ) if not present]
    return {
        "prediction": "likely_delayed" if probability >= threshold else "likely_on_time",
        "delay_probability": round(probability, 4), "threshold": threshold,
        "base_rate": support.get("delay_rate"), "distance_km": round(float(distance), 1),
        "distance_source": "provided" if request.distance_km else "great-circle from airport coordinates",
        "weather_context": weather, "model": metadata["selected_model"], "warnings": warnings,
        "limitations": metadata.get("limitations", []), "disclaimer": DISCLAIMER,
    }


def _route_checks(request: USPredictionRequest) -> None:
    if request.origin == request.destination:
        raise HTTPException(status_code=422, detail="Origin and destination must differ.")
    route_distance_km(request.origin, request.destination)  # raises ValueError for non-US/unknown codes


@app.get("/api/us/model-info")
def us_model_info() -> dict:
    _, metadata = load_us_artifacts()
    return metadata


@app.post("/api/us/predict")
def predict_us(request: USPredictionRequest) -> dict:
    # Fail clearly before spending a weather-provider request when training is incomplete or input is invalid.
    load_us_artifacts()
    try:
        _route_checks(request)
        weather = departure_weather(request.origin, request.flight_date, request.scheduled_departure_hour)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Weather provider (Open-Meteo) is unavailable: {type(error).__name__}. No estimate was made.") from error
    return us_prediction(request, weather)


@app.get("/api/live/config")
def live_configuration() -> dict:
    return {"flight_lookup_configured": live_configured(), "weather_provider": "Open-Meteo"}


@app.get("/api/live/flight/{flight_number}")
def live_flight(flight_number: str) -> dict:
    load_us_artifacts()
    if not re.fullmatch(r"[A-Za-z0-9]{2}\d{1,4}", flight_number.strip()):
        raise HTTPException(status_code=422, detail="Use an IATA flight number such as AA100.")
    try:
        flight = lookup_flight(flight_number.strip())
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="Live flight provider (Aviationstack) request failed.") from error
    if not all([flight.get("airline"), flight.get("origin"), flight.get("destination"), flight.get("scheduled_departure")]):
        raise HTTPException(status_code=502, detail="Live provider response lacks the fields needed for a risk estimate.")
    # Aviationstack reports airport-local wall-clock time (labelled +00:00), which matches BTS CRSDepTime.
    scheduled = datetime.fromisoformat(flight["scheduled_departure"].replace("Z", "+00:00"))
    try:
        request = USPredictionRequest(airline=flight["airline"], origin=flight["origin"], destination=flight["destination"], flight_date=scheduled.date(), scheduled_departure_hour=scheduled.hour)
        _route_checks(request)
        weather = departure_weather(request.origin, request.flight_date, request.scheduled_departure_hour)
    except ValueError as error:  # includes pydantic ValidationError, e.g. a non-US route
        return {"flight": flight, "risk": None, "outcome": departure_outcome(flight, None), "risk_unavailable_reason": f"No US-domestic estimate for this flight: {error}"}
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="Weather provider (Open-Meteo) is unavailable.") from error
    risk = us_prediction(request, weather)
    # For departed flights the estimate still uses only pre-departure inputs; the outcome is shown for comparison.
    return {"flight": flight, "risk": risk, "outcome": departure_outcome(flight, risk)}


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    model, metadata = load_artifacts()
    support = metadata["training_data"]
    route = f"{request.origin}-{request.destination}"
    unsupported: list[str] = []
    if request.airline not in support["airlines"]:
        unsupported.append("airline is not represented in the training sample")
    if request.origin not in support["origins"]:
        unsupported.append("origin airport is not represented in the training sample")
    if request.destination not in support["destinations"]:
        unsupported.append("destination airport is not represented in the training sample")
    if route not in support["route_counts"]:
        unsupported.append("exact route is not represented in the training sample")
    if f"{request.airline}|{route}" not in support["airline_route_counts"]:
        unsupported.append("airline-and-route combination is not represented in the training sample")
    if request.origin == request.destination:
        raise HTTPException(status_code=422, detail="Origin and destination must be different airports.")
    if unsupported:
        raise HTTPException(status_code=422, detail={"message": "Prediction unavailable for unsupported input.", "reasons": unsupported})

    row = pd.DataFrame([{
        "Airline": request.airline, "Origin": request.origin, "Dest": request.destination,
        "Month": request.flight_date.month, "DayOfWeek": request.flight_date.weekday(),
        "ScheduledDepartureHour": request.scheduled_departure_hour, "Distance_KM": request.distance_km,
    }])[FEATURE_COLUMNS]
    probability = float(model.predict_proba(row)[0][1])
    observed = int(support["route_counts"][route])
    evidence = "very limited" if observed < 3 else "limited"
    return {
        "prediction": "likely_delayed" if probability >= 0.5 else "likely_on_time",
        "delay_probability": round(probability, 4),
        "threshold": 0.5,
        "route_observations": observed,
        "evidence_level": evidence,
        "disclaimer": "Historical-pattern estimate only; it does not use live weather, aircraft rotation, or airport disruption data and is not a flight-status guarantee.",
    }
