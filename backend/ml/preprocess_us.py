"""Create a leak-free US BTS training dataset and enrich it with weather."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from backend.app.config import DATA_DIR
from backend.ml.preprocess import _scheduled_hour
from backend.ml.weather import WEATHER_FEATURES, enrich_historical_weather

OUTPUT_PATH = DATA_DIR / "bts_flights_weather_cleaned.csv"
REQUIRED = {"FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime", "Distance", "DepDel15", "Cancelled", "Diverted"}


def load_bts(path: Path, sample_rows: int = 150_000) -> pd.DataFrame:
    """Read one BTS ZIP/CSV and take a reproducible sample after cleaning."""
    def read_header_and_chunks():
        """Select the data CSV; the official BTS ZIP also includes readme.html."""
        if path.suffix != ".zip":
            yield pd.read_csv(path, nrows=0)
            yield from pd.read_csv(
                path, usecols=list(REQUIRED), chunksize=100_000, low_memory=False
            )
            return

        with zipfile.ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(f"Expected exactly one CSV in BTS ZIP; found: {csv_names}")
            with archive.open(csv_names[0]) as file:
                yield pd.read_csv(file, nrows=0)
            with archive.open(csv_names[0]) as file:
                yield from pd.read_csv(
                    file, usecols=list(REQUIRED), chunksize=100_000, low_memory=False
                )

    reader = read_header_and_chunks()
    header = next(reader)
    missing = REQUIRED.difference(header.columns)
    if missing:
        raise ValueError(f"BTS file missing columns: {', '.join(sorted(missing))}")
    frames = []
    for chunk in reader:
        chunk = chunk[(pd.to_numeric(chunk["Cancelled"], errors="coerce") == 0) & (pd.to_numeric(chunk["Diverted"], errors="coerce") == 0)].copy()
        chunk["FlightDate"] = pd.to_datetime(chunk["FlightDate"], errors="coerce")
        chunk["Distance_KM"] = pd.to_numeric(chunk["Distance"], errors="coerce") * 1.609344
        chunk["ScheduledDepartureHour"] = chunk["CRSDepTime"].map(_scheduled_hour)
        chunk["Delayed"] = pd.to_numeric(chunk["DepDel15"], errors="coerce")
        chunk = chunk.dropna(subset=["FlightDate", "Reporting_Airline", "Origin", "Dest", "Distance_KM", "ScheduledDepartureHour", "Delayed"])
        chunk = chunk[(chunk["Distance_KM"] > 0) & (chunk["Delayed"].isin([0, 1]))]
        chunk["Airline"] = chunk["Reporting_Airline"].astype(str).str.strip()
        chunk["Origin"] = chunk["Origin"].astype(str).str.strip().str.upper()
        chunk["Dest"] = chunk["Dest"].astype(str).str.strip().str.upper()
        chunk["Month"] = chunk["FlightDate"].dt.month.astype(int)
        chunk["DayOfWeek"] = chunk["FlightDate"].dt.dayofweek.astype(int)
        frames.append(chunk[["FlightDate", "Airline", "Origin", "Dest", "Month", "DayOfWeek", "ScheduledDepartureHour", "Distance_KM", "Delayed"]])
    clean = pd.concat(frames, ignore_index=True)
    if len(clean) > sample_rows:
        clean = clean.sample(sample_rows, random_state=42)
    return clean.sort_values("FlightDate").reset_index(drop=True)


def preprocess(path: Path, output: Path = OUTPUT_PATH, sample_rows: int = 150_000, max_airports: int = 40) -> pd.DataFrame:
    flights = load_bts(path, sample_rows)
    enriched = enrich_historical_weather(flights, max_airports=max_airports)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    return enriched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare weather-enriched US BTS flights.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--sample-rows", type=int, default=150_000)
    parser.add_argument("--max-airports", type=int, default=40)
    args = parser.parse_args()
    result = preprocess(args.input, args.output, args.sample_rows, args.max_airports)
    print(f"Saved {len(result):,} weather-enriched flights to {args.output}")
