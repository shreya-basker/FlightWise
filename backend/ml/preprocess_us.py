"""Create a leak-free US BTS training dataset and enrich it with weather."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import IO, Iterator

import pandas as pd

from backend.app.config import DATA_DIR
from backend.ml.weather import enrich_historical_weather

OUTPUT_PATH = DATA_DIR / "bts_flights_weather_cleaned.csv"
# Only pre-departure inputs plus the documented outcome/filters are ever read. Actual
# departure/arrival times, ArrDelay and the post-flight delay-cause columns never load.
REQUIRED = ["FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime", "Distance", "DepDel15", "Cancelled", "Diverted"]
OUTPUT_COLUMNS = ["FlightDate", "Airline", "Origin", "Dest", "Month", "DayOfWeek", "ScheduledDepartureHour", "Distance_KM", "Delayed"]
MILES_TO_KM = 1.609344
CHUNK_ROWS = 100_000


def select_csv_member(archive: zipfile.ZipFile) -> str:
    """Return the flight-data CSV inside a BTS ZIP; the official archive also ships readme.html."""
    csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]
    if len(csv_names) == 1:
        return csv_names[0]
    on_time = [name for name in csv_names if "on_time" in name.lower()]
    if len(on_time) == 1:
        return on_time[0]
    raise ValueError(f"Expected exactly one BTS CSV in {archive.filename}; found {csv_names or 'none'} among {archive.namelist()}")


def _validate_header(handle: IO[bytes] | Path, source: str) -> None:
    header = pd.read_csv(handle, nrows=0)
    missing = set(REQUIRED).difference(header.columns)
    if missing:
        raise ValueError(f"BTS file missing columns in {source}: {', '.join(sorted(missing))}")


def _read_chunks(handle: IO[bytes] | Path) -> Iterator[pd.DataFrame]:
    return pd.read_csv(handle, usecols=REQUIRED, chunksize=CHUNK_ROWS, dtype={"Reporting_Airline": str, "Origin": str, "Dest": str}, low_memory=False)


def scheduled_hours(values: pd.Series) -> pd.Series:
    """Vectorised HHMM -> hour; malformed or 2400-style values become NaN."""
    hhmm = pd.to_numeric(values, errors="coerce")
    hour, minute = hhmm // 100, hhmm % 100
    return hour.where((hour >= 0) & (hour <= 23) & (minute >= 0) & (minute <= 59))


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Remove cancelled/diverted flights and derive the model's pre-departure features."""
    keep = (pd.to_numeric(chunk["Cancelled"], errors="coerce") == 0) & (pd.to_numeric(chunk["Diverted"], errors="coerce") == 0)
    chunk = chunk[keep].copy()
    chunk["FlightDate"] = pd.to_datetime(chunk["FlightDate"], errors="coerce")
    chunk["Distance_KM"] = pd.to_numeric(chunk["Distance"], errors="coerce") * MILES_TO_KM
    chunk["ScheduledDepartureHour"] = scheduled_hours(chunk["CRSDepTime"])
    chunk["Delayed"] = pd.to_numeric(chunk["DepDel15"], errors="coerce")
    chunk["Airline"] = chunk["Reporting_Airline"].str.strip().str.upper()
    chunk["Origin"] = chunk["Origin"].str.strip().str.upper()
    chunk["Dest"] = chunk["Dest"].str.strip().str.upper()
    chunk = chunk.dropna(subset=["FlightDate", "Airline", "Origin", "Dest", "Distance_KM", "ScheduledDepartureHour", "Delayed"])
    chunk = chunk[(chunk["Distance_KM"] > 0) & chunk["Delayed"].isin([0, 1])].copy()
    chunk["Month"] = chunk["FlightDate"].dt.month.astype(int)
    chunk["DayOfWeek"] = chunk["FlightDate"].dt.dayofweek.astype(int)
    chunk["ScheduledDepartureHour"] = chunk["ScheduledDepartureHour"].astype(int)
    chunk["Delayed"] = chunk["Delayed"].astype(int)
    chunk["Distance_KM"] = chunk["Distance_KM"].round(1)
    return chunk[OUTPUT_COLUMNS]


def _load_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".zip":
        _validate_header(path, str(path))
        return pd.concat([clean_chunk(chunk) for chunk in _read_chunks(path)], ignore_index=True)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a valid ZIP archive. Delete it and re-run backend.ml.download_bts.")
    # The archive stays open for the whole chunk iteration; each member handle is a fresh stream.
    with zipfile.ZipFile(path) as archive:
        csv_name = select_csv_member(archive)
        with archive.open(csv_name) as handle:
            _validate_header(handle, f"{path}:{csv_name}")
        with archive.open(csv_name) as handle:
            return pd.concat([clean_chunk(chunk) for chunk in _read_chunks(handle)], ignore_index=True)


def load_bts(paths: Path | list[Path], sample_rows: int = 150_000) -> pd.DataFrame:
    """Read one or more BTS ZIP/CSV files and take a reproducible sample after cleaning."""
    paths = [paths] if isinstance(paths, Path) else list(paths)
    clean = pd.concat([_load_one(Path(path)) for path in paths], ignore_index=True)
    if clean.empty:
        raise ValueError("No usable (non-cancelled, non-diverted) flights were found.")
    if sample_rows and len(clean) > sample_rows:
        clean = clean.sample(sample_rows, random_state=42)
    return clean.sort_values(["FlightDate", "Origin", "ScheduledDepartureHour"], kind="stable").reset_index(drop=True)


def preprocess(paths: Path | list[Path], output: Path = OUTPUT_PATH, sample_rows: int = 150_000, max_airports: int = 40) -> pd.DataFrame:
    flights = load_bts(paths, sample_rows)
    print(f"Loaded {len(flights):,} clean BTS flights ({flights['FlightDate'].min().date()} to {flights['FlightDate'].max().date()}); delay rate {flights['Delayed'].mean():.3f}")
    enriched = enrich_historical_weather(flights, max_airports=max_airports)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    return enriched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare weather-enriched US BTS flights.")
    parser.add_argument("inputs", type=Path, nargs="+", help="One or more BTS monthly ZIP (or extracted CSV) files")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--sample-rows", type=int, default=150_000, help="0 keeps every clean flight")
    parser.add_argument("--max-airports", type=int, default=40)
    args = parser.parse_args()
    result = preprocess(args.inputs, args.output, args.sample_rows, args.max_airports)
    print(f"Saved {len(result):,} weather-enriched flights to {args.output}")
