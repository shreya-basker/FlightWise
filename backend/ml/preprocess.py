"""Clean the supplied educational sample without using post-departure features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backend.app.config import CLEANED_DATA_PATH, RAW_DATA_PATH
from backend.ml.schema import FEATURE_COLUMNS, TARGET_COLUMN

REQUIRED_COLUMNS = {
    "FlightDate", "Airline", "Origin", "Dest", "CRSDepTime", "DepDelay",
    "Cancelled", "Distance_KM",
}


def _scheduled_hour(value: object) -> float:
    """Return an hour from either HH:MM or a numeric HHMM value."""
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    if not pd.isna(parsed):
        return float(parsed.hour)
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return float("nan")
    number = int(digits)
    hour, minute = divmod(number, 100)
    return float(hour) if 0 <= hour <= 23 and 0 <= minute <= 59 else float("nan")


def preprocess_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    df = raw.copy()
    df["Cancelled"] = pd.to_numeric(df["Cancelled"], errors="coerce")
    df["DepDelay"] = pd.to_numeric(df["DepDelay"], errors="coerce")
    df["Distance_KM"] = pd.to_numeric(df["Distance_KM"], errors="coerce")
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    for column in ("Airline", "Origin", "Dest"):
        df[column] = df[column].astype("string").str.strip()
    for column in ("Origin", "Dest"):
        df[column] = df[column].str.upper().str.extract(r"^([A-Z]{3})$", expand=False)

    df["ScheduledDepartureHour"] = df["CRSDepTime"].map(_scheduled_hour)
    df = df[(df["Cancelled"] == 0) & (df["Distance_KM"] > 0)].copy()
    # Month and DayOfWeek are derived below, so validate only the source fields here.
    df = df.dropna(
        subset=[
            "FlightDate", "DepDelay", "Airline", "Origin", "Dest",
            "ScheduledDepartureHour", "Distance_KM",
        ]
    )

    # Monday=0 through Sunday=6; this derives from the date so inconsistent source labels
    # cannot silently become a training feature.
    df["Month"] = df["FlightDate"].dt.month.astype(int)
    df["DayOfWeek"] = df["FlightDate"].dt.dayofweek.astype(int)
    df["ScheduledDepartureHour"] = df["ScheduledDepartureHour"].astype(int)
    df[TARGET_COLUMN] = (df["DepDelay"] >= 15).astype(int)

    cleaned = df[["FlightDate", *FEATURE_COLUMNS, TARGET_COLUMN]].sort_values("FlightDate")
    return cleaned.reset_index(drop=True)


def preprocess_file(input_path: Path = RAW_DATA_PATH, output_path: Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    cleaned = preprocess_dataframe(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FlightWise training data.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=CLEANED_DATA_PATH)
    args = parser.parse_args()
    cleaned = preprocess_file(args.input, args.output)
    print(f"Cleaned records: {len(cleaned)}")
    print(cleaned[TARGET_COLUMN].value_counts().sort_index().to_string())
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
