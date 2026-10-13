"""Download a documented monthly BTS on-time performance archive."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import httpx

from backend.app.config import DATA_DIR

URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)


def is_valid_bts_zip(path: Path) -> bool:
    """A usable archive opens, passes CRC checks, and contains a data CSV (readme.html is expected too)."""
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            has_csv = any(name.lower().endswith(".csv") for name in archive.namelist())
            return has_csv and archive.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def download(year: int, month: int, output_dir: Path = DATA_DIR / "bts") -> Path:
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    if not 1987 <= year <= 2100:
        raise ValueError("BTS on-time data starts in 1987")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"bts_{year}_{month:02}.zip"
    if destination.exists():
        if is_valid_bts_zip(destination):
            return destination
        print(f"Existing {destination.name} is incomplete or corrupt; downloading again.", file=sys.stderr)
        destination.unlink()
    url = URL_TEMPLATE.format(year=year, month=month)
    temporary = destination.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    # BTS can be slow (tens of KB/s); the read timeout guards against a stalled connection.
    with httpx.stream("GET", url, timeout=httpx.Timeout(60.0, connect=30.0), follow_redirects=True) as response:
        if response.status_code == 404:
            raise ValueError(f"BTS has no published file for {year}-{month:02} ({url}).")
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        received, last_report = 0, -1
        with temporary.open("wb") as file:
            for block in response.iter_bytes(1 << 16):
                file.write(block)
                received += len(block)
                percent = received * 100 // total if total else -1
                if total and percent // 10 != last_report:
                    last_report = percent // 10
                    print(f"  {received / 1e6:.1f} / {total / 1e6:.1f} MB", file=sys.stderr)
    if not is_valid_bts_zip(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError("BTS download did not produce a valid ZIP archive containing a CSV.")
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download one BTS on-time performance month.")
    parser.add_argument("year", type=int)
    parser.add_argument("month", type=int)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "bts")
    arguments = parser.parse_args()
    print(download(arguments.year, arguments.month, arguments.output_dir))
