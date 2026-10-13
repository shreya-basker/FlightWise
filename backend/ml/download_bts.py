"""Download a documented monthly BTS on-time performance archive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from backend.app.config import DATA_DIR

URL_TEMPLATE = (
    "https://www.transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)


def download(year: int, month: int, output_dir: Path = DATA_DIR / "bts") -> Path:
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"bts_{year}_{month:02}.zip"
    if destination.exists() and zipfile.is_zipfile(destination):
        return destination
    url = URL_TEMPLATE.format(year=year, month=month)
    temporary = destination.with_suffix(".part")
    if temporary.exists():
        temporary.unlink()
    urlretrieve(url, temporary)
    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError("BTS download did not produce a valid ZIP archive.")
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download one BTS on-time performance month.")
    parser.add_argument("year", type=int)
    parser.add_argument("month", type=int)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "bts")
    arguments = parser.parse_args()
    print(download(arguments.year, arguments.month, arguments.output_dir))
