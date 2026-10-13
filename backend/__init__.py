"""FlightWise backend package."""

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        f"FlightWise needs Python 3.11 or newer (found {sys.version.split()[0]}). "
        "On macOS: brew install python@3.12, then recreate .venv with python3.12 -m venv .venv"
    )
