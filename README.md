# FlightWise — US domestic flight delay-risk estimate

FlightWise is an educational full-stack app that estimates whether a **US domestic** flight is likely to **depart 15+ minutes late** (BTS `DepDel15`). It uses only information available before departure: carrier, origin, destination, flight date, scheduled departure hour, route distance, and departure-airport weather.

> **This is a historical-pattern estimate, not a flight status, a guarantee, or a travel recommendation.** The probability comes from a model trained on past BTS flights. Weather is the only live input; air-traffic restrictions, aircraft rotation and crew issues are not modelled.

```text
Browser (frontend/)  →  FastAPI (backend/main.py)
                          ├── POST /api/us/predict        US model + Open-Meteo weather
                          ├── GET  /api/us/model-info     real training/evaluation metadata
                          ├── GET  /api/live/config       is flight lookup configured? (never returns the key)
                          ├── GET  /api/live/flight/{no}  optional Aviationstack lookup + estimate
                          └── legacy India prototype: /api/predict, /api/options, /api/model-info
ML pipeline (backend/ml/)
  download_bts.py → preprocess_us.py (+ weather.py) → train_us.py → backend/models/us_*
```

## Run it (macOS / VS Code)

**Requires Python 3.11+.** The `python3` that ships with macOS/Xcode Command Line Tools is 3.9 and will not work (current scikit-learn and airportsdata need 3.11+). Check with `python3 --version`. If it is older, install a newer Python and use it for the venv:

```bash
brew install python@3.12
python3.12 -m venv .venv
```

(Alternatively, use the python.org installer.) A `.venv` copied from Windows will not work on a Mac, so always create it on the Mac. The downloaded data and trained model are git-ignored, so run the full pipeline on the Mac as well.

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 -m backend.ml.download_bts 2024 1
python3 -m backend.ml.preprocess_us backend/data/bts/bts_2024_01.zip
python3 -m backend.ml.train_us
python3 -m pytest -q
python3 -m uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000> (API docs at <http://127.0.0.1:8000/docs>).

**Windows (PowerShell):** use `py -3.12 -m venv .venv` and `.\.venv\Scripts\Activate.ps1`, then run the same commands with `python` instead of `python3`. Use a python.org/Microsoft Store CPython, not MSYS2's Python, because MSYS2 Python cannot install the PyPI wheels for pandas/scikit-learn.

What each step does:

| Step | Output | Notes |
|---|---|---|
| `download_bts 2024 1` | `backend/data/bts/bts_2024_01.zip` (≈27.6 MB) | The BTS server can be slow (this run took ≈10 min). Existing archives are reused if they pass a CRC check; a corrupt or partial file is replaced. |
| `preprocess_us …zip` | `backend/data/bts_flights_weather_cleaned.csv` | Reads the CSV member inside the ZIP. It accepts several ZIPs, e.g. `…bts_2024_01.zip …bts_2024_02.zip`. Options: `--sample-rows` (default 150 000, `0` = all) and `--max-airports` (default 40). Weather is cached under `backend/data/weather_cache/`. |
| `train_us` | `backend/models/us_flight_delay_model.joblib`, `backend/models/us_metrics.json` | Prints the real held-out metrics. |

### Optional: live flight-number lookup

Open-Meteo weather needs no key. Flight-number lookup needs a personal [Aviationstack](https://aviationstack.com) key:

```bash
cp .env.example .env
# edit .env:  AVIATIONSTACK_API_KEY=your_key
```

The key is read only on the server, is never sent to the browser, and `.env` is git-ignored. Without a key, `/api/live/flight/{no}` returns `503` with "API key required", and the UI shows **API KEY REQUIRED** and disables the lookup button. If your plan rejects HTTPS (`https_access_restricted`), also set `AVIATIONSTACK_BASE_URL=http://api.aviationstack.com/v1/flights`. Lookups are cached for `LIVE_FLIGHT_CACHE_SECONDS` (default 300) to save quota. Aviationstack mixes several dates and partner codeshare records (e.g. BA5615 for AA1). FlightWise picks the operating flight's scheduled/active departure, and if only a codeshare number was entered it uses the operating carrier.

## Data pipeline

**Source.** The US DOT BTS *Reporting Carrier On-Time Performance (1987–present)* monthly PREZIP files. Each ZIP contains the data CSV **and `readme.html`**.

**Preprocessing (`backend/ml/preprocess_us.py`)**
- Opens the ZIP with `zipfile.ZipFile`, selects the single `.csv` member, and streams the header and 100k-row chunks through `archive.open(csv_name)`, keeping the archive open throughout. It does not use `pd.read_csv(zip)`, which failed with *"Multiple files found in ZIP file"*.
- Reads only `FlightDate, Reporting_Airline, Origin, Dest, CRSDepTime, Distance, DepDel15, Cancelled, Diverted`. Leakage columns such as actual departure/arrival times, `ArrDelay`, and the carrier/weather/NAS delay-cause columns are never loaded.
- Removes cancelled and diverted flights, and uses `DepDel15` as the target.
- Converts miles to km (× 1.609344) and derives month, day of week (Mon = 0), and scheduled local hour from `CRSDepTime` (HHMM; invalid values such as 2400 are dropped).
- Takes a deterministic sample (`random_state=42`).
- Joins hourly Open-Meteo **historical** weather (temperature, precipitation, wind, cloud cover) at the origin airport, in local time, for the busiest origin airports.

**Training (`backend/ml/train_us.py`)**
- Splits the data into three date-separated chronological periods: **fit → threshold validation → test**. No flight date appears in more than one period.
- Compares weather-aware **Logistic Regression** (one-hot + scaling) and **Histogram Gradient Boosting** (native categorical handling).
- Tunes each model's decision threshold for F1 on the validation dates only, then scores once on the untouched test dates.
- Selects the model by test F1, with ROC-AUC as the tie-breaker.
- Stores accuracy, precision, recall, F1, ROC-AUC, Brier score, 10-bin calibration, the periods, and the support lists in `us_metrics.json`.

## Actual results (run on 2026-10-13, BTS January 2024)

Preprocessing: 150,000 sampled clean flights (delay rate 23.2%). Weather was retrieved for all 40 busiest origins, giving **109,682** weather-enriched flights.

| Period | Dates | Flights | Delay rate |
|---|---|---|---|
| Fit | 2024-01-01 → 01-18 | 63,380 | 27.6% |
| Threshold validation | 2024-01-19 → 01-23 | 17,767 | 24.6% |
| Test | 2024-01-24 → 01-31 | 28,535 | 16.1% |

| Model (test set) | Threshold | F1 | ROC-AUC | Brier | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.19 | 0.3026 | 0.6001 | **0.1389** | 0.1995 | 0.6262 | 0.5348 |
| **Hist Gradient Boosting (selected)** | 0.25 | **0.3159** | **0.6281** | 0.1448 | 0.2251 | 0.5295 | 0.6303 |

How to read these numbers:
- The discrimination is modest: ROC-AUC 0.63, where 0.5 is chance.
- The test week had far fewer delays (16%) than the storm-affected fit period (28%). Mean predicted probability on test was 0.235 against 0.161 observed, so probabilities run high under this shift (see `calibration_bins`).
- With only January in the data, the `Month` feature carries no signal. Requests for other months get a warning.

Your numbers will match exactly only with the same input file, sample size and Open-Meteo data. Always use `GET /api/us/model-info` as the source of truth.

## API

`POST /api/us/predict`

```json
{"airline": "AA", "origin": "ATL", "destination": "LAX", "flight_date": "2026-10-14", "scheduled_departure_hour": 9, "distance_km": null}
```

The response includes:
- `prediction` and `delay_probability`; `threshold` and `base_rate`.
- `distance_km` and `distance_source` (great-circle if omitted).
- `weather_context`, with `source` and `kind`:
  - `forecast`: Open-Meteo hourly forecast for the scheduled hour, when the date is within about 16 days.
  - `recent_model_data`: for recent past dates.
  - `current_proxy`: current conditions, clearly labelled, when the date is outside that window.
- `model`, `warnings` (unseen airline/airport/month), `limitations`, and `disclaimer`.

| Situation | Response |
|---|---|
| US model not trained | `503` before any weather/provider call |
| Unknown or non-US airport, same origin/destination, hour outside 0–23 | `422` with the reason |
| Open-Meteo unreachable or incomplete | `502`, and no estimate is made |
| No Aviationstack key | `503` "API key required…" |
| Flight not found by provider | `404` |
| Provider flight is not US-domestic | `200` with flight data, `risk: null` and a reason |

`GET /api/health` returns `{status, us_model_ready, model_ready}` (`model_ready` refers to the legacy India prototype).

## Tests

```bash
python3 -m pytest -q
python3 -m compileall -q backend
```

There are 20 tests, and they run offline because providers are stubbed. They cover:
- A ZIP containing **both the BTS CSV and `readme.html`**, a ZIP with no CSV, a non-ZIP file, and a missing schema.
- Leakage-column exclusion, cancelled/diverted filtering, mile→km conversion, and HHMM parsing.
- DST de-duplication, the date-separated split, and end-to-end training artifacts.
- API 503/422/502 paths, a successful prediction, secret non-exposure, and frontend serving.

## Legacy India prototype

`backend/ml/preprocess.py`, `train.py` and `/api/predict` are the original 190-row, unverified India sample. They are kept for reference only and are not used by the UI. To rebuild it: `python3 -m backend.ml.preprocess && python3 -m backend.ml.train`.

## Remaining external requirements

- **Aviationstack key** (optional; configured locally in `.env` and verified on 2026-10-13 with a live AA1 JFK→LAX lookup over HTTPS). Each fresh lookup uses one request of your plan's monthly quota.
- **Internet access** to `transtats.bts.gov` (download), `archive-api.open-meteo.com` (preprocessing) and `api.open-meteo.com` (each prediction). All three were reachable during this run.
