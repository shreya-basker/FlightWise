# FlightWise

## US live-aware upgrade

FlightWise now has a US domestic workflow based on the official BTS Reporting Carrier On-Time Performance dataset. It creates a weather-aware model using historical hourly weather at the departure airport, then uses current departure-airport weather during a request.

Run the new workflow after installing requirements:

```bash
python3 -m backend.ml.download_bts 2024 1
python3 -m backend.ml.preprocess_us backend/data/bts/bts_2024_01.zip
python3 -m backend.ml.train_us
python3 -m pytest
python3 -m uvicorn backend.main:app --reload
```

The default preprocessing run takes a reproducible sample of up to 150,000 valid flights, restricts weather retrieval to the 40 busiest departure airports in that sample, and writes `backend/data/bts_flights_weather_cleaned.csv`. It deliberately excludes actual departure/arrival times and post-flight delay causes from features.

### Current flight lookup

Weather works without an API key through Open-Meteo. To enable flight-number lookup, copy `.env.example` to `.env` and add a personal-use Aviationstack key:

```bash
cp .env.example .env
# Open .env in VS Code and paste your key after AVIATIONSTACK_API_KEY=
```

Never commit `.env`. `GET /api/live/config` confirms whether lookup is enabled. `GET /api/live/flight/{flight_number}` obtains the provider's current flight record, collects current origin weather, and returns a model estimate. The API does not pretend a provider's estimated delay is a model prediction.

### US API

- `GET /api/us/model-info` — exact US model/evaluation metadata after training.
- `POST /api/us/predict` — context-aware US route estimate with live origin weather.
- `GET /api/live/flight/{flight_number}` — optional live lookup and estimate.

The new model uses a date-separated chronological holdout, compares weather-aware logistic regression with histogram gradient boosting, and records Accuracy, Precision, Recall, F1, ROC-AUC, Brier score, and calibration bins. Do not describe results until the training run writes them.

The original India sample workflow remains available only as the earlier educational prototype. The browser interface now uses the US workflow.

FlightWise is a local full-stack college prototype that estimates whether an Indian domestic flight is likely to depart **15 minutes or more late**. It uses only details that are knowable before departure: airline, route, calendar date, scheduled departure hour, and distance.

It includes a static frontend, FastAPI API, reproducible preprocessing/training scripts, model metrics, and tests.

> **Important:** This is an educational prototype, not a live flight-status system. The included source file has 190 rows and is not sufficiently documented to verify it as authentic flight-level operational data. It should not be used to make travel decisions or claimed to have production-grade accuracy.

## What is included

```text
backend/data/india_flight_delays_2025.csv  # supplied sample, preserved unchanged
backend/ml/preprocess.py                   # cleaning and target creation
backend/ml/train.py                        # chronological evaluation and model export
backend/main.py                            # API + frontend hosting
frontend/                                  # responsive browser interface
tests/                                     # preprocessing and API smoke tests
```

## Run on macOS / VS Code

Open a terminal in this project folder, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m backend.ml.preprocess
python3 -m backend.ml.train
python3 -m pytest
python3 -m uvicorn backend.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). The interactive API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The training command writes these local artifacts:

- `backend/data/flights_cleaned.csv`
- `backend/models/flight_delay_model.joblib`
- `backend/models/metrics.json`

The UI gets its dropdown choices and route-distance defaults from that saved metadata, so the API schema and model feature schema remain aligned.

## API

- `GET /api/health` — service/model status.
- `GET /api/options` — supported airlines, airports, routes, and sample route distances.
- `GET /api/model-info` — generated model metadata and actual holdout metrics.
- `POST /api/predict` — a historical-pattern delay-risk estimate.

Example request:

```json
{
  "airline": "AirAsia India",
  "origin": "LKO",
  "destination": "COK",
  "flight_date": "2025-06-01",
  "scheduled_departure_hour": 16,
  "distance_km": 1226
}
```

Unsupported airlines, airports, routes, and airline–route combinations return `422` rather than silently making a prediction from an unfamiliar input. Invalid values (such as a 25:00 hour or the same origin/destination) are also rejected.

## Data and evaluation notes

The included CSV was obtained from the public [TejShah804 India Flight Delay Data Analysis Project](https://github.com/TejShah804/India-Flight-Delay-Data-Analysis-Project). Its README says the data came from Kaggle, but it does not identify a primary source, license, collection method, or data dictionary. Therefore FlightWise records it as an **unverified educational sample**. The repository contains 190 rows; after preprocessing, the script reports the usable count.

`preprocess.py` removes cancelled/malformed rows, derives the target from actual `DepDelay`, derives weekday from `FlightDate`, and excludes post-departure fields such as actual departure/arrival times and arrival delay. This prevents target leakage.

`train.py` uses a chronological 75/25 holdout, compares Logistic Regression, Decision Tree, and Random Forest, then saves the model selected by test F1 score. It writes the exact Accuracy, Precision, Recall, F1, and ROC-AUC into `backend/models/metrics.json`. Do not report any metric until you have run this command; with such a small sample, a single holdout estimate is volatile and is not a generalization claim.

For a stronger future version, replace the raw CSV with a documented, licensed, flight-level source that includes observed departure outcomes, retain the same required columns, rerun preprocessing/training, and add independent temporal validation plus live weather/airport-disruption inputs.

## Test

```bash
python3 -m pytest
```

The tests cover preprocessing schema/target behavior and API health. Run the preprocessing and training commands before manually testing prediction endpoints.
