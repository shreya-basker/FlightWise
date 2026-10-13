from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODEL_DIR = BACKEND_DIR / "models"
FRONTEND_DIR = PROJECT_DIR / "frontend"
RAW_DATA_PATH = DATA_DIR / "india_flight_delays_2025.csv"
CLEANED_DATA_PATH = DATA_DIR / "flights_cleaned.csv"
MODEL_PATH = MODEL_DIR / "flight_delay_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
US_CLEANED_DATA_PATH = DATA_DIR / "bts_flights_weather_cleaned.csv"
US_MODEL_PATH = MODEL_DIR / "us_flight_delay_model.joblib"
US_METRICS_PATH = MODEL_DIR / "us_metrics.json"
