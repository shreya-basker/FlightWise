"""Train an evaluable, weather-aware US domestic flight delay model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from backend.app.config import US_CLEANED_DATA_PATH, US_METRICS_PATH, US_MODEL_PATH
from backend.ml.weather import WEATHER_FEATURES

US_CATEGORICAL = ["Airline", "Origin", "Dest"]
US_NUMERICAL = ["Month", "DayOfWeek", "ScheduledDepartureHour", "Distance_KM", *WEATHER_FEATURES]
US_FEATURES = [*US_CATEGORICAL, *US_NUMERICAL]
TARGET = "Delayed"


def logistic_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore", min_frequency=20), US_CATEGORICAL),
        ("number", StandardScaler(), US_NUMERICAL),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=3000, C=1.0, random_state=42))])


def boosted_pipeline() -> Pipeline:
    # HGB needs < 255 categories per feature; rare airports are grouped and unseen ones become missing (-1).
    preprocessor = ColumnTransformer([
        ("category", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1, min_frequency=20, max_categories=250), US_CATEGORICAL),
        ("number", "passthrough", US_NUMERICAL),
    ])
    classifier = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
        categorical_features=[0, 1, 2], early_stopping=False, random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def date_split(df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split on whole dates: every flight on a given date lands on exactly one side."""
    ordered = df.sort_values("FlightDate", kind="stable").reset_index(drop=True)
    dates = ordered["FlightDate"].dt.normalize()
    cumulative = dates.value_counts().sort_index().cumsum() / len(ordered)
    unique_dates = cumulative.index
    if len(unique_dates) < 2:
        raise ValueError("A date-separated split needs flights on at least two dates.")
    # Last date that keeps the earlier side at or below the target share (always leave one date for the later side).
    position = int(np.clip(np.searchsorted(cumulative.to_numpy(), fraction, side="right") - 1, 0, len(unique_dates) - 2))
    boundary = unique_dates[position]
    earlier, later = ordered[dates <= boundary].copy(), ordered[dates > boundary].copy()
    for name, part in (("earlier", earlier), ("later", later)):
        if part[TARGET].nunique() != 2:
            raise ValueError(f"The {name} side of the date split does not contain both classes.")
    return earlier, later


def best_threshold(y: pd.Series, probabilities: np.ndarray) -> float:
    """Threshold maximising F1 on the validation dates (never on the test dates)."""
    candidates = np.round(np.arange(0.05, 0.80, 0.01), 2)
    scores = [f1_score(y, (probabilities >= t).astype(int), zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def score(probabilities: np.ndarray, y: pd.Series, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    observed, predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y, predictions)), 4), "precision": round(float(precision_score(y, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 4), "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 4), "brier_score": round(float(brier_score_loss(y, probabilities)), 4),
        "mean_predicted_probability": round(float(probabilities.mean()), 4), "observed_delay_rate": round(float(y.mean()), 4),
        "calibration_bins": [{"mean_predicted": round(float(p), 4), "observed_rate": round(float(o), 4)} for o, p in zip(observed, predicted)],
    }


def _period(frame: pd.DataFrame) -> dict:
    return {"rows": int(len(frame)), "start": frame["FlightDate"].min().date().isoformat(), "end": frame["FlightDate"].max().date().isoformat(), "dates": int(frame["FlightDate"].dt.normalize().nunique()), "delay_rate": round(float(frame[TARGET].mean()), 4)}


def train(input_path: Path = US_CLEANED_DATA_PATH, model_path: Path = US_MODEL_PATH, metrics_path: Path = US_METRICS_PATH) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Run: python -m backend.ml.preprocess_us backend/data/bts/bts_2024_01.zip")
    df = pd.read_csv(input_path, parse_dates=["FlightDate"])
    required = {"FlightDate", TARGET, *US_FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Weather-enriched data missing: {', '.join(sorted(missing))}")
    if df[TARGET].nunique() != 2:
        raise ValueError("Training requires both delayed and on-time flights.")

    # Three date-separated periods: fit (earliest ~60%) -> threshold validation (~15%) -> untouched test (~25%).
    development, test_df = date_split(df, 0.75)
    fit_df, validation_df = date_split(development, 0.80)
    candidates = {"weather_logistic_regression": logistic_pipeline, "weather_hist_gradient_boosting": boosted_pipeline}
    results, fitted = {}, {}
    for name, build in candidates.items():
        model = build().fit(fit_df[US_FEATURES], fit_df[TARGET])
        threshold = best_threshold(validation_df[TARGET], model.predict_proba(validation_df[US_FEATURES])[:, 1])
        results[name] = score(model.predict_proba(test_df[US_FEATURES])[:, 1], test_df[TARGET], threshold)
        fitted[name] = model
        print(f"{name}: F1={results[name]['f1']} ROC-AUC={results[name]['roc_auc']} Brier={results[name]['brier_score']} (threshold {threshold})")
    selected = max(results, key=lambda name: (results[name]["f1"], results[name]["roc_auc"]))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected], model_path)
    metadata = {
        "selected_model": selected, "selection_metric": "test F1 at a validation-tuned threshold, then ROC-AUC",
        "decision_threshold": results[selected]["threshold"], "feature_columns": US_FEATURES, "target": "DepDel15 (departure delay >= 15 minutes)",
        "evaluation": {
            "split": "date-separated chronological: fit -> threshold-validation -> test; no date appears in two periods",
            "fit": _period(fit_df), "validation": _period(validation_df), "test": _period(test_df),
            "models": results,
        },
        "training_data": {
            "source": "US DOT BTS Reporting Carrier On-Time Performance (PREZIP monthly files)",
            "rows": int(len(df)), "start": df["FlightDate"].min().date().isoformat(), "end": df["FlightDate"].max().date().isoformat(),
            "months": sorted(df["FlightDate"].dt.strftime("%Y-%m").unique().tolist()), "delay_rate": round(float(df[TARGET].mean()), 4),
            "airlines": sorted(df["Airline"].unique().tolist()), "origins": sorted(df["Origin"].unique().tolist()), "destinations": sorted(df["Dest"].unique().tolist()),
        },
        "weather_source": "Open-Meteo Historical Weather API (hourly reanalysis at origin airport, local time); requests use Open-Meteo hourly forecast or current conditions.",
        "limitations": [
            "US domestic BTS sample only; trained months are listed in training_data.months, so seasonal effects outside them are not learned.",
            "Only the busiest origin airports have weather and therefore training data; other origins are treated as unseen.",
            "Historical reanalysis is not the same information as the forecast available before departure.",
            "Single chronological holdout; not production validation. Not a flight-status guarantee or travel recommendation.",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FlightWise US model.")
    parser.add_argument("--input", type=Path, default=US_CLEANED_DATA_PATH)
    parser.add_argument("--model", type=Path, default=US_MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=US_METRICS_PATH)
    args = parser.parse_args()
    result = train(args.input, args.model, args.metrics)
    summary = {name: {k: v for k, v in metrics.items() if k != "calibration_bins"} for name, metrics in result["evaluation"]["models"].items()}
    print(json.dumps({"selected_model": result["selected_model"], "periods": {k: result["evaluation"][k] for k in ("fit", "validation", "test")}, "models": summary}, indent=2))
