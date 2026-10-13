"""Train an evaluable, weather-aware US domestic flight delay model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from backend.app.config import US_CLEANED_DATA_PATH, US_METRICS_PATH, US_MODEL_PATH
from backend.ml.train import chronological_split
from backend.ml.weather import WEATHER_FEATURES

US_CATEGORICAL = ["Airline", "Origin", "Dest"]
US_NUMERICAL = ["Month", "DayOfWeek", "ScheduledDepartureHour", "Distance_KM", *WEATHER_FEATURES]
US_FEATURES = [*US_CATEGORICAL, *US_NUMERICAL]
TARGET = "Delayed"


def logistic_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore"), US_CATEGORICAL),
        ("number", StandardScaler(), US_NUMERICAL),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", solver="saga", n_jobs=-1, random_state=42))])


def boosted_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("category", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), US_CATEGORICAL),
        ("number", "passthrough", US_NUMERICAL),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=42))])


def score(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    observed, predicted = calibration_curve(y, probabilities, n_bins=8, strategy="quantile")
    return {
        "accuracy": round(float(accuracy_score(y, predictions)), 4), "precision": round(float(precision_score(y, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 4), "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 4), "brier_score": round(float(brier_score_loss(y, probabilities)), 4),
        "calibration_bins": [{"mean_predicted": round(float(p), 4), "observed_rate": round(float(o), 4)} for o, p in zip(observed, predicted)],
    }


def train(input_path: Path = US_CLEANED_DATA_PATH, model_path: Path = US_MODEL_PATH, metrics_path: Path = US_METRICS_PATH) -> dict:
    df = pd.read_csv(input_path, parse_dates=["FlightDate"])
    required = {"FlightDate", TARGET, *US_FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Weather-enriched data missing: {', '.join(sorted(missing))}")
    train_df, test_df = chronological_split(df.rename(columns={TARGET: "Delayed"}))
    models = {"weather_logistic_regression": logistic_pipeline(), "weather_hist_gradient_boosting": boosted_pipeline()}
    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(train_df[US_FEATURES], train_df[TARGET])
        results[name] = score(model, test_df[US_FEATURES], test_df[TARGET])
        fitted[name] = model
    selected = max(results, key=lambda name: (results[name]["f1"], results[name]["roc_auc"]))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected], model_path)
    metadata = {
        "selected_model": selected, "selection_metric": "f1_then_roc_auc", "feature_columns": US_FEATURES,
        "evaluation": {"split": "date-separated chronological 75% train / 25% test", "train_rows": len(train_df), "test_rows": len(test_df), "train_end": train_df["FlightDate"].max().date().isoformat(), "test_start": test_df["FlightDate"].min().date().isoformat(), "models": results},
        "training_data": {"rows": len(df), "airlines": sorted(df["Airline"].unique().tolist()), "origins": sorted(df["Origin"].unique().tolist()), "destinations": sorted(df["Dest"].unique().tolist())},
        "weather_source": "Open-Meteo Historical Weather API (hourly reanalysis); live requests use Open-Meteo current conditions.",
        "limitations": "US domestic BTS prototype. Historical reanalysis is not the same as the forecast available before departure; do not interpret the score as a live operational guarantee.",
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
    print(json.dumps(train(args.input, args.model, args.metrics)["evaluation"], indent=2))
