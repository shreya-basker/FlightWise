"""Train, evaluate, and save the FlightWise delay-risk model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from backend.app.config import CLEANED_DATA_PATH, METRICS_PATH, MODEL_PATH
from backend.ml.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERICAL_FEATURES, TARGET_COLUMN


def _make_pipeline(classifier: object) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numerical", StandardScaler(), NUMERICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("FlightDate").reset_index(drop=True)
    target_rows = int(len(ordered) * 0.75)
    # Consider only boundaries between dates: no flight date may appear in both sets.
    boundaries = [
        index for index in range(1, len(ordered))
        if ordered.loc[index - 1, "FlightDate"] < ordered.loc[index, "FlightDate"]
    ]
    eligible = []
    for boundary in boundaries:
        train, test = ordered.iloc[:boundary], ordered.iloc[boundary:]
        if train[TARGET_COLUMN].nunique() == 2 and test[TARGET_COLUMN].nunique() == 2:
            eligible.append(boundary)
    if not eligible:
        raise ValueError("No date-separated chronological split contains both classes in train and test.")
    split_at = min(eligible, key=lambda boundary: abs(boundary - target_rows))
    return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | None]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics: dict[str, float | None] = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": None,
    }
    if y_test.nunique() == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, probabilities)), 4)
    return metrics


def train(input_path: Path = CLEANED_DATA_PATH, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH) -> dict:
    df = pd.read_csv(input_path, parse_dates=["FlightDate"])
    required = {"FlightDate", *FEATURE_COLUMNS, TARGET_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Cleaned dataset missing columns: {', '.join(sorted(missing))}")
    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Training requires both delay classes.")

    train_df, test_df = chronological_split(df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, class_weight="balanced", random_state=42),
    }

    fitted: dict[str, Pipeline] = {}
    results: dict[str, dict] = {}
    for name, classifier in candidates.items():
        model = _make_pipeline(classifier)
        model.fit(X_train, y_train)
        fitted[name] = model
        results[name] = evaluate(model, X_test, y_test)

    selected_name = max(results, key=lambda name: (results[name]["f1"], results[name]["recall"]))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected_name], model_path)

    with_route = df.assign(route=df["Origin"] + "-" + df["Dest"])
    route_counts = with_route.groupby("route").size().to_dict()
    airline_route_counts = with_route.assign(
        airline_route=with_route["Airline"] + "|" + with_route["route"]
    ).groupby("airline_route").size().to_dict()
    artifact = {
        "selected_model": selected_name,
        "selection_metric": "f1",
        "evaluation": {
            "split": "chronological 75% train / 25% test",
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
            "train_end": train_df["FlightDate"].max().date().isoformat(),
            "test_start": test_df["FlightDate"].min().date().isoformat(),
            "models": results,
        },
        "training_data": {
            "rows": int(len(df)),
            "target_distribution": {str(key): int(value) for key, value in df[TARGET_COLUMN].value_counts().sort_index().items()},
            "feature_columns": FEATURE_COLUMNS,
            "airlines": sorted(df["Airline"].unique().tolist()),
            "origins": sorted(df["Origin"].unique().tolist()),
            "destinations": sorted(df["Dest"].unique().tolist()),
            "route_counts": route_counts,
            "airline_route_counts": airline_route_counts,
            "distance_by_route": {key: round(float(value), 2) for key, value in with_route.groupby("route")["Distance_KM"].median().items()},
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limitations": "Prototype trained on a small, unverified sample. Metrics are a single chronological holdout evaluation, not production validation.",
    }
    metrics_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlightWise models.")
    parser.add_argument("--input", type=Path, default=CLEANED_DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    args = parser.parse_args()
    result = train(args.input, args.model, args.metrics)
    print(json.dumps(result["evaluation"], indent=2))
    print(f"Selected model: {result['selected_model']}")


if __name__ == "__main__":
    main()
