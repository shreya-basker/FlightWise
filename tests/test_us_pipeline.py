import zipfile

import numpy as np
import pandas as pd
import pytest

from backend.ml.download_bts import is_valid_bts_zip
from backend.ml.preprocess_us import load_bts, scheduled_hours
from backend.ml.train_us import date_split, train
from backend.ml.weather import WEATHER_FEATURES, _normalise_hourly

BTS_CSV_NAME = "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_1.csv"


def bts_frame() -> pd.DataFrame:
    """A miniature BTS extract including leakage columns that must never reach the output."""
    return pd.DataFrame({
        "Year": [2024] * 5, "FlightDate": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
        "Reporting_Airline": ["AA", "DL", "UA", "WN", "AA"], "Origin": ["ATL", "ATL", "ORD", "DEN", "LAX"],
        "Dest": ["LAX", "JFK", "SFO", "PHX", "ATL"], "CRSDepTime": [905, 1430, 2359, 600, 2400],
        "DepTime": [930, None, 2359, 600, 5], "ArrDelay": [20.0, None, 0.0, -3.0, 4.0], "CarrierDelay": [15.0, None, None, None, None],
        "DepDel15": [1.0, None, 0.0, 0.0, 0.0], "Cancelled": [0.0, 1.0, 0.0, 0.0, 0.0], "Diverted": [0.0, 0.0, 0.0, 1.0, 0.0],
        "Distance": [1946.0, 760.0, 1846.0, 602.0, 1946.0], "Unnamed: 109": [None] * 5,
    })


def write_bts_zip(path, frame=None):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(BTS_CSV_NAME, (bts_frame() if frame is None else frame).to_csv(index=False))
        archive.writestr("readme.html", "<html><body>BTS field descriptions</body></html>")
    return path


def test_us_feature_contract_is_explicit():
    assert WEATHER_FEATURES == ["TemperatureC", "PrecipitationMM", "WindSpeedKPH", "CloudCoverPct"]


def test_bts_zip_with_readme_html_is_read_from_its_csv_member(tmp_path):
    path = write_bts_zip(tmp_path / "bts_2024_01.zip")
    assert is_valid_bts_zip(path)
    flights = load_bts(path)
    # Cancelled (DL), diverted (WN) and the invalid 2400 schedule are removed.
    assert flights[["Airline", "Origin", "Dest"]].values.tolist() == [["AA", "ATL", "LAX"], ["UA", "ORD", "SFO"]]
    assert flights["Delayed"].tolist() == [1, 0]
    assert flights["ScheduledDepartureHour"].tolist() == [9, 23]
    assert flights.loc[0, "Distance_KM"] == pytest.approx(1946 * 1.609344, abs=0.1)
    assert flights.loc[0, "DayOfWeek"] == 0 and flights.loc[0, "Month"] == 1
    for leaked in ("DepTime", "ArrDelay", "CarrierDelay", "DepDel15", "Cancelled", "Diverted"):
        assert leaked not in flights.columns


def test_bts_zip_without_a_csv_is_rejected(tmp_path):
    path = tmp_path / "broken.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.html", "<html></html>")
    assert not is_valid_bts_zip(path)
    with pytest.raises(ValueError, match="Expected exactly one BTS CSV"):
        load_bts(path)


def test_non_zip_file_with_zip_suffix_is_rejected(tmp_path):
    path = tmp_path / "bts_2024_01.zip"
    path.write_text("<html>error page</html>")
    with pytest.raises(ValueError, match="not a valid ZIP"):
        load_bts(path)


def test_bts_loader_rejects_missing_required_schema(tmp_path):
    path = tmp_path / "not_bts.csv"
    pd.DataFrame({"FlightDate": ["2024-01-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="BTS file missing columns"):
        load_bts(path)


def test_scheduled_hours_handles_bts_hhmm_values():
    assert scheduled_hours(pd.Series([5, 905, "1430", 2359, 2400, 1275, None])).tolist()[:4] == [0, 9, 14, 23]
    assert scheduled_hours(pd.Series([2400, 1275, None])).isna().all()


def test_hourly_weather_drops_repeated_dst_hour():
    payload = {"hourly": {"time": ["2024-11-03T01:00", "2024-11-03T01:00", "2024-11-03T02:00"], "temperature_2m": [1, 2, 3], "precipitation": [0, 0, 0], "wind_speed_10m": [5, 5, 5], "cloud_cover": [10, 10, 10]}}
    frame = _normalise_hourly(payload)
    assert frame["ScheduledDepartureHour"].tolist() == [1, 2]


def synthetic_training_frame(days: int = 20, per_day: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = days * per_day
    frame = pd.DataFrame({
        "FlightDate": np.repeat(pd.date_range("2024-01-01", periods=days), per_day),
        "Airline": rng.choice(["AA", "DL", "UA"], n), "Origin": rng.choice(["ATL", "ORD"], n), "Dest": rng.choice(["LAX", "JFK"], n),
        "ScheduledDepartureHour": rng.integers(5, 23, n), "Distance_KM": rng.uniform(500, 3000, n),
        "TemperatureC": rng.normal(5, 8, n), "PrecipitationMM": rng.exponential(0.3, n), "WindSpeedKPH": rng.uniform(0, 50, n), "CloudCoverPct": rng.uniform(0, 100, n),
    })
    frame["Month"], frame["DayOfWeek"] = frame["FlightDate"].dt.month, frame["FlightDate"].dt.dayofweek
    signal = 0.05 * frame["WindSpeedKPH"] + 0.1 * frame["ScheduledDepartureHour"] - 3.5
    frame["Delayed"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-signal))).astype(int)
    return frame


def test_date_split_never_shares_a_date():
    earlier, later = date_split(synthetic_training_frame(), 0.75)
    assert earlier["FlightDate"].max() < later["FlightDate"].min()
    assert 0.6 < len(earlier) / (len(earlier) + len(later)) <= 0.75


def test_training_writes_model_and_real_metrics(tmp_path):
    data, model, metrics = tmp_path / "clean.csv", tmp_path / "model.joblib", tmp_path / "metrics.json"
    synthetic_training_frame().to_csv(data, index=False)
    metadata = train(data, model, metrics)
    assert model.exists() and metrics.exists()
    ev = metadata["evaluation"]
    assert ev["fit"]["end"] < ev["validation"]["start"] <= ev["validation"]["end"] < ev["test"]["start"]
    assert set(ev["models"]) == {"weather_logistic_regression", "weather_hist_gradient_boosting"}
    for result in ev["models"].values():
        assert {"f1", "roc_auc", "brier_score", "calibration_bins", "threshold"} <= set(result)
