"""End-to-end training test on a small synthetic mart (no network, no real data)."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import paths
from src.config import load_config
from src.features import BASE_FEATURES
from src.timeutils import month_bounds
from src.train import error_metrics, paired_day_bootstrap, run

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]


def time_keys(node):
    """Every dict key, at any depth, whose name mentions a time measurement."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if any(word in key.lower() for word in ("second", "time", "duration")):
                found.append(key)
            found += time_keys(value)
    elif isinstance(node, list):
        for value in node:
            found += time_keys(value)
    return found


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train")
    rng = np.random.default_rng(1)
    mart_dir = tmp / "mart"
    for month in MONTHS:
        start, end = month_bounds(month)
        hours = pd.date_range(start, end, freq="h", inclusive="left")
        frames = []
        for zone, (borough, level) in enumerate([("Manhattan", 80), ("Queens", 20), ("Bronx", 3)], start=1):
            daily = 1 + 0.5 * np.sin(2 * np.pi * hours.hour / 24)
            frames.append(pd.DataFrame({
                "hour": hours, "zone_id": zone, "borough": borough,
                "trips_total": rng.poisson(level * daily),
            }))
        part = mart_dir / f"month={month}"
        part.mkdir(parents=True)
        pd.concat(frames).sort_values(["hour", "zone_id"]).to_parquet(part / "part.parquet", index=False)

    all_hours = pd.date_range("2026-01-01", "2026-06-01", freq="h", inclusive="left")
    weather_path = tmp / "weather.csv"
    pd.DataFrame({"hour": all_hours, "temperature_2m": rng.normal(10, 5, len(all_hours)),
                  "precipitation": 0.0}).to_csv(weather_path, index=False)

    cfg = load_config()
    cfg["split"] = {
        "train": {"start": "2026-01-08 00:00", "end": "2026-02-28 23:00"},
        "validation": {"start": "2026-03-01 00:00", "end": "2026-03-31 23:00"},
        "test_months": ["2026-04", "2026-05"],
    }
    grid = [{"name": "depth3", "max_depth": 3}, {"name": "depth5", "max_depth": 5}]
    metrics = run(cfg, mart_dir, weather_path, tmp / "models", tmp / "reports", param_grid=grid, max_rounds=40)
    return {"tmp": tmp, "metrics": metrics}


def test_outputs_are_written_and_consistent(trained):
    tmp, metrics = trained["tmp"], trained["metrics"]
    assert json.loads((tmp / "models" / "metrics.json").read_text(encoding="utf-8")) == metrics
    assert set(metrics["periods"]) == {"validation", "test_2026-04", "test_2026-05"}

    backtest = pd.read_parquet(tmp / "reports" / "backtest_hourly.parquet")
    assert list(backtest.columns) == ["hour", "zone_id", "actual", "baseline", "xgb"]
    assert len(backtest) == 3 * (30 + 31) * 24
    assert backtest["hour"].min() == pd.Timestamp("2026-04-01 00:00")
    assert (backtest["xgb"] >= 0).all()

    # metrics.json must be reproducible from the backtest file
    april = backtest[backtest["hour"] < "2026-05-01"]
    assert metrics["periods"]["test_2026-04"]["xgboost"]["mae"] == pytest.approx(
        (april["xgb"] - april["actual"]).abs().mean(), rel=1e-5)
    assert metrics["periods"]["test_2026-04"]["baseline"]["mae"] == pytest.approx(
        (april["baseline"] - april["actual"]).abs().mean(), rel=1e-5)

    boroughs = pd.read_csv(tmp / "reports" / "metrics_by_borough.csv")
    assert set(boroughs["borough"]) == {"Manhattan", "Queens", "Bronx"}
    assert len(boroughs) == 3 * 3

    saved = joblib.load(tmp / "models" / "xgb_demand.joblib")
    assert saved["features"] == metrics["final_model"]["features"]
    assert set(BASE_FEATURES) <= set(saved["features"])


def test_timings_are_written_to_their_own_file_not_to_the_metrics(trained):
    tmp, metrics = trained["tmp"], trained["metrics"]
    assert not time_keys(metrics)
    timings = json.loads((tmp / "reports" / "train_timings.json").read_text(encoding="utf-8"))
    assert set(timings["seconds"]["candidates_seconds"]) == {"depth3", "depth5"}
    assert {"build_features_seconds", "search_seconds", "weather_ablation_seconds", "final_fit_seconds"} <= set(timings["seconds"])
    assert timings["environment"] == metrics["environment"]


def test_committed_results_hold_no_wall_clock_fields():
    """Committed result files must be identical after a re-run, so they cannot carry timings."""
    for path in (paths.METRICS, paths.DATA_QUALITY):
        assert not time_keys(json.loads(path.read_text(encoding="utf-8"))), path


def test_selection_uses_validation_only(trained):
    metrics = trained["metrics"]
    best = min(metrics["search"]["candidates"], key=lambda c: c["validation_mae"])
    assert metrics["search"]["selected"] == best["name"]
    weather = metrics["weather"]
    assert weather["kept"] == (weather["validation_mae_with_weather"] < weather["validation_mae_without_weather"])
    assert metrics["rows"]["train"] == 3 * (24 + 28) * 24


def test_error_metrics():
    m = error_metrics([10, 0, 5], [8, 1, 5])
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert m["wape"] == pytest.approx(3 / 15)


def test_bootstrap_interval_brackets_a_clear_difference():
    hours = pd.date_range("2026-04-01", periods=24 * 20, freq="h")
    frame = pd.DataFrame({"hour": hours, "actual": 10.0, "baseline": 14.0, "xgb": 11.0})
    first = paired_day_bootstrap(frame, seed=42)
    assert first["mae_diff_xgb_minus_baseline"] == pytest.approx(-3.0)
    assert first["ci95_low"] <= -3.0 <= first["ci95_high"] < 0
    assert first["n_days"] == 20
    assert first == paired_day_bootstrap(frame, seed=42)  # seeded -> reproducible
