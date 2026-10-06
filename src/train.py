"""Trains the global XGBoost demand model and evaluates it against the last-week baseline.

Usage:
    python -m src.train

Protocol (splits come from config/pipeline.yaml):
  1. Candidate parameter sets are fitted on train with early stopping on validation and ranked
     by validation MAE.
  2. Weather features are kept only if they lower validation MAE.
  3. The chosen setup is refitted on train + validation with the selected number of trees.
  4. Each test month is predicted exactly once, by that refitted model.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import ROOT, load_config
from .features import BASE_FEATURES, BASELINE, TARGET, WEATHER_FEATURES, build_features, load_mart
from .timeutils import month_bounds

DEFAULT_MART = ROOT / "data" / "mart" / "demand_hourly"
DEFAULT_WEATHER = ROOT / "data" / "external" / "weather_hourly.csv"
DEFAULT_MODELS = ROOT / "models"
DEFAULT_REPORTS = ROOT / "reports"

# At most six candidates, fixed before any model was fitted.
PARAM_GRID = [
    {"name": "depth6", "max_depth": 6},
    {"name": "depth8", "max_depth": 8},
    {"name": "depth10", "max_depth": 10},
    {"name": "depth8_subsample", "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8},
    {"name": "depth8_min_child20", "max_depth": 8, "min_child_weight": 20},
    {"name": "depth8_poisson", "max_depth": 8, "objective": "count:poisson"},
]
LEARNING_RATE = 0.1
MAX_ROUNDS = 600
EARLY_STOPPING_ROUNDS = 30
N_BOOTSTRAP = 1000
MAX_MODEL_MB = 20


def make_model(params, n_estimators, cfg, early_stopping):
    params = {k: v for k, v in params.items() if k != "name"}
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=LEARNING_RATE,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="mae",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS if early_stopping else None,
        n_jobs=int(cfg["n_jobs"]),
        random_state=int(cfg["random_state"]),
        **params,
    )


def predict(model, frame, features):
    """Demand cannot be negative, so predictions are clipped at 0 everywhere."""
    return np.clip(model.predict(frame[features]), 0, None)


def error_metrics(actual, predicted):
    err = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape": float(np.abs(err).sum() / np.asarray(actual, dtype=float).sum()),
    }


def paired_day_bootstrap(frame, seed, n_boot=N_BOOTSTRAP):
    """95% CI of MAE(xgb) - MAE(baseline), resampling whole days with replacement.

    Both models are scored on the same resampled days (paired), and a day is kept as one
    block because errors within a day are strongly correlated.
    """
    days = frame.assign(
        day=frame["hour"].dt.normalize(),
        abs_xgb=(frame["xgb"] - frame["actual"]).abs(),
        abs_base=(frame["baseline"] - frame["actual"]).abs(),
    ).groupby("day").agg(abs_xgb=("abs_xgb", "sum"), abs_base=("abs_base", "sum"), n=("actual", "size"))
    diff_sum = (days["abs_xgb"] - days["abs_base"]).to_numpy()
    n = days["n"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(days), size=(n_boot, len(days)))
    diffs = diff_sum[picks].sum(axis=1) / n[picks].sum(axis=1)
    return {
        "mae_diff_xgb_minus_baseline": float(diff_sum.sum() / n.sum()),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "n_bootstrap": n_boot,
        "block": "day",
        "n_days": int(len(days)),
    }


def evaluate_period(frame, seed):
    """Metrics of one period; `frame` has hour, zone_id, borough, actual, baseline, xgb."""
    baseline = error_metrics(frame["actual"], frame["baseline"])
    model = error_metrics(frame["actual"], frame["xgb"])
    improvement = 1 - model["mae"] / baseline["mae"]
    return {
        "n_rows": int(len(frame)),
        "n_zones": int(frame["zone_id"].nunique()),
        "mean_actual": float(frame["actual"].mean()),
        "baseline": baseline,
        "xgboost": model,
        "mae_improvement_pct": float(100 * improvement),
        "bootstrap": paired_day_bootstrap(frame, seed),
        # Thresholds from the project rules: results this good must be checked for leakage first.
        "leakage_check_required": bool(model["wape"] < 0.10 or improvement > 0.60),
    }


def borough_table(frames):
    rows = []
    for period, frame in frames.items():
        for borough, part in frame.groupby("borough"):
            base = error_metrics(part["actual"], part["baseline"]) if part["actual"].sum() > 0 else None
            model = error_metrics(part["actual"], part["xgb"]) if part["actual"].sum() > 0 else None
            rows.append({
                "period": period,
                "borough": borough,
                "n_zones": part["zone_id"].nunique(),
                "n_rows": len(part),
                "mean_actual": round(part["actual"].mean(), 3),
                "mae_baseline": round(base["mae"], 4) if base else None,
                "mae_xgboost": round(model["mae"], 4) if model else None,
                "wape_baseline": round(base["wape"], 4) if base else None,
                "wape_xgboost": round(model["wape"], 4) if model else None,
            })
    return pd.DataFrame(rows)


def scored_frame(part, predictions):
    return pd.DataFrame({
        "hour": part["hour"].to_numpy(),
        "zone_id": part["zone_id"].astype("int16").to_numpy(),
        "borough": part["borough"].to_numpy(),
        "actual": part[TARGET].to_numpy(dtype="float32"),
        "baseline": part[BASELINE].to_numpy(dtype="float32"),
        "xgb": predictions.astype("float32"),
    })


def run(cfg, mart_dir=DEFAULT_MART, weather_path=DEFAULT_WEATHER, models_dir=DEFAULT_MODELS,
        reports_dir=DEFAULT_REPORTS, param_grid=PARAM_GRID, max_rounds=MAX_ROUNDS):
    seed = int(cfg["random_state"])
    timings = {}

    t0 = time.perf_counter()
    weather = pd.read_csv(weather_path, parse_dates=["hour"])
    data = build_features(load_mart(mart_dir), weather)
    timings["build_features_seconds"] = round(time.perf_counter() - t0, 1)

    split = cfg["split"]
    hour = data["hour"]
    train = data[(hour >= pd.Timestamp(split["train"]["start"])) & (hour <= pd.Timestamp(split["train"]["end"]))]
    val = data[(hour >= pd.Timestamp(split["validation"]["start"])) & (hour <= pd.Timestamp(split["validation"]["end"]))]
    tests = {}
    for month in split["test_months"]:
        start, end = month_bounds(month)
        tests[month] = data[(hour >= start) & (hour < end)]
    assert train["hour"].max() < val["hour"].min() < val["hour"].max() < min(t["hour"].min() for t in tests.values())

    all_features = BASE_FEATURES + WEATHER_FEATURES

    def fit_on_train(params, features):
        model = make_model(params, max_rounds, cfg, early_stopping=True)
        model.fit(train[features], train[TARGET], eval_set=[(val[features], val[TARGET])], verbose=False)
        return model, error_metrics(val[TARGET], predict(model, val, features))["mae"]

    # 1. Parameter search on validation (all features).
    t0 = time.perf_counter()
    candidates, fitted = [], {}
    for params in param_grid:
        t1 = time.perf_counter()
        model, val_mae = fit_on_train(params, all_features)
        fitted[params["name"]] = model
        candidates.append({
            "name": params["name"],
            "params": {k: v for k, v in params.items() if k != "name"},
            "best_iteration": int(model.best_iteration),
            "hit_round_limit": bool(model.best_iteration + 1 >= max_rounds),
            "validation_mae": val_mae,
            "seconds": round(time.perf_counter() - t1, 1),
        })
        print(f"[search] {params['name']}: val MAE {val_mae:.4f} at {model.best_iteration + 1} trees", flush=True)
    timings["search_seconds"] = round(time.perf_counter() - t0, 1)
    best = min(candidates, key=lambda c: c["validation_mae"])
    best_params = next(p for p in param_grid if p["name"] == best["name"])

    # 2. Weather ablation with the selected parameters.
    t0 = time.perf_counter()
    no_weather_model, no_weather_mae = fit_on_train(best_params, BASE_FEATURES)
    timings["weather_ablation_seconds"] = round(time.perf_counter() - t0, 1)
    keep_weather = best["validation_mae"] < no_weather_mae
    features = all_features if keep_weather else BASE_FEATURES
    selection_model = fitted[best["name"]] if keep_weather else no_weather_model
    n_trees = int(selection_model.best_iteration) + 1
    print(f"[weather] val MAE with {best['validation_mae']:.4f} / without {no_weather_mae:.4f} "
          f"-> {'kept' if keep_weather else 'dropped'}", flush=True)

    scored = {"validation": scored_frame(val, predict(selection_model, val, features))}

    # 3. Refit on train + validation with the selected number of trees.
    t0 = time.perf_counter()
    train_val = pd.concat([train, val])
    final = make_model(best_params, n_trees, cfg, early_stopping=False)
    final.fit(train_val[features], train_val[TARGET], verbose=False)
    timings["final_fit_seconds"] = round(time.perf_counter() - t0, 1)

    # 4. The only place where the test months are predicted.
    for month, part in tests.items():
        scored[f"test_{month}"] = scored_frame(part, predict(final, part, features))

    periods = {name: evaluate_period(frame, seed) for name, frame in scored.items()}

    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "xgb_demand.joblib"
    joblib.dump({"model": final, "features": features}, model_path, compress=3)
    model_mb = model_path.stat().st_size / 2**20

    backtest = pd.concat([scored[f"test_{m}"] for m in tests], ignore_index=True)
    backtest.drop(columns="borough").to_parquet(reports_dir / "backtest_hourly.parquet", index=False, compression="zstd")
    borough_table(scored).to_csv(reports_dir / "metrics_by_borough.csv", index=False, lineterminator="\n")

    metrics = {
        "generated_by": "python -m src.train",
        "target": "trips_total: pickups per zone and hour, summed over yellow, green, fhv and fhvhv",
        "notes": [
            "MAE and WAPE are the headline metrics. R2 is not reported: zones differ so much in volume that it is high for any model.",
            "WAPE = sum(|error|) / sum(actual) over all zone-hours of the period.",
            "Baseline = demand of the same zone at the same hour one week earlier (lag 168).",
            "Predictions are clipped at 0.",
            "Validation metrics come from the model fitted on train only; they were used for model selection and early stopping, so they are optimistic.",
            "Test metrics come from the model refitted on train + validation; each test month was predicted once.",
            "Weather is the observed value of the forecast hour, i.e. a perfect weather forecast is assumed. Real forecasts would be less accurate.",
            "Bootstrap: paired block bootstrap over days (both models scored on the same resampled days).",
        ],
        "split": split,
        "rows": {"train": int(len(train)), "validation": int(len(val)), **{f"test_{m}": int(len(p)) for m, p in tests.items()}},
        "search": {
            "learning_rate": LEARNING_RATE,
            "max_rounds": max_rounds,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "candidates": candidates,
            "selected": best["name"],
        },
        "weather": {
            "validation_mae_with_weather": best["validation_mae"],
            "validation_mae_without_weather": no_weather_mae,
            "kept": bool(keep_weather),
        },
        "final_model": {
            "params": {k: v for k, v in best_params.items() if k != "name"},
            "n_trees": n_trees,
            "features": features,
            "file": "models/xgb_demand.joblib",
            "size_mb": round(model_mb, 2),
        },
        "periods": periods,
        "timings": timings,
        "environment": {
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "n_jobs": int(cfg["n_jobs"]),
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
        },
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    metrics = run(load_config())
    for name, p in metrics["periods"].items():
        b = p["bootstrap"]
        print(f"[{name}] MAE baseline {p['baseline']['mae']:.3f} | xgboost {p['xgboost']['mae']:.3f} | "
              f"diff {b['mae_diff_xgb_minus_baseline']:.3f} [{b['ci95_low']:.3f}, {b['ci95_high']:.3f}] | "
              f"WAPE {p['baseline']['wape']:.3f} -> {p['xgboost']['wape']:.3f}")
    size = metrics["final_model"]["size_mb"]
    if size > MAX_MODEL_MB:
        print(f"Model file is {size} MB, above the {MAX_MODEL_MB} MB limit for committed files")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
