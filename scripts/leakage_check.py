"""Leakage checks for the demand model, run on the validation month only (test data is not touched).

Usage:
    python -m scripts.leakage_check

Writes reports/leakage_check.json with three independent checks:
  1. The lag features are recomputed with DuckDB window functions straight from the mart and
     compared with src.features.
  2. Naive references that need no model: last hour (lag 1) and last week (lag 168).
  3. The selected model is refitted without the short-term features (lag 1-3, rolling mean).
     If the reported accuracy came from a leak, it would survive this removal.
"""
import json
import sys

import duckdb
import numpy as np
import pandas as pd

from src import paths
from src.config import load_config
from src.features import BASE_FEATURES, LAGS, TARGET, build_features, load_mart
from src.train import MAX_ROUNDS, error_metrics, make_model, predict

SHORT_TERM = ["lag_1", "lag_2", "lag_3", "roll_mean_24"]


def main():
    cfg = load_config()
    metrics = json.loads(paths.METRICS.read_text(encoding="utf-8"))
    data = build_features(load_mart(paths.DEMAND_MART))
    split = cfg["split"]
    hour = data["hour"]
    train = data[(hour >= pd.Timestamp(split["train"]["start"])) & (hour <= pd.Timestamp(split["train"]["end"]))]
    val = data[(hour >= pd.Timestamp(split["validation"]["start"])) & (hour <= pd.Timestamp(split["validation"]["end"]))]

    # 1. Independent recomputation of the lags in SQL.
    lag_sql = ", ".join(
        f"lag(trips_total, {k}) OVER (PARTITION BY zone_id ORDER BY hour) AS lag_{k}" for k in LAGS
    )
    sql_lags = duckdb.sql(f"""
        SELECT hour, zone_id, {lag_sql},
               avg(trips_total) OVER (PARTITION BY zone_id ORDER BY hour
                                      ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS roll_mean_24
        FROM read_parquet('{paths.DEMAND_MART.as_posix()}/*/part.parquet')
        QUALIFY hour >= TIMESTAMP '{split["validation"]["start"]}' AND hour <= TIMESTAMP '{split["validation"]["end"]}'
        ORDER BY zone_id, hour
    """).df()
    ours = val.assign(zone_id=val["zone_id"].astype(int)).sort_values(["zone_id", "hour"]).reset_index(drop=True)
    assert (ours["hour"].to_numpy() == sql_lags["hour"].to_numpy()).all()
    lag_columns = [f"lag_{k}" for k in LAGS] + ["roll_mean_24"]
    max_abs_diff = {c: float(np.abs(ours[c].to_numpy(float) - sql_lags[c].to_numpy(float)).max()) for c in lag_columns}

    # 2. Naive references.
    naive = {
        "last_hour_lag_1": error_metrics(val[TARGET], val["lag_1"]),
        "last_week_lag_168": error_metrics(val[TARGET], val["lag_168"]),
    }

    # 3. Selected parameters without the short-term features.
    params = {"name": "selected", **metrics["final_model"]["params"]}
    reduced = [f for f in BASE_FEATURES if f not in SHORT_TERM]
    model = make_model(params, MAX_ROUNDS, cfg, early_stopping=True)
    model.fit(train[reduced], train[TARGET], eval_set=[(val[reduced], val[TARGET])], verbose=False)
    without_short_term = error_metrics(val[TARGET], predict(model, val, reduced))

    report = {
        "generated_by": "python -m scripts.leakage_check",
        "scope": "validation month only; the test months are not read by the model here",
        "lag_features_vs_sql_max_abs_diff": max_abs_diff,
        "validation": {
            "naive": naive,
            "xgboost_selected": metrics["periods"]["validation"]["xgboost"],
            "xgboost_without_short_term_features": {"features": reduced, **without_short_term},
        },
    }
    paths.LEAKAGE_CHECK.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
