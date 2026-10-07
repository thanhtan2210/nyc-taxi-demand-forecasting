"""Feature engineering for the hourly zone demand model.

Every lag and rolling feature of hour t is computed from hours <= t-1 of the same zone,
so a forecast for the next hour never sees the value it is predicting.
"""
from pathlib import Path

import pandas as pd

from .timeutils import hour_calendar

TARGET = "trips_total"
LAGS = [1, 2, 3, 24, 168]
ROLLING_WINDOW = 24
BASELINE = "lag_168"  # same hour, one week earlier
CALENDAR_FEATURES = ["hour_of_day", "day_of_week", "is_holiday"]
LAG_FEATURES = [f"lag_{k}" for k in LAGS] + [f"roll_mean_{ROLLING_WINDOW}"]
BASE_FEATURES = ["zone_id"] + CALENDAR_FEATURES + LAG_FEATURES
WEATHER_FEATURES = ["temperature_2m", "precipitation"]


def load_mart(mart_dir):
    """Reads every month partition of the demand mart into one frame."""
    parts = sorted(Path(mart_dir).glob("month=*/part.parquet"))
    if not parts:
        raise FileNotFoundError(f"No mart partitions under {mart_dir}; run `python -m src.warehouse` first")
    columns = ["hour", "zone_id", "borough", TARGET]
    return pd.concat([pd.read_parquet(p, columns=columns) for p in parts], ignore_index=True)


def build_features(demand, weather=None):
    """Returns one row per (zone, hour) with the target, the baseline and all features.

    `demand` must be a complete hourly grid per zone (the mart guarantees it), because lags
    are taken by row position. Rows whose lags reach before the first hour hold NaN.
    """
    df = demand.sort_values(["zone_id", "hour"]).reset_index(drop=True)
    steps = df.groupby("zone_id", sort=False)["hour"].diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError("demand is not a complete hourly grid per zone")

    by_zone = df.groupby("zone_id", sort=False)[TARGET]
    for k in LAGS:
        df[f"lag_{k}"] = by_zone.shift(k)
    # shift(1) first: the window of hour t covers hours t-24 .. t-1
    df[f"roll_mean_{ROLLING_WINDOW}"] = by_zone.transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW).mean()
    )

    calendar = hour_calendar(pd.DatetimeIndex(df["hour"].unique()).sort_values())
    calendar["is_holiday"] = calendar["is_holiday"].astype("int8")
    df = df.merge(calendar[["hour"] + CALENDAR_FEATURES], on="hour", how="left", validate="many_to_one")

    if weather is not None:
        df = df.merge(weather[["hour"] + WEATHER_FEATURES], on="hour", how="left", validate="many_to_one")

    df["zone_id"] = df["zone_id"].astype("category")
    return df
