"""Cached readers for the Streamlit app.

The app only reads files committed to the repository (mart, reports, metrics). It needs no
network, loads no model and does not import duckdb or xgboost.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MART_DIR = ROOT / "data" / "mart" / "demand_hourly"
DIM_ZONE = ROOT / "data" / "mart" / "dim_zone.csv"
DATA_QUALITY = ROOT / "reports" / "data_quality.json"
BACKTEST = ROOT / "reports" / "backtest_hourly.parquet"
METRICS = ROOT / "models" / "metrics.json"
BOROUGH_METRICS = ROOT / "reports" / "metrics_by_borough.csv"

SERVICES = ["fhvhv", "fhv", "green", "yellow"]
SERVICE_LABELS = {"fhvhv": "High-volume FHV (Uber, Lyft)", "fhv": "Other FHV", "green": "Green taxi", "yellow": "Yellow taxi"}
# One fixed colour per entity, reused on every chart.
SERVICE_COLORS = {"fhvhv": "#2a78d6", "fhv": "#eb6834", "green": "#1baf7a", "yellow": "#eda100"}
SERIES_COLORS = {"Actual": "#2a78d6", "Last-week baseline": "#eb6834", "XGBoost": "#1baf7a"}


@st.cache_data
def load_zones():
    # keep_default_na=False: the lookup uses the literal text "N/A"
    return pd.read_csv(DIM_ZONE, keep_default_na=False)


@st.cache_data
def load_overview():
    """Aggregates the hourly mart once into the small tables the Overview page draws."""
    columns = ["hour", "zone_id", "trips_total"] + [f"trips_{s}" for s in SERVICES]
    mart = pd.concat(
        [pd.read_parquet(p, columns=columns) for p in sorted(MART_DIR.glob("month=*/part.parquet"))],
        ignore_index=True,
    )
    service_columns = [f"trips_{s}" for s in SERVICES]
    daily = mart.groupby(mart["hour"].dt.normalize())[service_columns].sum()
    daily.index.name = "date"
    daily.columns = SERVICES

    citywide = mart.groupby("hour")["trips_total"].sum().reset_index()
    citywide["hour_of_day"] = citywide["hour"].dt.hour
    citywide["day_type"] = citywide["hour"].dt.dayofweek.map(lambda d: "Weekend" if d >= 5 else "Weekday")
    profile = citywide.groupby(["day_type", "hour_of_day"])["trips_total"].mean().reset_index()

    return {
        "total_trips": int(mart["trips_total"].sum()),
        "first_hour": mart["hour"].min(),
        "last_hour": mart["hour"].max(),
        "n_zones": int(mart["zone_id"].nunique()),
        "daily": daily.reset_index(),
        "service_totals": mart[service_columns].sum().rename(dict(zip(service_columns, SERVICES))),
        "zone_totals": mart.groupby("zone_id")["trips_total"].sum().reset_index(),
        "profile": profile,
    }


@st.cache_data
def load_fhv_null_zone_range():
    """Lowest and highest monthly share of FHV rows without a pickup zone, in percent."""
    entries = json.loads(DATA_QUALITY.read_text(encoding="utf-8"))["entries"]
    shares = [100 * e["rows_zone_null"] / e["rows_read"] for e in entries if e["service"] == "fhv"]
    return min(shares), max(shares)


@st.cache_data
def load_metrics():
    return json.loads(METRICS.read_text(encoding="utf-8"))


@st.cache_data
def load_backtest():
    backtest = pd.read_parquet(BACKTEST)
    zones = load_zones()[["zone_id", "borough", "zone"]]
    return backtest.merge(zones, on="zone_id", how="left", validate="many_to_one")


@st.cache_data
def load_borough_metrics():
    return pd.read_csv(BOROUGH_METRICS)
