"""Cached readers and shared display constants for the Streamlit app.

The app only reads files committed to the repository (mart, reports, metrics). It needs no
network, loads no model and does not import duckdb or xgboost. All computation lives in
src/analytics.py; this module reads files and caches results.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import analytics

ROOT = Path(__file__).resolve().parent.parent
MART_DIR = ROOT / "data" / "mart" / "demand_hourly"
DIM_ZONE = ROOT / "data" / "mart" / "dim_zone.csv"
DATA_QUALITY = ROOT / "reports" / "data_quality.json"
BACKTEST = ROOT / "reports" / "backtest_hourly.parquet"
METRICS = ROOT / "models" / "metrics.json"
BOROUGH_METRICS = ROOT / "reports" / "metrics_by_borough.csv"

SERVICES = analytics.SERVICES
SERVICE_LABELS = {"fhvhv": "High-volume FHV (Uber, Lyft)", "fhv": "Other FHV", "green": "Green taxi", "yellow": "Yellow taxi"}
# One fixed colour per entity, reused on every chart.
SERVICE_COLORS = {"fhvhv": "#2a78d6", "fhv": "#eb6834", "green": "#1baf7a", "yellow": "#eda100"}
# Boroughs in a fixed order, so a borough keeps its colour whatever is filtered.
BOROUGH_COLORS = {"Manhattan": "#2a78d6", "Brooklyn": "#eb6834", "Queens": "#1baf7a", "Bronx": "#eda100",
                  "Staten Island": "#e87ba4", "EWR": "#008300"}
# One-hue ramp for magnitudes, lightest step first.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BASELINE_COLOR = "#eb6834"
XGBOOST_COLOR = "#1baf7a"
PLOT_MARGIN = dict(l=10, r=10, t=10, b=10)
LEGEND_TOP = dict(orientation="h", y=1.12, x=0, title=None)


def neutral_ink():
    """A neutral line colour that stays readable on the active (light or dark) theme."""
    try:
        return "#ffffff" if st.context.theme.type == "dark" else "#0b0b0b"
    except Exception:  # no theme information outside a browser session
        return "#898781"


def is_dark_theme():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def magnitude_scale():
    """The blue ramp ordered so that low values recede into the active theme's background."""
    return BLUE_RAMP[::-1] if is_dark_theme() else BLUE_RAMP


@st.cache_data(show_spinner="Reading the demand mart")
def load_mart():
    """The whole hourly mart, read once per server process."""
    columns = ["hour", "zone_id", "borough"] + analytics.TRIP_COLUMNS
    mart = pd.concat(
        [pd.read_parquet(p, columns=columns) for p in sorted(MART_DIR.glob("month=*/part.parquet"))],
        ignore_index=True,
    )
    mart["borough"] = mart["borough"].astype("category")
    mart["zone_id"] = mart["zone_id"].astype("int16")
    return mart


@st.cache_data
def load_zones():
    # keep_default_na=False: the lookup uses the literal text "N/A"
    return pd.read_csv(DIM_ZONE, keep_default_na=False)


@st.cache_data
def load_quality_entries():
    return json.loads(DATA_QUALITY.read_text(encoding="utf-8"))["entries"]


@st.cache_data
def overview_tables():
    mart = load_mart()
    zone_table = analytics.zone_totals(mart, load_zones())
    return {
        "kpis": analytics.overview_kpis(mart),
        "daily": analytics.daily_totals(mart).reset_index(),
        "share": analytics.service_share(mart),
        "profile": analytics.hourly_profile(mart),
        "zones": zone_table,
    }


@st.cache_data
def borough_cube():
    return analytics.borough_hourly(load_mart())


@st.cache_data
def patterns_tables():
    cube = borough_cube()
    trend = analytics.monthly_trend(cube, by="borough")
    return {
        "boroughs": cube.groupby("borough", observed=True)["trips_total"].sum().sort_values(ascending=False).index.tolist(),
        "trend": trend,
        "month_over_month": analytics.month_over_month(trend, by="borough"),
        "service_mix": analytics.service_mix_by_borough(cube),
        "unusual_days": analytics.unusual_days(analytics.daily_totals(cube)),
    }


@st.cache_data
def weekly_rhythm(borough, service):
    """Weekday x hour table for one borough (None = whole city) and one service (None = all)."""
    cube = borough_cube()
    if borough is not None:
        cube = cube[cube["borough"] == borough]
    return analytics.weekday_hour_heatmap(cube, "trips_total" if service is None else f"trips_{service}")


@st.cache_data
def zone_frame(zone_id):
    mart = load_mart()
    return mart[mart["zone_id"] == zone_id].drop(columns="borough").reset_index(drop=True)


@st.cache_data
def load_metrics():
    return json.loads(METRICS.read_text(encoding="utf-8"))


@st.cache_data
def load_backtest():
    backtest = pd.read_parquet(BACKTEST)
    zones = load_zones()[["zone_id", "borough"]]
    backtest = backtest.merge(zones, on="zone_id", how="left", validate="many_to_one")
    backtest["borough"] = backtest["borough"].astype("category")
    return backtest


@st.cache_data
def load_borough_metrics():
    return pd.read_csv(BOROUGH_METRICS)


def zone_labels():
    """zone_id -> 'Zone name, Borough [id]' (zone names alone are not unique)."""
    zones = load_zones()
    return {z.zone_id: f"{z.zone}, {z.borough} [{z.zone_id}]" for z in zones.itertuples()}
