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
