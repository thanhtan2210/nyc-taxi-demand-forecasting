"""Cached readers and shared display constants for the Streamlit app.

The app only reads files committed to the repository (mart, reports, metrics). It needs no
network, loads no model and does not import duckdb or xgboost. All computation lives in
src/analytics.py; this module reads files and caches results.
"""
import json

import pandas as pd
import pyarrow.dataset as ds
import streamlit as st

from src import analytics, paths
from src.config import load_config

SERVICE_LABELS = {"fhvhv": "High-volume FHV (Uber, Lyft)", "fhv": "Other FHV", "green": "Green taxi", "yellow": "Yellow taxi"}
# One fixed colour per entity, reused on every chart.
SERVICE_COLORS = {"fhvhv": "#2a78d6", "fhv": "#eb6834", "green": "#1baf7a", "yellow": "#eda100"}
# The services come from config/pipeline.yaml; the label dict only fixes the order they are drawn in.
SERVICES = sorted(load_config()["services"], key=list(SERVICE_LABELS).index)
# Boroughs in a fixed order, so a borough keeps its colour whatever is filtered.
BOROUGH_COLORS = {"Manhattan": "#2a78d6", "Brooklyn": "#eb6834", "Queens": "#1baf7a", "Bronx": "#eda100",
                  "Staten Island": "#e87ba4", "EWR": "#008300"}
# One-hue ramp for magnitudes, lightest step first.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BASELINE_COLOR = "#eb6834"
XGBOOST_COLOR = "#1baf7a"
PLOT_MARGIN = dict(l=10, r=10, t=10, b=10)
# Shared view of the zone maps: a dark basemap centred on New York City.
MAP_VIEW = dict(map_style="carto-darkmatter", center={"lat": 40.70, "lon": -73.97}, zoom=9.5)
MAP_NO_DATA_COLOR = "#898781"
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
    # Read all partitions into one Arrow table and convert once, to keep peak memory low.
    # The borough is rebuilt from the zone id as a categorical instead of reading 2.7M strings.
    files = [str(p) for p in sorted(paths.DEMAND_MART.glob("month=*/part.parquet"))]
    table = ds.dataset(files, format="parquet").to_table(columns=["hour", "zone_id"] + analytics.TRIP_COLUMNS)
    mart = table.to_pandas(self_destruct=True, split_blocks=True)
    del table
    mart["zone_id"] = mart["zone_id"].astype("int16")
    borough_of_zone = load_zones().set_index("zone_id")["borough"]
    mart.insert(2, "borough", mart["zone_id"].map(borough_of_zone).astype("category"))
    return mart


@st.cache_data
def load_zones():
    # keep_default_na=False: the lookup uses the literal text "N/A"
    return pd.read_csv(paths.DIM_ZONE, keep_default_na=False)


@st.cache_data
def load_zone_shapes():
    """Zone boundaries as GeoJSON (built once by scripts/fetch_zone_shapes.py)."""
    return json.loads(paths.ZONE_SHAPES.read_text(encoding="utf-8"))


@st.cache_data
def load_quality_entries():
    return json.loads(paths.DATA_QUALITY.read_text(encoding="utf-8"))["entries"]


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
    return json.loads(paths.METRICS.read_text(encoding="utf-8"))


@st.cache_data
def load_backtest():
    backtest = pd.read_parquet(paths.BACKTEST)
    zones = load_zones()[["zone_id", "borough"]]
    backtest = backtest.merge(zones, on="zone_id", how="left", validate="many_to_one")
    backtest["borough"] = backtest["borough"].astype("category")
    return backtest


@st.cache_data
def load_borough_metrics():
    return pd.read_csv(paths.BOROUGH_METRICS)


def zone_labels():
    """zone_id -> 'Zone name, Borough [id]' (zone names alone are not unique)."""
    zones = load_zones()
    return {z.zone_id: f"{z.zone}, {z.borough} [{z.zone_id}]" for z in zones.itertuples()}
