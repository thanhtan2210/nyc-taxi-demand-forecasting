"""Smoke tests of the Streamlit app on the committed mart and reports (no network, no model)."""
import ast
import base64
import json
from datetime import date
from pathlib import Path

import numpy as np

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 120


def open_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()


def map_traces(at):
    """The choropleth map traces among the Plotly charts of the page."""
    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    return [trace for spec in specs for trace in spec["data"] if trace["type"] == "choroplethmap"]


def decoded(values):
    """Plotly sends numeric arrays either as lists or as base64 typed arrays."""
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"]).tolist()
    return list(values)


def test_overview_page_renders():
    at = open_app()  # Overview is the default page
    assert not at.exception
    assert at.title[0].value == "NYC ride demand"
    assert [m.label for m in at.metric] == ["Zone-attributed trips", "Period", "Taxi zones"]
    assert at.metric[2].value == "263"
    # the reconciliation line and the FHV note are built from reports/data_quality.json
    assert any("trips in the TLC source files" in m.value for m in at.markdown)
    assert any("have no pickup zone" in c.value for c in at.caption)
    assert "Source vs mart by service" in [e.label for e in at.expander]
    assert len(at.get("plotly_chart")) == 5
    # every chart has a table twin
    assert sum(e.label.endswith("as a table") for e in at.expander) == 5
    # the zone map draws one polygon value per zone of the mart
    maps = map_traces(at)
    assert len(maps) == 1
    assert sorted(decoded(maps[0]["locations"])) == list(range(1, 264))
    assert any("Zone boundaries: NYC TLC taxi zone shapefile" in c.value for c in at.caption)


def test_patterns_page_renders_and_reacts_to_selection():
    at = open_app().switch_page("src/app_pages/patterns.py").run()
    assert not at.exception
    assert at.title[0].value == "Demand patterns"
    assert [s.value for s in at.subheader] == [
        "Weekly rhythm", "Monthly trend by borough", "Service mix by borough", "Unusual days", "Zone explorer"]
    assert [s.label for s in at.selectbox] == ["Area", "Service", "Zone"]
    assert at.selectbox[0].options[0] == "City" and len(at.selectbox[0].options) == 7
    assert len(at.selectbox[2].options) == 263
    assert [m.label for m in at.metric] == ["Trips in the period", "Average trips per hour", "Busiest hour of the day"]
    # heatmap, borough trend, service mix, zone profile, zone trend, zone service mix
    assert len(at.get("plotly_chart")) == 6
    assert sum(e.label.endswith("as a table") for e in at.expander) == 6
    assert any("No cause is attributed" in c.value for c in at.caption)
    busiest_zone_trips = at.metric[0].value

    at.selectbox[0].select("Queens").run()
    at.selectbox[1].select("Yellow taxi").run()
    at.selectbox[2].select_index(200).run()
    assert not at.exception
    assert at.metric[0].value != busiest_zone_trips
    # a zone without any trip must not break the page
    at.selectbox[2].select_index(262).run()
    assert not at.exception

def test_forecast_page_renders_and_reacts_to_selection():
    at = open_app().switch_page("src/app_pages/forecast.py").run()
    assert not at.exception
    assert at.title[0].value == "Forecast backtest"
    assert [s.label for s in at.selectbox] == ["Test month", "Borough", "Week", "Hour"]
    assert at.selectbox[0].options == ["2026-05", "2026-06", "2026-07"]
    assert len(at.selectbox[2].options) == 5  # May 2026 in seven-day windows
    assert at.selectbox[2].index == 0         # the first week is the default
    assert [s.value.split(":")[0].split(",")[0] for s in at.subheader] == [
        "Manhattan", "Accuracy by period", "Accuracy by borough", "Error by hour of day", "Zone map", "Largest misses"]
    assert len(at.get("plotly_chart")) == 3  # backtest lines, error by hour of day, zone map
    assert any("not a live forecast" in c.value for c in at.caption)
    assert any("trips per zone-hour" in c.value for c in at.caption)
    assert any("No cause is attributed" in c.value for c in at.caption)

    at.selectbox[0].select("2026-07").run()
    at.selectbox[2].select_index(4).run()
    at.radio[0].set_value("Zone").run()
    assert not at.exception
    assert [s.label for s in at.selectbox] == ["Test month", "Zone", "Week", "Hour"]
    at.selectbox[1].select_index(3).run()
    assert not at.exception
    assert "July 2026" in at.subheader[0].value


def test_forecast_map_for_one_hour():
    at = open_app().switch_page("src/app_pages/forecast.py").run()
    assert [r.label for r in at.radio] == ["Level", "Map view"]
    assert at.radio[1].value == "Forecast for one hour"  # the default view
    assert at.date_input[0].value == date(2026, 5, 1)    # first day of the test month
    assert at.selectbox[3].value == "18:00"

    maps = map_traces(at)
    assert [m["name"] for m in maps] == ["XGBoost forecast"]
    assert sorted(decoded(maps[0]["locations"])) == list(range(1, 264))
    assert [m.label for m in at.metric] == ["Forecast, whole city", "Actual, whole city", "MAE per zone: XGBoost vs baseline"]
    assert any("the model saw data up to the previous hour only" in c.value for c in at.caption)
    assert "Forecast for one hour as a table" in [e.label for e in at.expander]
    evening_total = at.metric[0].value
    evening_colours = decoded(maps[0]["z"])

    # another date and hour: the page still renders and shows different numbers
    at.date_input[0].set_value(date(2026, 5, 20)).run()
    at.selectbox[3].select("04:00").run()
    assert not at.exception
    assert any("2026-05-20 04:00" in c.value for c in at.caption)
    assert at.metric[0].value != evening_total
    assert decoded(map_traces(at)[0]["z"]) != evening_colours

    # another month: the date goes back to the first day of that month
    at.selectbox[0].select("2026-06").run()
    assert not at.exception
    assert at.date_input[0].value == date(2026, 6, 1)


def test_forecast_map_of_monthly_error():
    at = open_app().switch_page("src/app_pages/forecast.py").run()
    at.radio[1].set_value("Error by zone (month)").run()
    assert not at.exception
    # a coloured trace and a grey one that together cover every zone once
    maps = map_traces(at)
    assert [m["name"] for m in maps] == ["WAPE", "Too few trips"]
    assert sorted(decoded(maps[0]["locations"]) + decoded(maps[1]["locations"])) == list(range(1, 264))
    assert any("too few trips for a stable percentage error" in c.value for c in at.caption)
    assert "Error by zone as a table" in [e.label for e in at.expander]
    assert len(at.date_input) == 0 and [s.label for s in at.selectbox] == ["Test month", "Borough", "Week"]
    grey_zones = len(decoded(maps[1]["locations"]))
    assert any(f"Grey: {grey_zones} zones" in c.value for c in at.caption)

    at.selectbox[0].select("2026-07").run()
    assert not at.exception
    assert len(map_traces(at)) == 2


def test_app_does_not_import_the_warehouse_or_the_model():
    """The deployed app must work from files alone: no duckdb, no xgboost, no model, no network."""
    sources = [ROOT / "app.py", ROOT / "src" / "app_data.py", ROOT / "src" / "analytics.py",
               *(ROOT / "src" / "app_pages").glob("*.py")]
    assert len(sources) == 6
    allowed_local = {"src", "src.app_data", "src.analytics"}
    for source in sources:
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.module == "src":
                    assert {alias.name for alias in node.names} <= {"analytics", "app_data"}, source
            else:
                continue
            for module in modules:
                assert module.split(".")[0] not in {"duckdb", "xgboost", "joblib", "requests"}, source
                assert not module.startswith("src") or module in allowed_local, f"{source} imports {module}"