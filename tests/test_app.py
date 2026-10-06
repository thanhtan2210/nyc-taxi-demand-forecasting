"""Smoke tests of the Streamlit app on the committed mart and reports (no network, no model)."""
import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 120


def open_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()


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
    assert len(at.get("plotly_chart")) == 4
    # every chart has a table twin
    assert sum(e.label.endswith("as a table") for e in at.expander) == 4


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
    assert [s.label for s in at.selectbox] == ["Test month", "Borough", "Week"]
    assert at.selectbox[0].options == ["2026-05", "2026-06", "2026-07"]
    assert len(at.selectbox[2].options) == 5  # May 2026 in seven-day windows
    assert at.selectbox[2].index == 0         # the first week is the default
    assert len(at.get("plotly_chart")) == 1
    assert len(at.dataframe) == 3  # hourly values, accuracy by period, accuracy by borough
    assert any("not a live forecast" in c.value for c in at.caption)
    assert any("trips per zone-hour" in c.value for c in at.caption)

    at.selectbox[0].select("2026-07").run()
    at.selectbox[2].select_index(4).run()
    at.radio[0].set_value("Zone").run()
    assert not at.exception
    assert [s.label for s in at.selectbox] == ["Test month", "Zone", "Week"]
    at.selectbox[1].select_index(3).run()
    assert not at.exception
    assert "July 2026" in at.subheader[0].value


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