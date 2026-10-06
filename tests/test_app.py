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
    assert [m.label for m in at.metric] == ["Total trips", "Period", "Taxi zones"]
    assert at.metric[2].value == "263"
    assert len(at.get("plotly_chart")) == 4
    # the FHV note is built from reports/data_quality.json, not typed in
    assert any("have no pickup zone" in c.value for c in at.caption)


def test_forecast_page_renders_and_reacts_to_selection():
    at = open_app().switch_page("src/app_pages/forecast.py").run()
    assert not at.exception
    assert at.title[0].value == "Forecast backtest"
    assert at.selectbox[0].options == ["2026-05", "2026-06", "2026-07"]
    assert len(at.get("plotly_chart")) == 1
    assert len(at.dataframe) == 3  # hourly values, accuracy by period, accuracy by borough
    assert any("not a live forecast" in c.value for c in at.caption)

    at.selectbox[0].select("2026-07").run()
    at.radio[0].set_value("Zone").run()
    assert not at.exception
    assert at.selectbox[1].label == "Zone"
    assert "July 2026" in at.subheader[0].value


def test_app_does_not_import_the_warehouse_or_the_model():
    """The deployed app must work from files alone: no duckdb, no xgboost, no joblib model."""
    sources = [ROOT / "app.py", ROOT / "src" / "app_data.py", *(ROOT / "src" / "app_pages").glob("*.py")]
    assert len(sources) == 4
    for source in sources:
        imported = set()
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
                if node.module and node.module.startswith("src."):
                    assert node.module == "src.app_data", f"{source} imports {node.module}"
        assert not imported & {"duckdb", "xgboost", "joblib", "requests"}, source
