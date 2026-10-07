import pandas as pd

from src.config import load_config, month_range, parse_months


def test_config_covers_four_services_and_fourteen_months():
    cfg = load_config()
    assert list(cfg["services"]) == ["yellow", "green", "fhv", "fhvhv"]
    for columns in cfg["services"].values():
        assert set(columns) == {"pickup_column", "zone_column"}
    months = month_range(cfg["months"]["start"], cfg["months"]["end"])
    assert len(months) == 14
    assert months[0] == "2025-06" and months[-1] == "2026-07"


def test_split_is_chronological_and_does_not_overlap():
    split = load_config()["split"]
    train_end = pd.Timestamp(split["train"]["end"])
    val_start = pd.Timestamp(split["validation"]["start"])
    val_end = pd.Timestamp(split["validation"]["end"])
    first_test = pd.Timestamp(split["test_months"][0])
    assert pd.Timestamp(split["train"]["start"]) < train_end < val_start < val_end < first_test


def test_parse_months():
    assert parse_months("2025-11:2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert parse_months("2026-07") == ["2026-07"]


def test_every_configured_service_has_a_label_and_a_colour():
    from src import analytics, app_data

    services = list(load_config()["services"])
    assert set(app_data.SERVICES) == set(services)
    assert set(analytics.SERVICES) == set(services)
    for service in services:
        assert app_data.SERVICE_LABELS[service]
        assert app_data.SERVICE_COLORS[service].startswith("#")


def test_default_paths_point_inside_the_repository():
    from src import paths

    assert (paths.ROOT / "app.py").is_file()
    for path in (paths.CONFIG, paths.DIM_ZONE, paths.ZONE_SHAPES, paths.WEATHER, paths.METRICS, paths.DATA_QUALITY,
                 paths.BACKTEST, paths.BOROUGH_METRICS, paths.LEAKAGE_CHECK, paths.DOUBLE_COUNT_CHECK,
                 paths.UNUSUAL_DAYS):
        assert path.is_file(), path
    assert paths.SQL_DIR.is_dir() and paths.DEMAND_MART.is_dir()
