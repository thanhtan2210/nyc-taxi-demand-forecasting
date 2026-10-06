"""Default locations of every file the project reads or writes, relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "config" / "pipeline.yaml"
SQL_DIR = ROOT / "sql"

# Local DuckDB file, rebuilt by `python -m src.warehouse` and never committed
WAREHOUSE_DB = ROOT / "warehouse" / "nyc_taxi.duckdb"

# Committed outputs of the warehouse build
MART_DIR = ROOT / "data" / "mart"
DEMAND_MART = MART_DIR / "demand_hourly"
DIM_ZONE = MART_DIR / "dim_zone.csv"
ZONE_SHAPES = MART_DIR / "taxi_zones.geojson"
WEATHER = ROOT / "data" / "external" / "weather_hourly.csv"

MODELS_DIR = ROOT / "models"
METRICS = MODELS_DIR / "metrics.json"

REPORTS_DIR = ROOT / "reports"
DATA_QUALITY = REPORTS_DIR / "data_quality.json"
BACKTEST = REPORTS_DIR / "backtest_hourly.parquet"
BOROUGH_METRICS = REPORTS_DIR / "metrics_by_borough.csv"
LEAKAGE_CHECK = REPORTS_DIR / "leakage_check.json"
DOUBLE_COUNT_CHECK = REPORTS_DIR / "double_count_check.json"
UNUSUAL_DAYS = REPORTS_DIR / "unusual_days.csv"
