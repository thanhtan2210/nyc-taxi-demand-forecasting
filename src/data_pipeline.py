import os
from .extractors.extract import get_files
from .transformers import bq_engine
from .loaders.bigquery import load_parquet_to_bq
from .loaders.bigquery_dims import (
    load_dim_location_to_bq, load_dim_time_to_bq, 
    load_dim_service_type_to_bq, load_dim_weather_to_bq
)

def run_pipeline(load_raw=False, load_clean=False, load_dims=False, target_cat=None):
    """
    Main Orchestrator: runs the BigQuery ELT (staging -> Fact_Trips -> Fact_Demand_Hourly).

    Args:
        load_raw (bool): Whether to ingest raw data into the Staging area.
        load_clean (bool): Unused by the BigQuery ELT; the SQL transform always runs.
        load_dims (bool): Whether to refresh dimension tables.
        target_cat (str): Specific taxi category to process (e.g., 'yellow').
    """
    print("="*60)
    print("NYC TAXI - ELT PIPELINE (BigQuery)")
    print("="*60)

    # --- 1. DIMENSION LOADING (Shared logic) ---
    if load_dims:
        run_dimensions_load()

    # --- 2. FACT DATA PROCESSING ---
    all_categories = ["yellow", "green", "fhv", "fhvhv"]
    categories = [target_cat] if target_cat else all_categories

    # Initialize Table Schemas before fact processing (prevents type mismatch)
    bq_engine.execute_sql_file("sql/ddl_create_tables.sql")

    for cat in categories:
        run_cloud_bq_elt(cat, load_raw)

def run_dimensions_load():
    """Triggers the loading sequence for all dimension tables."""
    lookup_csv = os.getenv("TAXI_ZONE_LOOKUP", "dataset/taxi_zone_lookup.csv")
    print("\n>>> Initializing Dimension Table Refresh...")
    try:
        load_dim_location_to_bq(lookup_csv)
        load_dim_time_to_bq()
        load_dim_service_type_to_bq()
        load_dim_weather_to_bq()
    except Exception as e:
        print(f"   [ERROR] Dimension refresh failed: {e}")

def run_cloud_bq_elt(cat, load_raw):
    """
    Cloud ELT: loads raw parquet into staging, then transforms and aggregates with BigQuery SQL.
    """
    input_base = os.getenv("RAW_DATA_DIR", "dataset/Trip_Record")
    files = get_files(input_base, cat)

    # Naming convention mapping for diverse source columns
    col_mapping = {
        "yellow": {
            "pickup": "tpep_pickup_datetime", "dropoff": "tpep_dropoff_datetime",
            "dist": "trip_distance", "fare": "fare_amount", "pass": "passenger_count"
        },
        "green": {
            "pickup": "lpep_pickup_datetime", "dropoff": "lpep_dropoff_datetime",
            "dist": "trip_distance", "fare": "fare_amount", "pass": "passenger_count"
        },
        "fhvhv": {
            "pickup": "pickup_datetime", "dropoff": "dropoff_datetime",
            "dist": "trip_miles", "fare": "base_passenger_fare", "pass": "CAST(NULL AS INT64)"
        },
        "fhv": {
            "pickup": "pickup_datetime", "dropoff": "dropoff_datetime",
            "dist": "CAST(NULL AS FLOAT64)", "fare": "CAST(NULL AS FLOAT64)", "pass": "CAST(NULL AS INT64)"
        }
    }

    mapping = col_mapping.get(cat.lower())
    service_map = {"yellow": 1, "green": 2, "fhv": 3, "fhvhv": 4}
    service_key = service_map.get(cat.lower(), 0)

    print(f"\n>>> [BigQuery] Processing Category: {cat.upper()}")

    # 1. Ingest Raw Data into Staging Area
    if load_raw:
        print(f"    [STAGING] Ingesting {len(files)} raw files to cloud...")
        for file_path in files:
            load_parquet_to_bq(file_path, cat, is_raw=True)
    else:
        print(f"    [STAGING] Skipping ingestion (Data already exists in Staging).")

    # 2. Transform & Aggregate via Cloud SQL
    params = {
        "CATEGORY": cat, 
        "SERVICE_TYPE_KEY": service_key,
        "COL_PICKUP": mapping["pickup"],
        "COL_DROPOFF": mapping["dropoff"],
        "COL_DIST": mapping["dist"],
        "COL_FARE": mapping["fare"],
        "COL_PASS": mapping["pass"]
    }

    # Staging to Fact_Trips Transformation
    bq_engine.execute_sql_file("sql/transform_generic.sql", params=params)

    # Fact_Trips to Fact_Demand_Hourly Aggregation
    bq_engine.execute_sql_file("sql/aggregate_demand.sql", params=params)
