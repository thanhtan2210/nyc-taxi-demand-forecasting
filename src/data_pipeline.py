import os
import polars as pl
import pandas as pd
import gc
from .extractors.extract import get_files, scan_data
from .transformers import polars_engine, bq_engine
from .loaders.local import ensure_output_dir, save_data, sink_data
from .loaders.bigquery import load_parquet_to_bq
from .loaders.bigquery_dims import (
    load_dim_location_to_bq, load_dim_time_to_bq, 
    load_dim_service_type_to_bq, load_dim_weather_to_bq
)
from .loaders.bigquery_facts import load_to_fact_trips, load_to_fact_demand_hourly

def run_pipeline(engine="polars", load_raw=False, load_clean=False, load_dims=False, target_cat=None):
    """
    Main Orchestrator: Dispatches workloads to either the Local (Polars) or Cloud (BigQuery) engine.
    
    Args:
        engine (str): The processing engine ('polars' or 'bigquery').
        load_raw (bool): Whether to ingest raw data into the Staging area.
        load_clean (bool): Whether to process and ingest cleaned data into the DW.
        load_dims (bool): Whether to refresh dimension tables.
        target_cat (str): Specific taxi category to process (e.g., 'yellow').
    """
    print("="*60)
    print(f"NYC TAXI - ETL PIPELINE (Engine: {engine.upper()})")
    print("="*60)

    # --- 1. DIMENSION LOADING (Shared logic) ---
    if load_dims:
        run_dimensions_load()

    # --- 2. FACT DATA PROCESSING ---
    all_categories = ["yellow", "green", "fhv", "fhvhv"]
    categories = [target_cat] if target_cat else all_categories

    # Initialize Table Schemas before fact processing (prevents type mismatch)
    if load_clean or engine == "bigquery":
        bq_engine.execute_sql_file("sql/ddl_create_tables.sql")

    if engine == "polars":
        for cat in categories:
            run_local_polars_etl(cat, load_raw, load_clean)
    elif engine == "bigquery":
        for cat in categories:
            run_cloud_bq_elt(cat, load_raw)
    else:
        print(f"[ERROR] Engine '{engine}' is not supported.")

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

def run_local_polars_etl(cat, load_raw, load_clean):
    """
    STRATEGY 1: Local ETL utilizing Polars Streaming API for memory efficiency.
    """
    input_base = os.getenv("RAW_DATA_DIR", "dataset/Trip_Record")
    output_base = os.getenv("PROCESSED_DATA_DIR", "dataset/processed")
    
    files = get_files(input_base, cat)
    output_dir = ensure_output_dir(output_base, cat)
    agg_output_dir = ensure_output_dir(output_base, f"aggregated/{cat}")
    
    print(f"\n>>> [Polars] Processing Category: {cat.upper()} ({len(files)} files)")
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            # 1. Extract & Optional Raw Ingestion
            lf_raw = scan_data(file_path)
            if load_raw:
                load_parquet_to_bq(file_path, cat, is_raw=True)
            
            # 2. Transform (Applying domain-specific cleaning rules)
            lf_std = polars_engine.standardize_columns(lf_raw)
            lf_cleaned = polars_engine.apply_cleaning_logic(lf_std, cat)
            
            # Schema Validation using Pandera (validates a sample to be memory-safe)
            try:
                from .validators.schemas import CleanTripSchema
                sample_df = lf_cleaned.slice(0, 1000).collect().to_pandas()
                if len(sample_df) > 0:
                    CleanTripSchema.validate(sample_df)
                    print(f"    [PANDERA] Clean schema validated successfully for {file_name}.")
            except Exception as ve:
                print(f"    [PANDERA] WARNING: Schema validation failed for {file_name}: {ve}")

            # 3. Load (Streaming Sink to Disk)
            saved_path = sink_data(lf_cleaned, output_dir, file_name)
            
            if load_clean:
                # Load Transactional Fact into BigQuery
                load_to_fact_trips(saved_path)
                
                # 4. Atomic Aggregation (Two-Pass Strategy)
                lf_agg = polars_engine.aggregate_trips(pl.scan_parquet(saved_path))
                df_agg_final = lf_agg.collect()
                agg_saved_path = save_data(df_agg_final, agg_output_dir, f"agg_{file_name}")
                
                # Load Aggregated Fact (Feature Store) into BigQuery
                load_to_fact_demand_hourly(agg_saved_path)
            
            gc.collect()
        except Exception as e:
            print(f"   [ERROR] Processing failure for {file_name}: {e}")

def run_cloud_bq_elt(cat, load_raw):
    """
    STRATEGY 2: Cloud ELT utilizing BigQuery SQL for massive scalability.
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
