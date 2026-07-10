import pytest
import polars as pl
from datetime import datetime
from src.transformers import transform, polars_engine

# --- Tests for src/transformers/transform.py ---

def test_transform_standardize_columns(sample_yellow_raw):
    lf_std = transform.standardize_columns(sample_yellow_raw)
    schema = lf_std.collect_schema()
    
    # Assert renamed columns exist
    assert "pickup_time" in schema.names()
    assert "dropoff_time" in schema.names()
    assert "distance" in schema.names()
    assert "fare" in schema.names()
    
    # Assert old column names do not exist
    assert "tpep_pickup_datetime" not in schema.names()
    assert "fare_amount" not in schema.names()

def test_transform_apply_cleaning_logic_yellow(sample_yellow_raw):
    lf_std = transform.standardize_columns(sample_yellow_raw)
    lf_cleaned = transform.apply_cleaning_logic(lf_std, "yellow")
    df_cleaned = lf_cleaned.collect()
    
    # 1. Row count should be reduced because of filters
    # - Out of date bounds (May 2025) -> filtered
    # - Distance 0.0 -> filtered
    # - Fare 1.0 -> filtered
    # - Duration 0 (same pickup/dropoff) -> filtered
    assert len(df_cleaned) == 3 # Only valid rows remain
    
    # 2. Schema check
    assert "time_key" in df_cleaned.columns
    assert "is_weekend" in df_cleaned.columns
    assert "duration_minutes" in df_cleaned.columns
    
    # 3. Imputation check: Null pulocationid or dolocationid mapped to 264
    # The row with null pulocationid (6th row in fixture) has valid dates, distance, fare, so it should be retained but imputed
    # Let's verify that pulocationid=264 exists in the output
    assert 264 in df_cleaned["pulocationid"].to_list()

def test_transform_apply_cleaning_logic_fhvhv(sample_fhvhv_raw):
    # FHVHV columns: pickup_datetime, base_passenger_fare, trip_miles
    # Need to standardize first
    lf_std = transform.standardize_columns(sample_fhvhv_raw)
    lf_cleaned = transform.apply_cleaning_logic(lf_std, "fhvhv")
    df_cleaned = lf_cleaned.collect()
    
    # Raw FHVHV has 3 rows:
    # 1. Valid
    # 2. Valid
    # 3. Distance outlier (150 miles > 100 limit) -> filtered
    assert len(df_cleaned) == 2
    
    # Flags shared_request_flag/shared_match_flag null imputation
    assert "shared_request_flag" in df_cleaned.columns
    assert "shared_match_flag" in df_cleaned.columns
    # check that null was filled with "N" (second row had None for shared_request_flag)
    assert "N" in df_cleaned["shared_request_flag"].to_list()

def test_transform_aggregate_trips(sample_yellow_raw):
    lf_std = transform.standardize_columns(sample_yellow_raw)
    lf_cleaned = transform.apply_cleaning_logic(lf_std, "yellow")
    lf_agg = transform.aggregate_trips(lf_cleaned)
    df_agg = lf_agg.collect()
    
    # Check aggregation schema
    expected_cols = {"pulocationid", "time_key", "trip_count", "total_revenue", "total_distance", "avg_passengers"}
    assert expected_cols.issubset(set(df_agg.columns))
    
    # Keys should not be null
    assert df_agg["pulocationid"].null_count() == 0
    assert df_agg["time_key"].null_count() == 0


# --- Tests for src/transformers/polars_engine.py ---

def test_polars_engine_standardize_columns(sample_yellow_raw):
    lf_std = polars_engine.standardize_columns(sample_yellow_raw)
    schema = lf_std.collect_schema()
    assert "pickup_time" in schema.names()
    assert "dropoff_time" in schema.names()
    assert "pulocationid" in schema.names()
    assert "dolocationid" in schema.names()

def test_polars_engine_apply_cleaning_logic_yellow(sample_yellow_raw):
    lf_std = polars_engine.standardize_columns(sample_yellow_raw)
    lf_cleaned = polars_engine.apply_cleaning_logic(lf_std, "yellow")
    df_cleaned = lf_cleaned.collect()
    
    # Check unified columns are created
    assert "ml_unified_fare" in df_cleaned.columns
    assert "ml_unified_distance" in df_cleaned.columns
    assert "ml_unified_duration" in df_cleaned.columns
    assert "pickup_time_key" in df_cleaned.columns
    assert "service_type_key" in df_cleaned.columns
    
    # Yellow service type key is 1
    assert (df_cleaned["service_type_key"] == 1).all()

def test_polars_engine_aggregate_trips(sample_yellow_raw):
    lf_std = polars_engine.standardize_columns(sample_yellow_raw)
    lf_cleaned = polars_engine.apply_cleaning_logic(lf_std, "yellow")
    lf_agg = polars_engine.aggregate_trips(lf_cleaned)
    df_agg = lf_agg.collect()
    
    expected_cols = {"pickup_time_key", "pulocationid", "service_type_key", "total_demand", "total_revenue_generated", "average_trip_distance", "average_duration"}
    assert expected_cols.issubset(set(df_agg.columns))

def test_pandera_validation(sample_yellow_raw):
    from src.validators.schemas import CleanTripSchema
    lf_std = polars_engine.standardize_columns(sample_yellow_raw)
    lf_cleaned = polars_engine.apply_cleaning_logic(lf_std, "yellow")
    
    # Collect data, convert to pandas
    df_pd = lf_cleaned.collect().to_pandas()
    
    # Schema should validate successfully
    validated_df = CleanTripSchema.validate(df_pd)
    assert validated_df is not None

