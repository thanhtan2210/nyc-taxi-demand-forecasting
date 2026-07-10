import os
import polars as pl
from datetime import datetime
from google.cloud import bigquery as bq
from ..utils.database import get_bq_client

def load_dim_location_to_bq(csv_path):
    """Ingests the Dim_Location table from a lookup CSV source."""
    client = get_bq_client()
    project_id = os.getenv("BQ_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET_ID", "nyc_taxi_dw")
    table_id = f"{project_id}.{dataset_id}.Dim_Location"
    
    if not os.path.exists(csv_path):
        print(f"[BQ] ERROR: Source file {csv_path} not found.")
        return

    df = pl.read_csv(csv_path)
    df.columns = ["Location_ID", "Borough", "Zone", "Service_Zone"]
    
    # Refresh the dimension table with standard configuration
    job_config = bq.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df.to_pandas(), table_id, job_config=job_config).result()
    print(f"[BQ] Dim_Location synchronized successfully.")

def load_dim_time_to_bq(start_date='2025-06-01', end_date='2025-11-30'):
    """Generates and ingests a high-fidelity Dim_Time table for analytical and ML purposes."""
    client = get_bq_client()
    project_id = os.getenv("BQ_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET_ID", "nyc_taxi_dw")
    table_id = f"{project_id}.{dataset_id}.Dim_Time"
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate continuous hourly sequence
    df = pl.datetime_range(start, end, interval="1h", eager=True).alias("Full_Date").to_frame()
    
    df = df.with_columns([
        pl.col("Full_Date").dt.strftime("%Y%m%d%H").cast(pl.Int64).alias("Time_Key"),
        pl.col("Full_Date").dt.hour().alias("Hour"),
        pl.col("Full_Date").dt.day().alias("Day"),
        pl.col("Full_Date").dt.month().alias("Month"),
        pl.col("Full_Date").dt.weekday().alias("Day_of_Week_Number"),
        (pl.col("Full_Date").dt.weekday() >= 6).alias("Is_Weekend")
    ])
    
    # Feature Engineering: Peak Demand (Rush Hour) identification
    df = df.with_columns(
        pl.when((pl.col("Hour").is_between(7, 9)) | (pl.col("Hour").is_between(16, 19)))
        .then(True).otherwise(False).alias("Is_Rush_Hour")
    )
    
    # Feature Engineering: Operational Shifts
    df = df.with_columns(
        pl.when(pl.col("Hour").is_between(5, 11)).then(pl.lit("Morning"))
        .when(pl.col("Hour").is_between(12, 16)).then(pl.lit("Afternoon"))
        .when(pl.col("Hour").is_between(17, 21)).then(pl.lit("Evening"))
        .otherwise(pl.lit("Night")).alias("Shift_Name")
    )
    
    # UI Localization: Mapping ISO weekdays to names
    days_map = {
        1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 
        5: "Friday", 6: "Saturday", 7: "Sunday"
    }
    df = df.with_columns(
        pl.col("Day_of_Week_Number").cast(pl.Utf8).replace(days_map).alias("Day_of_Week_Name")
    )

    job_config = bq.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df.to_pandas(), table_id, job_config=job_config).result()
    print(f"[BQ] Dim_Time synchronized successfully.")

def load_dim_service_type_to_bq():
    """Initializes and loads the static Dim_Service_Type dimension."""
    client = get_bq_client()
    project_id = os.getenv("BQ_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET_ID", "nyc_taxi_dw")
    table_id = f"{project_id}.{dataset_id}.Dim_Service_Type"
    
    data = {
        "Service_Type_Key": [1, 2, 3, 4],
        "Service_Name": ["Yellow", "Green", "FHV", "FHVHV"],
        "Category": ["Street-hail", "Street-hail", "App-based", "App-based"]
    }
    df = pl.DataFrame(data)
    
    job_config = bq.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df.to_pandas(), table_id, job_config=job_config).result()
    print(f"[BQ] Dim_Service_Type initialized successfully.")

def load_dim_weather_to_bq():
    """Synchronizes Dim_Weather from CSV or initializes a validated schema."""
    client = get_bq_client()
    project_id = os.getenv("BQ_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET_ID", "nyc_taxi_dw")
    table_id = f"{project_id}.{dataset_id}.Dim_Weather"

    csv_path = "dataset/nyc_weather_2025.csv"
    if not os.path.exists(csv_path):
        # Fallback: Initialize an empty table with the validated dimension schema
        schema = [
            bq.SchemaField("Weather_Key", "INT64", mode="REQUIRED"),
            bq.SchemaField("Temperature", "FLOAT64"),
            bq.SchemaField("Precipitation", "FLOAT64"),
            bq.SchemaField("Condition", "STRING"),
        ]
        table = bq.Table(table_id, schema=schema)
        client.create_table(table, exists_ok=True)
        print(f"[BQ] Dim_Weather schema placeholder initialized.")
    else:
        df = pl.read_csv(csv_path)
        job_config = bq.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        client.load_table_from_dataframe(df.to_pandas(), table_id, job_config=job_config).result()
        print(f"[BQ] Dim_Weather synchronized from local CSV.")
