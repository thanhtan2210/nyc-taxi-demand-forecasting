import os
import pandas as pd
import numpy as np
import joblib
from google.cloud import bigquery
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

from ..features.build_features import engineer_features

def fetch_data_from_bq():
    """Extracts cleaned aggregation data from BigQuery for model training."""
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env'))
    load_dotenv(dotenv_path)
    
    client = bigquery.Client()
    project_id = os.getenv("BQ_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET_ID", "nyc_taxi_dw")

    query = f"""
    SELECT 
        f.pickup_time_key as Time_Key, 
        f.pulocationid as PULocation_Key, 
        f.total_demand,
        w.Temperature,
        w.Precipitation
    FROM `{project_id}.{dataset_id}.Fact_Demand_Hourly` f
    LEFT JOIN `{project_id}.{dataset_id}.Dim_Weather` w ON f.pickup_time_key = w.Weather_Key
    WHERE f.service_type_key = 1
    ORDER BY PULocation_Key, Time_Key
    """
    print(">>> Extracting high-signal data from BigQuery...")
    return client.query(query).to_dataframe()

def evaluate_metrics(model_name, y_true, y_pred):
    """Logs standard regression performance metrics to the console."""
    print(f"--- {model_name} Performance ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_true, y_pred):.2f}\n")

def run_automated_training():
    """Executes the full training cycle for the XGBoost demand model."""
    
    # 1. Ingestion
    df_raw = fetch_data_from_bq()
    
    # 2. Advanced Feature Engineering
    df_features = engineer_features(df_raw)
    
    # 3. Chronological Train/Test Splitting (Preventing Data Leakage)
    train_df = df_features[df_features['Datetime'] < '2025-11-01']
    test_df = df_features[df_features['Datetime'] >= '2025-11-01']

    features_cols = [
        'PULocation_Key', 'Hour', 'DayOfWeek', 'Is_Weekend', 'hour_sin', 'hour_cos',
        'lag_1h', 'lag_2h', 'lag_24h', 'lag_168h', 'rolling_mean_6h',
        'Temperature', 'Precipitation'
    ]
    target_col = 'total_demand'

    X_train, y_train = train_df[features_cols], train_df[target_col]
    X_test, y_test = test_df[features_cols], test_df[target_col]

    print(">>> Training XGBoost...")
    xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    evaluate_metrics("XGBoost", y_test, xgb_model.predict(X_test))

    # --- ARTIFACT PERSISTENCE ---
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    os.makedirs(os.path.join(root_path, 'saved_models/xgboost'), exist_ok=True)
    joblib.dump(xgb_model, os.path.join(root_path, 'saved_models/xgboost/xgb_demand_model_v1.joblib'))
    
    print(f">>> Artifacts stored in {os.path.join(root_path, 'saved_models')}")

if __name__ == "__main__":
    run_automated_training()
