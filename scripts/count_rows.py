import os
import glob
import polars as pl
import pandas as pd
from src.transformers import polars_engine

def get_files(base_dir, category):
    search_pattern = os.path.join(base_dir, category, "*.parquet")
    return sorted(glob.glob(search_pattern))

def main():
    base_dir = "dataset/Trip_Record"
    categories = ["yellow", "green", "fhv", "fhvhv"]
    
    results = []
    
    print("Starting row counts calculations (this may take a minute as we load parquet files)...")
    for cat in categories:
        files = get_files(base_dir, cat)
        print(f"\nCategory: {cat.upper()} ({len(files)} files)")
        for file_path in files:
            file_name = os.path.basename(file_path)
            try:
                # 1. Scan raw
                lf_raw = pl.scan_parquet(file_path)
                raw_rows = lf_raw.select(pl.len()).collect().item()
                
                # 2. Standardize & clean
                lf_std = polars_engine.standardize_columns(lf_raw)
                lf_cleaned = polars_engine.apply_cleaning_logic(lf_std, cat)
                cleaned_rows = lf_cleaned.select(pl.len()).collect().item()
                
                retention_rate = (cleaned_rows / raw_rows * 100) if raw_rows > 0 else 0.0
                
                print(f"  - {file_name}: Raw={raw_rows:,} | Cleaned={cleaned_rows:,} ({retention_rate:.2f}%)")
                
                results.append({
                    "category": cat,
                    "file_name": file_name,
                    "raw_rows": raw_rows,
                    "cleaned_rows": cleaned_rows,
                    "status": "Success"
                })
            except Exception as e:
                print(f"  - {file_name}: ERROR {e}")
                results.append({
                    "category": cat,
                    "file_name": file_name,
                    "raw_rows": 0,
                    "cleaned_rows": 0,
                    "status": f"Failed: {str(e)}"
                })
                
    # Save results to csv
    df = pd.DataFrame(results)
    df.to_csv("etl_report_summary.csv", index=False)
    print("\nSuccessfully updated etl_report_summary.csv with real row counts!")

if __name__ == "__main__":
    main()
