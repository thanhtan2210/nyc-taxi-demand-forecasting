import argparse
from src.data_pipeline import run_pipeline
from dotenv import load_dotenv

def main():
    """Entry point of the BigQuery ELT pipeline (staging -> star schema)."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="NYC Taxi ELT & Star Schema Master Script")
    parser.add_argument("--raw", action="store_true", help="Load RAW parquet files into the BigQuery staging dataset")
    parser.add_argument("--clean", action="store_true", help="Kept for compatibility: the SQL transform always runs")
    parser.add_argument("--dims", action="store_true", help="Load the Dimension tables (Time, Location, etc.) into BigQuery")
    parser.add_argument("--all", action="store_true", help="Run ALL steps (Dims + Raw + Clean)")
    parser.add_argument("--cat", type=str, choices=["yellow", "green", "fhv", "fhvhv"], help="Process a single category only")

    args = parser.parse_args()

    load_raw = args.raw or args.all
    load_clean = args.clean or args.all
    load_dims = args.dims or args.all

    print(f"[*] Config: Category={args.cat or 'ALL'}")
    print(f"[*] Actions: Raw={load_raw}, Clean={load_clean}, Dims={load_dims}")

    run_pipeline(
        load_raw=load_raw,
        load_clean=load_clean,
        load_dims=load_dims,
        target_cat=args.cat
    )

if __name__ == "__main__":
    main()
