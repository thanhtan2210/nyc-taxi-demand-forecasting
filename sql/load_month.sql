-- Reads ONE (service, month) parquet file and aggregates it in a single scan.
-- Only two columns are read: the pickup timestamp and the pickup zone.
-- {pickup_column} / {zone_column} come from config/pipeline.yaml; $url, $month_start, $month_end are bound parameters.
CREATE OR REPLACE TEMP TABLE month_raw AS
SELECT
    coalesce({pickup_column} >= $month_start AND {pickup_column} < $month_end, FALSE) AS in_month,
    date_trunc('hour', {pickup_column}) AS hour,
    TRY_CAST({zone_column} AS INTEGER) AS zone_id,
    count(*) AS trips
FROM read_parquet($url)
GROUP BY ALL;
