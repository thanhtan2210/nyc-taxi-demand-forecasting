-- Star schema of the local DuckDB warehouse.
-- Dimensions are rebuilt on every run (see src/warehouse.py); the fact table is loaded month by month.
CREATE TABLE IF NOT EXISTS fact_pickups_hourly (
    hour       TIMESTAMP NOT NULL,  -- pickup hour, New York local wall time as published by TLC
    zone_id    INTEGER   NOT NULL,  -- dim_zone.zone_id (1-265)
    service_id TINYINT   NOT NULL,  -- dim_service.service_id
    trips      INTEGER   NOT NULL
);
