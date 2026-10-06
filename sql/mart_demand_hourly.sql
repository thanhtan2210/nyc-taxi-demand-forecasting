-- Hourly demand mart for one month: a complete zone x hour grid, hours without trips are 0.
-- Zones 264 (Unknown) and 265 (Outside of NYC) are excluded; they are counted in reports/data_quality.json.
CREATE OR REPLACE TEMP TABLE mart_month AS
WITH grid AS (
    SELECT h.hour, z.zone_id, z.borough
    FROM dim_hour h
    CROSS JOIN dim_zone z
    WHERE h.hour >= $month_start AND h.hour < $month_end
      AND z.zone_id BETWEEN 1 AND 263
),
agg AS (
    SELECT
        f.hour,
        f.zone_id,
        sum(f.trips) AS trips_total,
        sum(f.trips) FILTER (WHERE s.service = 'yellow') AS trips_yellow,
        sum(f.trips) FILTER (WHERE s.service = 'green') AS trips_green,
        sum(f.trips) FILTER (WHERE s.service = 'fhv') AS trips_fhv,
        sum(f.trips) FILTER (WHERE s.service = 'fhvhv') AS trips_fhvhv
    FROM fact_pickups_hourly f
    JOIN dim_service s USING (service_id)
    WHERE f.hour >= $month_start AND f.hour < $month_end
    GROUP BY f.hour, f.zone_id
)
SELECT
    g.hour,
    g.zone_id,
    g.borough,
    CAST(coalesce(a.trips_total, 0) AS INTEGER) AS trips_total,
    CAST(coalesce(a.trips_yellow, 0) AS INTEGER) AS trips_yellow,
    CAST(coalesce(a.trips_green, 0) AS INTEGER) AS trips_green,
    CAST(coalesce(a.trips_fhv, 0) AS INTEGER) AS trips_fhv,
    CAST(coalesce(a.trips_fhvhv, 0) AS INTEGER) AS trips_fhvhv
FROM grid g
LEFT JOIN agg a USING (hour, zone_id)
ORDER BY g.hour, g.zone_id;
