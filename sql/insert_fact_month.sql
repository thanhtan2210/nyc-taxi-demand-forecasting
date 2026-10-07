-- Appends the in-month rows of month_raw that have a known zone id (1-265) to the fact table.
INSERT INTO fact_pickups_hourly
SELECT hour, zone_id, $service_id AS service_id, CAST(sum(trips) AS INTEGER) AS trips
FROM month_raw
WHERE in_month AND zone_id BETWEEN 1 AND 265
GROUP BY hour, zone_id;
