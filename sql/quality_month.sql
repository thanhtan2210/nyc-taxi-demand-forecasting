-- Row accounting for the (service, month) held in month_raw.
-- The five categories after rows_read are mutually exclusive and add up to rows_read.
SELECT
    coalesce(sum(trips), 0) AS rows_read,
    coalesce(sum(trips) FILTER (WHERE NOT in_month), 0) AS rows_outside_month,
    coalesce(sum(trips) FILTER (WHERE in_month AND zone_id IS NULL), 0) AS rows_zone_null,
    coalesce(sum(trips) FILTER (WHERE in_month AND zone_id IN (264, 265)), 0) AS rows_zone_264_265,
    coalesce(sum(trips) FILTER (WHERE in_month AND zone_id NOT BETWEEN 1 AND 265), 0) AS rows_zone_out_of_range,
    coalesce(sum(trips) FILTER (WHERE in_month AND zone_id BETWEEN 1 AND 263), 0) AS rows_in_mart
FROM month_raw;
