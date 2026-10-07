# Power BI report on the demand mart

Power BI reads the same committed files as the Streamlit app. No database or gateway is needed.

| File | Grain | Columns |
| --- | --- | --- |
| `data/mart/demand_hourly/month=YYYY-MM/part.parquet` | one row per zone and hour (zones 1-263, hours without trips are 0) | `hour`, `zone_id`, `borough`, `trips_total`, `trips_yellow`, `trips_green`, `trips_fhv`, `trips_fhvhv` |
| `data/mart/dim_zone.csv` | one row per taxi zone | `zone_id`, `borough`, `zone`, `service_zone` |

## 1. Load the mart with the Folder connector

1. **Get data → Folder**, choose the local path of `data/mart/demand_hourly`, then **Transform data**.
2. Keep only the parquet files: filter `Extension` to `.parquet`.
3. Click the **Combine Files** button on the `Content` column and accept the sample file. Power Query creates the helper queries and appends the 14 monthly files.
4. Remove the generated `Source.Name` column and the `borough` column (the borough comes from `dim_zone`).
5. Unpivot the four service columns so the service becomes a dimension:
   - remove `trips_total` (it is the sum of the four service columns and would double count),
   - select `trips_yellow`, `trips_green`, `trips_fhv`, `trips_fhvhv` → **Transform → Unpivot Columns**,
   - rename `Attribute` to `service` and `Value` to `trips`, then replace the prefix `trips_` with nothing in `service`.
6. Add a date column: **Add Column → Custom Column**, name `date`, formula `DateTime.Date([hour])`.
7. Set the types: `hour` Date/Time, `date` Date, `zone_id` Whole Number, `service` Text, `trips` Whole Number.
8. Rename the query to `demand`.

## 2. Load the zone table

1. **Get data → Text/CSV**, choose `data/mart/dim_zone.csv`.
2. Check that `zone_id` is a Whole Number and name the query `dim_zone`.

Zones 264 (Unknown) and 265 (Outside of NYC) are listed in `dim_zone` but have no rows in `demand`.

## 3. Date table and relationships

Create the date table in **Modeling → New table**:

```DAX
dim_date =
ADDCOLUMNS (
    CALENDAR ( MIN ( demand[date] ), MAX ( demand[date] ) ),
    "Year Month", FORMAT ( [Date], "YYYY-MM" ),
    "Weekday", FORMAT ( [Date], "ddd" ),
    "Is Weekend", WEEKDAY ( [Date], 2 ) >= 6
)
```

Mark it as a date table on `Date`, then create two single-direction, many-to-one relationships in the model view:

| From (many) | To (one) |
| --- | --- |
| `demand[zone_id]` | `dim_zone[zone_id]` |
| `demand[date]` | `dim_date[Date]` |

## 4. Measures

```DAX
Total Trips = SUM ( demand[trips] )

Trips by Service =
-- use with demand[service] on an axis or legend; shown here for one service
CALCULATE ( [Total Trips], demand[service] = "fhvhv" )

Service Share =
DIVIDE ( [Total Trips], CALCULATE ( [Total Trips], REMOVEFILTERS ( demand[service] ) ) )

Avg Trips per Zone-Hour =
-- the mart is a complete zone x hour grid, so zone-hours = zones x hours
DIVIDE (
    [Total Trips],
    DISTINCTCOUNT ( demand[zone_id] ) * DISTINCTCOUNT ( demand[hour] )
)

MoM Change =
VAR previous = CALCULATE ( [Total Trips], DATEADD ( dim_date[Date], -1, MONTH ) )
RETURN DIVIDE ( [Total Trips] - previous, previous )
```

Format `Service Share` and `MoM Change` as percentages.

## 5. Suggested one-page layout

| Area | Visual | Fields |
| --- | --- | --- |
| Top row | 3 cards | `Total Trips`, `Avg Trips per Zone-Hour`, `MoM Change` (filtered to the last month) |
| Left, wide | Stacked area chart | Axis `dim_date[Date]`, values `Total Trips`, legend `demand[service]` |
| Right | Bar chart | Axis `demand[service]`, values `Service Share` |
| Bottom left | Bar chart, top 15 | Axis `dim_zone[zone]`, values `Total Trips`, Top N filter = 15 |
| Bottom right | Matrix | Rows `dim_zone[borough]`, columns `dim_date[Year Month]`, values `Total Trips` |
| Slicers | | `dim_date[Year Month]`, `dim_zone[borough]` |

## Notes

- `fhv` is under-counted: most of its trips have no pickup zone in the source files. The exact share per month is in `reports/data_quality.json`.
- Timestamps are New York wall-clock time. The hour 01:00 on 2025-11-02 contains two real hours (clocks fall back) and 02:00 on 2026-03-08 is nearly empty (clocks spring forward).
- These steps were written from the file schema. Check the generated column names after the Combine Files step and adjust the measures if they differ.
