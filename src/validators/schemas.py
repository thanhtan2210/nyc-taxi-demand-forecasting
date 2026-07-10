import pandera as pa

# Cleaned data validation schema
CleanTripSchema = pa.DataFrameSchema(
    columns={
        "pulocationid": pa.Column(pa.Int64, nullable=False, checks=pa.Check.in_range(1, 265)),
        "dolocationid": pa.Column(pa.Int64, nullable=False, checks=pa.Check.in_range(1, 265)),
        "service_type_key": pa.Column(pa.Int64, nullable=False, checks=pa.Check.in_range(1, 4)),
        "ml_unified_fare": pa.Column(pa.Float64, nullable=True),
        "ml_unified_distance": pa.Column(pa.Float64, nullable=True, checks=pa.Check.greater_than_or_equal_to(0)),
        "ml_unified_duration": pa.Column(pa.Float64, nullable=True),
        "pickup_time_key": pa.Column(pa.Int64, nullable=False),
        "dropoff_time_key": pa.Column(pa.Int64, nullable=True)
    },
    coerce=True,
    strict=False # Allow other original/metadata columns to remain
)
