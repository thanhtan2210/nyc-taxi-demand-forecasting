import numpy as np
import pandas as pd
import pytest

from src.features import BASE_FEATURES, LAG_FEATURES, TARGET, WEATHER_FEATURES, build_features

N_HOURS = 400


@pytest.fixture
def demand():
    rng = np.random.default_rng(0)
    hours = pd.date_range("2026-01-01", periods=N_HOURS, freq="h")
    frames = [
        pd.DataFrame({"hour": hours, "zone_id": zone, "borough": "Queens",
                      TARGET: rng.integers(0, 50, N_HOURS)})
        for zone in (1, 2)
    ]
    # shuffled on purpose: build_features must not depend on the input order
    return pd.concat(frames).sample(frac=1, random_state=0).reset_index(drop=True)


def zone_frame(features, zone):
    return features[features["zone_id"] == zone].sort_values("hour").reset_index(drop=True)


def test_features_at_hour_t_use_only_data_up_to_t_minus_1(demand):
    """Changing the target at hour t and at every later hour must not change any feature of hour <= t."""
    t = pd.Timestamp("2026-01-10 09:00")
    tampered = demand.copy()
    tampered.loc[tampered["hour"] >= t, TARGET] = 10_000

    original = build_features(demand)
    changed = build_features(tampered)

    upto_t = original["hour"] <= t
    pd.testing.assert_frame_equal(original.loc[upto_t, BASE_FEATURES], changed.loc[upto_t, BASE_FEATURES])
    # the tampering is real: the very next hour does see it through lag_1
    after = changed["hour"] == t + pd.Timedelta(hours=1)
    assert (changed.loc[after, "lag_1"] == 10_000).all()


def test_lag_and_rolling_values_are_exact(demand):
    z = zone_frame(build_features(demand), 1)
    y = z[TARGET].to_numpy(dtype=float)
    i = 300
    assert z.loc[i, "lag_1"] == y[i - 1]
    assert z.loc[i, "lag_2"] == y[i - 2]
    assert z.loc[i, "lag_3"] == y[i - 3]
    assert z.loc[i, "lag_24"] == y[i - 24]
    assert z.loc[i, "lag_168"] == y[i - 168]
    assert z.loc[i, "roll_mean_24"] == pytest.approx(y[i - 24:i].mean())
    # no history yet -> NaN, never a value borrowed from the future or from another zone
    assert z.loc[:167, "lag_168"].isna().all()
    assert z.loc[168:, LAG_FEATURES].notna().all().all()


def test_zones_do_not_leak_into_each_other(demand):
    tampered = demand.copy()
    tampered.loc[tampered["zone_id"] == 2, TARGET] = 10_000
    original = zone_frame(build_features(demand), 1)
    changed = zone_frame(build_features(tampered), 1)
    pd.testing.assert_frame_equal(original[LAG_FEATURES], changed[LAG_FEATURES])


def test_calendar_and_weather_features(demand):
    hours = pd.date_range("2026-01-01", periods=N_HOURS, freq="h")
    weather = pd.DataFrame({"hour": hours, "temperature_2m": np.arange(N_HOURS, dtype=float), "precipitation": 0.0})
    z = zone_frame(build_features(demand, weather), 1)
    assert len(z) == N_HOURS
    assert z.loc[5, "hour_of_day"] == 5
    assert z.loc[0, "day_of_week"] == 3          # 2026-01-01 is a Thursday
    assert z.loc[0, "is_holiday"] == 1           # New Year's Day
    assert z.loc[48, "is_holiday"] == 0
    assert z.loc[7, "temperature_2m"] == 7.0     # weather of the same hour
    assert z["zone_id"].dtype == "category"
    assert set(WEATHER_FEATURES) <= set(z.columns)


def test_incomplete_grid_is_rejected(demand):
    with pytest.raises(ValueError):
        build_features(demand.drop(index=demand.index[10]))
