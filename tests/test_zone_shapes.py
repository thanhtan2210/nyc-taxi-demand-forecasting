"""Checks of the committed zone boundaries (no network, no pyshp or pyproj needed)."""
import json
import math

import pytest

from src.paths import ZONE_SHAPES

# Airport reference points published by the FAA, as listed on AirNav
# (https://www.airnav.com/airport/KJFK and https://www.airnav.com/airport/KLGA, "Lat/Long").
AIRPORTS = {
    132: ("JFK Airport", 40.6399281, -73.7786922),
    138: ("LaGuardia Airport", 40.7772422, -73.8726056),
}


@pytest.fixture(scope="module")
def features():
    return json.loads(ZONE_SHAPES.read_text(encoding="utf-8"))["features"]


def points_of(feature):
    assert feature["geometry"]["type"] == "MultiPolygon"
    return [point for polygon in feature["geometry"]["coordinates"] for ring in polygon for point in ring]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def test_every_zone_has_exactly_one_feature(features):
    ids = [f["properties"]["location_id"] for f in features]
    assert sorted(ids) == list(range(1, 264))
    assert all(isinstance(i, int) for i in ids)
    assert all(list(f["properties"]) == ["location_id"] for f in features)


def test_all_coordinates_are_inside_the_new_york_area(features):
    for feature in features:
        for lon, lat in points_of(feature):
            assert -74.30 <= lon <= -73.65, feature["properties"]
            assert 40.48 <= lat <= 40.95, feature["properties"]


def test_coordinates_are_rounded_and_rings_are_closed(features):
    for feature in features:
        for polygon in feature["geometry"]["coordinates"]:
            for ring in polygon:
                assert len(ring) >= 4 and ring[0] == ring[-1]
                assert all(a != b for a, b in zip(ring, ring[1:]))  # no repeated consecutive point
                assert all(round(v, 5) == v for point in ring for v in point)


@pytest.mark.parametrize("zone_id", sorted(AIRPORTS))
def test_airport_zones_sit_on_the_real_airports(features, zone_id):
    name, lat, lon = AIRPORTS[zone_id]
    points = points_of(next(f for f in features if f["properties"]["location_id"] == zone_id))
    lons, lats = [p[0] for p in points], [p[1] for p in points]
    centre_lon, centre_lat = (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2
    distance = haversine_km(lat, lon, centre_lat, centre_lon)
    assert distance <= 3.0, f"{name}: bounding-box centre is {distance:.2f} km from the airport reference point"
