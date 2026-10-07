"""Converts the official TLC taxi zone shapefile to a small WGS84 GeoJSON for the app's maps.

Usage (one-off; the result is committed, so the app and the tests never need these libraries):
    pip install -r requirements-geo.txt
    python -m scripts.fetch_zone_shapes

Source: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip (NYC TLC, same CDN as the
trip records). The shapefile is read with pyshp. Its coordinate system is taken from the .prj
file (NAD83 / New York Long Island, US survey feet, EPSG:2263) and converted to WGS84 longitude
and latitude with pyproj (always_xy=True).

Older copies of the TLC file duplicated the LocationID attribute (the shape of zone 57 was
labelled 56; the shapes of zones 104 and 105 were labelled 103). OBJECTID is unique and runs
from 1 to 263, so location_id is taken from OBJECTID and every shape whose LocationID differs
is printed (the copy dated 2026-02-18 has none). To make sure the ids line up, the zone name and
borough of every shape are compared with data/mart/dim_zone.csv and the script fails on any
mismatch.

Output: data/mart/taxi_zones.geojson, one MultiPolygon feature per zone with the single
property `location_id`. Coordinates are rounded to 5 decimals (about 1 m) and consecutive
duplicate points are dropped.
"""
import csv
import io
import json
import sys
import urllib.request
import zipfile

import pyproj
import shapefile

from src import paths

SOURCE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
DECIMALS = 5
MAX_BYTES = 5 * 2**20


def read_zip_member(archive, suffix):
    name = next(n for n in archive.namelist() if n.lower().endswith(suffix))
    return archive.read(name)


def signed_area(ring):
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:])) / 2


def convert_ring(ring, transformer, exterior):
    """Projects a ring to WGS84, rounds it, drops repeated points and fixes its winding order."""
    xs, ys = transformer.transform([p[0] for p in ring], [p[1] for p in ring])
    points = []
    for lon, lat in zip(xs, ys):
        point = [round(lon, DECIMALS), round(lat, DECIMALS)]
        if not points or point != points[-1]:
            points.append(point)
    if points[0] != points[-1]:
        points.append(points[0])
    if len(points) < 4:
        return None  # collapsed to a line or a point after rounding
    # GeoJSON (RFC 7946): exterior rings counter-clockwise, holes clockwise
    if (signed_area(points) > 0) != exterior:
        points.reverse()
    return points


def convert_polygons(geometry, transformer):
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    converted = []
    for polygon in polygons:
        exterior = convert_ring(polygon[0], transformer, exterior=True)
        if exterior is None:
            continue
        holes = [convert_ring(r, transformer, exterior=False) for r in polygon[1:]]
        converted.append([exterior] + [h for h in holes if h is not None])
    return converted


def main():
    print(f"Downloading {SOURCE_URL}")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    source_crs = pyproj.CRS.from_wkt(read_zip_member(archive, ".prj").decode("utf-8"))
    print(f"Source coordinate system (.prj): {source_crs.name}, EPSG:{source_crs.to_epsg()}, "
          f"unit {source_crs.axis_info[0].unit_name}")
    transformer = pyproj.Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

    reader = shapefile.Reader(
        shp=io.BytesIO(read_zip_member(archive, ".shp")),
        dbf=io.BytesIO(read_zip_member(archive, ".dbf")),
        shx=io.BytesIO(read_zip_member(archive, ".shx")),
    )
    with open(paths.DIM_ZONE, encoding="utf-8", newline="") as fh:
        dim_zone = {int(row["zone_id"]): row for row in csv.DictReader(fh)}

    polygons_by_zone, id_mismatches, name_mismatches = {}, [], []
    for item in reader.iterShapeRecords():
        record = item.record.as_dict()
        location_id = int(record["OBJECTID"])
        if int(record["LocationID"]) != location_id:
            id_mismatches.append((location_id, int(record["LocationID"]), record["zone"]))
        expected = dim_zone.get(location_id)
        if expected is None or (record["zone"], record["borough"]) != (expected["zone"], expected["borough"]):
            name_mismatches.append((location_id, record["zone"], record["borough"],
                                    expected and expected["zone"], expected and expected["borough"]))
        polygons_by_zone.setdefault(location_id, []).extend(
            convert_polygons(item.shape.__geo_interface__, transformer))

    print(f"Shapes read: {len(reader)}; zones: {len(polygons_by_zone)}")
    print("Shapes whose LocationID differs from OBJECTID (OBJECTID, LocationID, zone):")
    for row in id_mismatches:
        print(f"  {row}")
    print(f"Zone or borough names that differ from dim_zone.csv: {len(name_mismatches)}")
    for row in name_mismatches:
        print(f"  {row}")

    mart_zones = {z for z in dim_zone if z <= 263}
    missing = sorted(mart_zones - set(polygons_by_zone))
    extra = sorted(set(polygons_by_zone) - mart_zones)
    print(f"Mart zones without a polygon: {missing}; polygons without a mart zone: {extra}")
    if name_mismatches or missing or extra:
        print("Refusing to write the GeoJSON: the shapes do not line up with dim_zone.csv")
        return 1

    features = [
        {"type": "Feature", "properties": {"location_id": zone},
         "geometry": {"type": "MultiPolygon", "coordinates": polygons_by_zone[zone]}}
        for zone in sorted(polygons_by_zone)
    ]
    text = json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":"))
    size = len(text.encode("utf-8"))
    points = sum(len(ring) for f in features for polygon in f["geometry"]["coordinates"] for ring in polygon)
    print(f"{len(features)} features, {points:,} points, {size / 2**20:.2f} MB")
    if size > MAX_BYTES:
        print(f"Refusing to write: larger than {MAX_BYTES / 2**20:.0f} MB")
        return 1
    paths.ZONE_SHAPES.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {paths.ZONE_SHAPES.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
