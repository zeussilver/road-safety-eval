"""Raw CSV / GeoJSON -> Parquet / GeoParquet.

Crash tables go through DuckDB. Each Speed Zones snapshot becomes a slim attribute table
(one row per feature id, plus a geometry hash so later snapshots can be checked for re-drawn
segments); full geometry is kept only for the base snapshot and the legacy Nov 2021 snapshot.
"""
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
import shapely
import duckdb

from rse import config
from rse.util import save_metrics, timed

SZ_COLS = ["id", "speed_limit", "zone_conditions", "speed_zone_change_request_id", "direction", "zone_length"]


def _mb(p) -> float:
    return round(p.stat().st_size / 1e6, 1)


def ingest_crashes(stats: dict) -> None:
    src = config.RAW / "crash"
    out = config.INTERIM
    con = duckdb.connect()
    tables = {}
    for name in config.CRASH_RESOURCES:
        csv, pq = src / f"{name}.csv", out / f"{name}.parquet"
        con.execute(f"COPY (SELECT * FROM read_csv('{csv}', all_varchar=true, header=true)) "
                    f"TO '{pq}' (FORMAT parquet, COMPRESSION zstd)")
        n = con.execute(f"SELECT count(*) FROM '{pq}'").fetchone()[0]
        tables[name] = {"rows": n, "csv_mb": _mb(csv), "parquet_mb": _mb(pq)}

    # node.csv repeats most rows verbatim, and ~1.5% of crashes carry two different urban-area
    # labels for the same point; keep one location per crash (lowest label alphabetically).
    node_rows, node_unique = con.execute(
        f"SELECT count(*), count(DISTINCT ACCIDENT_NO) FROM '{out}/node.parquet'").fetchone()
    node_conflicts = con.execute(f"""
        SELECT count(*) FROM (SELECT ACCIDENT_NO FROM '{out}/node.parquet' GROUP BY 1
                              HAVING count(DISTINCT (NODE_TYPE, LATITUDE, LONGITUDE, LGA_NAME, DEG_URBAN_NAME)) > 1)
    """).fetchone()[0]
    con.execute(f"""
        COPY (
          WITH node AS (
            SELECT ACCIDENT_NO, NODE_TYPE, LATITUDE, LONGITUDE, LGA_NAME, DEG_URBAN_NAME, POSTCODE_CRASH
            FROM '{out}/node.parquet'
            QUALIFY row_number() OVER (PARTITION BY ACCIDENT_NO ORDER BY DEG_URBAN_NAME, NODE_TYPE, LGA_NAME) = 1),
          loc AS (SELECT * FROM '{out}/accident_location.parquet')
          SELECT a.ACCIDENT_NO AS accident_no,
                 CAST(a.ACCIDENT_DATE AS DATE) AS date,
                 strftime(CAST(a.ACCIDENT_DATE AS DATE), '%Y-%m') AS month,
                 CAST(a.SEVERITY AS INTEGER) AS severity,
                 CASE WHEN a.SPEED_ZONE IN ('777','888','999') THEN NULL
                      ELSE CAST(a.SPEED_ZONE AS INTEGER) END AS crash_speed_zone,
                 a.ROAD_GEOMETRY_DESC AS road_geometry, a.RMA AS rma, a.DCA_DESC AS dca,
                 CAST(a.NO_PERSONS_KILLED AS INTEGER) AS killed,
                 CAST(a.NO_PERSONS_INJ_2 AS INTEGER) AS serious_inj,
                 n.NODE_TYPE AS node_type, CAST(n.LATITUDE AS DOUBLE) AS lat, CAST(n.LONGITUDE AS DOUBLE) AS lon,
                 n.LGA_NAME AS crash_lga, n.DEG_URBAN_NAME AS deg_urban, n.POSTCODE_CRASH AS postcode,
                 loc.ROAD_NAME AS road_name, loc.ROAD_TYPE AS road_type, loc.ROAD_NAME_INT AS road_name_int
          FROM '{out}/accident.parquet' a
          LEFT JOIN node n USING (ACCIDENT_NO)
          LEFT JOIN loc USING (ACCIDENT_NO)
        ) TO '{config.PROCESSED}/crashes.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    c = con.execute(f"""SELECT count(*), count(DISTINCT accident_no), min(date), max(date),
                               sum(CASE WHEN lat IS NULL THEN 1 ELSE 0 END)
                        FROM '{config.PROCESSED}/crashes.parquet'""").fetchone()
    monthly = con.execute(f"""SELECT month, count(*) FROM '{config.PROCESSED}/crashes.parquet'
                              WHERE month >= '2025-07' GROUP BY 1 ORDER BY 1""").fetchall()
    # Datum check: node.csv also carries VicGrid94 metres (AMG_X/AMG_Y). Projecting LATITUDE/LONGITUDE
    # under each candidate datum and comparing tells us which datum the lat/lon are in.
    import pyproj
    pts = con.execute(f"""SELECT DISTINCT CAST(LATITUDE AS DOUBLE), CAST(LONGITUDE AS DOUBLE),
                                 CAST(AMG_X AS DOUBLE), CAST(AMG_Y AS DOUBLE)
                          FROM '{out}/node.parquet' USING SAMPLE 20000 ROWS""").fetchnumpy()
    lat, lon, ax, ay = (pts[k] for k in pts)
    datum = {}
    for name, epsg in (("gda94", 4283), ("gda2020", 7844)):
        x, y = pyproj.Transformer.from_crs(epsg, 3111, always_xy=True).transform(lon, lat)
        datum[f"median_offset_m_if_{name}"] = round(float(np.median(np.hypot(x - ax, y - ay))), 2)
    stats["crash_datum_check"] = datum
    stats["crash_tables"] = tables
    stats["node_rows"], stats["node_unique_crashes"] = node_rows, node_unique
    stats["node_conflicting_crashes"] = node_conflicts
    stats["crashes"] = {"rows": c[0], "unique": c[1], "first_date": str(c[2]), "last_date": str(c[3]),
                        "no_location": c[4], "recent_monthly_counts": dict(monthly)}


def _conditions(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (list, tuple, np.ndarray)):
        return "|".join(sorted(str(x) for x in v))
    return str(v)


def _geom_hash(geoms) -> np.ndarray:
    wkb = shapely.to_wkb(shapely.set_precision(geoms, 1e-7))
    return pd.util.hash_array(np.asarray(wkb, dtype=object))


def ingest_snapshot(key: str) -> dict:
    src = config.RAW / "speed_zones" / f"speed_zones_{key}.geojson"
    df = pyogrio.read_dataframe(src, columns=SZ_COLS, use_arrow=True)
    df = df.set_crs(config.CRS_SZ, allow_override=True)
    attrs = pd.DataFrame({
        "id": df["id"].astype(str),
        "speed_limit": pd.to_numeric(df["speed_limit"], errors="coerce").astype("Int64"),
        "conditions": df["zone_conditions"].map(_conditions),
        "change_request": df["speed_zone_change_request_id"].astype("string"),
        "direction": df["direction"].astype("string"),
        "geom_hash": _geom_hash(df.geometry.values),
    })
    attrs.to_parquet(config.INTERIM / f"sz_attrs_{key}.parquet", index=False)
    if key == config.BASE_SNAPSHOT:
        g = df[["id", "direction", "geometry"]].to_crs(config.CRS)
        g.to_parquet(config.INTERIM / "sz_base_geom.parquet", index=False)
    return {"features": len(df), "geojson_mb": _mb(src),
            "parquet_mb": _mb(config.INTERIM / f"sz_attrs_{key}.parquet")}


def ingest_legacy(key: str) -> dict:
    """Nov 2021: different schema, VicGrid94 coordinates. Effective limit = lowest at any time."""
    src = config.RAW / "speed_zones" / f"speed_zones_{key}.geojson"
    cols = ["OBJECTID", "SIGN_SPEED", "DIRECTION", "Variable_S", "Seasonal_S"]
    df = pyogrio.read_dataframe(src, columns=cols, use_arrow=True)
    df = df.set_crs(config.CRS_SZ_2021, allow_override=True).to_crs(config.CRS)
    lim = df[["SIGN_SPEED", "Variable_S", "Seasonal_S"]].astype(float).where(lambda x: x > 0)
    out = df[["OBJECTID", "geometry"]].copy()
    out["speed_limit"] = lim.min(axis=1).astype("Int64")
    out["conditions"] = np.where((df["Variable_S"] > 0) | (df["Seasonal_S"] > 0), "Time-varying", "Normal Operation")
    out.to_parquet(config.INTERIM / "sz_legacy_geom.parquet", index=False)
    return {"features": len(df), "geojson_mb": _mb(src), "parquet_mb": _mb(config.INTERIM / "sz_legacy_geom.parquet")}


def _cached(key: str) -> dict | None:
    """Stats for a snapshot whose Parquet is newer than its GeoJSON (skip re-parsing ~1 min/file)."""
    src = config.RAW / "speed_zones" / f"speed_zones_{key}.geojson"
    out = config.INTERIM / ("sz_legacy_geom.parquet" if key == config.LEGACY_SNAPSHOT else f"sz_attrs_{key}.parquet")
    extra = [] if key != config.BASE_SNAPSHOT else [config.INTERIM / "sz_base_geom.parquet"]
    if os.environ.get("FORCE") or not all(p.exists() and p.stat().st_mtime > src.stat().st_mtime for p in [out, *extra]):
        return None
    return {"features": pq.ParquetFile(out).metadata.num_rows, "geojson_mb": _mb(src), "parquet_mb": _mb(out)}


def main():
    stats, timings = {}, {}
    with timed("crash tables", timings):
        ingest_crashes(stats)
    snaps = {}
    for key in config.SNAPSHOTS_QUARTERLY:
        if (cached := _cached(key)) is not None:
            snaps[key] = cached
            print(f"[snapshot {key}] up to date")
            continue
        with timed(f"snapshot {key}", timings):
            snaps[key] = ingest_legacy(key) if key == config.LEGACY_SNAPSHOT else ingest_snapshot(key)
    stats["snapshots"] = snaps
    stats["snapshot_geojson_gb"] = round(sum(s["geojson_mb"] for s in snaps.values()) / 1000, 2)
    stats["snapshot_parquet_mb"] = round(sum(s["parquet_mb"] for s in snaps.values()), 1)
    stats["snapshot_features_total"] = sum(s["features"] for s in snaps.values())
    stats["timings_s"] = timings
    save_metrics("ingest", stats)
    print({k: v for k, v in stats.items() if k != "snapshots"})


if __name__ == "__main__":
    main()
