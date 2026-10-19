"""Carriageway units and their attributes.

Speed Zones stores one feature per travel direction, so most roads appear twice with the same
geometry. A *unit* is the set of base-snapshot features that share a geometry (orientation
ignored). Each unit gets: its limit in every snapshot, LGA, metro/regional, declared-road flag
and 2019 AADT, and its limit in the legacy Nov 2021 snapshot.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

from rse import config
from rse.util import save_metrics, timed

NORMAL = "Normal Operation"


def unit_keys(geoms: np.ndarray, precision: float = 0.01) -> np.ndarray:
    """Integer key shared by features with the same line geometry, whatever their direction."""
    g = shapely.normalize(shapely.line_merge(shapely.set_precision(geoms, precision)))
    return pd.factorize(pd.Series(shapely.to_wkb(g)))[0]


def build_units(base: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    base = base.copy()
    base["unit_id"] = unit_keys(base.geometry.values)
    fmap = base[["id", "unit_id"]]
    units = base.groupby("unit_id").agg(n_features=("id", "size")).reset_index()
    geom = base.drop_duplicates("unit_id").set_index("unit_id").geometry
    units = gpd.GeoDataFrame(units, geometry=geom.loc[units.unit_id].values, crs=base.crs)
    units["length_m"] = units.length
    return units, fmap


def snapshot_limits(fmap: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Wide table: per unit and snapshot, the limit (NA if directions disagree) plus flags."""
    base_hash = pd.read_parquet(config.INTERIM / f"sz_attrs_{config.BASE_SNAPSHOT}.parquet",
                                columns=["id", "geom_hash"]).rename(columns={"geom_hash": "h0"})
    out = pd.DataFrame(index=pd.Index(np.sort(fmap.unit_id.unique()), name="unit_id"))
    for k in keys:
        a = pd.read_parquet(config.INTERIM / f"sz_attrs_{k}.parquet")
        a = fmap.merge(a, on="id", how="left").merge(base_hash, on="id", how="left")
        a["timevarying"] = a.conditions.fillna("") != NORMAL
        a["redrawn"] = a.geom_hash != a.h0
        g = a.groupby("unit_id")
        lo, hi = g.speed_limit.min(), g.speed_limit.max()
        out[f"lim_{k}"] = lo.where(lo == hi)
        out[f"dirmix_{k}"] = (lo != hi).fillna(True)
        out[f"tv_{k}"] = g.timevarying.any()
        out[f"redrawn_{k}"] = g.redrawn.any()
        cr = a.dropna(subset=["change_request"]).groupby("unit_id").change_request.first()
        out[f"cr_{k}"] = cr.reindex(out.index)
    return out.reset_index()


def sample_points(lines: gpd.GeoSeries, fracs=(0.25, 0.5, 0.75)) -> gpd.GeoDataFrame:
    parts = []
    for f in fracs:
        parts.append(gpd.GeoDataFrame({"row": np.arange(len(lines)), "frac": f},
                                      geometry=lines.interpolate(f, normalized=True).values, crs=lines.crs))
    return pd.concat(parts, ignore_index=True)


def assign_lga(units: gpd.GeoDataFrame, lga: gpd.GeoDataFrame) -> pd.Series:
    mid = gpd.GeoDataFrame({"row": np.arange(len(units))},
                           geometry=units.geometry.interpolate(0.5, normalized=True).values, crs=units.crs)
    j = gpd.sjoin(mid, lga[["name", "geometry"]], how="inner", predicate="within").drop_duplicates("row")
    rest = mid[~mid.row.isin(j.row)]  # on the coast or the state border: nearest LGA within 1 km
    k = gpd.sjoin_nearest(rest, lga[["name", "geometry"]], how="inner", max_distance=1000).drop_duplicates("row")
    names = pd.concat([j, k]).set_index("row")["name"]
    return names.reindex(np.arange(len(units))).fillna("UNKNOWN").values


def assign_declared(units: gpd.GeoDataFrame, aadt: gpd.GeoDataFrame) -> pd.DataFrame:
    """Declared if at least 2 of 3 points along the unit lie within 15 m of the AADT (declared) network."""
    pts = sample_points(units.geometry)
    j = gpd.sjoin_nearest(pts, aadt[["aadt", "road_name", "geometry"]], how="inner",
                          max_distance=config.DECLARED_BUFFER_M)
    j = j.sort_values("aadt", ascending=False).drop_duplicates(["row", "frac"])
    per = j.groupby("row").agg(hits=("frac", "size"), aadt_2019=("aadt", "first"),
                               aadt_road=("road_name", "first"))
    per = per.reindex(np.arange(len(units)))
    per["declared"] = per.hits.fillna(0) >= 2
    per.loc[~per.declared, ["aadt_2019", "aadt_road"]] = np.nan
    return per[["declared", "aadt_2019", "aadt_road"]].reset_index(drop=True)


def legacy_limits(units: gpd.GeoDataFrame, legacy: gpd.GeoDataFrame) -> pd.DataFrame:
    """Limit in Nov 2021 at each unit's midpoint (features within 5 m; both directions)."""
    mid = gpd.GeoDataFrame({"row": np.arange(len(units))},
                           geometry=units.geometry.interpolate(0.5, normalized=True).values, crs=units.crs)
    j = gpd.sjoin_nearest(mid, legacy[["speed_limit", "conditions", "geometry"]], how="inner",
                          max_distance=config.LEGACY_MATCH_M, distance_col="d")
    j["tv"] = j.conditions != NORMAL
    g = j.groupby("row")
    lo, hi = g.speed_limit.min(), g.speed_limit.max()
    res = pd.DataFrame({"lim_legacy": lo.where(lo == hi), "tv_legacy": g.tv.any()})
    res = res.reindex(np.arange(len(units)))
    res["legacy_matched"] = res.index.isin(g.groups.keys())
    return res.reset_index(drop=True)


def main():
    stats, timings = {}, {}
    keys = [k for k in config.SNAPSHOTS_QUARTERLY if k != config.LEGACY_SNAPSHOT]
    with timed("units", timings):
        base = gpd.read_parquet(config.INTERIM / "sz_base_geom.parquet")
        units, fmap = build_units(base)
        fmap.to_parquet(config.PROCESSED / "feature_unit.parquet", index=False)
    with timed("snapshot limits", timings):
        lim = snapshot_limits(fmap, keys)
    with timed("lga", timings):
        lga = pyogrio.read_dataframe(config.RAW / "vicmap" / "lga.geojson", columns=["name", "state"])
        lga = lga[lga.state == "VIC"].to_crs(config.CRS)
        units["lga"] = assign_lga(units, lga)
        units["region"] = np.where(units.lga.isin(config.METRO_LGAS), "metro", "regional")
    with timed("declared roads", timings):
        aadt = pyogrio.read_dataframe(config.RAW / "aadt" / "aadt_2019.geojson")
        aadt = aadt.rename(columns={"Average Annual Daily Traffic Volume": "aadt", "Road Name": "road_name"})
        aadt = aadt.set_crs(4283, allow_override=True).to_crs(config.CRS)
        units = pd.concat([units, assign_declared(units, aadt)], axis=1)
    with timed("legacy match", timings):
        legacy = gpd.read_parquet(config.INTERIM / "sz_legacy_geom.parquet")
        units = pd.concat([units, legacy_limits(units, legacy)], axis=1)
    units = units.merge(lim, on="unit_id", how="left")
    units = gpd.GeoDataFrame(units, geometry="geometry", crs=config.CRS)
    units.to_parquet(config.PROCESSED / "units.parquet", index=False)

    stats.update({
        "base_features": len(base), "units": len(units),
        "units_by_n_features": units.n_features.value_counts().sort_index().to_dict(),
        "network_km": round(units.length_m.sum() / 1000),
        "declared_units": int(units.declared.sum()),
        "declared_km": round(units.loc[units.declared, "length_m"].sum() / 1000),
        "aadt_features": len(aadt),
        "lga_unknown_units": int((units.lga == "UNKNOWN").sum()),
        "legacy_match_rate": round(float(units.legacy_matched.mean()), 4),
        "base_direction_mismatch_units": int(units[f"dirmix_{config.BASE_SNAPSHOT}"].sum()),
        "timings_s": timings,
    })
    save_metrics("network", stats)
    print(stats)


if __name__ == "__main__":
    main()
