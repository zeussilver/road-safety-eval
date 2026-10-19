"""Crash -> carriageway unit assignment with match-quality QA.

Every unit within the buffer is a candidate. Candidates within TIE_TOL_M of the nearest one are
treated as a near-tie (typical at intersections, where several units meet at the crash node).
A near-tie is broken in favour of the unit whose posted limit at the crash date equals the
police-recorded SPEED_ZONE; otherwise the nearest unit wins.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from rse import config
from rse.treatment import analysis_keys
from rse.util import month_index, save_metrics, timed

TIE_TOL_M = 2.0
BUFFERS_M = (15, 30, 50)


def crash_points() -> gpd.GeoDataFrame:
    c = pd.read_parquet(config.PROCESSED / "crashes.parquet")
    c = c[c.lat.notna()]
    return gpd.GeoDataFrame(c, geometry=gpd.points_from_xy(c.lon, c.lat), crs=config.CRS_CRASH).to_crs(config.CRS)


def limit_at(unit_rows: np.ndarray, months: np.ndarray, lim: np.ndarray, snap_months: np.ndarray) -> np.ndarray:
    """Posted limit of each unit at each crash month: the latest snapshot at or before that month
    (the base snapshot for anything earlier)."""
    k = np.clip(np.searchsorted(snap_months, months, side="right") - 1, 0, len(snap_months) - 1)
    return lim[unit_rows, k]


def snap(crashes: gpd.GeoDataFrame, units: gpd.GeoDataFrame, max_distance: float,
         lim: np.ndarray, snap_months: np.ndarray, tie_tol: float = TIE_TOL_M) -> pd.DataFrame:
    """Return one row per crash: unit_id (NA if nothing within max_distance), dist_m, n_near, tie_rule."""
    if crashes.crs != units.crs or not units.crs.is_projected:
        raise ValueError(f"crashes ({crashes.crs}) and units ({units.crs}) must share a projected CRS")
    u = units[["unit_id", "geometry"]].reset_index(drop=True)
    c = crashes[["accident_no", "month", "crash_speed_zone", "geometry"]].reset_index(drop=True)
    j = gpd.sjoin(c, u, how="inner", predicate="dwithin", distance=max_distance)
    j = j.rename(columns={"index_right": "urow"})
    j["dist_m"] = shapely.distance(j.geometry.values, u.geometry.values[j.urow.to_numpy()])
    j["dmin"] = j.groupby(level=0).dist_m.transform("min")
    j = j[j.dist_m <= j.dmin + tie_tol].copy()
    j["n_near"] = j.groupby(level=0).dist_m.transform("size")
    months = np.array([month_index(m) for m in j.month.to_numpy()])
    j["unit_limit"] = limit_at(j.urow.to_numpy(), months, lim, snap_months)
    j["agree"] = (j.unit_limit == j.crash_speed_zone.astype("float64")).fillna(False).astype(bool)
    n_agree = j.groupby(level=0).agree.sum()
    j = j.sort_values(["agree", "dist_m", "unit_id"], ascending=[False, True, True])
    best = j[~j.index.duplicated(keep="first")]
    rule = np.where(best.n_near == 1, "single",
                    np.where(best.agree & (n_agree.reindex(best.index) == 1), "limit_match", "nearest"))
    best = best.assign(tie_rule=rule).reindex(pd.RangeIndex(len(c)))
    return pd.DataFrame({
        "accident_no": c.accident_no.to_numpy(),
        "unit_id": best.unit_id.astype("Int64").to_numpy(),
        "dist_m": best.dist_m.astype("float64").to_numpy(),
        "n_near": best.n_near.fillna(0).astype(int).to_numpy(),
        "tie_rule": best.tie_rule.fillna("unmatched").astype(str).to_numpy(),
        "unit_limit": best.unit_limit.astype("float64").to_numpy(),
    })


def qa(crashes: pd.DataFrame, s: pd.DataFrame) -> dict:
    m = crashes[["accident_no", "node_type", "crash_speed_zone", "month"]].merge(s, on="accident_no")
    matched = m.unit_id.notna()
    known = matched & m.crash_speed_zone.notna()
    recent = known & (m.month >= config.snapshot_month(config.BASE_SNAPSHOT))
    single = recent & (m.tie_rule == "single")
    agree = m.unit_limit == m.crash_speed_zone.astype("float64")
    by_node = {nt: round(float(matched[m.node_type == nt].mean()), 4) for nt in ("I", "N")}
    tv_units = pd.read_parquet(config.PROCESSED / "treatment.parquet", columns=["unit_id", "reason"])
    tv_units = set(tv_units.loc[tv_units.reason == "time_of_day_zone", "unit_id"])
    normal = recent & ~m.unit_id.isin(tv_units)
    return {
        "crashes_with_location": len(m),
        "matched_share": round(float(matched.mean()), 4),
        "matched_share_by_node_type": by_node,
        "dist_m_quantiles": {q: round(float(m.loc[matched, "dist_m"].quantile(q)), 2) for q in (0.5, 0.9, 0.95, 0.99)},
        "tie_rule_counts": m.loc[matched, "tie_rule"].value_counts().to_dict(),
        "near_tie_share": round(float((m.loc[matched, "n_near"] > 1).mean()), 4),
        # Agreement is most meaningful for crashes after the base snapshot (posted limits are current)
        # and for crashes with a single candidate (the tie-breaker did not use SPEED_ZONE).
        "limit_agreement_since_base": round(float(agree[recent].mean()), 4),
        "limit_agreement_since_base_single": round(float(agree[single].mean()), 4),
        "limit_agreement_since_base_n": int(recent.sum()),
        # School / shop zones store the lowest limit of the day, so police often record the higher one.
        "limit_agreement_since_base_normal_zones": round(float(agree[normal].mean()), 4),
        "limit_agreement_since_base_time_of_day_zones": round(float(agree[recent & ~normal].mean()), 4),
        "limit_agreement_since_base_single_n": int(single.sum()),
    }


def datum_check(crashes: gpd.GeoDataFrame, units: gpd.GeoDataFrame, n: int = 20000) -> dict:
    """Median crash-to-road distance under the configured datum vs if Speed Zones were GDA2020.
    A constant ~1.6 m shift is the GDA94 -> GDA2020 offset, so the smaller one is the right pairing."""
    sample = crashes.sample(min(n, len(crashes)), random_state=0)
    out = {}
    for label, pts in (("configured", sample),
                       ("if_speed_zones_gda2020", sample.to_crs(config.CRS_CRASH).set_crs(7844, allow_override=True)
                        .to_crs(config.CRS))):
        j = gpd.sjoin_nearest(pts[["accident_no", "geometry"]], units[["unit_id", "geometry"]], max_distance=30,
                              distance_col="d").drop_duplicates("accident_no")
        out[f"median_dist_m_{label}"] = round(float(j.d.median()), 2)
    return out


def main():
    keys = analysis_keys()
    units = gpd.read_parquet(config.PROCESSED / "units.parquet")
    lim = units[[f"lim_{k}" for k in keys]].astype("float64").to_numpy()
    snap_months = np.array([month_index(k) for k in keys])
    crashes = crash_points()
    stats, timings = {}, {}
    for b in BUFFERS_M:
        with timed(f"snap {b} m", timings):
            s = snap(crashes, units, b, lim, snap_months)
        s.to_parquet(config.PROCESSED / f"snap_{b}m.parquet", index=False)
        stats[f"{b}m"] = qa(crashes, s)
        if b == config.SNAP_BUFFER_M:
            d = s.dist_m.dropna()
            pd.cut(d, bins=np.arange(0, b + 1, 1)).value_counts(sort=False).rename("crashes") \
                .to_csv(config.TABLES / "snap_distance_hist.csv")
    stats["datum_check"] = datum_check(crashes, units)
    stats["tie_tolerance_m"] = TIE_TOL_M
    stats["timings_s"] = timings
    save_metrics("snap", stats)
    print(stats)


if __name__ == "__main__":
    main()
