"""Case study: Yarra 30 km/h trial expansion (Fitzroy + Collingwood, from 2024-05-09).

The expansion is not recorded in the Speed Zones snapshots, so the treated set is defined from the
council's published scope: every street in Fitzroy and Collingwood except five state-managed
arterials (also dropped from the comparison suburbs). Police-coded SPEED_ZONE on crashes is used to check that the new limit was in force.
Counts are small; this is descriptive and the power calculation says how little it can detect.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy import stats as sps

from rse import config
from rse.evaluate import _binomial
from rse.util import month_index, save_metrics

ELIGIBLE_LIMITS = (40, 50)
PRE_TRIAL_SNAPSHOT = "2024_03"


def area_units() -> gpd.GeoDataFrame:
    u = gpd.read_parquet(config.PROCESSED / "units.parquet",
                         columns=["unit_id", "geometry", "length_m", "aadt_road", "declared", f"lim_{PRE_TRIAL_SNAPSHOT}"])
    tr = pd.read_parquet(config.PROCESSED / "treatment.parquet", columns=["unit_id", "status"])
    loc = pyogrio.read_dataframe(config.RAW / "vicmap" / "localities.geojson", columns=["name", "state"])
    loc = loc[loc.state == "VIC"].to_crs(config.CRS)
    mid = gpd.GeoDataFrame(u[["unit_id"]], geometry=u.geometry.interpolate(0.5, normalized=True), crs=config.CRS)
    j = gpd.sjoin(mid, loc[["name", "geometry"]], predicate="within").drop_duplicates("unit_id")
    u = u.merge(j[["unit_id", "name"]], on="unit_id").merge(tr, on="unit_id")
    u["arm"] = np.where(u.name.isin(config.YARRA_TREATED_LOCALITIES), "treated", "comparison")
    road = u.aadt_road.fillna("").str.upper().str.strip()
    u["state_arterial"] = road.isin(config.YARRA_EXCLUDED_ROADS)  # excluded in both arms, for like-for-like
    u["limit_before"] = u[f"lim_{PRE_TRIAL_SNAPSHOT}"]
    return u


def eligible(u: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Unchanged in every snapshot, 40/50 km/h before the trial, not one of the excluded arterials."""
    ok = (u.status == "comparison") & u.limit_before.isin(ELIGIBLE_LIMITS) & ~u.state_arterial
    return u[ok]


def power(yb_t: float, yb_c: float, ya_c: float, mb: int, ma: int, true_rr: float, n_sim: int = 20000,
          seed: int = 1) -> float:
    """Share of simulated studies whose Wald 95% CI for the DiD rate ratio excludes 1, assuming the
    observed before counts and comparison trend are the truth."""
    rng = np.random.default_rng(seed)
    trend = ya_c / yb_c
    a_t = rng.poisson(yb_t * trend * true_rr, n_sim)
    b_t = rng.poisson(yb_t, n_sim)
    a_c = rng.poisson(ya_c, n_sim)
    b_c = rng.poisson(yb_c, n_sim)
    ok = (a_t > 0) & (b_t > 0) & (a_c > 0) & (b_c > 0)
    lrr = np.log(a_t / b_t) - np.log(a_c / b_c)
    se = np.sqrt(1 / a_t + 1 / b_t + 1 / a_c + 1 / b_c)
    return float(np.mean(ok & ((lrr + 1.96 * se) < 0)))


def main():
    u = eligible(area_units())
    c = pd.read_parquet(config.PROCESSED / "crashes.parquet", columns=["accident_no", "month", "severity", "crash_speed_zone"])
    s = pd.read_parquet(config.PROCESSED / f"snap_{config.SNAP_BUFFER_M}m.parquet", columns=["accident_no", "unit_id"])
    c = c.merge(s, on="accident_no")
    c = c[c.unit_id.isin(u.unit_id)].merge(u[["unit_id", "arm"]], on="unit_id")
    c["m"] = c.month.map(month_index)
    b0, b1 = map(month_index, config.YARRA_BEFORE)
    a0, a1 = map(month_index, config.YARRA_AFTER)
    c["period"] = np.select([(c.m >= b0) & (c.m <= b1), (c.m >= a0) & (c.m <= a1)], ["before", "after"], None)
    c = c[c.period.notna()]
    mb, ma = b1 - b0 + 1, a1 - a0 + 1

    km = u.groupby("arm").length_m.sum() / 1000
    tab = c.groupby(["arm", "period"]).size().unstack(fill_value=0).reindex(index=["treated", "comparison"])
    yb_t, ya_t = int(tab.loc["treated", "before"]), int(tab.loc["treated", "after"])
    yb_c, ya_c = int(tab.loc["comparison", "before"]), int(tab.loc["comparison", "after"])

    # naive before-after on the trial streets: exact CI from the binomial split of crashes
    n = yb_t + ya_t
    lo_p, hi_p = sps.beta.ppf(0.025, ya_t, n - ya_t + 1), sps.beta.ppf(0.975, ya_t + 1, n - ya_t)
    to_rr = lambda p: p / (1 - p) * mb / ma
    naive = {"rr": (ya_t / ma) / (yb_t / mb), "lo": to_rr(lo_p), "hi": to_rr(hi_p)}

    # DiD as the unit-level conditional Poisson model, clustered by unit
    per = c.groupby(["unit_id", "period"]).size().unstack(fill_value=0)
    d = u[["unit_id", "arm"]].merge(per, on="unit_id", how="left").fillna({"before": 0, "after": 0})
    d = d.rename(columns={"before": "yb", "after": "ya"})
    d = d[(d.yb + d.ya) > 0].assign(mb=mb, ma=ma, cluster=lambda x: x.unit_id.astype(str))
    X = pd.DataFrame({"const": 1.0, "treated": (d.arm == "treated").astype(float)}, index=d.index)
    did = _binomial(d, X, "treated")

    coded30 = c.assign(is30=c.crash_speed_zone == 30).groupby(["arm", "period"]).is30.agg(["sum", "size"])
    fidelity = {f"{a}_{p}": {"coded_30": int(r["sum"]), "crashes": int(r["size"]), "share": float(r["sum"] / r["size"])}
                for (a, p), r in coded30.iterrows()}
    out = {
        "start": config.YARRA_START, "before": list(config.YARRA_BEFORE), "after": list(config.YARRA_AFTER),
        "before_months": mb, "after_months": ma,
        "units": u.arm.value_counts().to_dict(), "km": km.round(1).to_dict(),
        "treated_localities": config.YARRA_TREATED_LOCALITIES, "comparison_localities": config.YARRA_COMPARISON_LOCALITIES,
        "crashes": {"treated_before": yb_t, "treated_after": ya_t, "comparison_before": yb_c, "comparison_after": ya_c},
        "rate_per_km_yr": {f"{a}_{p}": float(tab.loc[a, p] / km[a] / ((mb if p == "before" else ma) / 12))
                           for a in tab.index for p in ("before", "after")},
        "naive": naive, "did": did, "fidelity_coded_30": fidelity,
        "power_wald_95": {f"true_{int(round((1 - rr) * 100))}pct_reduction": power(yb_t, yb_c, ya_c, mb, ma, rr)
                          for rr in (0.8, 0.6)},
    }
    pd.DataFrame([{"arm": a, **{p: int(tab.loc[a, p]) for p in ("before", "after")}, "km": round(km[a], 1)}
                  for a in tab.index]).to_csv(config.TABLES / "yarra_counts.csv", index=False)
    u.drop(columns="geometry").to_parquet(config.PROCESSED / "yarra_units.parquet", index=False)
    save_metrics("yarra", out)
    print(out)


if __name__ == "__main__":
    main()
