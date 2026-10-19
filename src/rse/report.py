"""Figures, interactive map, and README/BRIEF rendered from outputs/metrics.json.

Every number in the rendered documents comes from `facts()` below, which reads only
metrics.json and the pipeline's tables. Templates live in docs/templates/.
"""
import json
from string import Template

import folium
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rse import config  # noqa: E402
from rse.util import load_metrics, month_index  # noqa: E402

# Reference palette (dataviz skill): categorical slots in fixed order + chart chrome.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
TYPE_COLORS = {"50→40": BLUE, "60→40/50": ORANGE, "70/80→lower": AQUA, "90-110→lower": VIOLET, "other": MUTED}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_estimates(m: dict) -> None:
    ev = m["evaluate"]
    rows = [("Naive before–after (treated roads only)", ev["injury"]["naive"], MUTED),
            ("Trend on comparison roads (no change)", ev["injury"]["trend"], MUTED),
            ("Comparison-group DiD", ev["injury"]["did"], BLUE),
            ("Empirical Bayes before–after", ev["injury"]["eb"], BLUE),
            ("DiD, fatal + serious injury only", ev["fsi"]["did"], BLUE)]
    for ct, r in ev["by_type"].items():
        if "skipped" not in r:
            rows.append((f"DiD, {ct} km/h", r["did"], BLUE))
    rows.append(("Placebo: change moved 3 years earlier", ev["sensitivity"]["placebo (change moved 36 months earlier)"], MUTED))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    y = np.arange(len(rows))[::-1]
    for yi, (label, r, col) in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=col, linewidth=2, solid_capstyle="round")
        ax.plot(r["rr"], yi, "o", color=col, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(1.02, yi, f"{r['rr']:.2f}  ({r['lo']:.2f}–{r['hi']:.2f})", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9, color=INK2)
    ax.axvline(1, color=INK2, linewidth=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlabel("Rate ratio, after vs before (1 = no change; < 1 = fewer injury crashes)", fontsize=9, color=INK2)
    fig.suptitle("Injury crashes after a limit reduction: rate ratio by method (95% CI)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "estimates.png", facecolor=SURFACE)
    plt.close(fig)


def fig_event_time() -> None:
    q = pd.read_csv(config.TABLES / "event_time.csv")
    # Half-years: quarterly counts on treated roads (~40) are too noisy to read a trend from.
    q["h"] = np.where(q.q < 0, -((-q.q + 1) // 2), (q.q + 1) // 2)
    e = q.groupby(["treated", "h"]).agg(crashes=("crashes", "sum"), km_months=("km_months", "sum")).reset_index()
    e["rate"] = e.crashes / e.km_months
    e["index"] = e.rate / e.treated.map(e[e.h < 0].groupby("treated").apply(
        lambda d: d.crashes.sum() / d.km_months.sum(), include_groups=False))
    e = e.rename(columns={"h": "q"})
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for grp, col, label in ((1, BLUE, "Roads whose limit was lowered"), (0, ORANGE, "Comparison roads")):
        d = e[e.treated == grp].sort_values("q")
        for part in (d[d.q < 0], d[d.q > 0]):
            ax.plot(part.q, part["index"], color=col, linewidth=2, marker="o", markersize=4)
        last = d[d.q > 0].iloc[-1]
        ax.text(last.q + 0.2, last["index"], label, color=INK2, fontsize=9, va="center")
    ax.axvspan(-0.5, 0.5, color=GRID, alpha=0.6, linewidth=0)
    ax.text(0, ax.get_ylim()[1], "change\n(dropped)", ha="center", va="top", fontsize=8, color=MUTED)
    ax.axhline(1, color=AXIS, linewidth=1)
    ax.set_xlabel("Half-years relative to the limit change (transition months dropped)", fontsize=9, color=INK2)
    ax.set_xticks(range(int(e.q.min()), int(e.q.max()) + 1))
    ax.set_ylabel("Injury-crash rate\n(before average = 1)", fontsize=9, color=INK2)
    ax.set_title("Crash rate around the change, indexed to each group's own before period",
                 fontsize=11, color=INK, loc="left")
    ax.set_xlim(e.q.min() - 0.5, e.q.max() + 2.2)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "event_time.png", facecolor=SURFACE)
    plt.close(fig)


def fig_snap_distance() -> None:
    h = pd.read_csv(config.TABLES / "snap_distance_hist.csv")
    fig, ax = plt.subplots(figsize=(6, 2.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.bar(np.arange(len(h)) + 0.5, h.crashes, width=0.85, color=BLUE)
    ax.set_yscale("log")  # almost every crash is within 1 m; the log scale shows the tail
    ax.set_xlabel("Distance from crash point to assigned road segment (m)", fontsize=9, color=INK2)
    ax.set_ylabel("Crashes (log scale)", fontsize=9, color=INK2)
    ax.set_title("Crash-to-road snapping distance (30 m buffer, log scale)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "snap_distance.png", facecolor=SURFACE)
    plt.close(fig)


def build_map() -> None:
    units = gpd.read_parquet(config.PROCESSED / "units.parquet", columns=["unit_id", "geometry"])
    st = pd.read_parquet(config.PROCESSED / "stack_injury.parquet")
    t = st[st.treated == 1][["unit_id", "cohort", "from_limit", "to_limit", "change_type", "scheme", "lga",
                             "yb", "ya", "before", "after", "km"]]
    tu = units.merge(t, on="unit_id")
    tu["geometry"] = tu.geometry.simplify(3)
    tu = tu.to_crs(4326)
    tu["label"] = tu.from_limit.astype(int).astype(str) + "→" + tu.to_limit.astype(int).astype(str) + " km/h"
    tu["length"] = (tu.km * 1000).round().astype(int).astype(str) + " m"

    fmap = folium.Map(location=[-37.81, 144.96], zoom_start=11, tiles="OpenStreetMap", control_scale=True)
    for ct, col in TYPE_COLORS.items():
        d = tu[tu.change_type == ct]
        if d.empty:
            continue
        fg = folium.FeatureGroup(name=f"Lowered {ct} km/h ({len(d)} segments)")
        folium.GeoJson(
            d[["geometry", "label", "cohort", "scheme", "lga", "length", "yb", "ya", "before", "after"]],
            style_function=lambda _, c=col: {"color": c, "weight": 4, "opacity": 0.9},
            highlight_function=lambda _: {"weight": 7},
            tooltip=folium.GeoJsonTooltip(
                fields=["label", "scheme", "lga", "length", "before", "yb", "after", "ya"],
                aliases=["Change", "Change request", "LGA", "Length", "Before window", "Injury crashes before",
                         "After window", "Injury crashes after"]),
        ).add_to(fg)
        fg.add_to(fmap)

    # crashes on treated segments inside their windows
    c = pd.read_parquet(config.PROCESSED / "crashes.parquet", columns=["accident_no", "month", "severity", "lat", "lon", "dca"])
    s = pd.read_parquet(config.PROCESSED / f"snap_{config.SNAP_BUFFER_M}m.parquet", columns=["accident_no", "unit_id"])
    c = c.merge(s, on="accident_no").merge(t[["unit_id", "before", "after"]], on="unit_id")
    c = c[c.severity.isin((1, 2, 3))]
    inside = lambda m, w: (m >= w.str[:7]) & (m <= w.str[-7:])
    for period, col in (("before", MUTED), ("after", RED)):
        d = c[inside(c.month, c[period])]
        fg = folium.FeatureGroup(name=f"Injury crashes {period} the change ({len(d)})", show=period == "after")
        for r in d.itertuples():
            folium.CircleMarker([r.lat, r.lon], radius=3, color=col, fill=True, fill_opacity=0.8, weight=1,
                                tooltip=f"{r.month} · {'fatal' if r.severity == 1 else 'serious' if r.severity == 2 else 'other injury'} · {r.dca}"
                                ).add_to(fg)
        fg.add_to(fmap)

    # Yarra case study
    y = pd.read_parquet(config.PROCESSED / "yarra_units.parquet", columns=["unit_id", "arm", "limit_before"])
    yu = units.merge(y, on="unit_id")
    yu["geometry"] = yu.geometry.simplify(2)
    yu = yu.to_crs(4326)
    for arm, col, name in (("treated", MAGENTA, "Yarra 30 km/h trial streets (May 2024)"),
                            ("comparison", GREEN, "Yarra case: comparison streets")):
        d = yu[yu.arm == arm]
        fg = folium.FeatureGroup(name=f"{name} ({len(d)})", show=False)
        folium.GeoJson(d[["geometry", "limit_before"]], style_function=lambda _, c=col: {"color": c, "weight": 3},
                       tooltip=folium.GeoJsonTooltip(fields=["limit_before"], aliases=["Limit before trial"])).add_to(fg)
        fg.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(config.OUTPUTS / "map.html")


def _f(x, nd=2):
    return f"{x:,.{nd}f}"


def _pct(r):
    """'a 12% reduction' style phrase from a rate-ratio dict."""
    p = r["pct_change"]
    return f"{abs(p):.0f}% {'lower' if p < 0 else 'higher'}"


def _range(r) -> str:
    """95% CI of a rate ratio in words: 'between 1% and 34% fewer' / 'from 42% fewer to 10% more'."""
    fewer, more = round(100 * (1 - r["lo"])), round(100 * (r["hi"] - 1))
    if r["hi"] < 1:
        return f"between {round(100 * (1 - r['hi']))}% and {fewer}% fewer"
    return f"anywhere from {fewer}% fewer to {more}% more"


def facts(m: dict) -> dict:
    ing, net, trt, snp, ev, ya = (m[k] for k in ("ingest", "network", "treatment", "snap", "evaluate", "yarra"))
    s30 = snp[f"{config.SNAP_BUFFER_M}m"]
    inj, fsi = ev["injury"], ev["fsi"]
    f = {
        "crash_rows": f"{ing['crashes']['rows']:,}", "crash_first": ing["crashes"]["first_date"],
        "crash_last": ing["crashes"]["last_date"], "no_location": f"{ing['crashes']['no_location']:,}",
        "node_rows": f"{ing['node_rows']:,}", "node_unique": f"{ing['node_unique_crashes']:,}",
        "node_conflicts": f"{ing['node_conflicting_crashes']:,}",
        "datum_gda94": _f(ing["crash_datum_check"]["median_offset_m_if_gda94"]),
        "datum_gda2020": _f(ing["crash_datum_check"]["median_offset_m_if_gda2020"]),
        "n_snapshots": len(ing["snapshots"]), "snap_gb": _f(ing["snapshot_geojson_gb"], 1),
        "snap_parquet_mb": _f(ing["snapshot_parquet_mb"], 0),
        "snap_features": f"{ing['snapshot_features_total']:,}",
        "base_features": f"{net['base_features']:,}", "units": f"{net['units']:,}",
        "network_km": f"{net['network_km']:,}", "declared_km": f"{net['declared_km']:,}",
        "legacy_match": _f(100 * net["legacy_match_rate"], 2),
        "ch_from": trt["all_changes_context"]["from"].replace("_", "-"),
        "ch_to": trt["all_changes_context"]["to"].replace("_", "-"),
        "ch_changed": f"{trt['all_changes_context']['changed']:,}",
        "ch_dec": f"{trt['all_changes_context']['decreases']:,}",
        "ch_inc": f"{trt['all_changes_context']['increases']:,}",
        "treated_units": f"{trt['treated_units']:,}", "treated_km": _f(trt["treated_km"], 0),
        "treated_schemes": f"{trt['treated_schemes']:,}",
        "comparison_units": f"{trt['comparison_units']:,}", "comparison_km": f"{trt['comparison_km']:,}",
        "match_share": _f(100 * s30["matched_share"], 1),
        "match_I": _f(100 * s30["matched_share_by_node_type"]["I"], 1),
        "match_N": _f(100 * s30["matched_share_by_node_type"]["N"], 1),
        "dist_p50": _f(s30["dist_m_quantiles"]["0.5"], 1), "dist_p95": _f(s30["dist_m_quantiles"]["0.95"], 1),
        "near_tie": _f(100 * s30["near_tie_share"], 1),
        "agree": _f(100 * s30["limit_agreement_since_base"], 1),
        "agree_single": _f(100 * s30["limit_agreement_since_base_single"], 1),
        "agree_n": f"{s30['limit_agreement_since_base_n']:,}",
        "agree_normal": _f(100 * s30["limit_agreement_since_base_normal_zones"], 0),
        "agree_tod": _f(100 * s30["limit_agreement_since_base_time_of_day_zones"], 0),
        "dist_med_ok": _f(snp["datum_check"]["median_dist_m_configured"]),
        "dist_med_bad": _f(snp["datum_check"]["median_dist_m_if_speed_zones_gda2020"]),
        "excl_tod": f"{trt['excluded_reasons'].get('time_of_day_zone', 0):,}",
        "excl_window": f"{trt['excluded_reasons'].get('change_outside_window', 0):,}",
        "excl_2123": f"{trt['excluded_reasons'].get('changed_2021_2023', 0):,}",
        "excl_inc": f"{trt['excluded_reasons'].get('increase', 0):,}",
        "excl_other": f"{sum(v for k, v in trt['excluded_reasons'].items() if k not in ('time_of_day_zone', 'change_outside_window', 'changed_2021_2023', 'increase')):,}",
        "metro_share": _f(100 * trt["treated_by_region"].get("metro", 0) / trt["treated_units"], 0),
        "declared_share": _f(100 * trt["treated_declared_share"], 0),
        "top_lga": next(iter(trt["treated_top_lgas"])).title(),
        "top_lga_n": f"{next(iter(trt['treated_top_lgas'].values())):,}",
    }
    for key, r in (("naive", inj["naive"]), ("trend", inj["trend"]), ("did", inj["did"]), ("cells", inj["cells"]),
                   ("eb", inj["eb"]), ("fsi", fsi["did"]), ("fsi_naive", fsi["naive"])):
        f.update({f"{key}_rr": _f(r["rr"]), f"{key}_lo": _f(r["lo"]), f"{key}_hi": _f(r["hi"]),
                  f"{key}_pct": _f(abs(r["pct_change"]), 0), f"{key}_dir": "fewer" if r["rr"] < 1 else "more",
                  f"{key}_phrase": _pct(r)})
    d = inj["did"]
    f.update({"did_p": _f(d["p"], 3), "did_units": f"{d['treated_units']:,}", "did_km": _f(d["treated_km"], 0),
              "did_tb": f"{d['treated_before']:,}", "did_ta": f"{d['treated_after']:,}",
              "did_cu": f"{d['comparison_units']:,}", "did_cb": f"{d['comparison_before']:,}",
              "did_ca": f"{d['comparison_after']:,}", "did_clusters": f"{d['clusters']['assignment']:,}",
              "naive_tb": f"{inj['naive']['treated_before']:,}", "naive_ta": f"{inj['naive']['treated_after']:,}",
              "eb_obs": f"{inj['eb']['observed_after']:,}", "eb_exp": _f(inj["eb"]["expected_after"], 0),
              "fsi_tb": f"{fsi['did']['treated_before']:,}", "fsi_ta": f"{fsi['did']['treated_after']:,}"})
    sens = ev["sensitivity"]
    order = ["buffer 15 m", "buffer 50 m", "midblock crashes only", "transition padded 3 months each side",
             "region strata only", "12-month before window", "24-month before window", "60-month before window"]
    f["sens_table"] = "\n".join(
        f"| {label} | {sens[label]['rr']:.2f} ({sens[label]['lo']:.2f}–{sens[label]['hi']:.2f}) | "
        f"{sens[label]['treated_before']:,} / {sens[label]['treated_after']:,} |" for label in order)
    fmt = lambda r: f"{r['rr']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})"
    f["sens_pad"] = fmt(sens["transition padded 3 months each side"])
    f["sens_mid"] = fmt(sens["midblock crashes only"])
    rc = ing["crashes"]["recent_monthly_counts"]
    f.update({"m_dec25": f"{rc['2025-12']:,}", "m_jan26": f"{rc['2026-01']:,}", "m_feb26": f"{rc['2026-02']:,}"})
    span = lambda w: month_index(w.split("..")[1]) - month_index(w.split("..")[0]) + 1
    after = [span(c["after"]) for c in ev["cohorts"]]
    f.update({"after_min": min(after), "after_max": max(after), "y_n_comp": len(ya["comparison_localities"])})
    f["sens_min"] = _f(min(sens[k]["rr"] for k in order))
    f["sens_max"] = _f(max(sens[k]["rr"] for k in order))
    f["bw_min"] = _f(min(sens[k]["rr"] for k in order if "before window" in k))
    f["bw_max"] = _f(max(sens[k]["rr"] for k in order if "before window" in k))
    f["sens_cross1"] = sum(sens[k]["hi"] >= 1 for k in order)
    pct_fewer = lambda x: _f(100 * (1 - x), 0)
    f["sens_fewer_lo"] = pct_fewer(max(sens[k]["rr"] for k in order))
    f["sens_fewer_hi"] = pct_fewer(min(sens[k]["rr"] for k in order))
    f["bw_fewer_lo"] = pct_fewer(max(sens[k]["rr"] for k in order if "before window" in k))
    f["bw_fewer_hi"] = pct_fewer(min(sens[k]["rr"] for k in order if "before window" in k))
    f["sens_n"] = len(order)
    pl_all = ev["injury_placebo"]
    for key in ("naive", "trend", "did", "eb"):
        r = pl_all[key]
        f.update({f"pl_{key}_rr": _f(r["rr"]), f"pl_{key}_lo": _f(r["lo"]), f"pl_{key}_hi": _f(r["hi"]),
                  f"pl_{key}_pct": _f(abs(r["pct_change"]), 0)})
    f["pl_trend_pct"] = _f(abs(pl_all["trend"]["pct_change"]), 0)
    f["eb_obs_pl"] = f"{pl_all['eb']['observed_after']:,}"
    f["eb_exp_pl"] = _f(pl_all["eb"]["expected_after"], 0)
    # plain-English ranges: "between X% and Y% fewer"
    for key, r in (("did", inj["did"]), ("fsi", fsi["did"]), ("y_did", ya["did"])):
        f[f"{key}_range"] = _range(r)
    for key, ct in (("t5040", "50→40"), ("t60", "60→40/50"), ("t7080", "70/80→lower"), ("t90", "90-110→lower")):
        r = ev["by_type"][ct]["did"]
        f.update({f"{key}_rr": _f(r["rr"]), f"{key}_pct": _f(abs(r["pct_change"]), 0),
                  f"{key}_dir": "fewer" if r["rr"] < 1 else "more", f"{key}_range": _range(r)})
    f["t90_pl_pct"] = _f(abs(ev["by_type"]["90-110→lower"]["placebo_did"]["pct_change"]), 0)
    f["y_miss20"] = _f(100 * (1 - ya["power_wald_95"]["true_20pct_reduction"]), 0)
    f["y_did_p"] = _f(ya["did"]["p"], 2)
    f["y_did_pct"] = _f(abs(ya["did"]["pct_change"]), 0)
    f["type_table"] = "\n".join(
        f"| {ct} | {r['did']['treated_units']:,} | {r['did']['treated_before']:,} / {r['did']['treated_after']:,} | "
        f"{r['naive']['rr']:.2f} ({r['naive']['lo']:.2f}–{r['naive']['hi']:.2f}) | "
        f"{r['did']['rr']:.2f} ({r['did']['lo']:.2f}–{r['did']['hi']:.2f}) | "
        f"{r['eb']['rr']:.2f} ({r['eb']['lo']:.2f}–{r['eb']['hi']:.2f}) | "
        f"{r['placebo_did']['rr']:.2f} ({r['placebo_did']['lo']:.2f}–{r['placebo_did']['hi']:.2f}) |"
        if "skipped" not in r else f"| {ct} | {r['treated_units']:,} | {r['treated_before']:,} / {r['treated_after']:,} | – | too few crashes | – | – |"
        for ct, r in ev["by_type"].items())
    f["pre_table"] = "\n".join(
        f"| {r['change_type']} | {r['treated_rate_per_km_yr']:.2f} | {r['comparison_rate_per_km_yr']:.2f} | {r['ratio']:.1f}× |"
        for r in ev["pre_period_rates"])
    f["cohort_table"] = "\n".join(
        f"| {c['cohort'].replace('_', '-')} | {c['units']:,} | {c['before']} | {c['after']} | {c['crashes_before']:,} / {c['crashes_after']:,} |"
        for c in ev["cohorts"])
    yc, yf = ya["crashes"], ya["fidelity_coded_30"]
    f.update({
        "y_units_t": f"{ya['units']['treated']:,}", "y_units_c": f"{ya['units']['comparison']:,}",
        "y_km_t": _f(ya["km"]["treated"], 1), "y_km_c": _f(ya["km"]["comparison"], 1),
        "y_tb": yc["treated_before"], "y_ta": yc["treated_after"], "y_cb": yc["comparison_before"], "y_ca": yc["comparison_after"],
        "y_mb": ya["before_months"], "y_ma": ya["after_months"],
        "y_naive_rr": _f(ya["naive"]["rr"]), "y_naive_lo": _f(ya["naive"]["lo"]), "y_naive_hi": _f(ya["naive"]["hi"]),
        "y_did_rr": _f(ya["did"]["rr"]), "y_did_lo": _f(ya["did"]["lo"]), "y_did_hi": _f(ya["did"]["hi"]),
        "y_30_before": _f(100 * yf["treated_before"]["share"], 0), "y_30_after": _f(100 * yf["treated_after"]["share"], 0),
        "y_30_after_n": yf["treated_after"]["coded_30"], "y_after_n": yf["treated_after"]["crashes"],
        "y_pow20": _f(100 * ya["power_wald_95"]["true_20pct_reduction"], 0),
        "y_pow40": _f(100 * ya["power_wald_95"]["true_40pct_reduction"], 0),
    })
    return f


def render(f: dict) -> None:
    for name, dest in (("README.md.tmpl", config.ROOT / "README.md"), ("BRIEF.md.tmpl", config.ROOT / "docs" / "BRIEF.md")):
        tmpl = config.ROOT / "docs" / "templates" / name
        if tmpl.exists():
            dest.write_text(Template(tmpl.read_text()).substitute(f))


def main():
    m = load_metrics()
    fig_estimates(m)
    fig_event_time()
    fig_snap_distance()
    build_map()
    f = facts(m)
    (config.OUTPUTS / "facts.json").write_text(json.dumps(f, indent=2, ensure_ascii=False))
    render(f)
    print(f"map {(config.OUTPUTS / 'map.html').stat().st_size / 1e6:.1f} MB; {len(f)} facts")


if __name__ == "__main__":
    main()
