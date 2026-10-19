"""Stacked comparison-group evaluation of speed-limit reductions.

For each cohort (snapshot in which a decrease first appears) the treated units of that cohort
are paired with clean comparison units from the same stratum (original limit x declared/local
x LGA, falling back to region when the LGA has too few comparison units). Each unit has a
before count and an after count over the cohort's windows.

Estimators
- naive before-after on treated units only;
- comparison-group DiD: Poisson with unit fixed effects. With two periods this is exactly a
  binomial GLM for the after-share of each unit's crashes, offset log(after/before months),
  with cohort x stratum effects; exp(treated coef) is the DiD rate ratio;
- the same DiD as a cell-level Poisson GLM (identical point estimate; cross-check);
- Empirical Bayes before-after (Hauer): safety performance function on comparison units (Poisson
  mean, NB2 overdispersion by moments), comparison-group trend for the after period.

Standard errors: cluster-robust SEs are computed three ways (by scheme for treated / LGA for
comparison units, by unit, by LGA) and the largest is reported. Some subsets have few, very
unequal clusters (e.g. 18 schemes for 50->40), where a single clustering can understate the SE.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from rse import config
from rse.util import month_index, month_str, save_metrics, timed
from rse.windows import cohort_windows

OUTCOMES = {"injury": (1, 2, 3), "fsi": (1, 2)}
Z = 1.959964


@dataclass(frozen=True)
class Spec:
    outcome: str = "injury"
    buffer_m: int = config.SNAP_BUFFER_M
    midblock_only: bool = False
    pad: int = 0
    shift: int = 0
    before_months: int = config.BEFORE_MONTHS
    strata: str = "lga"            # "lga" (with region fallback) or "region"


def load_crashes(spec: Spec) -> pd.DataFrame:
    c = pd.read_parquet(config.PROCESSED / "crashes.parquet", columns=["accident_no", "month", "severity", "node_type"])
    s = pd.read_parquet(config.PROCESSED / f"snap_{spec.buffer_m}m.parquet", columns=["accident_no", "unit_id"])
    c = c.merge(s, on="accident_no")
    c = c[c.unit_id.notna() & c.severity.isin(OUTCOMES[spec.outcome])]
    if spec.midblock_only:
        c = c[c.node_type == "N"]
    return pd.DataFrame({"unit_id": c.unit_id.astype("int64").to_numpy(),
                         "m": [month_index(x) for x in c.month]})


def _counts(crashes: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    sel = crashes[(crashes.m >= lo) & (crashes.m <= hi)]
    return sel.groupby("unit_id").size()


def build_stack(tr: pd.DataFrame, crashes: pd.DataFrame, spec: Spec, treated_mask=None) -> pd.DataFrame:
    t_all = tr[tr.status == "treated"] if treated_mask is None else tr[(tr.status == "treated") & treated_mask]
    comp = tr[tr.status == "comparison"].copy()
    comp["k_lga"] = comp.from_limit.astype(int).astype(str) + "|" + comp.declared.astype(str) + "|" + comp.lga
    comp["k_reg"] = comp.from_limit.astype(int).astype(str) + "|" + comp.declared.astype(str) + "|" + comp.region
    n_lga = comp.k_lga.value_counts()
    rows = []
    for cohort, t in t_all.groupby("cohort"):
        w = cohort_windows(t.prev_snapshot.iloc[0], cohort, config.STUDY_END, spec.before_months,
                           pad=spec.pad, shift=spec.shift)
        t = t.copy()
        t["k_lga"] = t.from_limit.astype(int).astype(str) + "|" + t.declared.astype(str) + "|" + t.lga
        t["k_reg"] = t.from_limit.astype(int).astype(str) + "|" + t.declared.astype(str) + "|" + t.region
        enough = t.k_lga.map(n_lga).fillna(0) >= config.MIN_COMPARISON_UNITS
        t["stratum"] = np.where(enough & (spec.strata == "lga"), "L:" + t.k_lga, "R:" + t.k_reg)
        used = set(t.stratum)
        c_l = comp[("L:" + comp.k_lga).isin(used)].assign(stratum=lambda d: "L:" + d.k_lga)
        c_r = comp[("R:" + comp.k_reg).isin(used)].assign(stratum=lambda d: "R:" + d.k_reg)
        g = pd.concat([t.assign(treated=1), c_l.assign(treated=0), c_r.assign(treated=0)], ignore_index=True)
        yb, ya = _counts(crashes, *w.before), _counts(crashes, *w.after)
        g["yb"] = g.unit_id.map(yb).fillna(0).astype(int)
        g["ya"] = g.unit_id.map(ya).fillna(0).astype(int)
        g["mb"], g["ma"] = w.before_months, w.after_months
        g["cohort"] = cohort
        g["before"] = f"{month_str(w.before[0])}..{month_str(w.before[1])}"
        g["after"] = f"{month_str(w.after[0])}..{month_str(w.after[1])}"
        rows.append(g)
    st = pd.concat(rows, ignore_index=True)
    st["cs"] = st.cohort + "#" + st.stratum
    st["km"] = st.length_m / 1000
    st["cluster"] = np.where(st.treated == 1, "S:" + st.scheme.fillna("na"), "L:" + st.lga)
    keep = ["cohort", "unit_id", "treated", "stratum", "cs", "yb", "ya", "mb", "ma", "km", "lga", "region",
            "declared", "from_limit", "to_limit", "change_type", "scheme", "cluster", "before", "after", "aadt_2019"]
    return st[keep]


def _ci(beta: float, se: float) -> dict:
    return {"rr": float(np.exp(beta)), "lo": float(np.exp(beta - Z * se)), "hi": float(np.exp(beta + Z * se)),
            "pct_change": float(100 * (np.exp(beta) - 1))}


def _binomial(d: pd.DataFrame, X: pd.DataFrame, term: str) -> dict:
    """Binomial GLM on (after, before) counts; reports the largest of the available cluster-robust SEs."""
    endog = np.column_stack([d.ya.to_numpy(), d.yb.to_numpy()]).astype(float)
    offset = np.log(d.ma.to_numpy() / d.mb.to_numpy())
    model = sm.GLM(endog, X.astype(float), family=sm.families.Binomial(), offset=offset)
    ses, k, beta = {}, {}, None
    for name, col in (("assignment", "cluster"), ("unit", "unit_id"), ("lga", "lga")):
        if col not in d:
            continue
        groups = pd.factorize(d[col])[0]
        res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        beta, ses[name], k[name] = res.params[term], float(res.bse[term]), int(groups.max() + 1)
    se = max(ses.values())
    out = _ci(beta, se)
    out.update({"se_log": se, "se_by_clustering": ses, "clusters": k,
                "p": float(2 * stats.norm.sf(abs(beta) / se))})
    return out


def fit_did(st: pd.DataFrame) -> dict:
    """Conditional (unit fixed-effect) Poisson DiD. Strata that cannot inform the contrast are dropped."""
    d = st[(st.yb + st.ya) > 0]
    g = d.groupby("cs")
    ok = (g.treated.transform("nunique") == 2) & (g.ya.transform("sum") > 0) & (g.yb.transform("sum") > 0)
    d = d[ok]
    X = pd.get_dummies(d.cs, dtype=float)
    X["treated"] = d.treated.astype(float)
    out = _binomial(d, X, "treated")
    used = st[st.cs.isin(d.cs.unique())]
    out.update(_sample(used))
    out["strata_used"] = int(d.cs.nunique())
    return out


def fit_naive(st: pd.DataFrame) -> dict:
    d = st[(st.treated == 1) & ((st.yb + st.ya) > 0)]
    out = _binomial(d, pd.DataFrame({"const": np.ones(len(d))}, index=d.index), "const")
    out.update(_sample(st[st.treated == 1]))
    return out


def fit_trend(st: pd.DataFrame) -> dict:
    """Comparison units only: how much crashes moved on untreated roads (same windows)."""
    d = st[(st.treated == 0) & ((st.yb + st.ya) > 0)].drop_duplicates(["cohort", "unit_id"])
    return _binomial(d, pd.DataFrame({"const": np.ones(len(d))}, index=d.index), "const")


def fit_cells(st: pd.DataFrame) -> dict:
    """Cell-level Poisson GLM on (cohort x stratum x group x period); same strata as fit_did."""
    d = st[(st.yb + st.ya) > 0]
    g = d.groupby("cs")
    ok = (g.treated.transform("nunique") == 2) & (g.ya.transform("sum") > 0) & (g.yb.transform("sum") > 0)
    cs_ok = set(d[ok].cs)
    s = st[st.cs.isin(cs_ok)]
    cells = s.groupby(["cs", "treated"]).agg(yb=("yb", "sum"), ya=("ya", "sum"), km=("km", "sum"),
                                             mb=("mb", "first"), ma=("ma", "first")).reset_index()
    long = pd.concat([
        cells.assign(post=0, y=cells.yb, expo=cells.km * cells.mb / 12),
        cells.assign(post=1, y=cells.ya, expo=cells.km * cells.ma / 12)], ignore_index=True)
    post_fe = pd.get_dummies(long.cs, dtype=float).mul(long.post, axis=0).add_suffix("|post")
    X = pd.concat([pd.get_dummies(long.cs + "|g" + long.treated.astype(str), dtype=float), post_fe], axis=1)
    X["did"] = (long.treated * long.post).astype(float)
    res = sm.GLM(long.y.astype(float), X, family=sm.families.Poisson(), offset=np.log(long.expo)).fit()
    out = _ci(res.params["did"], res.bse["did"])
    out["se_log_model_based"] = float(res.bse["did"])
    return out


def fit_eb(st: pd.DataFrame) -> dict:
    """Hauer EB before-after with an NB SPF per cohort and a stratum comparison-group trend."""
    def design(d: pd.DataFrame, limits: list[str]) -> pd.DataFrame:
        X = pd.DataFrame({f"lim_{v}": (d.from_limit.astype(int).astype(str) == v).astype(float) for v in limits},
                         index=d.index)
        X["declared"] = d.declared.astype(float)
        X["metro"] = (d.region == "metro").astype(float)
        X["log_km"] = np.log(d.km.clip(lower=0.005))
        return X

    lam = pi = var_pi = 0.0
    for cohort, s in st.groupby("cohort"):
        comp = s[s.treated == 0].drop_duplicates("unit_id")
        limits = sorted(comp.from_limit.astype(int).astype(str).unique())[1:]   # first level is the reference
        X = design(comp, limits)
        X = sm.add_constant(X.loc[:, X.std() > 0])                              # drop constant covariates
        y = comp.yb.astype(float).to_numpy()
        # SPF mean from a Poisson GLM; NB2 overdispersion by method of moments. Sturdier than the NB
        # likelihood when a small stratum has all-zero counts for some limit level.
        spf = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        mu_c = spf.fittedvalues
        alpha = max(float(((y - mu_c) ** 2 - mu_c).sum() / (mu_c ** 2).sum()), 1e-6)
        t = s[s.treated == 1]
        Xt = sm.add_constant(design(t, limits), has_constant="add")[X.columns]
        mu = np.exp(Xt.to_numpy() @ spf.params.to_numpy())
        w = 1.0 / (1.0 + alpha * mu)
        eb = w * mu + (1 - w) * t.yb.to_numpy()
        var_eb = (1 - w) * eb
        cs = s[s.treated == 0].groupby("stratum")[["yb", "ya"]].sum()
        pooled = cs.ya.sum() / max(cs.yb.sum(), 1)
        ratio = (cs.ya / cs.yb).where(cs.yb > 0)
        r = t.stratum.map(ratio).fillna(pooled).to_numpy()
        lam += t.ya.sum()
        pi += float((r * eb).sum())
        var_pi += float((r ** 2 * var_eb).sum())
    theta = (lam / pi) / (1 + var_pi / pi ** 2)
    var_theta = theta ** 2 * (1 / lam + var_pi / pi ** 2) / (1 + var_pi / pi ** 2) ** 2
    se = np.sqrt(var_theta)
    return {"rr": float(theta), "lo": float(theta - Z * se), "hi": float(theta + Z * se),
            "pct_change": float(100 * (theta - 1)), "observed_after": int(lam), "expected_after": float(pi)}


def _sample(s: pd.DataFrame) -> dict:
    t, c = s[s.treated == 1], s[s.treated == 0]
    return {"treated_units": int(t.unit_id.nunique()), "treated_km": float(t.km.sum()),
            "treated_before": int(t.yb.sum()), "treated_after": int(t.ya.sum()),
            "comparison_units": int(c.unit_id.nunique()),
            "comparison_before": int(c.drop_duplicates(["cohort", "unit_id"]).yb.sum()),
            "comparison_after": int(c.drop_duplicates(["cohort", "unit_id"]).ya.sum())}


def pre_period_rates(st: pd.DataFrame) -> pd.DataFrame:
    """Before-window crash rate per km-year, treated vs comparison, by change type."""
    s = st.drop_duplicates(["cohort", "unit_id", "treated"]).copy()
    s["expo"] = s.km * s.mb / 12
    t = s[s.treated == 1].groupby("change_type").agg(units=("unit_id", "nunique"), crashes=("yb", "sum"), expo=("expo", "sum"))
    t["rate"] = t.crashes / t.expo
    out = []
    for ct in t.index:
        strata = st[(st.treated == 1) & (st.change_type == ct)].cs.unique()
        c = s[(s.treated == 0) & s.cs.isin(strata)].drop_duplicates(["cohort", "unit_id"])
        out.append({"change_type": ct, "treated_units": int(t.loc[ct, "units"]),
                    "treated_rate_per_km_yr": float(t.loc[ct, "rate"]),
                    "comparison_rate_per_km_yr": float(c.yb.sum() / (c.km * c.mb / 12).sum()),
                    })
    df = pd.DataFrame(out)
    df["ratio"] = df.treated_rate_per_km_yr / df.comparison_rate_per_km_yr
    return df


def event_time(st: pd.DataFrame, crashes: pd.DataFrame, max_after_q: int = 4) -> pd.DataFrame:
    """Quarterly crash rate around the change for treated and comparison units, each indexed to its
    own before-window mean (=1). Quarters after +max_after_q are dropped so every cohort contributes."""
    rows = []
    for cohort, s in st.groupby("cohort"):
        w0 = month_index(s.before.iloc[0].split("..")[0])
        p = month_index(s.before.iloc[0].split("..")[1])
        a0 = month_index(s.after.iloc[0].split("..")[0])
        for grp, g in s.drop_duplicates(["unit_id", "treated"]).groupby("treated"):
            km = g.km.sum()
            cm = crashes[crashes.unit_id.isin(g.unit_id)]
            for m in range(w0, month_index(config.STUDY_END) + 1):
                if p < m < a0:
                    continue
                q = (m - p + 2) // 3 - 1 if m <= p else (m - a0) // 3 + 1
                rows.append({"cohort": cohort, "treated": grp, "q": q, "crashes": int((cm.m == m).sum()), "km_months": km})
    e = pd.DataFrame(rows).groupby(["treated", "q"]).agg(crashes=("crashes", "sum"), km_months=("km_months", "sum")).reset_index()
    e["rate_per_km_yr"] = e.crashes / e.km_months * 12
    base = e[e.q < 0].groupby("treated").apply(lambda d: d.crashes.sum() / d.km_months.sum() * 12, include_groups=False)
    e["index"] = e.rate_per_km_yr / e.treated.map(base)
    return e[e.q <= max_after_q]


def run(tr: pd.DataFrame, spec: Spec, treated_mask=None, with_extras: bool = False) -> dict:
    crashes = load_crashes(spec)
    st = build_stack(tr, crashes, spec, treated_mask)
    res = {"did": fit_did(st)}
    if with_extras:
        res["naive"] = fit_naive(st)
        res["trend"] = fit_trend(st)
        res["cells"] = fit_cells(st)
        res["eb"] = fit_eb(st)
        res["_stack"] = st
        res["_crashes"] = crashes
    return res


def main():
    tr = pd.read_parquet(config.PROCESSED / "treatment.parquet")
    metrics, timings, est_rows = {}, {}, []

    def record(label, subset, outcome, r, estimator):
        est_rows.append({"analysis": label, "subset": subset, "outcome": outcome, "estimator": estimator,
                         **{k: v for k, v in r.items() if not k.startswith("_")}})

    with timed("main", timings):
        for outcome in OUTCOMES:
            r = run(tr, Spec(outcome=outcome), with_extras=True)
            for est in ("naive", "trend", "did", "cells", "eb"):
                record("main", "all", outcome, r[est], est)
            metrics[outcome] = {k: v for k, v in r.items() if not k.startswith("_")}
            if outcome == "injury":
                st, crashes = r["_stack"], r["_crashes"]
                st.to_parquet(config.PROCESSED / "stack_injury.parquet", index=False)
                pre = pre_period_rates(st)
                pre.to_csv(config.TABLES / "pre_period_rates.csv", index=False)
                metrics["pre_period_rates"] = pre.to_dict(orient="records")
                ev = event_time(st, crashes)
                ev.to_csv(config.TABLES / "event_time.csv", index=False)
                coh = st[st.treated == 1].groupby("cohort").agg(
                    units=("unit_id", "nunique"), before=("before", "first"), after=("after", "first"),
                    crashes_before=("yb", "sum"), crashes_after=("ya", "sum")).reset_index()
                coh.to_csv(config.TABLES / "cohorts.csv", index=False)
                metrics["cohorts"] = coh.to_dict(orient="records")
                # Every estimator on placebo windows (change moved 3 years earlier; true effect is none).
                # A biased estimator shows up here, e.g. an SPF that cannot explain why treated roads are riskier.
                pl = run(tr, Spec(outcome=outcome, shift=config.PLACEBO_SHIFT_MONTHS), with_extras=True)
                metrics["injury_placebo"] = {k: pl[k] for k in ("naive", "trend", "did", "eb")}
                for est in ("naive", "trend", "did", "eb"):
                    record("placebo", "all", outcome, pl[est], est)

    with timed("by change type", timings):
        by_type = {}
        for ct, n in tr[tr.status == "treated"].change_type.value_counts().items():
            mask = tr.change_type == ct
            r = run(tr, Spec(), treated_mask=mask, with_extras=True)
            if r["did"]["treated_before"] < 30:
                by_type[ct] = {"skipped": "fewer than 30 treated before-crashes", **r["did"]}
                continue
            by_type[ct] = {k: v for k, v in r.items() if not k.startswith("_")}
            by_type[ct]["placebo_did"] = run(tr, Spec(shift=config.PLACEBO_SHIFT_MONTHS), treated_mask=mask)["did"]
            for est in ("naive", "did", "eb"):
                record("by_type", ct, "injury", r[est], est)
            record("by_type_placebo", ct, "injury", by_type[ct]["placebo_did"], "did")
        metrics["by_type"] = by_type

    with timed("placebo + sensitivity", timings):
        sens = {
            "placebo (change moved 36 months earlier)": Spec(shift=config.PLACEBO_SHIFT_MONTHS),
            "buffer 15 m": Spec(buffer_m=15),
            "buffer 50 m": Spec(buffer_m=50),
            "midblock crashes only": Spec(midblock_only=True),
            "transition padded 3 months each side": Spec(pad=3),
            "region strata only": Spec(strata="region"),
            # If schemes were triggered by a recent crash spike (regression to the mean), short before
            # windows would show bigger reductions than long ones.
            "12-month before window": Spec(before_months=12),
            "24-month before window": Spec(before_months=24),
            "60-month before window": Spec(before_months=60),
        }
        metrics["sensitivity"] = {}
        for label, spec in sens.items():
            r = run(tr, spec)
            metrics["sensitivity"][label] = r["did"]
            record("sensitivity", label, "injury", r["did"], "did")

    est = pd.DataFrame(est_rows)
    est.to_csv(config.TABLES / "estimates.csv", index=False)
    metrics["timings_s"] = timings
    save_metrics("evaluate", metrics)
    print(est[["analysis", "subset", "outcome", "estimator", "rr", "lo", "hi", "treated_before", "treated_after"]]
          .to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
