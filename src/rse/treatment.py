"""Treated and comparison units from the snapshot sequence.

treated    : exactly one limit decrease, first visible in a snapshot inside TREAT_FIRST_SEEN,
             no other change up to CLEAN_THROUGH, Normal Operation throughout.
comparison : no change in any snapshot (Nov 2021 -> CLEAN_THROUGH), Normal Operation throughout.
Everything else is excluded with a reason, checked in the order below.
"""
import numpy as np
import pandas as pd

from rse import config
from rse.util import month_index, save_metrics


def analysis_keys() -> list[str]:
    keys = [k for k in config.SNAPSHOTS_QUARTERLY if k != config.LEGACY_SNAPSHOT]
    return [k for k in keys if month_index(k) <= month_index(config.CLEAN_THROUGH)]


def change_type(frm: float, to: float) -> str:
    if frm == 50 and to == 40:
        return "50→40"
    if frm == 60 and to in (40, 50):
        return "60→40/50"
    if frm in (70, 80):
        return "70/80→lower"
    if frm >= 90:
        return "90-110→lower"
    return "other"


def classify(u: pd.DataFrame, keys: list[str], first_seen=config.TREAT_FIRST_SEEN,
             min_limit=config.MIN_LIMIT) -> pd.DataFrame:
    L = u[[f"lim_{k}" for k in keys]].astype("float64").to_numpy()
    any_of = lambda prefix: u[[f"{prefix}_{k}" for k in keys]].fillna(True).astype(bool).any(axis=1).to_numpy()
    base = L[:, 0]
    legacy = u["lim_legacy"].astype("float64").to_numpy()
    diffs = np.diff(L, axis=1)
    changed = (diffs != 0) & ~np.isnan(diffs)
    n_changes = changed.sum(axis=1)
    first = np.where(n_changes > 0, changed.argmax(axis=1) + 1, -1)  # index into keys
    step = np.take_along_axis(diffs, np.maximum(first - 1, 0)[:, None], axis=1)[:, 0]

    lo, hi = month_index(first_seen[0]), month_index(first_seen[1])
    seen_month = np.array([month_index(keys[i]) if i >= 0 else -1 for i in first])
    in_window = (seen_month >= lo) & (seen_month <= hi)

    rules = [
        ("limit_below_30_or_missing", np.isnan(base) | (base < min_limit)),
        ("direction_mismatch", any_of("dirmix")),
        ("time_of_day_zone", any_of("tv") | u["tv_legacy"].fillna(False).astype(bool).to_numpy()),
        ("geometry_redrawn", any_of("redrawn")),
        ("no_2021_match", ~u["legacy_matched"].astype(bool).to_numpy() | np.isnan(legacy)),
        ("changed_2021_2023", legacy != base),
        ("missing_later_limit", np.isnan(L).any(axis=1)),
        ("multiple_changes", n_changes > 1),
        ("increase", (n_changes == 1) & (step > 0)),
        ("change_outside_window", (n_changes == 1) & ~in_window),
    ]
    reason = np.full(len(u), "", dtype=object)
    for name, mask in rules:
        reason[(reason == "") & mask] = name
    status = np.where(reason != "", "excluded", np.where(n_changes == 0, "comparison", "treated"))

    t = status == "treated"
    cohort = np.array([keys[i] if ok else None for i, ok in zip(first, t)], dtype=object)
    prev = np.array([keys[i - 1] if ok else None for i, ok in zip(first, t)], dtype=object)
    to = np.where(t, base + step, np.nan)
    scheme = np.full(len(u), None, dtype=object)
    for k in keys:
        sel = cohort == k
        scheme[sel] = [c if isinstance(c, str) else f"unknown-{k}"
                       for c in u.loc[sel, f"cr_{k}"].to_numpy(dtype=object)]
    return pd.DataFrame({
        "unit_id": u["unit_id"].to_numpy(), "status": status, "reason": reason,
        "cohort": pd.array(cohort, dtype="string"), "prev_snapshot": pd.array(prev, dtype="string"),
        "from_limit": np.where(t | (status == "comparison"), base, np.nan), "to_limit": to,
        "change_type": pd.array([change_type(f, x) if ok else None for f, x, ok in zip(base, to, t)], dtype="string"),
        "scheme": pd.array(scheme, dtype="string"),
    })


def all_changes(first: str, last: str) -> dict:
    """Feature-level count of limit changes between two snapshots (context, not the study sample)."""
    a = pd.read_parquet(config.INTERIM / f"sz_attrs_{first}.parquet", columns=["id", "speed_limit"])
    b = pd.read_parquet(config.INTERIM / f"sz_attrs_{last}.parquet", columns=["id", "speed_limit"])
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    ch = m[m.speed_limit_a != m.speed_limit_b]
    return {"from": first, "to": last, "features_first": len(a), "features_last": len(b),
            "ids_kept": len(m), "changed": len(ch),
            "decreases": int((ch.speed_limit_b < ch.speed_limit_a).sum()),
            "increases": int((ch.speed_limit_b > ch.speed_limit_a).sum())}


def main():
    import geopandas as gpd
    u = gpd.read_parquet(config.PROCESSED / "units.parquet")
    keys = analysis_keys()
    tr = classify(u, keys)
    tr = tr.merge(u[["unit_id", "length_m", "lga", "region", "declared", "aadt_2019"]], on="unit_id")
    tr.to_parquet(config.PROCESSED / "treatment.parquet", index=False)

    t = tr[tr.status == "treated"]
    c = tr[tr.status == "comparison"]
    summary = (t.groupby(["change_type", "cohort"]).agg(units=("unit_id", "size"), km=("length_m", "sum"))
               .assign(km=lambda d: d.km / 1000).round(1).reset_index())
    summary.to_csv(config.TABLES / "treated_by_type_cohort.csv", index=False)
    stats = {
        "snapshots_used": keys,
        "status_counts": tr.status.value_counts().to_dict(),
        "excluded_reasons": tr.loc[tr.status == "excluded", "reason"].value_counts().to_dict(),
        "treated_units": len(t), "treated_km": round(t.length_m.sum() / 1000, 1),
        "treated_schemes": int(t.scheme.nunique()),
        "treated_unknown_scheme_units": int(t.scheme.str.startswith("unknown").sum()),
        "treated_by_type": t.change_type.value_counts().to_dict(),
        "treated_by_cohort": t.cohort.value_counts().sort_index().to_dict(),
        "treated_by_region": t.region.value_counts().to_dict(),
        "treated_declared_share": round(float(t.declared.mean()), 3),
        "treated_top_lgas": t.lga.value_counts().head(10).to_dict(),
        "comparison_units": len(c), "comparison_km": round(c.length_m.sum() / 1000),
        "all_changes_context": all_changes(config.BASE_SNAPSHOT, config.SNAPSHOTS_QUARTERLY[-1]),
    }
    save_metrics("treatment", stats)
    print(stats)


if __name__ == "__main__":
    main()
