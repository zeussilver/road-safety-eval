import numpy as np
import pandas as pd
import shapely

from rse.network import unit_keys
from rse.treatment import change_type, classify

KEYS = ["2023_06", "2023_09", "2024_01", "2024_03"]
WINDOW = ("2023_09", "2024_01")


def make_units(rows):
    """rows: list of dicts with limits (list), plus optional overrides."""
    recs = []
    for i, r in enumerate(rows):
        rec = {"unit_id": i, "lim_legacy": r.get("legacy", r["lims"][0]), "tv_legacy": False, "legacy_matched": True}
        for k, v in zip(KEYS, r["lims"]):
            rec[f"lim_{k}"] = v
            rec[f"dirmix_{k}"] = r.get("dirmix", False)
            rec[f"tv_{k}"] = r.get("tv", False)
            rec[f"redrawn_{k}"] = False
            rec[f"cr_{k}"] = r.get("cr") if k != KEYS[0] else None
        recs.append(rec)
    return pd.DataFrame(recs)


def run(rows):
    return classify(make_units(rows), KEYS, first_seen=WINDOW).set_index("unit_id")


def test_unchanged_unit_is_comparison():
    out = run([{"lims": [50, 50, 50, 50]}])
    assert out.loc[0, "status"] == "comparison"
    assert out.loc[0, "from_limit"] == 50


def test_single_decrease_in_window_is_treated_with_cohort():
    out = run([{"lims": [50, 40, 40, 40], "cr": "MOA-1"}])
    r = out.loc[0]
    assert r.status == "treated"
    assert (r.cohort, r.prev_snapshot) == ("2023_09", "2023_06")
    assert (r.from_limit, r.to_limit, r.change_type, r.scheme) == (50, 40, "50→40", "MOA-1")


def test_exclusion_reasons():
    out = run([
        {"lims": [50, 60, 60, 60]},                    # increase
        {"lims": [50, 50, 50, 40]},                    # decrease seen after the window
        {"lims": [60, 50, 40, 40]},                    # two changes
        {"lims": [60, 60, 60, 60], "tv": True},        # school / shop zone
        {"lims": [50, 50, 50, 50], "legacy": 60},      # changed between Nov 2021 and Jun 2023
        {"lims": [50, 50, 50, 50], "dirmix": True},    # directions disagree
        {"lims": [20, 20, 20, 20]},                    # shared zone
    ])
    assert list(out.reason) == ["increase", "change_outside_window", "multiple_changes", "time_of_day_zone",
                                "changed_2021_2023", "direction_mismatch", "limit_below_30_or_missing"]
    assert (out.status == "excluded").all()
    assert out.cohort.isna().all()


def test_missing_scheme_gets_placeholder():
    out = run([{"lims": [80, 80, 60, 60]}])
    assert out.loc[0, "scheme"] == "unknown-2024_01"
    assert out.loc[0, "change_type"] == "70/80→lower"


def test_change_type_bands():
    assert change_type(60, 50) == "60→40/50"
    assert change_type(100, 80) == "90-110→lower"
    assert change_type(40, 30) == "other"


def test_directional_features_collapse_to_one_unit():
    a = shapely.LineString([(0, 0), (100, 0), (100, 50)])
    a_rev = shapely.LineString([(100, 50), (100, 0), (0, 0)])
    multi = shapely.MultiLineString([[(0, 0), (100, 0)], [(100, 0), (100, 50)]])  # same road, stored in parts
    other = shapely.LineString([(0, 10), (100, 10)])
    k = unit_keys(np.array([a, a_rev, multi, other]))
    assert k[0] == k[1] == k[2]
    assert k[3] != k[0]
