import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from rse import config
from rse.snap import limit_at, snap
from rse.util import month_index

SNAP_MONTHS = np.array([month_index("2023-06"), month_index("2024-03")])


@pytest.fixture
def units():
    # unit 10: east-west road (50 then 40 km/h); unit 20: north-south road (60) meeting it at (50, 0)
    geoms = [shapely.LineString([(0, 0), (100, 0)]), shapely.LineString([(50, 0), (50, 100)]),
             shapely.LineString([(1000, 1000), (1100, 1000)])]
    return gpd.GeoDataFrame({"unit_id": [10, 20, 30]}, geometry=geoms, crs=config.CRS)


LIM = np.array([[50.0, 40.0], [60.0, 60.0], [80.0, 80.0]])


def crashes(rows):
    df = pd.DataFrame(rows, columns=["accident_no", "x", "y", "month", "crash_speed_zone"])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=config.CRS)


def test_nearest_within_buffer_and_unmatched_beyond(units):
    c = crashes([("A", 20, 5, "2024-06", 40), ("D", 10, 45, "2024-06", 50)])
    s = snap(c, units, 30, LIM, SNAP_MONTHS).set_index("accident_no")
    assert s.loc["A", "unit_id"] == 10 and s.loc["A", "tie_rule"] == "single"
    assert s.loc["A", "dist_m"] == pytest.approx(5)
    assert pd.isna(s.loc["D", "unit_id"]) and s.loc["D", "tie_rule"] == "unmatched"


def test_intersection_near_tie_uses_recorded_limit(units):
    # 0.3 m from unit 10, 0.5 m from unit 20: a near-tie at the intersection
    c = crashes([("B", 50.5, 0.3, "2024-06", 60), ("C", 50.5, 0.3, "2024-06", None)])
    s = snap(c, units, 30, LIM, SNAP_MONTHS).set_index("accident_no")
    assert s.loc["B", "unit_id"] == 20 and s.loc["B", "tie_rule"] == "limit_match"
    assert s.loc["C", "unit_id"] == 10 and s.loc["C", "tie_rule"] == "nearest"
    assert s.loc["B", "n_near"] == 2


def test_far_candidate_is_not_a_tie(units):
    # 1 m from unit 10, 10 m from unit 20: recorded 60 must not pull it onto unit 20
    c = crashes([("E", 40, 1, "2024-06", 60)])
    s = snap(c, units, 30, LIM, SNAP_MONTHS).set_index("accident_no")
    assert s.loc["E", "unit_id"] == 10 and s.loc["E", "tie_rule"] == "single"


def test_limit_at_crash_month():
    rows = np.array([0, 0, 0])
    months = np.array([month_index("2019-01"), month_index("2023-12"), month_index("2024-03")])
    assert list(limit_at(rows, months, LIM, SNAP_MONTHS)) == [50, 50, 40]


def test_crs_mismatch_is_rejected(units):
    c = crashes([("A", 20, 5, "2024-06", 40)]).set_crs(4326, allow_override=True)
    with pytest.raises(ValueError):
        snap(c, units, 30, LIM, SNAP_MONTHS)
