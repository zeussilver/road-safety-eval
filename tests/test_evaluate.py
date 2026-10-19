import numpy as np
import pandas as pd
import pytest

from rse.evaluate import fit_cells, fit_did, fit_naive


def synthetic_stack(true_rr=0.7, trend=0.9, n_units=4000, seed=0):
    """Two strata with different base rates and trends; treated units are selected with higher rates."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, (base, tr_s) in enumerate([(0.6, trend), (1.5, trend * 1.1)]):
        for i in range(n_units):
            treated = int(i < n_units // 4)
            lam = rng.gamma(2.0, base / 2.0) * (1.5 if treated else 1.0)   # heterogeneous units
            yb = rng.poisson(lam * 3)                                       # 36 months
            ya = rng.poisson(lam * 2 * tr_s * (true_rr if treated else 1))  # 24 months
            rows.append({"cohort": "2024_03", "unit_id": s * n_units + i, "treated": treated, "stratum": f"S{s}",
                         "cs": f"2024_03#S{s}", "yb": yb, "ya": ya, "mb": 36, "ma": 24, "km": 1.0,
                         "cluster": f"U{s * n_units + i}"})
    return pd.DataFrame(rows)


def test_conditional_poisson_equals_cell_poisson_and_recovers_effect():
    st = synthetic_stack()
    a, b = fit_did(st), fit_cells(st)
    assert a["rr"] == pytest.approx(b["rr"], rel=1e-6)
    assert a["lo"] < 0.7 < a["hi"]


def test_naive_estimate_absorbs_the_trend():
    st = synthetic_stack(true_rr=1.0, trend=0.8)
    naive, did = fit_naive(st), fit_did(st)
    assert naive["rr"] < 0.9            # looks like a reduction ...
    assert did["lo"] < 1.0 < did["hi"]  # ... that the comparison group explains
