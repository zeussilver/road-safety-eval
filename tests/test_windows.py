from rse.util import month_index, month_str
from rse.windows import cohort_windows, period_of


def ms(w):
    return month_str(w[0]), month_str(w[1])


def test_windows_drop_transition_and_end_at_study_end():
    w = cohort_windows("2023_06", "2023_09", "2025-12", 36)
    assert ms(w.before) == ("2020-07", "2023-06")
    assert ms(w.after) == ("2023-10", "2025-12")
    assert (w.before_months, w.after_months) == (36, 27)


def test_period_assignment_at_boundaries():
    w = cohort_windows("2023_06", "2023_09", "2025-12", 36)
    assert period_of(month_index("2020-06"), w) is None      # before the before window
    assert period_of(month_index("2020-07"), w) == "before"
    assert period_of(month_index("2023-06"), w) == "before"   # last month with the old limit
    for m in ("2023-07", "2023-08", "2023-09"):               # transition: change happened in here
        assert period_of(month_index(m), w) is None
    assert period_of(month_index("2023-10"), w) == "after"
    assert period_of(month_index("2025-12"), w) == "after"
    assert period_of(month_index("2026-01"), w) is None       # incomplete crash data


def test_padding_widens_transition_both_sides():
    w = cohort_windows("2023_06", "2023_09", "2025-12", 36, pad=2)
    assert ms(w.before) == ("2020-05", "2023-04")
    assert w.after[0] == month_index("2023-12")


def test_placebo_never_reaches_real_change():
    real = cohort_windows("2024_03", "2024_06", "2025-12", 36)
    pl = cohort_windows("2024_03", "2024_06", "2025-12", 36, shift=36)
    assert ms(pl.before) == ("2018-04", "2021-03")
    assert pl.after[0] == month_index("2021-07")
    assert pl.after[1] == real.before[1]                     # stops at the last real before month
    assert pl.after[1] < real.after[0]
