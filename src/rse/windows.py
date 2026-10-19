"""Before / after windows in whole months.

A change first seen in snapshot S_k (old limit still in S_{k-1}) happened at some point in the
months after S_{k-1} up to and including S_k. Those months are the transition period and are
dropped. `pad` widens the transition on both sides (sensitivity to snapshot timing lag).
"""
from dataclasses import dataclass

from rse.util import month_index


@dataclass(frozen=True)
class Windows:
    before: tuple[int, int]   # inclusive month indices
    after: tuple[int, int]

    @property
    def before_months(self) -> int:
        return self.before[1] - self.before[0] + 1

    @property
    def after_months(self) -> int:
        return max(0, self.after[1] - self.after[0] + 1)


def cohort_windows(prev_snapshot: str, first_seen: str, study_end: str, before_months: int,
                   pad: int = 0, shift: int = 0) -> Windows:
    """`shift` moves the change earlier by that many months (placebo). The placebo after window
    then ends at the last real before month, so it never overlaps the real transition."""
    last_old = month_index(prev_snapshot) - pad - shift
    first_new = month_index(first_seen) + pad - shift
    end = month_index(study_end) if shift == 0 else month_index(prev_snapshot) - pad
    return Windows(before=(last_old - before_months + 1, last_old), after=(first_new + 1, end))


def period_of(month: int, w: Windows) -> str | None:
    if w.before[0] <= month <= w.before[1]:
        return "before"
    if w.after[0] <= month <= w.after[1]:
        return "after"
    return None
