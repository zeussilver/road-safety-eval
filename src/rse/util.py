"""Small shared helpers: month arithmetic, metrics file, timing."""
import json
import time
from contextlib import contextmanager

import numpy as np

from rse import config


def month_index(ym: str) -> int:
    """'2024-03' or '2024_03' -> months since year 0 (an integer that subtracts cleanly)."""
    y, m = ym.replace("_", "-").split("-")[:2]
    return int(y) * 12 + int(m) - 1


def month_str(idx: int) -> str:
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def save_metrics(section: str, data: dict) -> None:
    """Replace one top-level section of outputs/metrics.json, keeping the others."""
    path = config.METRICS
    allm = json.loads(path.read_text()) if path.exists() else {}
    allm[section] = data
    path.write_text(json.dumps(allm, indent=2, default=_jsonable, sort_keys=True))


def load_metrics() -> dict:
    return json.loads(config.METRICS.read_text())


@contextmanager
def timed(label: str, sink: dict | None = None):
    t0 = time.perf_counter()
    print(f"[{label}] ...", flush=True)
    yield
    dt = time.perf_counter() - t0
    print(f"[{label}] done in {dt:.1f}s", flush=True)
    if sink is not None:
        sink[label] = round(dt, 1)
