"""Download raw data. Resource URLs are resolved from the CKAN API, not hard-coded.

Files already on disk with the server's Content-Length are skipped, so re-runs are cheap.
Set SNAPSHOTS=monthly to fetch every Speed Zones snapshot instead of the quarterly set.
"""
import http.client
import json
import os
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from rse import config

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def resources(package: str) -> dict[str, str]:
    """Map resource title -> download URL for a CKAN package."""
    res = _get_json(config.CKAN + package)["result"]["resources"]
    return {r["name"].strip(): r["url"] for r in res}


def has_nul(path) -> bool:
    """A size-correct file can still be zero-filled (seen once on a snapshot); text data never has NUL."""
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            if b"\x00" in chunk:
                return True
    return False


def fetch(url: str, dest, check_size: bool = True) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and has_nul(dest):
        print(f"corrupt {dest.name}: NUL bytes, downloading again", flush=True)
        dest.unlink()
    if dest.exists():
        if not check_size:
            return f"skip {dest.name}"
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=120) as r:
            size = int(r.headers.get("Content-Length") or -1)
        if size == dest.stat().st_size:
            return f"skip {dest.name} ({size / 1e6:.0f} MB)"
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        # Resume a partial file when the server reports a length (CKAN files); WFS output restarts.
        have = part.stat().st_size if part.exists() and check_size else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=600) as r, open(part, "ab" if r.status == 206 else "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            if has_nul(part):
                part.unlink()
                raise OSError("downloaded file contains NUL bytes")
            part.rename(dest)
            return f"got  {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)"
        except (OSError, http.client.HTTPException) as e:
            print(f"retry {attempt + 1} {dest.name}: {e!r}", flush=True)
    raise SystemExit(f"failed to download {url}")


def snapshot_key(title: str) -> str | None:
    """'Speed Zones March 2024' -> '2024_03'."""
    parts = title.lower().split()
    if len(parts) != 4 or parts[:2] != ["speed", "zones"] or parts[2] not in MONTHS:
        return None
    return f"{parts[3]}_{MONTHS.index(parts[2]) + 1:02d}"


def plan_downloads() -> list[tuple[str, object, bool]]:
    jobs = []
    crash = resources(config.CRASH_PACKAGE)
    for key, title in config.CRASH_RESOURCES.items():
        jobs.append((crash[title], config.RAW / "crash" / f"{key}.csv", True))

    sz = {snapshot_key(t): u for t, u in resources(config.SZ_PACKAGE).items() if snapshot_key(t)}
    wanted = sorted(sz) if os.environ.get("SNAPSHOTS") == "monthly" else config.SNAPSHOTS_QUARTERLY
    missing = [k for k in wanted if k not in sz]
    if missing:
        raise SystemExit(f"snapshots not on the portal: {missing}")
    for k in wanted:
        jobs.append((sz[k], config.RAW / "speed_zones" / f"speed_zones_{k}.geojson", True))

    aadt = resources(config.AADT_PACKAGE)
    jobs.append((aadt[config.AADT_RESOURCE], config.RAW / "aadt" / "aadt_2019.geojson", True))

    jobs.append((config.WFS + config.LGA_LAYER + "&propertyName=name,state,geom",
                 config.RAW / "vicmap" / "lga.geojson", False))
    names = config.YARRA_TREATED_LOCALITIES + config.YARRA_COMPARISON_LOCALITIES
    cql = "name IN (" + ",".join(f"'{n}'" for n in names) + ")"
    jobs.append((config.WFS + config.LOCALITY_LAYER + "&CQL_FILTER=" + urllib.parse.quote(cql),
                 config.RAW / "vicmap" / "localities.geojson", False))
    return jobs


def main():
    jobs = plan_downloads()
    print(f"{len(jobs)} files")
    with ThreadPoolExecutor(max_workers=5) as ex:
        for msg in ex.map(lambda j: fetch(*j), jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
