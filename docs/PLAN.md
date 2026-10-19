# PLAN — Did lowering speed limits change injury crashes on Victorian roads?

Status: **approved 2026-10-19** (defaults: Yarra kept as a low-power descriptive case study; quarterly snapshots).
Deviations found while building (all documented in the README): the Jan 2024 snapshot on the portal is corrupt,
so Nov 2023 + Feb 2024 are used; Speed Zones lon/lat are GDA94 like the crash data; Empirical Bayes failed its
placebo, so the comparison-group DiD is the main estimate; SEs use the largest of three clusterings.
Target: finished by 2026-10-25 for DTP Road Safety Research & Data Science intern (job 1859004, closes 2026-10-26).

## 1. Question

Did road segments whose posted speed limit was **lowered** between mid-2023 and end-2024 see a change in
police-reported **injury crashes**, compared with similar segments whose limit did **not** change?

This is an observational evaluation. The design can support statements like "after accounting for the
trend on comparable roads, injury crashes on treated segments changed by X% (95% CI a–b)". It cannot
prove the limit change *caused* the change, because limit changes are not randomly assigned and often
come with other works.

## 2. Data, verified on the live portals on 2026-10-19

All CC BY 4.0, from discover.data.vic.gov.au / opendata.transport.vic.gov.au (CKAN API `package_show`).

| Dataset | What is really there | Consequence for the design |
|---|---|---|
| **Victoria road crash data** (`victoria-road-crash-data`, updated 2026-10-12) | `accident.csv` 202,297 crashes (34.7 MB), 2012-01-01 → 2026-02-28. Jan–Feb 2026 are clearly incomplete (138 and 90 crashes vs ~1,100–1,500/month). Severity 1 fatal / 2 serious / 3 other injury (only 4 non-injury rows). `SPEED_ZONE` coded `030…110`, plus `777/888/999` (other / off-road / unknown). `node.csv` has `LATITUDE/LONGITUDE`, `NODE_TYPE` (I = intersection, N = non-intersection), `LGA_NAME`, `DEG_URBAN_NAME`; it has 407,410 rows but these are exact duplicates: 201,003 unique crashes with coordinates (1,294 crashes have no location). | Study period ends **2025-12-31**. Outcomes: all injury crashes (primary), fatal + serious (secondary, small counts). Deduplicate the node table and report that. |
| **Speed Zones** (`speed-zones`, monthly) | 36 snapshots, ~475 MB GeoJSON each: **Nov 2021**, then **Jun 2023 → Sep 2026** monthly with gaps (Dec 2023, Jul 2024, Nov 2025, Dec 2025 missing). From Jun 2023 the feature `id` (UUID) is **stable**: all 470,175 Jun-2023 ids still exist in Sep 2026, 469,138 with identical geometry. **9,977 segments changed limit Jun 2023 → Sep 2026: 9,485 decreases, 492 increases** (e.g. 50→40: 5,353; 60→50: 969; 80→60: 752; 100→80: 646). 99.7% of changed segments carry a `speed_zone_change_request_id` (e.g. `MOA-00002782`). `road_name`, `lgas`, `road_type` are filled only for changed segments (~3%). Features are **directional**: 471,826 features collapse to 289,572 unique geometries (mostly N/S or E/W pairs; only 662 pairs differ by direction). `zone_conditions` flags School / Shop / Seasonal / Side-road-activated zones (the stored limit is the lowest at any time of day). Nov 2021 has a different schema (`SIGN_SPEED`, `Variable_C` school codes, no UUID). | **Snapshot diff works; the crash-record fallback is not needed for detection.** Change-request id groups segments into *schemes*. Collapse directional pairs into one carriageway unit. Exclude time-of-day-only changes (school/shop zones). LGA and road class must be derived spatially. |
| **Yarra 30 km/h expansion** (Fitzroy + Collingwood, started 2024-05-09) | **Not recorded in Speed Zones.** In a Fitzroy/Collingwood bounding box the Jun 2023 and Sep 2026 snapshots both show the same 136 segments at 30 km/h (the older 2018 trial area) and ~720–750 at 40. But police crash records do pick it up: crashes coded 30 km/h in postcodes 3065/3066 went 3 (2023) → 16 (2024) → 20 (2025). | Case study defines treatment from the **council-published area** (all streets in Fitzroy and Collingwood except state arterials), not from snapshots. Crash-record `SPEED_ZONE` serves as a fidelity check. Local-street injury crashes there are ~45–65 a year, so power is low and the case study is descriptive. |
| **Historical AADT 2001–2019** | Yearly GeoJSON (2019: 32 MB), **declared (arterial) roads only**, per direction, with `Average Annual Daily Traffic Volume`, heavy-vehicle %, `Local Government Area`. Nothing after 2019. | Use 2019 to (a) flag declared vs local roads and (b) give a volume band for matching. It **cannot** be time-varying exposure for 2020–2025, so exposure = segment length × time. |
| **Vicmap LGA / locality polygons** | Open WFS at `opendata.maps.vic.gov.au/geoserver/wfs` (`ad_lga_area_polygon`, `ad_locality_area_polygon`) returns GeoJSON. | Assign LGA to segments; define the Fitzroy/Collingwood area. |

Sources: [Speed Zones](https://discover.data.vic.gov.au/dataset/speed-zones), [Victoria road crash data](https://discover.data.vic.gov.au/dataset/victoria-road-crash-data), [Historical AADT](https://discover.data.vic.gov.au/dataset/historical-annual-average-daily-traffic-volume), Vicmap Admin LGA/locality polygons. Yarra trial dates: [Bicycle Network, 2024-05-09](https://bicyclenetwork.com.au/newsroom/2024/05/09/30km-h-speed-limits-expand-across-fitzroy-and-collingwood/), [Your Say Yarra](https://yoursayyarra.com.au/30km).

**No connected-vehicle data is public.** None will be simulated. The README will say how the pipeline would
take CV speed data (operating speed before/after per segment) as future work, and nothing more.

## 3. Pipeline (one command: `make all`)

```
download → ingest → network → treatment → snap → evaluate → yarra → report
```

1. **download**: resolve resource URLs from the CKAN API (not hard-coded), skip files already present with matching size.
   Default snapshot set is quarterly: Nov 2021, Jun/Sep/Nov 2023, Feb/Mar/Jun/Sep/Dec 2024, Mar/Jun/Oct 2025, Jan/Sep 2026 (14 files, ~6.7 GB, about 30–40 min).
   `SNAPSHOTS=monthly` uses every snapshot (~17 GB).
2. **ingest** (DuckDB + pyogrio → Parquet/GeoParquet): crash tables to Parquet; each snapshot to a slim attribute
   Parquet (`id, speed_limit, zone_conditions, change_request_id`) plus one GeoParquet of geometry. Report row counts,
   raw vs Parquet sizes and timings. This is the "large datasets" evidence.
3. **network** (GeoPandas, EPSG:7855 GDA2020 / MGA55):
   collapse directional features into carriageway units; assign LGA (centroid in Vicmap polygon); flag declared
   road and 2019 AADT band (overlap with the AADT network within 15 m); match Nov 2021 features to units by
   geometry so that units changed between Nov 2021 and Jun 2023 can be excluded.
4. **treatment**: diff consecutive snapshots per unit. A unit is **treated** if it has exactly one limit decrease,
   both sides "Normal Operation" (time-of-day zones excluded), first seen between the Sep 2023 and Dec 2024 snapshots,
   and no further change up to Jan 2026. The implementation date is only known to lie between two snapshots. That
   interval is the **transition period** and is dropped from both before and after. A unit is a **clean comparison**
   if its limit never changed in any snapshot (Nov 2021 → Jan 2026).
5. **snap**: each crash goes to the nearest carriageway unit within 30 m (`sjoin_nearest`). QA reported:
   unmatched %, distance distribution, intersection vs midblock share, ties at intersections. A validation check:
   agreement between the crash record's `SPEED_ZONE` and the snapped unit's limit at that date. Sensitivity runs:
   15 m / 50 m buffers and midblock-only crashes.
6. **evaluate** (stacked comparison-group DiD):
   - Each treatment interval is a *cohort*. Its before window is 36 months ending at the interval start. Its after window
     runs from the interval end to 2025-12-31, which gives 12–27 months.
   - Comparison units are clean units in the same stratum: **original limit × declared/local × LGA**. When a stratum has
     too few comparison units, LGA is coarsened to metro/regional. Matching balance is reported (length, pre-period
     crash rate per km).
   - **Estimator**: a Poisson model with unit fixed effects. With two periods per unit this is estimated exactly as a
     binomial GLM on the share of each unit's crashes falling in the after period, with offset log(after/before
     exposure) and cohort×stratum effects. The treated coefficient gives the DiD rate ratio. Cluster-robust SEs by scheme
     (treated) / LGA (comparison). A cell-level Poisson GLM is a cross-check, and negative binomial is a sensitivity run.
   - Shown side by side: **(a) naive before–after** on treated units only, **(b) comparison-group DiD**, and
     **(c) Empirical Bayes before–after** (NB safety performance function fitted on comparison units; this is the
     stretch goal that targets regression to the mean directly). Also reported: pre-period crash rate of treated vs
     comparison (were treated roads picked for high crashes?) and a **placebo** test (fake treatment date three years
     earlier; the estimate should be ~1).
   - Results pooled and by change type where counts allow: urban local 50→40, urban arterial 60→50/40, rural 100/80→lower.
7. **yarra** case study: treated area = units inside Fitzroy + Collingwood locality polygons, non-declared, limit ≤ 50,
   not already in the 2018 30 km/h area. Comparison = 40/50 non-declared units in nearby inner suburbs that never changed.
   Before Jan 2021 → Apr 2024, after 2024-05-09 → 2025-12-31. Counts and rate ratio are reported with exact Poisson CIs,
   with a plain statement that power is low. The share of crashes coded 30 km/h is reported as a fidelity check.
8. **report**: `outputs/metrics.json` + CSV tables + PNG figures; folium map `outputs/map.html` (treated units by
   change type, comparison units, crashes before/after); `docs/BRIEF.md`; README.

**Tests** (pytest, small synthetic fixtures, no downloads): crash snapping (nearest within buffer, outside buffer
unmatched, tie handling, CRS), window assignment (transition dropped, boundary dates, exposure months), treatment
detection (single decrease, increase ignored, school-zone change excluded, double change excluded, directional
collapse).

## 4. Honesty rules built into the outputs

- Every figure in README/BRIEF is read from `outputs/metrics.json` or a CSV that the pipeline writes. No hand-typed numbers.
- Observational wording only: "associated with", "after accounting for comparison-road trends".
- Limitations section: implementation date known only to within a snapshot interval; co-interventions (traffic calming,
  road works) not observable; snapping error at intersections; recent crashes are provisional; AADT stops in 2019;
  COVID-era volumes in some before windows (DiD absorbs common shocks only if they hit both groups alike); EPSG:7855 is
  ~0.3% scale-distorted in far-west Victoria (irrelevant at 30 m buffers).

## 5. Layout

```
road-safety-eval/
  Makefile  requirements.txt  README.md
  src/rse/{config,download,ingest,network,treatment,snap,evaluate,yarra,report}.py
  tests/{test_snap,test_windows,test_treatment}.py
  data/{raw,interim,processed}/   (git-ignored)
  outputs/{metrics.json,tables/,figures/,map.html}
  docs/{PLAN.md,BRIEF.md}
```

Python 3.12 venv via `uv`; geopandas, pyogrio, shapely 2, pyproj, duckdb, pyarrow, statsmodels, folium, matplotlib, pytest.

## 6. Schedule (lean, ~4 working days)

| Day | Work |
|---|---|
| 10-19/20 | download + ingest + network + treatment detection |
| 10-21 | snapping + QA, evaluation (naive / DiD / placebo) |
| 10-22 | EB, Yarra case study, map, tests |
| 10-23 | README, BRIEF, resume bullet candidates |
| 10-24/25 | buffer |

## 7. Decisions needed

1. Approve this plan (required before code).
2. Yarra case study: snapshots do not show the expansion, so treatment comes from the council's published area definition
   and is checked with crash records. Recommended: keep it as a clearly labelled low-power descriptive case study.
3. Snapshot resolution: quarterly by default (recommended: ~30 min download, up to ~3-month transition periods dropped)
   vs monthly (~17 GB, sharper dating).

Not doing without explicit OK: git commit, GitHub repo, push.
