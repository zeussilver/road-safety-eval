"""Paths, data sources and study-design constants. Everything tunable lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
METRICS = OUTPUTS / "metrics.json"

CRS = 7855            # GDA2020 / MGA zone 55 (metres)
CRS_CRASH = 4283      # crash LATITUDE/LONGITUDE: GDA94 (checked against node.csv VicGrid94 metres in ingest)
CRS_SZ = 4283         # 2023+ Speed Zones GeoJSON lon/lat: also GDA94, not the WGS84 the GeoJSON spec implies
                      # (as GDA2020 every crash sits ~1.6 m off its road; see snap QA "datum_check")
CRS_SZ_2021 = 3111    # Nov 2021 Speed Zones GeoJSON carries VicGrid94 metres despite no CRS tag

# --- sources (resolved through the CKAN API at download time) ---
CKAN = "https://discover.data.vic.gov.au/api/3/action/package_show?id="
CRASH_PACKAGE = "victoria-road-crash-data"
CRASH_RESOURCES = {"accident": "Accident", "node": "Node", "accident_location": "Accident Location"}
SZ_PACKAGE = "speed-zones"
AADT_PACKAGE = "historical-annual-average-daily-traffic-volume"
AADT_RESOURCE = "Annual Average Daily Traffic Volume - 2019"
WFS = ("https://opendata.maps.vic.gov.au/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature"
       "&outputFormat=application/json&srsName=EPSG:4326&typeNames=open-data-platform:")
LGA_LAYER = "ad_lga_area_polygon"
LOCALITY_LAYER = "ad_locality_area_polygon"

# Snapshot keys are YYYY_MM of the month named in the resource title. A snapshot is
# taken to describe the network at the end of that month. Jan 2024 is skipped: the file on the
# portal was mostly zero bytes on four downloads (2026-10-19); Nov 2023 + Feb 2024 bracket it.
SNAPSHOTS_QUARTERLY = ["2021_11", "2023_06", "2023_09", "2023_11", "2024_02", "2024_03", "2024_06",
                       "2024_09", "2024_12", "2025_03", "2025_06", "2025_10", "2026_01", "2026_09"]
BASE_SNAPSHOT = "2023_06"          # first snapshot with stable feature ids
LEGACY_SNAPSHOT = "2021_11"        # different schema; used only to drop units changed before the base
CLEAN_THROUGH = "2026_01"          # comparison units must be unchanged up to here
TREAT_FIRST_SEEN = ("2023_09", "2024_12")  # a decrease first visible in these snapshots is a treatment

# --- study design ---
STUDY_END = "2025-12"              # last complete month of crash data (Jan-Feb 2026 are partial)
BEFORE_MONTHS = 36
MIN_LIMIT = 30                     # below this: shared zones / car parks, not part of the study
SNAP_BUFFER_M = 30
DECLARED_BUFFER_M = 15
LEGACY_MATCH_M = 5
MIN_COMPARISON_UNITS = 30          # per stratum before falling back from LGA to region
PLACEBO_SHIFT_MONTHS = 36

METRO_LGAS = {
    "BANYULE", "BAYSIDE", "BOROONDARA", "BRIMBANK", "CARDINIA", "CASEY", "DAREBIN", "FRANKSTON",
    "GLEN EIRA", "GREATER DANDENONG", "HOBSONS BAY", "HUME", "KINGSTON", "KNOX", "MANNINGHAM",
    "MARIBYRNONG", "MAROONDAH", "MELBOURNE", "MELTON", "MERRI-BEK", "MONASH", "MOONEE VALLEY",
    "MORNINGTON PENINSULA", "NILLUMBIK", "PORT PHILLIP", "STONNINGTON", "WHITEHORSE", "WHITTLESEA",
    "WYNDHAM", "YARRA", "YARRA RANGES",
}

# --- Yarra 30 km/h case study (council-published scope, not visible in Speed Zones) ---
YARRA_START = "2024-05-09"
YARRA_TREATED_LOCALITIES = ["FITZROY", "COLLINGWOOD"]
YARRA_COMPARISON_LOCALITIES = ["CARLTON", "CARLTON NORTH", "FITZROY NORTH", "CLIFTON HILL", "ABBOTSFORD",
                               "RICHMOND", "EAST MELBOURNE", "BRUNSWICK EAST", "NORTHCOTE"]
# The council lists exactly these five as state-managed and outside the trial. Other declared roads
# in the area (Smith, Brunswick, Gertrude St) were in it: their post-May-2024 crashes are coded 30.
YARRA_EXCLUDED_ROADS = ["JOHNSTON STREET", "NICHOLSON STREET", "HODDLE STREET", "ALEXANDRA PARADE", "VICTORIA PARADE"]
YARRA_BEFORE = ("2021-01", "2024-04")
YARRA_AFTER = ("2024-06", STUDY_END)  # May 2024 is the transition month


def snapshot_month(key: str) -> str:
    """'2024_03' -> '2024-03'."""
    return key.replace("_", "-")


for _p in (RAW, INTERIM, PROCESSED, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
