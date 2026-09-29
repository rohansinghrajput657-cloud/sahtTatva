"""Shared constants for sahtTatva v2 — UP-wide Hybrid AI-NWP blending prototype.

SIH 2026 PS SIH26081: "Hybrid AI-NWP Multi-Model Forecast Blending System"
Ministry of Earth Sciences (MoES), Software category.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

APP_NAME = "sahtTatva"

# 12 representative training cities spread across Uttar Pradesh.
# (name -> (lat, lon)); coords are city centres used for API queries.
CITIES = {
    "Lucknow": (26.85, 80.95),
    "Varanasi": (25.32, 82.99),   # pilot city from v1
    "Prayagraj": (25.45, 81.85),
    "Kanpur": (26.45, 80.35),
    "Agra": (27.18, 78.02),
    "Meerut": (28.98, 77.71),
    "Gorakhpur": (26.76, 83.37),
    "Bareilly": (28.37, 79.43),
    "Jhansi": (25.45, 78.57),
    "Ayodhya": (26.80, 82.20),
    "Mirzapur": (25.15, 82.57),
    "Saharanpur": (29.97, 77.55),
}

# UP bounding box for validating searched locations (generous margins).
UP_BBOX = {"lat_min": 23.5, "lat_max": 31.0, "lon_min": 77.0, "lon_max": 85.0}

# Model identifiers as named by Open-Meteo. These stand in for the full
# operational NWP suite; the blending framework is model-agnostic.
#
# Data-quality notes (verified empirically, Sep 2026):
#  * ecmwf_ifs04 returns all-null values whenever past_days is requested.
#  * ecmwf_ifs "past analyses" are bit-identical to ERA5 (backfilled), so a
#    bias corrector trained on them is circular (raw RMSE = 0).
#  * gem_seamless (Environment Canada GEM) has genuine 30-day past analyses.
MODELS = ["gfs_seamless", "icon_seamless", "gem_seamless"]
MODEL_LABELS = {
    "gfs_seamless": "GFS (NWP)",
    "icon_seamless": "ICON (NWP)",
    "gem_seamless": "GEM (NWP)",
}

VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m"]
VAR_LABELS = {
    "temperature_2m": "Temperature 2m (°C)",
    "precipitation": "Precipitation (mm/h)",
    "wind_speed_10m": "Wind speed 10m (km/h)",
}

TIMEZONE = "Asia/Kolkata"
FORECAST_DAYS = 7      # live forecast horizon
ARCHIVE_LAG_DAYS = 6   # ERA5 archive lags ~5 days; stay safe with 6
TRAIN_WINDOW_DAYS = 30  # bias-correction training window (analysis vs ERA5)

LEAD_BANDS = {
    "0-24h": (0, 24),
    "24-72h": (24, 72),
    "72-168h": (72, 169),
}

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
