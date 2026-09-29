"""Step 1 — Ingest real data for all 12 UP training cities (Open-Meteo, no key).

For each city:
  * Forecast API: past_days=30 of each model's own analysis + 7-day hourly
    forecasts (temperature_2m, precipitation, wind_speed_10m) for
    GFS / ICON / ECMWF-IFS.
  * Archive API: ERA5 reanalysis for the same 30-day window as ground truth.

Saves raw JSON + tidy CSVs (with a `city` column) into data/.
All numbers come from the APIs; missing values stay NaN and are counted.
"""
import json
import time
from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import (
    ARCHIVE_LAG_DAYS, ARCHIVE_URL, CITIES, DATA_DIR, FORECAST_DAYS,
    FORECAST_URL, MODELS, TIMEZONE, TRAIN_WINDOW_DAYS, VARIABLES,
)

IST = ZoneInfo(TIMEZONE)


def _get(url, params, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - network flakiness, retry all
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{retries}] {e} -> waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"API failed after {retries} retries: {url}")


def _cities():
    return list(CITIES.items())


def fetch_forecast(past_days):
    """One API call per model (multi-model merges are unreliable for
    ecmwf_ifs04) — returns {model: raw_json}."""
    cities = _cities()
    out = {}
    for model in MODELS:
        params = {
            "latitude": ",".join(str(lat) for _, (lat, _) in cities),
            "longitude": ",".join(str(lon) for _, (_, lon) in cities),
            "hourly": ",".join(VARIABLES),
            "models": model,
            "past_days": past_days,
            "forecast_days": FORECAST_DAYS,
            "timezone": TIMEZONE,
        }
        print(f"  fetching {model} for {len(cities)} cities "
              f"(past {past_days}d + next {FORECAST_DAYS}d) ...")
        out[model] = _get(FORECAST_URL, params)
        time.sleep(1)  # be polite to the free API
    return out


def fetch_era5(start_date, end_date):
    cities = _cities()
    params = {
        "latitude": ",".join(str(lat) for _, (lat, _) in cities),
        "longitude": ",".join(str(lon) for _, (_, lon) in cities),
        "hourly": ",".join(VARIABLES),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": TIMEZONE,
    }
    print(f"Fetching ERA5 archive {start_date} -> {end_date} ...")
    return _get(ARCHIVE_URL, params)


def tidy_forecast(raw_by_model):
    """Flatten per-model multi-location JSON -> long dataframe.

    Columns: city, time, lat, lon, variable, model, value, kind
    kind = 'analysis' (past, model's own analysis) or 'forecast' (future).
    Handles both suffixed (var_model) and plain (var) hourly keys.
    """
    rows = []
    cities = _cities()
    for model, raw in raw_by_model.items():
        locs = raw if isinstance(raw, list) else [raw]
        for i, loc in enumerate(locs):
            city = cities[i][0] if i < len(cities) else f"loc{i}"
            lat, lon = round(loc["latitude"], 4), round(loc["longitude"], 4)
            hourly = loc["hourly"]
            # API returns local-time strings for the requested timezone
            times = pd.to_datetime(hourly["time"]).tz_localize(IST)
            for var in VARIABLES:
                key = f"{var}_{model}" if f"{var}_{model}" in hourly else var
                vals = hourly.get(key)
                if vals is None:
                    print(f"  WARNING: missing key {key} @ {city}")
                    continue
                for t, v in zip(times, vals):
                    rows.append({"city": city, "time": t, "lat": lat,
                                 "lon": lon, "variable": var,
                                 "model": model, "value": v})
    df = pd.DataFrame(rows)
    today = pd.Timestamp.now(tz=IST).normalize()
    df["kind"] = (df["time"] >= today).map({True: "forecast", False: "analysis"})
    return df


def tidy_era5(raw):
    rows = []
    locs = raw if isinstance(raw, list) else [raw]
    cities = _cities()
    for i, loc in enumerate(locs):
        city = cities[i][0] if i < len(cities) else f"loc{i}"
        lat, lon = round(loc["latitude"], 4), round(loc["longitude"], 4)
        hourly = loc["hourly"]
        times = pd.to_datetime(hourly["time"]).tz_localize(IST)
        for var in VARIABLES:
            vals = hourly.get(var)
            if vals is None:
                print(f"  WARNING: missing ERA5 key {var} @ {city}")
                continue
            for t, v in zip(times, vals):
                rows.append({"city": city, "time": t, "lat": lat, "lon": lon,
                             "variable": var, "value": v})
    return pd.DataFrame(rows)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz=IST).normalize()

    era_end = (today - timedelta(days=ARCHIVE_LAG_DAYS)).date()
    era_start = era_end - timedelta(days=TRAIN_WINDOW_DAYS - 1)
    past_days = (today.date() - era_start).days + 1

    forecast_raw = fetch_forecast(past_days)
    era5_raw = fetch_era5(era_start, era_end)

    with open(DATA_DIR / "forecast_raw.json", "w") as f:
        json.dump(forecast_raw, f)
    with open(DATA_DIR / "era5_raw.json", "w") as f:
        json.dump(era5_raw, f)

    fc = tidy_forecast(forecast_raw)
    era = tidy_era5(era5_raw)
    fc.to_csv(DATA_DIR / "forecast_tidy.csv", index=False)
    era.to_csv(DATA_DIR / "era5_tidy.csv", index=False)

    n_missing = int(fc["value"].isna().sum())
    print(f"Saved: {len(fc)} forecast rows "
          f"({(fc['kind'] == 'forecast').sum()} forecast / "
          f"{(fc['kind'] == 'analysis').sum()} analysis), "
          f"{len(era)} ERA5 rows, NaN forecast values: {n_missing}")
    print(f"ERA5 window: {era['time'].min()} -> {era['time'].max()}")
    print(f"Forecast window: {fc['time'].min()} -> {fc['time'].max()}")
    return fc, era


if __name__ == "__main__":
    main()
