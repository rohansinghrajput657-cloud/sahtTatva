"""Step 5 — IMD extreme-weather alerts on blended forecasts.

Thresholds (IMD):
  * Heavy rainfall: >= 64.5 mm in 24 h (hourly values accumulated)
  * Heatwave: Tmax >= 45 C, or >= 4.5 C above normal
      (normal = ERA5 30-day mean daily max per city)
  * High wind: >= 60 km/h (hourly)

compute_alerts() is reused by the Streamlit app for searched locations.
main() runs on the 12-city blended snapshots -> outputs/alerts.csv
"""
import numpy as np
import pandas as pd

from config import DATA_DIR, OUT_DIR

RAIN_24H = 64.5
HEAT_ABS = 45.0
HEAT_ANOM = 4.5
WIND = 60.0


def compute_alerts(blended, normals):
    """blended: df with time, city, lat, lon, variable, blended.
    normals: dict city -> ERA5 30-day mean daily max temp (C).
    Returns alerts dataframe (may be empty).
    """
    alerts = []
    for city, g in blended.groupby("city"):
        lat = float(g["lat"].iloc[0])
        lon = float(g["lon"].iloc[0])
        normal_max = normals.get(city)

        rain = g[g["variable"] == "precipitation"].sort_values("time")
        if len(rain):
            r24 = (rain.set_index("time")["blended"]
                   .rolling("24h", min_periods=24).sum())
            for t, v in r24[r24 >= RAIN_24H].items():
                alerts.append({"city": city, "lat": lat, "lon": lon,
                               "window_start": t - pd.Timedelta(hours=24),
                               "window_end": t, "type": "Heavy rainfall",
                               "severity": "High" if v >= 115.5 else "Moderate",
                               "value": round(float(v), 1),
                               "threshold": RAIN_24H, "unit": "mm/24h"})

        temp = g[g["variable"] == "temperature_2m"].copy()
        if len(temp):
            temp["date"] = temp["time"].dt.date
            daily_max = temp.groupby("date")["blended"].max()
            for d, vmax in daily_max.items():
                thresh = HEAT_ABS
                if normal_max is not None:
                    thresh = min(HEAT_ABS, normal_max + HEAT_ANOM)
                if vmax >= thresh:
                    alerts.append({"city": city, "lat": lat, "lon": lon,
                                   "window_start": pd.Timestamp(d),
                                   "window_end": pd.Timestamp(d) + pd.Timedelta(days=1),
                                   "type": "Heatwave",
                                   "severity": "High" if vmax >= 47 else "Moderate",
                                   "value": round(float(vmax), 1),
                                   "threshold": round(float(thresh), 1),
                                   "unit": "°C (daily max)"})

        wind = g[g["variable"] == "wind_speed_10m"].sort_values("time")
        if len(wind):
            for r in wind[wind["blended"] >= WIND].itertuples():
                alerts.append({"city": city, "lat": lat, "lon": lon,
                               "window_start": r.time, "window_end": r.time,
                               "type": "High wind",
                               "severity": "High" if r.blended >= 80 else "Moderate",
                               "value": round(float(r.blended), 1),
                               "threshold": WIND, "unit": "km/h"})
    cols = ["city", "lat", "lon", "window_start", "window_end", "type",
            "severity", "value", "threshold", "unit"]
    return pd.DataFrame(alerts, columns=cols)


def era5_normals():
    """Per-city ERA5 30-day mean daily max temperature (the 'normal')."""
    era = pd.read_csv(DATA_DIR / "era5_tidy.csv", parse_dates=["time"])
    t = era[era["variable"] == "temperature_2m"].copy()
    t["date"] = t["time"].dt.date
    daily_max = t.groupby(["city", "date"])["value"].max()
    return daily_max.groupby("city").mean().to_dict()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = pd.read_csv(OUT_DIR / "snapshots_blended.csv", parse_dates=["time"])
    normals = era5_normals()
    alerts = compute_alerts(snap, normals)
    alerts.to_csv(OUT_DIR / "alerts.csv", index=False)
    print(f"Saved outputs/alerts.csv: {len(alerts)} alerts")
    if len(alerts):
        print(alerts.groupby(["city", "type"]).size().to_string())
    return alerts


if __name__ == "__main__":
    main()
