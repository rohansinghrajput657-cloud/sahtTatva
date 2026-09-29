"""Step 6 — Live 7-day blended snapshots for the 12 training cities.

One forecast-API call for all cities -> bias-correct with the pooled
global-UP correctors -> blend with global weights.
Feeds the UP overview map in the app and the alerts step.

Output: outputs/snapshots_blended.csv
"""
import json
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import (CITIES, DATA_DIR, FORECAST_DAYS, FORECAST_URL, MODELS,
                    OUT_DIR, TIMEZONE, VARIABLES)
from blend import correct_and_blend, load_correctors

IST = ZoneInfo(TIMEZONE)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cities = list(CITIES.items())
    raws = {}
    for model in MODELS:
        params = {
            "latitude": ",".join(str(lat) for _, (lat, _) in cities),
            "longitude": ",".join(str(lon) for _, (_, lon) in cities),
            "hourly": ",".join(VARIABLES),
            "models": model,
            "forecast_days": FORECAST_DAYS,
            "timezone": TIMEZONE,
        }
        print(f"  fetching live {model} ...")
        for attempt in range(4):
            try:
                r = requests.get(FORECAST_URL, params=params, timeout=120)
                r.raise_for_status()
                raws[model] = r.json()
                break
            except Exception as e:  # noqa: BLE001
                print(f"    [retry {attempt + 1}/4] {e}")
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Snapshot forecast fetch failed for {model}")
        time.sleep(1)

    rows = []
    for model, raw in raws.items():
        locs = raw if isinstance(raw, list) else [raw]
        for i, loc in enumerate(locs):
            city = cities[i][0] if i < len(cities) else f"loc{i}"
            lat, lon = round(loc["latitude"], 4), round(loc["longitude"], 4)
            hourly = loc["hourly"]
            times = pd.to_datetime(hourly["time"]).tz_localize(IST)
            for var in VARIABLES:
                key = f"{var}_{model}" if f"{var}_{model}" in hourly else var
                vals = hourly.get(key) or []
                for t, v in zip(times, vals):
                    rows.append({"city": city, "time": t, "lat": lat,
                                 "lon": lon, "variable": var,
                                 "model": model, "value": v})
    live = pd.DataFrame(rows)

    correctors = load_correctors()["global"]
    with open(DATA_DIR / "weights.json") as f:
        wdata = json.load(f)
    gweights = wdata["weights"]["global"]
    guse = wdata["use_corrected"]["global"]

    out = []
    for city, g in live.groupby("city"):
        b = correct_and_blend(g, correctors, gweights, guse)
        b["city"] = city
        out.append(b)
    snap = pd.concat(out, ignore_index=True)
    snap.to_csv(OUT_DIR / "snapshots_blended.csv", index=False)
    print(f"Saved outputs/snapshots_blended.csv: {len(snap)} rows, "
          f"NaN blended: {int(snap['blended'].isna().sum())}")
    return snap


if __name__ == "__main__":
    main()
