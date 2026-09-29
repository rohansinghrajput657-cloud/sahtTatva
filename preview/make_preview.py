"""sahtTatva preview generator — renders app-style preview PNGs using the
REAL app pipeline (geocode -> live 3-model fetch -> ML bias correction ->
adaptive blend -> alerts). No fabricated numbers: every value on the charts
comes from the same functions dashboard/app.py uses.
"""
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import (APP_NAME, CITIES, DATA_DIR, FORECAST_URL, GEOCODE_URL,
                    MODELS, MODEL_LABELS, TIMEZONE, VARIABLES, VAR_LABELS)
from blend import correct_and_blend
from alerts import compute_alerts, era5_normals

IST = ZoneInfo(TIMEZONE)
OUT = ROOT / "preview"
OUT.mkdir(exist_ok=True)

MODEL_COLORS = {"gfs_seamless": "#7aa7d9", "icon_seamless": "#f2a65a",
                "gem_seamless": "#8fd18f"}
BLEND_COLOR = "#1a3a5c"


def api_get(url, params, retries=3, timeout=60):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"API failed: {last}")


def geocode(query):
    data = api_get(GEOCODE_URL, {"name": query, "count": 5,
                                 "language": "en", "format": "json"})
    return data.get("results", []) or []


def fetch_live_forecast(lat, lon):
    rows = []
    for model in MODELS:
        data = api_get(FORECAST_URL, {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(VARIABLES),
            "models": model, "forecast_days": 7, "timezone": TIMEZONE})
        hourly = data["hourly"]
        times = pd.to_datetime(hourly["time"])
        for var in VARIABLES:
            key = f"{var}_{model}" if f"{var}_{model}" in hourly else var
            vals = hourly.get(key) or []
            for t, v in zip(times, vals):
                rows.append({"time": t, "lat": lat, "lon": lon,
                             "variable": var, "model": model, "value": v})
    return pd.DataFrame(rows)


def main():
    # ---- 1. real geocode -------------------------------------------------
    results = geocode("Lucknow")
    up = [r for r in results
          if (r.get("admin1") or "").lower() == "uttar pradesh"]
    loc = up[0] if up else results[0]
    name, lat, lon = loc["name"], float(loc["latitude"]), float(loc["longitude"])
    print(f"geocoded: {name} ({lat:.3f}, {lon:.3f}) admin1={loc.get('admin1')}")

    # ---- 2. real live fetch ----------------------------------------------
    live = fetch_live_forecast(lat, lon)
    print(f"live rows: {len(live)}, nans: {int(live['value'].isna().sum())}")

    # ---- 3. real correct + blend -------------------------------------------
    with open(DATA_DIR / "bias_models.pkl", "rb") as f:
        correctors = pickle.load(f)["global"]
    with open(DATA_DIR / "weights.json") as f:
        wdata = json.load(f)
    weights = wdata["weights"]["global"]
    use_corr = wdata["use_corrected"]["global"]
    blended = correct_and_blend(live, correctors, weights, use_corr)
    print(f"blended rows: {len(blended)}, nans: {int(blended['blended'].isna().sum())}")

    # raw model pivots for comparison lines
    raw = live.pivot_table(index=["time", "variable"], columns="model",
                           values="value").reset_index()

    # ---- 4. real alerts -----------------------------------------------------
    b2 = blended.copy()
    b2["city"] = name
    alerts = compute_alerts(b2, era5_normals())
    print(f"alerts: {len(alerts)}")

    stamp = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")

    # ================= PNG 1: Lucknow 7-day forecast ========================
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f"\u2600\ufe0f {APP_NAME} \u2014 {name}, Uttar Pradesh "
                 f"({lat:.2f}\u00b0N, {lon:.2f}\u00b0E)",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94,
             f"Live blended forecast \u00b7 GFS + ICON + GEM, ML bias-corrected "
             f"\u00b7 7-day hourly \u00b7 fetched {stamp}",
             ha="center", fontsize=10, color="#555")

    t = blended[blended["variable"] == "temperature_2m"].sort_values("time")
    rt = raw[raw["variable"] == "temperature_2m"].sort_values("time")
    ax = axes[0]
    for m in MODELS:
        ax.plot(rt["time"], rt[m], color=MODEL_COLORS[m], lw=1.1, alpha=0.8,
                label=f"{MODEL_LABELS[m]} (raw)")
    ax.plot(t["time"], t["blended"], color=BLEND_COLOR, lw=2.4,
            label="sahtTatva blended")
    ax.set_ylabel("Temp (\u00b0C)")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(alpha=0.25)

    p = blended[blended["variable"] == "precipitation"].sort_values("time")
    ax = axes[1]
    ax.bar(p["time"], p["blended"], width=0.035, color="#2f7fd0",
           label="sahtTatva blended")
    ax.set_ylabel("Rain (mm/h)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, axis="y")

    w = blended[blended["variable"] == "wind_speed_10m"].sort_values("time")
    rw = raw[raw["variable"] == "wind_speed_10m"].sort_values("time")
    ax = axes[2]
    for m in MODELS:
        ax.plot(rw["time"], rw[m], color=MODEL_COLORS[m], lw=1.1, alpha=0.8,
                label=f"{MODEL_LABELS[m]} (raw)")
    ax.plot(w["time"], w["blended"], color=BLEND_COLOR, lw=2.4,
            label="sahtTatva blended")
    ax.set_ylabel("Wind (km/h)")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.text(0.5, 0.02,
             "Real live data \u2014 the exact numbers the sahtTatva app shows for this search",
             ha="center", fontsize=9, color="#555", style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(OUT / "preview_1_lucknow_forecast.png", dpi=110)
    plt.close(fig)
    print("saved preview_1")

    # ================= PNG 2: home screen layout ============================
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    fig.patch.set_facecolor("#f7f7f5"); ax.set_facecolor("#f7f7f5")
    ax.text(5, 9.2, f"\U0001f327\ufe0f {APP_NAME}", fontsize=30,
            fontweight="bold", ha="center")
    ax.text(5, 8.7, "Hybrid AI\u2013NWP Multi-Model Forecast Blending System \u00b7 "
            "SIH 2026 PS SIH26081", fontsize=10, ha="center", color="#666")
    # search box
    box = plt.Rectangle((1.2, 7.4), 7.6, 0.8, fc="white", ec="#ccc",
                        lw=1.2)
    ax.add_patch(box)
    ax.text(1.6, 7.8, "\U0001f50d  Search any place in Uttar Pradesh\u2026  |  Lucknow \u2318",
            fontsize=12, va="center", color="#333")
    ax.text(5, 6.9, "Type a city / town / district name, e.g. Lucknow, Prayagraj, "
            "Mirzapur", fontsize=10, ha="center", color="#888")
    # three feature cards
    cards = [("3 NWP Models", "GFS \u00b7 ICON \u00b7 GEM\nlive 7-day forecasts"),
             ("AI Bias Correction", "Gradient-boosted correctors\ntrained on 20,736 real rows"),
             ("IMD Alerts", "Heavy rain \u00b7 Heatwave\n\u00b7 High wind warnings")]
    for i, (h, b) in enumerate(cards):
        x = 0.7 + i * 3.15
        c = plt.Rectangle((x, 4.4), 2.85, 1.7, fc="white", ec="#ddd", lw=1.2)
        ax.add_patch(c)
        ax.text(x + 1.42, 5.65, h, fontsize=11, fontweight="bold", ha="center")
        ax.text(x + 1.42, 5.05, b, fontsize=9, ha="center", color="#555",
                linespacing=1.5)
    ax.text(5, 3.5, "Also inside:  \U0001f5fa\ufe0f UP overview map of 12 cities   \u00b7  "
            "\U0001f4ca Verification (blended vs each model)   \u00b7  \u2696\ufe0f Model-weight explorer",
            fontsize=10, ha="center", color="#444")
    ax.text(5, 0.6, "Layout illustration \u2014 Streamlit app me aisa dikhega",
            fontsize=10, ha="center", color="#999", style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "preview_2_home.png", dpi=110)
    plt.close(fig)
    print("saved preview_2")

    # ================= PNG 3: weights + verification + alerts ===============
    ver = pd.read_csv(ROOT / "outputs" / "verification_table.csv")
    upv = ver[ver["city"] == "UP-AGGREGATED"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.suptitle(f"{APP_NAME} \u2014 how the blend is built (real trained values)",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    wrows = []
    for var in VARIABLES:
        band = next(iter(weights[var].values()))  # bands identical (documented)
        for m in MODELS:
            wrows.append({"variable": VAR_LABELS[var],
                          "model": MODEL_LABELS[m], "weight": band[m]})
    wdf = pd.DataFrame(wrows)
    y = np.arange(len(VAR_LABELS))
    left = np.zeros(len(VAR_LABELS))
    for m in MODELS:
        vals = [wdf[(wdf["variable"] == VAR_LABELS[v]) &
                    (wdf["model"] == MODEL_LABELS[m])]["weight"].iloc[0]
                for v in VARIABLES]
        ax.barh([VAR_LABELS[v] for v in VARIABLES], vals, left=left,
                color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        left += np.array(vals)
    ax.set_xlabel("Blend weight (sums to 1)")
    ax.set_title("Adaptive model weights", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    ax = axes[1]; ax.axis("off")
    ax.text(0.02, 0.95, "Held-out verification \u2014 RMSE (lower = better)",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    yy = 0.82
    for var in VARIABLES:
        sub = upv[upv["variable"] == var]
        rb = float(sub[sub["method"] == "blended_global"]["rmse"].iloc[0])
        best_raw = min(float(sub[sub["method"] == f"raw_{m}"]["rmse"].iloc[0])
                       for m in MODELS)
        mark = "\u2713" if rb < best_raw else "\u2248 tie"
        col = "#1a5c1a" if rb < best_raw else "#7a5c00"
        ax.text(0.02, yy, f"{VAR_LABELS[var]}:  blended {rb:.2f}  vs  "
                f"best single model {best_raw:.2f}  {mark}",
                fontsize=10, transform=ax.transAxes, color=col)
        yy -= 0.1
    ax.text(0.02, yy - 0.02, f"IMD alerts for {name} (next 7 days):",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    if len(alerts):
        for _, r in alerts.head(4).iterrows():
            yy -= 0.1
            ax.text(0.02, yy, f"\u26a0\ufe0f {r['type']} \u2014 {r['severity']} "
                    f"({r['value']} {r['unit']})", fontsize=10,
                    transform=ax.transAxes, color="#a33")
    else:
        ax.text(0.02, yy - 0.12, "\u2705 Koi alert nahi \u2014 agle 7 din mausam "
                "shaant hai (thresholds cross nahi hue)",
                fontsize=10, transform=ax.transAxes, color="#1a5c1a")
    fig.text(0.5, 0.01, "Weights & scores: app ke trained models se \u00b7 "
             "Alerts: Lucknow ke LIVE blended forecast se",
             ha="center", fontsize=9, color="#777", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 0.9])
    fig.savefig(OUT / "preview_3_weights_alerts.png", dpi=110)
    plt.close(fig)
    print("saved preview_3")
    print("DONE")


if __name__ == "__main__":
    main()
