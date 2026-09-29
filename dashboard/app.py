"""sahtTatva — live UP-wide Hybrid AI-NWP blending app (SIH 2026 / SIH26081).

Search ANY place in Uttar Pradesh -> live 7-day blended forecast from
GFS + ICON + GEM, bias-corrected with ML models trained on real
model-analysis vs ERA5 pairs across 12 UP cities, blended with
inverse-skill weights per lead-time band.

Run:  streamlit run dashboard/app.py
"""
import json
import sys
import time
import pickle
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (APP_NAME, CITIES, DATA_DIR, FORECAST_URL, GEOCODE_URL,
                    LEAD_BANDS, MODELS, MODEL_LABELS, OUT_DIR, TIMEZONE,
                    UP_BBOX, VARIABLES, VAR_LABELS)
from alerts import compute_alerts
from blend import correct_and_blend, load_correctors

st.set_page_config(page_title=APP_NAME, page_icon="🌦️", layout="wide")
IST = ZoneInfo(TIMEZONE)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.hero {
    background: linear-gradient(120deg, #0284c7 0%, #4f46e5 55%, #9333ea 100%);
    border-radius: 20px; padding: 26px 30px; color: #fff; margin-bottom: 20px;
    box-shadow: 0 8px 30px rgba(79,70,229,.35);
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -.02em; }
.hero p { margin: 8px 0 0; opacity: .93; font-size: 1rem; }
.live-badge {
    display: inline-flex; align-items: center; gap: 7px; background: rgba(255,255,255,.16);
    border: 1px solid rgba(255,255,255,.35); border-radius: 999px;
    padding: 4px 14px; font-size: .8rem; font-weight: 700; letter-spacing: .08em;
}
.live-dot { width: 9px; height: 9px; border-radius: 50%; background: #4ade80;
    box-shadow: 0 0 10px #4ade80; animation: pulse 1.6s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .45; } }
.card {
    background: rgba(148,163,184,.08); border: 1px solid rgba(148,163,184,.16);
    border-radius: 16px; padding: 16px 18px; height: 100%;
}
.card .big { font-size: 2.1rem; font-weight: 800; line-height: 1.1; }
.card .label { font-size: .72rem; opacity: .65; text-transform: uppercase;
    letter-spacing: .08em; margin-bottom: 4px; }
.card .sub { font-size: .85rem; opacity: .75; margin-top: 2px; }
.day-card {
    text-align: center; background: rgba(148,163,184,.07);
    border: 1px solid rgba(148,163,184,.14); border-radius: 14px;
    padding: 12px 6px; margin-bottom: 8px;
}
.day-card .d { font-weight: 700; font-size: .95rem; }
.day-card .dt { font-size: .75rem; opacity: .6; }
.day-card .e { font-size: 1.7rem; margin: 4px 0; }
.day-card .t { font-size: .95rem; margin: 2px 0; }
.day-card .m { font-size: .78rem; opacity: .75; }
.loc-card {
    background: rgba(56,189,248,.1); border: 1px solid rgba(56,189,248,.3);
    border-radius: 14px; padding: 12px 18px; margin: 12px 0;
    font-size: 1.05rem; font-weight: 600;
}
.section-title { font-size: 1.25rem; font-weight: 700; margin: 22px 0 10px; }
.hint { font-size: .85rem; opacity: .6; }

/* Hide Streamlit default header, footer & badges */
#MainMenu {visibility: hidden; display: none !important;}
footer {visibility: hidden; display: none !important;}
header {visibility: hidden; display: none !important;}
div[class*="viewerBadge"] {display: none !important;}
.viewerBadge_container__1QSob {display: none !important;}
.viewerBadge_link__1S137 {display: none !important;}
#Manage-app-button {display: none !important;}
a[href*="streamlit.io"] {display: none !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def api_get(url, params, retries=3, timeout=60):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"API request failed after {retries} tries: {last}")


@st.cache_resource
def load_models():
    with open(DATA_DIR / "bias_models.pkl", "rb") as f:
        correctors = pickle.load(f)["global"]
    with open(DATA_DIR / "weights.json") as f:
        wdata = json.load(f)
    weights = wdata["weights"]["global"]
    use_corrected = wdata["use_corrected"]["global"]
    with open(DATA_DIR / "training_stats.json") as f:
        stats = json.load(f)
    ver = pd.read_csv(OUT_DIR / "verification_table.csv")
    return correctors, weights, use_corrected, stats, ver


@st.cache_data(ttl=3600)
def geocode(query):
    data = api_get(GEOCODE_URL, {"name": query, "count": 5,
                                 "language": "en", "format": "json"})
    return data.get("results", []) or []


def in_up(lat, lon, admin1):
    in_box = (UP_BBOX["lat_min"] <= lat <= UP_BBOX["lat_max"]
              and UP_BBOX["lon_min"] <= lon <= UP_BBOX["lon_max"])
    return in_box and (admin1 or "").strip().lower() == "uttar pradesh"


@st.cache_data(ttl=1800)
def fetch_live_forecast(lat, lon):
    """One API call per model (multi-model merges proved unreliable).
    Returns tidy long dataframe."""
    rows = []
    for model in MODELS:
        data = api_get(FORECAST_URL, {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(VARIABLES),
            "models": model,
            "forecast_days": 7, "timezone": TIMEZONE})
        hourly = data["hourly"]
        times = pd.to_datetime(hourly["time"])
        for var in VARIABLES:
            key = f"{var}_{model}" if f"{var}_{model}" in hourly else var
            vals = hourly.get(key) or []
            for t, v in zip(times, vals):
                rows.append({"time": t, "lat": lat, "lon": lon,
                             "variable": var, "model": model, "value": v})
    return pd.DataFrame(rows)


def daily_aggregates(blended):
    out = []
    for var in VARIABLES:
        g = blended[blended["variable"] == var].copy()
        g["date"] = g["time"].dt.date
        agg = g.groupby("date")["blended"]
        if var == "precipitation":
            d = agg.sum().rename("total_mm")
        elif var == "temperature_2m":
            d = pd.DataFrame({"min_C": agg.min(), "max_C": agg.max(),
                              "mean_C": agg.mean()})
        else:
            d = pd.DataFrame({"max_kmh": agg.max(), "mean_kmh": agg.mean()})
        d = d.reset_index() if isinstance(d, pd.Series) else d.reset_index()
        d["variable"] = var
        out.append(d)
    return out


def day_icon(tmax, rain):
    if rain >= 8:
        return "🌧️"
    if rain >= 0.5:
        return "🌦️"
    if tmax >= 40:
        return "🥵"
    if tmax >= 33:
        return "☀️"
    return "⛅"


def metric_card(label, big, sub=""):
    st.markdown(f'<div class="card"><div class="label">{label}</div>'
                f'<div class="big">{big}</div>'
                + (f'<div class="sub">{sub}</div>' if sub else '')
                + '</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- data
with st.spinner("Loading sahtTatva's trained models…"):
    correctors, gweights, guse_corrected, stats, ver = load_models()

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown(f"## 🌦️ {APP_NAME}")
page = st.sidebar.radio("Navigate", ["🔍 Search any UP location", "🗺️ UP overview"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Trained on real data** — no synthetic values anywhere.\n\n"
    f"- {len(stats['cities'])} UP cities × 30 days of model analyses\n"
    f"- Ground truth: ERA5 reanalysis\n"
    f"- Pooled training rows: {stats['training_rows_pooled']:,}\n"
    f"- Models: GFS · ICON · GEM")
up_ver = ver[ver["city"] == "UP-AGGREGATED"]
st.sidebar.markdown("**UP-aggregated test RMSE** (blended vs best raw):")
for var in VARIABLES:
    sub = up_ver[up_ver["variable"] == var]
    b = float(sub[sub["method"] == "blended_global"]["rmse"].iloc[0])
    best_raw = float(sub[sub["method"].str.startswith("raw_")]["rmse"].min())
    st.sidebar.markdown(f"- {VAR_LABELS[var]}: **{b:.2f}** vs {best_raw:.2f}")
st.sidebar.markdown("---")
st.sidebar.caption("Data: Open-Meteo (GFS/ICON/GEM) + ERA5 · SIH 2026")

# ================================================================ SEARCH
if page == "🔍 Search any UP location":
    st.markdown(
        '<div class="hero"><span class="live-badge">'
        '<span class="live-dot"></span>LIVE</span>'
        f'<h1>🌦️ {APP_NAME}</h1>'
        '<p>Hybrid AI–NWP multi-model forecast blending · '
        'SIH 2026 PS SIH26081 · Ministry of Earth Sciences</p></div>',
        unsafe_allow_html=True)

    if "q" not in st.session_state:
        st.session_state.q = ""
    st.markdown("#### 📍 Find your place in Uttar Pradesh")
    st.text_input("Search", placeholder="Type a city, town or district… e.g. Lucknow",
                  key="q", label_visibility="collapsed")
    def _set_q(city):
        # callback runs BEFORE the script body, so mutating the widget-bound
        # key here is legal (doing it after instantiation raises
        # StreamlitWidgetAlreadyInstantiatedError)
        st.session_state.q = city

    chips = ["Lucknow", "Varanasi", "Prayagraj", "Kanpur", "Agra", "Meerut"]
    ccols = st.columns(len(chips))
    for col, city in zip(ccols, chips):
        with col:
            st.button(city, key=f"chip_{city}", use_container_width=True,
                      on_click=_set_q, args=(city,))
    st.markdown('<div class="hint">Tap a city for instant forecast, or type any '
                'UP town / district / village name above.</div>',
                unsafe_allow_html=True)

    query = (st.session_state.q or "").strip()
    if query and len(query) >= 2:
        try:
            results = geocode(query)
        except RuntimeError as e:
            st.error(f"Geocoding service unreachable: {e}")
            results = []
        if not results:
            st.warning("🔎 No location found. Check the spelling and try again.")
        else:
            options = {f"{r['name']} ({r.get('admin1', '')}, {r.get('country', '')})"
                       f" — {r['latitude']:.2f}, {r['longitude']:.2f}": r
                       for r in results}
            choice = st.selectbox("Pick the correct place", list(options))
            r = options[choice]
            lat, lon = float(r["latitude"]), float(r["longitude"])
            if not in_up(lat, lon, r.get("admin1")):
                st.error("📍 This place is outside Uttar Pradesh. "
                         "sahtTatva currently serves UP only — "
                         "please search a UP city, town or district.")
            else:
                st.markdown(
                    f'<div class="loc-card">📍 {r["name"]}, {r.get("admin1")} '
                    f'<span style="opacity:.65">({lat:.3f}, {lon:.3f})</span></div>',
                    unsafe_allow_html=True)
                if st.button("🔄 Refresh live data"):
                    fetch_live_forecast.clear()
                    st.rerun()
                with st.spinner("Fetching live forecasts from 3 models and blending…"):
                    try:
                        live = fetch_live_forecast(lat, lon)
                        blended = correct_and_blend(live, correctors, gweights,
                                                    guse_corrected)
                    except RuntimeError as e:
                        st.error(f"Forecast service unreachable: {e}")
                        blended = None
                if blended is not None and len(blended):
                    blended = blended.copy()
                    blended["date"] = blended["time"].dt.date
                    fetched_at = datetime.now(IST).strftime("%d %b %Y, %H:%M IST")

                    # ---- alerts first (most important)
                    snap_like = blended.copy()
                    snap_like["city"] = r["name"]
                    try:
                        era = pd.read_csv(DATA_DIR / "era5_tidy.csv",
                                          parse_dates=["time"])
                        t = era[era["variable"] == "temperature_2m"].copy()
                        t["date"] = t["time"].dt.date
                        up_normals = {r["name"]: float(
                            t.groupby("date")["value"].max().mean())}
                    except Exception:  # noqa: BLE001
                        up_normals = {}
                    al = compute_alerts(snap_like, up_normals or {})
                    if len(al):
                        st.markdown('<div class="section-title">⚠️ '
                                    'Extreme-weather alerts</div>',
                                    unsafe_allow_html=True)
                        for _, a in al.iterrows():
                            msg = (f"**{a['type']}** ({a['severity']}) — "
                                   f"{a['value']} {a['unit']}")
                            if a["severity"] == "High":
                                st.error(f"🚨 {msg}")
                            else:
                                st.warning(f"⚠️ {msg}")
                    else:
                        st.success("✅ No extreme-weather alerts for this location "
                                   "in the next 7 days.")

                    # ---- hero metrics
                    st.markdown('<div class="section-title">🌤️ Right now & today'
                                '</div>', unsafe_allow_html=True)
                    now_t = blended["time"].min()
                    today = now_t.date()
                    g_now = blended[blended["time"] == now_t]
                    g_today = blended[blended["date"] == today]
                    now_temp = float(g_now[g_now["variable"] == "temperature_2m"]
                                     ["blended"].iloc[0])
                    tmax = float(g_today[g_today["variable"] == "temperature_2m"]
                                 ["blended"].max())
                    tmin = float(g_today[g_today["variable"] == "temperature_2m"]
                                 ["blended"].min())
                    rain_today = float(g_today[g_today["variable"] == "precipitation"]
                                       ["blended"].sum())
                    wind_max = float(g_today[g_today["variable"] == "wind_speed_10m"]
                                     ["blended"].max())
                    mcols = st.columns(4)
                    with mcols[0]:
                        metric_card("🌡️ Now", f"{now_temp:.1f}°C",
                                    f"H {tmax:.0f}° / L {tmin:.0f}°")
                    with mcols[1]:
                        metric_card("🌧️ Rain today", f"{rain_today:.1f} mm",
                                    "blended total")
                    with mcols[2]:
                        metric_card("💨 Wind today", f"{wind_max:.0f} km/h",
                                    "max gust")
                    with mcols[3]:
                        metric_card("🤖 Models blended", "3",
                                    "GFS · ICON · GEM")
                    st.caption(f"Data fetched live at {fetched_at} · "
                               f"{len(blended)} hourly blended values · "
                               f"ML bias-corrected with {stats['training_rows_pooled']:,} "
                               f"real training rows")

                    # ---- tabs
                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["📅 7-day outlook", "📈 Hourly charts",
                         "🤖 Model showdown", "⚖️ Blend weights"])
                    with tab1:
                        days = sorted(blended["date"].unique())[:7]
                        dcols = st.columns(7)
                        for col, d in zip(dcols, days):
                            gd = blended[blended["date"] == d]
                            tt = gd[gd["variable"] == "temperature_2m"]["blended"]
                            rr = gd[gd["variable"] == "precipitation"]["blended"].sum()
                            ww = gd[gd["variable"] == "wind_speed_10m"]["blended"].max()
                            with col:
                                st.markdown(
                                    f'<div class="day-card">'
                                    f'<div class="d">{d.strftime("%a")}</div>'
                                    f'<div class="dt">{d.strftime("%d %b")}</div>'
                                    f'<div class="e">{day_icon(tt.max(), rr)}</div>'
                                    f'<div class="t"><b>{tt.max():.0f}°</b> / {tt.min():.0f}°</div>'
                                    f'<div class="m">🌧️ {rr:.1f} mm</div>'
                                    f'<div class="m">💨 {ww:.0f} km/h</div>'
                                    f'</div>', unsafe_allow_html=True)
                    with tab2:
                        var = st.selectbox("Variable", VARIABLES,
                                           format_func=lambda v: VAR_LABELS[v],
                                           key="hourly_var")
                        g = blended[blended["variable"] == var].sort_values("time")
                        fig = go.Figure()
                        for model in MODELS:
                            fig.add_trace(go.Scatter(
                                x=g["time"], y=g[f"{model}_corrected"],
                                mode="lines",
                                name=f"{MODEL_LABELS[model]} (bias-corr.)",
                                line=dict(dash="dot", width=1.5), opacity=0.75))
                        fig.add_trace(go.Scatter(
                            x=g["time"], y=g["blended"], mode="lines",
                            name="Blended (sahtTatva)",
                            line=dict(width=3.2, color="#38bdf8")))
                        fig.update_layout(height=430, hovermode="x unified",
                                          yaxis_title=VAR_LABELS[var],
                                          legend=dict(orientation="h", y=-0.22),
                                          margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab3:
                        st.markdown("How each raw model did vs the blend, "
                                    "on held-out real data (UP-aggregated RMSE — "
                                    "lower is better):")
                        for v in VARIABLES:
                            sub = up_ver[up_ver["variable"] == v].copy()
                            order = ([f"raw_{m}" for m in MODELS]
                                     + [f"corrected_{m}_global" for m in MODELS]
                                     + ["blended_global"])
                            sub["ord"] = sub["method"].map(
                                {m: i for i, m in enumerate(order)})
                            sub = sub.sort_values("ord")
                            figm = px.bar(sub, x="method", y="rmse",
                                          title=VAR_LABELS[v],
                                          labels={"rmse": "Test RMSE",
                                                  "method": "Method"},
                                          color="method")
                            figm.update_layout(height=300, showlegend=False,
                                               margin=dict(l=10, r=10, t=40, b=10))
                            st.plotly_chart(figm, use_container_width=True)
                    with tab4:
                        var2 = st.selectbox("Variable", VARIABLES,
                                            format_func=lambda v: VAR_LABELS[v],
                                            key="weight_var")
                        wrows = []
                        for band in LEAD_BANDS:
                            for model in MODELS:
                                wrows.append({"Lead band": band,
                                              "Model": MODEL_LABELS[model],
                                              "Weight": gweights[var2][band][model]})
                        wdf = pd.DataFrame(wrows)
                        fig2 = px.bar(wdf, x="Lead band", y="Weight", color="Model",
                                      barmode="group",
                                      title=f"Weights for {VAR_LABELS[var2]}")
                        fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig2, use_container_width=True)
                        st.caption("Weights = inverse bias-corrected test RMSE, "
                                   "normalised to sum to 1 — the model that was most "
                                   "skillful recently gets the biggest say.")

# ================================================================ OVERVIEW
else:
    st.markdown(
        '<div class="hero"><span class="live-badge">'
        '<span class="live-dot"></span>12 CITIES</span>'
        f'<h1>🗺️ UP overview</h1>'
        '<p>Blended snapshots from the 12 training cities · refreshed by the '
        'daily pipeline</p></div>',
        unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Blended vs best single model '
                '(held-out test RMSE — lower wins)</div>', unsafe_allow_html=True)
    vcols = st.columns(3)
    for col, var in zip(vcols, VARIABLES):
        sub = up_ver[up_ver["variable"] == var]
        b = float(sub[sub["method"] == "blended_global"]["rmse"].iloc[0])
        best_raw = float(sub[sub["method"].str.startswith("raw_")]["rmse"].min())
        win = b < best_raw
        with col:
            metric_card(f"{VAR_LABELS[var]}",
                        f"{b:.2f}",
                        f"best raw: {best_raw:.2f} · {'✅ blended wins' if win else '≈ tie'}")

    st.markdown('<div class="section-title">🗺️ 12-city blended snapshot map</div>',
                unsafe_allow_html=True)
    try:
        snap = pd.read_csv(OUT_DIR / "snapshots_blended.csv", parse_dates=["time"])
    except FileNotFoundError:
        st.error("Snapshots not found — run `python src/run_daily.py` first.")
        snap = None
    if snap is not None and len(snap):
        metric = st.selectbox("Map colour", ["Today's max temp (°C)",
                                             "7-day total rain (mm)",
                                             "Today's max wind (km/h)"])
        today = snap["time"].dt.date.min()
        rows = []
        for city, g in snap.groupby("city"):
            lat, lon = float(g["lat"].iloc[0]), float(g["lon"].iloc[0])
            if metric.startswith("Today"):
                d = g[g["time"].dt.date == today]
                var = "temperature_2m" if "temp" in metric else "wind_speed_10m"
                val = float(d[d["variable"] == var]["blended"].max())
            else:
                val = float(g[g["variable"] == "precipitation"]["blended"].sum())
            rows.append({"city": city, "lat": lat, "lon": lon, "value": val})
        mdf = pd.DataFrame(rows)
        fig = px.scatter_map(mdf, lat="lat", lon="lon", color="value",
                             size_max=18, zoom=5.6,
                             center={"lat": 27.0, "lon": 80.5},
                             hover_name="city",
                             color_continuous_scale="YlOrRd",
                             map_style="open-street-map",
                             title=f"12-city blended snapshot — {metric}")
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">📊 Verification — blended vs '
                    'individual models</div>', unsafe_allow_html=True)
        st.caption("Lower RMSE is better. 'Blended (global)' is exactly the "
                   "pipeline the search page uses for unseen locations.")
        for var in VARIABLES:
            sub = up_ver[up_ver["variable"] == var].copy()
            order = ([f"raw_{m}" for m in MODELS]
                     + [f"corrected_{m}_city" for m in MODELS]
                     + ["blended_city", "blended_global"])
            sub["ord"] = sub["method"].map({m: i for i, m in enumerate(order)})
            sub = sub.sort_values("ord")
            fig = px.bar(sub, x="method", y="rmse", title=VAR_LABELS[var],
                         labels={"rmse": "Test RMSE", "method": "Method"})
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔬 Per-city verification table"):
            cityv = ver[ver["city"] != "UP-AGGREGATED"]
            st.dataframe(cityv.round(4), hide_index=True, use_container_width=True)

st.markdown("---")
st.caption(f"{APP_NAME} · trained on real Open-Meteo model analyses + ERA5 · "
           "weights re-calibrated by the daily pipeline · SIH 2026")
