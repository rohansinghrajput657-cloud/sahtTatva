"""Step 3 — Adaptive blending weights + reusable correct-and-blend routine.

Weights: per (scope, variable, lead-time band), inverse bias-corrected test
RMSE, normalised to sum to 1. Scopes: each training city, plus "global"
(UP-aggregated) used by the live app for unseen locations.

Honest note: training pairs are model-analysis vs ERA5 (no true lead-time
dimension), so weights are identical across lead bands in this prototype.
The structure fully supports band-varying weights once true forecast
archives (e.g. TIGGE) are used — see README.

Outputs: data/weights.json, data/weights.csv
Also exposes correct_and_blend() reused by the Streamlit app.
"""
import json
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import CITIES, DATA_DIR, LEAD_BANDS, MODELS, TIMEZONE, VARIABLES


def load_correctors():
    with open(DATA_DIR / "bias_models.pkl", "rb") as f:
        return pickle.load(f)


def _weights_from_skill(skill_sub):
    """Inverse-RMSE weights from a (variable-filtered) skill slice.

    Do-no-harm gate: a model's corrected output is used only if its test
    RMSE beats its raw RMSE; otherwise the raw value is used. Returns
    (weights, use_corrected) dicts keyed by model.
    """
    inv, use_corrected = {}, {}
    for model in MODELS:
        r = skill_sub[skill_sub["model"] == model]
        if len(r) == 0:
            continue
        raw = float(r["rmse_raw"].iloc[0])
        corr = float(r["rmse_corrected"].iloc[0])
        use = bool(corr < raw) if raw and raw > 0 else False
        use_corrected[model] = use
        best = corr if use else raw
        inv[model] = 1.0 / best if best and best > 0 else 0.0
    total = sum(inv.values())
    if total <= 0:
        n = max(len(inv), 1)
        return ({m: 1.0 / n for m in (inv or MODELS)},
                {m: False for m in (inv or MODELS)})
    return ({m: w / total for m, w in inv.items()}, use_corrected)


def compute_weights():
    skill = pd.read_csv(DATA_DIR / "bias_skill.csv")
    test = skill[skill["split"] == "test"]
    weights = {"per_city": {}, "global": {}}
    use_corr = {"per_city": {}, "global": {}}
    for city in CITIES:
        weights["per_city"][city] = {}
        use_corr["per_city"][city] = {}
        for var in VARIABLES:
            sub = test[(test["scope"] == "city") & (test["city"] == city)
                       & (test["variable"] == var)]
            w, u = _weights_from_skill(sub)
            weights["per_city"][city][var] = {b: dict(w) for b in LEAD_BANDS}
            use_corr["per_city"][city][var] = u
    for var in VARIABLES:
        sub = test[(test["scope"] == "global") & (test["city"] == "ALL-UP")
                   & (test["variable"] == var)]
        w, u = _weights_from_skill(sub)
        weights["global"][var] = {b: dict(w) for b in LEAD_BANDS}
        use_corr["global"][var] = u
    # validate sums
    for scope_dict in (weights["per_city"], {"_": weights["global"]}):
        for _, g in scope_dict.items():
            for var, bands in g.items():
                for band, w in bands.items():
                    s = sum(w.values())
                    assert abs(s - 1.0) < 1e-9, f"weights sum != 1: {s}"
    return weights, use_corr


def lead_band(hours):
    for band, (lo, hi) in LEAD_BANDS.items():
        if lo <= hours < hi:
            return band
    return "72-168h" if hours >= 72 else "0-24h"


def correct_and_blend(live_df, correctors, weights_scope, use_corrected_scope=None):
    """Bias-correct + blend a live forecast dataframe.

    live_df columns: time, lat, lon, variable, model, value
    correctors: dict keyed (model, variable) -> regressor (global set)
    weights_scope: {variable: {band: {model: weight}}}
    use_corrected_scope: {variable: {model: bool}} — do-no-harm gate;
        when False the model's raw value is used instead of corrected.
    Returns df: time, lat, lon, variable, lead_hours, lead_band,
                <model>_corrected..., blended
    """
    use_corrected_scope = use_corrected_scope or {}
    df = live_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["lead_hours"] = ((df["time"] - df["time"].min())
                        .dt.total_seconds() / 3600.0)
    df["lead_band"] = df["lead_hours"].map(lead_band)

    pivot = df.pivot_table(index=["time", "lat", "lon", "lead_hours",
                                  "lead_band", "variable"],
                           columns="model", values="value").reset_index()
    out = []
    for r in pivot.itertuples():
        corr_vals, used_w = {}, {}
        band_w = weights_scope.get(r.variable, {})
        gate = use_corrected_scope.get(r.variable, {})
        for model in MODELS:
            raw = getattr(r, model, np.nan)
            if pd.isna(raw):
                continue
            if gate.get(model, True):
                reg = correctors.get((model, r.variable))
                if reg is None:
                    continue
                feat = pd.DataFrame([{model: raw, "hour": r.time.hour,
                                      "doy": r.time.dayofyear,
                                      "lat": r.lat, "lon": r.lon}])
                corr_vals[model] = float(reg.predict(feat)[0])
            else:
                corr_vals[model] = float(raw)  # do-no-harm: keep raw
            used_w[model] = band_w.get(r.lead_band, {}).get(model, 0.0)
        wsum = sum(used_w.values())
        blended = (sum(corr_vals[m] * used_w[m] for m in corr_vals) / wsum
                   if wsum > 0 else float("nan"))
        row = {"time": r.time, "lat": r.lat, "lon": r.lon,
               "variable": r.variable, "lead_hours": round(float(r.lead_hours), 1),
               "lead_band": r.lead_band, "blended": blended}
        for model in MODELS:
            row[f"{model}_corrected"] = corr_vals.get(model, np.nan)
        out.append(row)
    return (pd.DataFrame(out)
            .sort_values(["time", "lat", "lon", "variable"])
            .reset_index(drop=True))


def main():
    weights, use_corr = compute_weights()
    payload = {
        "generated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "method": ("inverse test-RMSE (corrected if it beats raw, else raw "
                   ") — do-no-harm gate, normalised to sum 1"),
        "note": ("Weights are identical across lead bands in this prototype "
                 "because training pairs are model-analysis vs ERA5 (no true "
                 "lead-time dimension). The structure supports band-varying "
                 "weights with true forecast archives."),
        "weights": weights,
        "use_corrected": use_corr,
    }
    with open(DATA_DIR / "weights.json", "w") as f:
        json.dump(payload, f, indent=2)
    flat = []
    for scope, group in (("global", weights["global"]),
                         *[(f"city:{c}", g) for c, g in weights["per_city"].items()]):
        for var, bands in group.items():
            for band, w in bands.items():
                for model, val in w.items():
                    city = scope.split(":", 1)[1] if ":" in scope else None
                    gate = (use_corr["per_city"][city][var][model]
                            if city else use_corr["global"][var][model])
                    flat.append({"scope": scope, "variable": var,
                                 "lead_band": band, "model": model,
                                 "weight": val,
                                 "bias_correction_used": gate})
    pd.DataFrame(flat).to_csv(DATA_DIR / "weights.csv", index=False)
    print("Global weights (bias-correction gate in brackets):")
    gw = pd.DataFrame(flat)
    piv = gw[gw["scope"] == "global"].pivot_table(
        index=["variable", "lead_band"], columns="model",
        values="weight").round(3)
    print(piv.to_string())
    print("Saved data/weights.json, data/weights.csv")
    return weights


if __name__ == "__main__":
    main()
