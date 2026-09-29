"""Step 2 — Train ML bias correctors (the "AI" in hybrid AI-NWP).

For every (city, model, variable): a HistGradientBoostingRegressor learns
raw model output -> ERA5 ground truth, from model *analysis* values paired
with ERA5 truth (NOT archived forecasts — limitation documented in README).

Additionally, ONE pooled "global-UP" corrector per (model, variable) is
trained on all cities combined; the live app uses the global correctors for
never-seen searched locations.

Features: model value, hour-of-day, day-of-year, lat, lon.
Train on first N days, test on last 6 (or 5) days, per city.

Outputs: data/bias_models.pkl, data/bias_skill.csv, data/training_stats.json
"""
import json
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import CITIES, DATA_DIR, MODELS, TIMEZONE, VARIABLES

FEATURES_BASE = ["hour", "doy", "lat", "lon"]


def features_for(model):
    """Feature columns for a model's corrector: its own value + time/space."""
    return [model] + FEATURES_BASE


def _add_features(df):
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["doy"] = df["time"].dt.dayofyear
    return df


def load_aligned():
    """Join model analysis values with ERA5 truth.

    NOTE: each model snaps to its own grid, so API-returned lat/lon differ
    slightly per model. We therefore join on (city, time, variable) and use
    the nominal city coordinates from CITIES as the lat/lon features.

    Returns df with columns: city, time, variable, lat, lon, <model cols>,
    truth, split.
    """
    fc = pd.read_csv(DATA_DIR / "forecast_tidy.csv", parse_dates=["time"])
    era = pd.read_csv(DATA_DIR / "era5_tidy.csv", parse_dates=["time"])
    ana = fc[fc["kind"] == "analysis"].copy()
    ana["date"] = ana["time"].dt.date
    era["date"] = era["time"].dt.date

    frames, common_by_city = [], {}
    for city in CITIES:
        a = ana[ana["city"] == city]
        e = era[era["city"] == city]
        common = sorted(set(a["date"]) & set(e["date"]))
        if not common:
            print(f"  WARNING: no common dates for {city}, skipping")
            continue
        common_by_city[city] = common
        a = a[a["date"].isin(common)]
        e = e[e["date"].isin(common)]
        pivot = a.pivot_table(index=["time", "variable"],
                              columns="model", values="value").reset_index()
        truth = (e.rename(columns={"value": "truth"})
                 .groupby(["time", "variable"], as_index=False)["truth"].mean())
        df = pivot.merge(truth, on=["time", "variable"], how="inner")
        df["city"] = city
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["lat"] = full["city"].map(lambda c: CITIES[c][0])
    full["lon"] = full["city"].map(lambda c: CITIES[c][1])
    full = _add_features(full)
    # train/test split per city (first N days train, rest test)
    full["split"] = "test"
    for city, common in common_by_city.items():
        train_dates, _ = _split_dates(common)
        mask = (full["city"] == city) & (full["time"].dt.date.isin(train_dates))
        full.loc[mask, "split"] = "train"
    n_models_ok = sum(1 for m in MODELS if m in full.columns)
    print(f"Aligned rows: {len(full)} across {len(common_by_city)} cities "
          f"({n_models_ok}/{len(MODELS)} model columns present)")
    return full, common_by_city


def _split_dates(common):
    n = len(common)
    n_train = 24 if n >= 30 else (16 if n >= 21 else max(1, n - 5))
    return set(common[:n_train]), set(common[n_train:])


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else float("nan")


def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m]))) if m.sum() else float("nan")


def _fit_and_score(sub, model):
    """Fit corrector on sub[sub.split=='train'], score on both splits."""
    features = features_for(model)
    tr = sub[sub["split"] == "train"].dropna(subset=features + ["truth"])
    te = sub[sub["split"] == "test"].dropna(subset=features + ["truth"])
    if len(tr) < 100 or len(te) < 20:
        return None, None
    reg = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=10, random_state=42)
    reg.fit(tr[features], tr["truth"])
    return reg, (tr, te)


def _score_rows(scope, city, model, var, tr, te, reg):
    features = features_for(model)
    rows = []
    for split, d in (("train", tr), ("test", te)):
        raw = d[model].to_numpy()
        corr = reg.predict(d[features])
        truth = d["truth"].to_numpy()
        rows.append({"scope": scope, "city": city, "model": model,
                     "variable": var, "split": split, "n": len(d),
                     "rmse_raw": rmse(raw, truth), "mae_raw": mae(raw, truth),
                     "rmse_corrected": rmse(corr, truth),
                     "mae_corrected": mae(corr, truth)})
    return rows


def main():
    df, common_by_city = load_aligned()

    city_models, global_models, skill_rows = {}, {}, []

    # 1) per-city correctors
    for city in common_by_city:
        for model in MODELS:
            for var in VARIABLES:
                sub = df[(df["city"] == city) & (df["variable"] == var)].copy()
                sub = sub.dropna(subset=features_for(model) + ["truth"])
                reg, parts = _fit_and_score(sub, model)
                if reg is None:
                    print(f"  SKIP city {city}/{model}/{var}: too few rows")
                    continue
                city_models[(city, model, var)] = reg
                tr, te = parts
                skill_rows.extend(_score_rows("city", city, model, var, tr, te, reg))

    # 2) pooled global-UP correctors (what the app uses for unseen locations)
    for model in MODELS:
        for var in VARIABLES:
            sub = df[df["variable"] == var].copy()
            sub = sub.dropna(subset=features_for(model) + ["truth"])
            reg, parts = _fit_and_score(sub, model)
            if reg is None:
                print(f"  SKIP global {model}/{var}: too few rows")
                continue
            global_models[(model, var)] = reg
            tr, te = parts
            skill_rows.extend(_score_rows("global", "ALL-UP", model, var, tr, te, reg))
            # also score the global corrector per city (honest app-path evaluation)
            for city in common_by_city:
                csub = sub[sub["city"] == city]
                ctr = csub[csub["split"] == "train"]
                cte = csub[csub["split"] == "test"]
                if len(ctr) >= 20 and len(cte) >= 10:
                    skill_rows.extend(
                        _score_rows("global", city, model, var, ctr, cte, reg))

    with open(DATA_DIR / "bias_models.pkl", "wb") as f:
        pickle.dump({"city": city_models, "global": global_models}, f)
    skill = pd.DataFrame(skill_rows)
    skill.to_csv(DATA_DIR / "bias_skill.csv", index=False)

    # headline: did correction help on test?
    t = skill[(skill["split"] == "test")]
    for scope in ("city", "global"):
        s = t[t["scope"] == scope]
        if len(s):
            imp = (s["rmse_raw"] - s["rmse_corrected"]).mean()
            print(f"  [{scope}] mean test RMSE improvement: {imp:.4f} "
                  f"over {len(s)} (model,var,city) combos")

    stats = {
        "generated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "cities": {c: {"lat": lat, "lon": lon} for c, (lat, lon) in CITIES.items()},
        "models": MODELS,
        "variables": VARIABLES,
        "per_city_train_days": {c: len(_split_dates(d)[0])
                                for c, d in common_by_city.items()},
        "per_city_test_days": {c: len(d) - len(_split_dates(d)[0])
                               for c, d in common_by_city.items()},
        "n_city_correctors": len(city_models),
        "n_global_correctors": len(global_models),
        "training_rows_pooled": int(df[df["split"] == "train"]
                                    .dropna(subset=["truth"]).shape[0]),
    }
    with open(DATA_DIR / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {len(city_models)} city + {len(global_models)} global correctors")
    return city_models, global_models, skill


if __name__ == "__main__":
    main()
