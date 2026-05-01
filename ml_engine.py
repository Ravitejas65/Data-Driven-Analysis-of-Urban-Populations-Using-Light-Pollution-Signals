"""
models/ml_engine.py
===================
Machine-learning layer for the Urban Light Pollution dashboard.

Models
------
1. Random-Forest regressor  – predict radiance from economic indicators
2. K-Means clustering       – group cities by light/economic profile
3. PCA                      – 2-D projection for cluster visualisation
4. Isolation Forest         – anomaly detection (over/under-lit cities)
5. Linear trend forecast    – per-city radiance extrapolation to 2030
6. Feature importance       – SHAP-style bar chart data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ── Feature catalogue ─────────────────────────────────────────────────────────
ECON_FEATURES  = ["gdp_per_capita", "electricity_pct", "urban_pct", "co2_pc"]
LIGHT_FEATURES = ["radiance_2013", "radiance_growth", "radiance_5yr_avg"]
ALL_FEATURES   = ECON_FEATURES + LIGHT_FEATURES
TARGET         = "radiance_2023"

CLUSTER_LABELS = {
    0: "Mega-Cities",
    1: "Advanced Urban",
    2: "Emerging Metros",
    3: "Developing Hubs",
    4: "Mid-Tier Cities",
}


def _clean(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Drop rows with NaN in required columns and return a copy."""
    return df.dropna(subset=cols).copy()


# ── 1. Random Forest – radiance predictor ────────────────────────────────────
def train_radiance_model(df: pd.DataFrame) -> dict:
    available = [c for c in ALL_FEATURES if c in df.columns]
    sub = _clean(df, available + [TARGET])
    if len(sub) < 8:
        return {}

    X = sub[available].values
    y = sub[TARGET].values

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=300, max_depth=6,
                               min_samples_leaf=2, random_state=42)
    cv_scores = cross_val_score(rf, X_s, y, cv=min(5, len(sub) // 3),
                                scoring="r2")

    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.25,
                                               random_state=42)
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)

    importances = pd.Series(rf.feature_importances_,
                            index=available).sort_values(ascending=False)

    return {
        "model":       rf,
        "scaler":      scaler,
        "features":    available,
        "r2":          r2_score(y_te, preds),
        "mae":         mean_absolute_error(y_te, preds),
        "cv_mean":     float(cv_scores.mean()),
        "cv_std":      float(cv_scores.std()),
        "importances": importances,
        "preds":       preds,
        "actuals":     y_te,
        "city_names":  sub["city"].values,
    }


# ── 2. K-Means clustering ─────────────────────────────────────────────────────
def cluster_cities(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    feats = [c for c in ["radiance_2023", "gdp_per_capita",
                          "electricity_pct", "population",
                          "radiance_growth"] if c in df.columns]
    sub = _clean(df, feats)
    if len(sub) < n_clusters:
        return df.copy()

    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(sub[feats])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    labels = km.fit_predict(X_s)

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_s)

    sub = sub.copy()
    sub["cluster"]          = labels
    sub["pca_x"]            = coords[:, 0]
    sub["pca_y"]            = coords[:, 1]
    sub["var_explained"]    = str([round(v, 3) for v in
                                    pca.explained_variance_ratio_])

    # Assign human-readable labels based on cluster centroid character
    label_map = {}
    centroids = pd.DataFrame(km.cluster_centers_, columns=feats)
    for c in range(n_clusters):
        rad = centroids.loc[c, "radiance_2023"] if "radiance_2023" in centroids else 0
        gdp = centroids.loc[c, "gdp_per_capita"] if "gdp_per_capita" in centroids else 0
        if rad > 55 and gdp > 40000:
            lbl = "Mega-Cities"
        elif rad > 45 and gdp > 20000:
            lbl = "Advanced Urban"
        elif rad > 35:
            lbl = "Emerging Metros"
        elif gdp < 8000:
            lbl = "Developing Hubs"
        else:
            lbl = "Mid-Tier Cities"
        label_map[c] = lbl

    sub["cluster_label"] = sub["cluster"].map(label_map)
    return sub


# ── 3. Anomaly detection ──────────────────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in ["radiance_2023", "gdp_per_capita",
                          "electricity_pct", "population"] if c in df.columns]
    sub = _clean(df, feats)

    scaler   = StandardScaler()
    X_s      = scaler.fit_transform(sub[feats])

    iso = IsolationForest(contamination=0.15, random_state=42)
    sub = sub.copy()
    sub["anomaly_flag"]  = iso.fit_predict(X_s)   # -1 = anomaly
    sub["anomaly_score"] = iso.score_samples(X_s)
    sub["is_anomaly"]    = sub["anomaly_flag"] == -1

    # Expected radiance from economic profile (Ridge regression)
    if "gdp_per_capita" in feats and "radiance_2023" in sub.columns:
        econ_idx = [i for i, f in enumerate(feats) if f != "radiance_2023"]
        if econ_idx:
            X_econ   = X_s[:, econ_idx]
            y_rad    = X_s[:, feats.index("radiance_2023")]
            rr       = Ridge(alpha=1.0).fit(X_econ, y_rad)
            expected = rr.predict(X_econ)
            sub["radiance_residual"] = y_rad - expected   # >0 = over-lit
    return sub


# ── 4. Radiance forecast 2024-2030 ───────────────────────────────────────────
def forecast_radiance(ts_df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Per-city linear trend extrapolation."""
    all_rows = []
    for city in ts_df["city"].unique():
        sub = ts_df[ts_df["city"] == city].sort_values("year")
        x   = sub["year"].values.reshape(-1, 1)
        y   = sub["radiance"].values
        lr  = LinearRegression().fit(x, y)

        last_yr  = int(sub["year"].max())
        fut_yrs  = np.arange(last_yr + 1, last_yr + horizon + 1).reshape(-1, 1)
        preds    = np.clip(lr.predict(fut_yrs), 0, None)

        for yr, p in zip(fut_yrs.flatten(), preds):
            all_rows.append({
                "city":     city,
                "region":   sub["region"].iloc[0],
                "year":     int(yr),
                "radiance": float(p),
                "type":     "forecast",
            })

    hist         = ts_df.copy()
    hist["type"] = "historical"
    return pd.concat([hist, pd.DataFrame(all_rows)], ignore_index=True)


# ── 5. Efficiency index ───────────────────────────────────────────────────────
def compute_efficiency_index(df: pd.DataFrame) -> pd.DataFrame:
    """Light efficiency = economic output per unit radiance."""
    sub = _clean(df, ["radiance_2023", "gdp_per_capita"])
    sub = sub.copy()
    sub["gdp_per_radiance"]  = sub["gdp_per_capita"] / (sub["radiance_2023"] + 1)
    # Normalise 0-100
    mn, mx = sub["gdp_per_radiance"].min(), sub["gdp_per_radiance"].max()
    sub["efficiency_score"]  = (sub["gdp_per_radiance"] - mn) / (mx - mn + 1e-9) * 100
    return sub.sort_values("efficiency_score", ascending=False)
