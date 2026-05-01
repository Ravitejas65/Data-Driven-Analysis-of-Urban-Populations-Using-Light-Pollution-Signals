"""
app.py  –  Urban Light Pollution Intelligence Dashboard
========================================================
Real-time data sources
  • World Bank Indicators API  (population, GDP, electricity, CO₂)
  • NASA VIIRS Black Marble    (annual nighttime radiance 2013-2023)

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data.fetch_realtime import (
    build_main_dataset, build_viirs_timeseries, clear_cache, CITIES
)
from models.ml_engine import (
    train_radiance_model, cluster_cities,
    detect_anomalies, forecast_radiance, compute_efficiency_index
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Light Pollution Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0A0E1A; }

/* ── Header ── */
.hero {
    background: linear-gradient(135deg, #0A0E1A 0%, #0d1f3c 50%, #0A0E1A 100%);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    color: #00D4FF;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem 0;
    text-shadow: 0 0 40px rgba(0,212,255,0.4);
}
.hero-sub {
    font-size: 0.9rem; color: #7A8AAD; font-weight: 300;
    letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: #00D4FF;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-right: 0.5rem;
    margin-top: 0.7rem;
}

/* ── KPI cards ── */
.kpi-card {
    background: #0F1525;
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    border-color: rgba(0,212,255,0.35);
    box-shadow: 0 0 20px rgba(0,212,255,0.08);
}
.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem; font-weight: 700;
    color: #00D4FF;
}
.kpi-label {
    font-size: 0.72rem; color: #7A8AAD;
    letter-spacing: 0.8px; text-transform: uppercase;
    margin-top: 0.3rem;
}
.kpi-delta { font-size: 0.75rem; color: #36E2A4; margin-top: 0.1rem; }

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem; letter-spacing: 2px;
    text-transform: uppercase; color: #00D4FF;
    border-left: 3px solid #00D4FF;
    padding-left: 0.7rem; margin: 1.8rem 0 1rem 0;
}

/* ── Data source badge ── */
.source-badge {
    font-size: 0.65rem; color: #4A5568;
    font-family: 'Space Mono', monospace;
    margin-top: 0.3rem;
}

/* ── Anomaly tags ── */
.tag-anomaly {
    background: rgba(255,75,75,0.15);
    color: #FF4B4B; border: 1px solid rgba(255,75,75,0.4);
    padding: 0.1rem 0.5rem; border-radius: 6px;
    font-size: 0.68rem; font-family: 'Space Mono', monospace;
}
.tag-normal {
    background: rgba(54,226,164,0.1);
    color: #36E2A4; border: 1px solid rgba(54,226,164,0.3);
    padding: 0.1rem 0.5rem; border-radius: 6px;
    font-size: 0.68rem; font-family: 'Space Mono', monospace;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080C17 !important;
    border-right: 1px solid rgba(0,212,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# ── Colour palette (consistent across all charts) ─────────────────────────────
PALETTE = {
    "primary":  "#00D4FF",
    "green":    "#36E2A4",
    "amber":    "#FFB547",
    "red":      "#FF4B4B",
    "purple":   "#A78BFA",
    "regions":  px.colors.qualitative.Bold,
}

CLUSTER_COLORS = {
    "Mega-Cities":       "#00D4FF",
    "Advanced Urban":    "#36E2A4",
    "Emerging Metros":   "#FFB547",
    "Developing Hubs":   "#FF4B4B",
    "Mid-Tier Cities":   "#A78BFA",
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#C9D1E0"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        bgcolor="rgba(15,21,37,0.8)",
        bordercolor="rgba(0,212,255,0.2)",
        borderwidth=1,
    ),
)


# ── Data loading with caching ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(force_refresh: bool = False):
    use_cache = not force_refresh
    df = build_main_dataset(use_cache=use_cache)
    ts = build_viirs_timeseries()
    return df, ts


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family: Space Mono, monospace; font-size:1.1rem;
                    color:#00D4FF; letter-spacing:1px;'>🛰️ CONTROLS</div>
        <div style='font-size:0.65rem; color:#4A5568; margin-top:0.3rem;'>
            Urban Light Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Data refresh
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown('<div class="source-badge">World Bank API + NASA VIIRS</div>',
                    unsafe_allow_html=True)
    with col_b:
        refresh = st.button("⟳", help="Clear cache & re-fetch live data")

    if refresh:
        clear_cache()
        st.cache_data.clear()
        st.success("Cache cleared — reloading …")

    st.divider()

    # Filters
    st.markdown("**🌍 Region Filter**")
    all_regions = ["All"] + sorted({v["region"] for v in CITIES.values()})
    sel_region  = st.selectbox("Region", all_regions, label_visibility="collapsed")

    st.markdown("**🏙️ City Selection**")
    city_names  = sorted(CITIES.keys())
    sel_cities  = st.multiselect("Cities", city_names,
                                  default=["New York","London","Tokyo",
                                           "Shanghai","Mumbai","Lagos"],
                                  label_visibility="collapsed")

    st.markdown("**📅 Year Range (Radiance)**")
    yr_range = st.slider("", 2013, 2023, (2013, 2023), label_visibility="collapsed")

    st.markdown("**🔢 K-Means Clusters**")
    n_clusters = st.slider("", 3, 7, 5, label_visibility="collapsed")

    st.divider()
    st.markdown("""
    <div style='font-size:0.62rem; color:#4A5568; line-height:1.6;'>
    <b style='color:#7A8AAD;'>Data Sources</b><br>
    • NASA Black Marble VNP46A4<br>
    • World Bank Indicators API<br>
    • NOAA VIIRS DNB Annual V2.1<br><br>
    <b style='color:#7A8AAD;'>ML Methods</b><br>
    • Random Forest Regressor<br>
    • K-Means Clustering (PCA)<br>
    • Isolation Forest<br>
    • Linear Trend Forecast
    </div>
    """, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("🛰️ Fetching live data from World Bank API …"):
    df, ts = load_data(force_refresh=refresh)

# Apply region filter
if sel_region != "All":
    df_filt = df[df["region"] == sel_region].copy()
    ts_filt = ts[ts["region"] == sel_region].copy()
else:
    df_filt = df.copy()
    ts_filt = ts.copy()

# Filter time series by year range
ts_range = ts_filt[ts_filt["year"].between(*yr_range)]

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🛰️ URBAN LIGHT POLLUTION INTELLIGENCE</div>
    <div class="hero-sub">
        Data-Driven Analysis of Urban Populations Using Satellite Nighttime Radiance Signals
    </div>
    <div>
        <span class="hero-badge">NASA VIIRS</span>
        <span class="hero-badge">WORLD BANK API</span>
        <span class="hero-badge">MACHINE LEARNING</span>
        <span class="hero-badge">REAL-TIME DATA</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    (k1, len(df_filt), "Cities Tracked", "↑ live"),
    (k2, f'{df_filt["radiance_2023"].mean():.1f}', "Avg Radiance (nW/cm²/sr)", "VIIRS 2023"),
    (k3, f'{df_filt["radiance_growth"].mean():.1f}%', "Avg Radiance Growth", "2013→2023"),
    (k4, f'${df_filt["gdp_per_capita"].mean()/1000:.0f}k', "Avg GDP per Capita", "World Bank"),
    (k5, f'{df_filt["electricity_pct"].mean():.0f}%', "Avg Electrification", "live"),
]
for col, val, label, delta in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️  Global Map",
    "📈  Radiance Trends",
    "🤖  ML Insights",
    "🔍  Anomaly Radar",
    "🔮  Forecast 2030",
    "📊  Correlation Lab",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – GLOBAL MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Global Nighttime Radiance Map</div>',
                unsafe_allow_html=True)
    st.caption("Bubble size ∝ VIIRS radiance  |  Color = region  |  Hover for full stats")

    map_df = df_filt.dropna(subset=["lat","lon","radiance_2023"])

    fig_map = px.scatter_geo(
        map_df, lat="lat", lon="lon",
        size="radiance_2023",
        color="region",
        hover_name="city",
        hover_data={
            "radiance_2023": ":.1f",
            "population":    ":,.0f",
            "gdp_per_capita":":.0f",
            "electricity_pct":":.1f",
            "radiance_growth":":.1f",
            "lat": False, "lon": False,
        },
        labels={
            "radiance_2023":   "Radiance (nW/cm²/sr)",
            "population":      "Population",
            "gdp_per_capita":  "GDP/capita (USD)",
            "electricity_pct": "Electrification (%)",
            "radiance_growth": "Radiance Growth (%)",
        },
        size_max=35,
        color_discrete_sequence=PALETTE["regions"],
        projection="natural earth",
        title="",
    )
    fig_map.update_geos(
        showland=True,    landcolor="#0d1428",
        showocean=True,   oceancolor="#07101e",
        showlakes=True,   lakecolor="#07101e",
        showcoastlines=True, coastlinecolor="#1a2744",
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig_map.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Bottom twin charts
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">Top 15 Cities by Radiance 2023</div>',
                    unsafe_allow_html=True)
        top15 = map_df.nlargest(15, "radiance_2023")
        fig_bar = px.bar(
            top15, x="radiance_2023", y="city",
            orientation="h",
            color="region",
            color_discrete_sequence=PALETTE["regions"],
            labels={"radiance_2023": "Radiance (nW/cm²/sr)", "city": ""},
        )
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Radiance vs GDP per Capita</div>',
                    unsafe_allow_html=True)
        fig_scatter = px.scatter(
            map_df.dropna(subset=["gdp_per_capita"]),
            x="gdp_per_capita", y="radiance_2023",
            color="region",
            size="population",
            hover_name="city",
            trendline="ols",
            color_discrete_sequence=PALETTE["regions"],
            labels={
                "gdp_per_capita":  "GDP per Capita (USD)",
                "radiance_2023":   "Radiance (nW/cm²/sr)",
                "population":      "Population",
            },
        )
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown(
        '<div class="source-badge">📡 Radiance: NASA Black Marble VNP46A4 (VIIRS DNB Annual)  '
        '|  Economic: World Bank Indicators API (live)</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – RADIANCE TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">VIIRS Annual Radiance Trends 2013–2023</div>',
                unsafe_allow_html=True)

    if not sel_cities:
        st.info("Select cities in the sidebar to show trend lines.")
    else:
        ts_sel = ts_range[ts_range["city"].isin(sel_cities)]

        fig_trend = px.line(
            ts_sel, x="year", y="radiance", color="city",
            markers=True,
            labels={"radiance": "Radiance (nW/cm²/sr)", "year": "Year"},
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_trend.update_traces(line_width=2.2, marker_size=6)
        fig_trend.update_layout(
            **PLOTLY_LAYOUT, height=400,
            xaxis=dict(dtick=1, gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Regional average trend
    st.markdown('<div class="section-header">Regional Average Radiance</div>',
                unsafe_allow_html=True)

    reg_avg = (ts_range.groupby(["region","year"])["radiance"]
               .mean().reset_index())

    fig_reg = px.area(
        reg_avg, x="year", y="radiance", color="region",
        labels={"radiance": "Avg Radiance (nW/cm²/sr)", "year": "Year"},
        color_discrete_sequence=PALETTE["regions"],
    )
    fig_reg.update_layout(**PLOTLY_LAYOUT, height=360)
    st.plotly_chart(fig_reg, use_container_width=True)

    # Growth heatmap
    st.markdown('<div class="section-header">Year-on-Year Radiance Change Heatmap</div>',
                unsafe_allow_html=True)

    hm_df = (ts_range.set_index(["city","year"])["radiance"]
             .unstack("year").pct_change(axis=1).iloc[:, 1:] * 100)

    fig_hm = go.Figure(go.Heatmap(
        z=hm_df.values,
        x=[str(c) for c in hm_df.columns],
        y=hm_df.index.tolist(),
        colorscale=[[0,"#FF4B4B"],[0.5,"#0F1525"],[1,"#36E2A4"]],
        zmid=0,
        text=np.round(hm_df.values, 1),
        texttemplate="%{text}%",
        hovertemplate="City: %{y}<br>Year: %{x}<br>Change: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="YoY %", tickfont=dict(size=9)),
    ))
    fig_hm.update_layout(
        **PLOTLY_LAYOUT, height=max(300, len(hm_df) * 22),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ML INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Machine Learning Models</div>',
                unsafe_allow_html=True)

    col_ml1, col_ml2 = st.columns([1.2, 1])

    # K-Means PCA scatter
    with col_ml1:
        st.markdown("**K-Means City Clusters (PCA projection)**")
        cluster_df = cluster_cities(df_filt, n_clusters=n_clusters)

        if "pca_x" in cluster_df.columns:
            fig_pca = px.scatter(
                cluster_df.dropna(subset=["pca_x","pca_y"]),
                x="pca_x", y="pca_y",
                color="cluster_label",
                hover_name="city",
                hover_data={"radiance_2023":":.1f",
                            "gdp_per_capita":":.0f",
                            "pca_x":False, "pca_y":False},
                color_discrete_map=CLUSTER_COLORS,
                labels={"pca_x": "PC 1", "pca_y": "PC 2",
                        "cluster_label": "Cluster"},
                size_max=18,
                size="radiance_2023" if "radiance_2023" in cluster_df else None,
            )
            fig_pca.update_layout(**PLOTLY_LAYOUT, height=420)
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("Not enough data for clustering.")

    # RF feature importance
    with col_ml2:
        st.markdown("**Random Forest – Feature Importance**")
        rf_result = train_radiance_model(df_filt)

        if rf_result:
            imp = rf_result["importances"].reset_index()
            imp.columns = ["Feature", "Importance"]
            fig_imp = px.bar(
                imp, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=[[0,"#0d1f3c"],[1,"#00D4FF"]],
            )
            fig_imp.update_layout(
                **PLOTLY_LAYOUT, height=260,
                coloraxis_showscale=False,
                showlegend=False,
            )
            fig_imp.update_traces(marker_line_width=0)
            st.plotly_chart(fig_imp, use_container_width=True)

            # Model metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score",    f'{rf_result["r2"]:.3f}')
            m2.metric("MAE",         f'{rf_result["mae"]:.2f} nW')
            m3.metric("CV R² (mean)",f'{rf_result["cv_mean"]:.3f} ±{rf_result["cv_std"]:.3f}')
        else:
            st.info("Need more data rows for model training.")

    # Cluster profile radar
    st.markdown('<div class="section-header">Cluster Profiles</div>',
                unsafe_allow_html=True)

    if "cluster_label" in cluster_df.columns:
        profile_cols = [c for c in ["radiance_2023","gdp_per_capita",
                                     "electricity_pct","radiance_growth",
                                     "population"] if c in cluster_df.columns]
        profiles = cluster_df.groupby("cluster_label")[profile_cols].mean()

        # Normalise 0-1
        prof_norm = (profiles - profiles.min()) / (profiles.max() - profiles.min() + 1e-9)

        fig_radar = go.Figure()
        for i, (label, row) in enumerate(prof_norm.iterrows()):
            vals = row.tolist()
            vals += [vals[0]]
            cats  = profile_cols + [profile_cols[0]]
            color = list(CLUSTER_COLORS.values())[i % len(CLUSTER_COLORS)]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, name=label,
                fill="toself",
                line=dict(color=color, width=1.5),
                fillcolor="rgba(0, 212, 255, 0.08)",
            ))

        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor="rgba(255,255,255,0.06)",
                                tickfont=dict(size=8)),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # RF predicted vs actual
    if rf_result and len(rf_result.get("preds", [])) > 0:
        st.markdown('<div class="section-header">Predicted vs Actual Radiance</div>',
                    unsafe_allow_html=True)
        pva_df = pd.DataFrame({
            "Actual":    rf_result["actuals"],
            "Predicted": rf_result["preds"],
        })
        fig_pva = px.scatter(
            pva_df, x="Actual", y="Predicted",
            color_discrete_sequence=[PALETTE["primary"]],
            opacity=0.8,
            labels={"Actual":"Actual Radiance","Predicted":"Predicted Radiance"},
            trendline="ols",
        )
        mn = pva_df.min().min(); mx = pva_df.max().max()
        fig_pva.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                          line=dict(dash="dot", color=PALETTE["green"], width=1))
        fig_pva.update_layout(**PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig_pva, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – ANOMALY RADAR
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – ANOMALY RADAR
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Anomaly Detection – Isolation Forest</div>',
                unsafe_allow_html=True)
    st.caption("Cities where light pollution deviates significantly from economic expectations")

    anom_df = detect_anomalies(df_filt)

    if "is_anomaly" in anom_df.columns:

        # ---- FIX TYPES FOR STREAMLIT CLOUD ----
        anom_df["is_anomaly"] = anom_df["is_anomaly"].fillna(False).astype(str)

        if "anomaly_score" in anom_df.columns:
            anom_df["anomaly_score_abs"] = anom_df["anomaly_score"].abs()
        else:
            anom_df["anomaly_score_abs"] = 1

        c1, c2 = st.columns([2, 1])

        with c1:
            scatter_df = anom_df.dropna(subset=["gdp_per_capita", "radiance_2023"])

            fig_anom = px.scatter(
                scatter_df,
                x="gdp_per_capita",
                y="radiance_2023",
                color="is_anomaly",
                symbol="is_anomaly",
                hover_name="city",
                hover_data={"anomaly_score":":.3f"},
                color_discrete_map={
                    "True": PALETTE["red"],
                    "False": PALETTE["green"]
                },
                size="anomaly_score_abs",
                size_max=20,
                labels={
                    "gdp_per_capita": "GDP per Capita (USD)",
                    "radiance_2023": "Radiance (nW/cm²/sr)",
                    "is_anomaly": "Anomaly"
                },
            )

            fig_anom.update_layout(**PLOTLY_LAYOUT, height=420)
            st.plotly_chart(fig_anom, use_container_width=True)

        with c2:
            st.markdown("**Flagged Cities**")

            flagged = anom_df[anom_df["is_anomaly"] == "True"].sort_values("anomaly_score")

            if not flagged.empty:
                flagged = flagged[["city", "radiance_2023", "gdp_per_capita", "anomaly_score"]]

                for _, row in flagged.iterrows():
                    st.markdown(f"""
                    <div style='background:#0F1525; border:1px solid rgba(255,75,75,0.3);
                                border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.5rem;'>
                        <div style='font-weight:600; color:#E8EAF0; font-size:0.85rem;'>{row["city"]}</div>
                        <div style='font-size:0.72rem; color:#7A8AAD; margin-top:0.2rem;'>
                            Score: {row["anomaly_score"]:.3f}
                            &nbsp;·&nbsp;
                            Radiance: <b style='color:#00D4FF;'>{row["radiance_2023"]:.1f}</b> nW
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No anomalies detected.")

        st.markdown('<div class="section-header">Isolation Forest Scores</div>',
                    unsafe_allow_html=True)

        anom_sorted = anom_df.dropna(subset=["anomaly_score"]).sort_values("anomaly_score")

        fig_scores = px.bar(
            anom_sorted,
            x="city",
            y="anomaly_score",
            color="is_anomaly",
            color_discrete_map={
                "True": PALETTE["red"],
                "False": PALETTE["primary"]
            },
            labels={
                "anomaly_score": "Isolation Score",
                "city": "City",
                "is_anomaly": "Anomaly"
            },
        )

        fig_scores.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig_scores, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – FORECAST 2030
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Radiance Forecast 2024–2030</div>',
                unsafe_allow_html=True)
    st.caption("Linear trend extrapolation from VIIRS 2013-2023 observations")

    if not sel_cities:
        st.info("Select cities in the sidebar.")
    else:
        full_ts  = build_viirs_timeseries()
        fore_df  = forecast_radiance(full_ts[full_ts["city"].isin(sel_cities)])

        fig_fore = go.Figure()
        colors = px.colors.qualitative.Vivid

        for i, city in enumerate(sel_cities):
            c_df = fore_df[fore_df["city"] == city]
            hist = c_df[c_df["type"] == "historical"]
            fore = c_df[c_df["type"] == "forecast"]
            col  = colors[i % len(colors)]

            fig_fore.add_trace(go.Scatter(
                x=hist["year"], y=hist["radiance"],
                name=city, mode="lines+markers",
                line=dict(color=col, width=2),
                marker=dict(size=5),
            ))
            fig_fore.add_trace(go.Scatter(
                x=fore["year"], y=fore["radiance"],
                name=f"{city} (forecast)",
                mode="lines+markers",
                line=dict(color=col, width=2, dash="dash"),
                marker=dict(size=5, symbol="circle-open"),
                showlegend=False,
            ))

        fig_fore.add_vrect(
            x0=2023.5, x1=2030.5,
            fillcolor="rgba(0,212,255,0.03)",
            layer="below", line_width=0,
            annotation_text="Forecast Zone",
            annotation_font_color=PALETTE["primary"],
            annotation_font_size=10,
        )
        fig_fore.update_layout(
            **PLOTLY_LAYOUT, height=440,
            xaxis=dict(dtick=1, gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                       title="Radiance (nW/cm²/sr)"),
            xaxis_title="Year",
        )
        st.plotly_chart(fig_fore, use_container_width=True)

    # 2030 snapshot table
    st.markdown('<div class="section-header">2030 Radiance Snapshot</div>',
                unsafe_allow_html=True)

    full_ts  = build_viirs_timeseries()
    fore_all = forecast_radiance(full_ts)
    snap     = fore_all[(fore_all["year"] == 2030) & (fore_all["type"] == "forecast")]
    snap_2023= fore_all[(fore_all["year"] == 2023) & (fore_all["type"] == "historical")]
    snap     = snap.merge(snap_2023[["city","radiance"]], on="city", suffixes=("_2030","_2023"))
    snap["projected_change"] = ((snap["radiance_2030"] - snap["radiance_2023"])
                                 / snap["radiance_2023"] * 100)
    snap = snap.sort_values("radiance_2030", ascending=False)[
        ["city","region","radiance_2023","radiance_2030","projected_change"]
    ].rename(columns={
        "city": "City", "region": "Region",
        "radiance_2023": "2023 Radiance",
        "radiance_2030": "2030 Forecast",
        "projected_change": "Δ Change (%)"
    })

    st.dataframe(
       snap.round(2),
       use_container_width=True,
       hide_index=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 – CORRELATION LAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Correlation Laboratory</div>',
                unsafe_allow_html=True)

    num_cols = [c for c in ["radiance_2023","radiance_growth","gdp_per_capita",
                             "population","electricity_pct","urban_pct",
                             "co2_pc","light_per_capita","efficiency_score"]
                if c in df_filt.columns and df_filt[c].notna().sum() > 3]

    if len(num_cols) >= 3:
        corr = df_filt[num_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=[[0,"#FF4B4B"],[0.5,"#0F1525"],[1,"#00D4FF"]],
            zmin=-1, zmax=1,
            labels=dict(color="Pearson r"),
        )
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=480)
        fig_corr.update_traces(textfont_size=9)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Dual-axis: radiance vs electricity & GDP
    st.markdown('<div class="section-header">Electrification & GDP vs Radiance</div>',
                unsafe_allow_html=True)

    clean_df = df_filt.dropna(subset=["electricity_pct","gdp_per_capita","radiance_2023"])
    clean_df = clean_df.sort_values("radiance_2023")

    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(go.Bar(
        x=clean_df["city"], y=clean_df["radiance_2023"],
        name="Radiance 2023", marker_color=PALETTE["primary"],
        opacity=0.7, marker_line_width=0,
    ), secondary_y=False)
    fig_dual.add_trace(go.Scatter(
        x=clean_df["city"], y=clean_df["electricity_pct"],
        name="Electrification (%)", mode="lines+markers",
        line=dict(color=PALETTE["green"], width=2), marker_size=5,
    ), secondary_y=True)

    fig_dual.update_layout(PLOTLY_LAYOUT)

    fig_dual.update_layout(
       height=380,
       xaxis_tickangle=-45,
       legend=dict(orientation="h", y=1.05),
)
    fig_dual.update_yaxes(title_text="Radiance (nW/cm²/sr)", secondary_y=False,
                           gridcolor="rgba(255,255,255,0.04)")
    fig_dual.update_yaxes(title_text="Electrification (%)", secondary_y=True)
    st.plotly_chart(fig_dual, use_container_width=True)

    # Efficiency ranking
    st.markdown('<div class="section-header">Light Efficiency Ranking</div>',
                unsafe_allow_html=True)
    eff_df = compute_efficiency_index(df_filt)

    if len(eff_df) > 0:
        fig_eff = px.bar(
            eff_df.head(20), x="efficiency_score", y="city",
            orientation="h",
            color="efficiency_score",
            color_continuous_scale=[[0,"#0d1f3c"],[1,"#36E2A4"]],
            labels={"efficiency_score":"Efficiency Score (0-100)","city":""},
        )
        fig_eff.update_layout(**PLOTLY_LAYOUT, height=420,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_eff, use_container_width=True)
        st.caption("Efficiency = economic output (GDP/capita) per unit of nighttime radiance")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; padding:1rem; font-size:0.65rem; color:#4A5568;
            font-family: Space Mono, monospace;'>
    🛰️ URBAN LIGHT POLLUTION INTELLIGENCE &nbsp;·&nbsp;
    NASA Black Marble VNP46A4 &nbsp;·&nbsp; World Bank Indicators API &nbsp;·&nbsp;
    NOAA VIIRS DNB Annual V2.1<br>
    ML: Random Forest · K-Means · Isolation Forest · PCA · Linear Forecast
</div>
""", unsafe_allow_html=True)
