# 🛰️ Urban Light Pollution Intelligence Dashboard

> **Data-Driven Analysis of Urban Populations Using Satellite Nighttime Radiance Signals**

A production-grade interactive dashboard combining **real-time World Bank economic data** with **NASA VIIRS satellite nighttime radiance** observations to analyse urban growth, energy consumption, and light pollution trends across 30 global cities — powered by machine learning.

---

## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📡 Real-Time Data Sources

| Source | Data | Access |
|--------|------|--------|
| **NASA Black Marble (VNP46A4)** | Annual nighttime radiance 2013–2023 (nW/cm²/sr) | Public, no key required |
| **NOAA VIIRS DNB Annual V2.1** | Cloud-free BRDF-corrected radiance composites | Public domain |
| **World Bank Indicators API** | Population, GDP/capita, electrification, CO₂, urbanisation | Free REST API, no key |

> The World Bank data is fetched **live** on first load and cached for 1 hour.  
> NASA VIIRS radiance is pre-tabulated from the VNP46A4 product (0.1° city extracts).

---

## 🤖 Machine Learning Models

| Model | Purpose |
|-------|---------|
| **Random Forest Regressor** | Predict city radiance from economic signals; feature importance |
| **K-Means Clustering (k=3-7)** | Group cities by light+economic profile |
| **PCA (2-D)** | Visualise cluster separation |
| **Isolation Forest** | Flag over/under-lit cities relative to economic baseline |
| **Linear Trend Extrapolation** | Per-city radiance forecast to 2030 |

---

## 📊 Dashboard Tabs

1. **🗺️ Global Map** — Bubble map of radiance with hover stats; top-15 bar chart; radiance vs GDP scatter
2. **📈 Radiance Trends** — Multi-city line trends; regional area chart; YoY heatmap
3. **🤖 ML Insights** — PCA cluster plot; RF feature importance; predicted vs actual; radar profiles
4. **🔍 Anomaly Radar** — Isolation Forest scatter; anomaly score bar; over/under-lit flags
5. **🔮 Forecast 2030** — Historical + forecast lines; 2030 snapshot table
6. **📊 Correlation Lab** — Pearson correlation heatmap; dual-axis radiance/electrification; efficiency ranking

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/urban-light-pollution.git
cd urban-light-pollution

# 2. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.  
On first run it fetches live World Bank data (~15 s), then caches it for 1 hour.

---

## 🌍 Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, file `app.py`.
4. Click **Deploy** — done!

No secrets or environment variables are required (all APIs are open).

---

## 📁 Project Structure

```
urban-light-pollution/
│
├── app.py                        # Main Streamlit dashboard
│
├── data/
│   ├── __init__.py
│   ├── fetch_realtime.py         # World Bank API + VIIRS data layer
│   └── cache/                    # Auto-created; holds API response cache
│
├── models/
│   ├── __init__.py
│   └── ml_engine.py              # All ML models (RF, KMeans, IsoForest, etc.)
│
├── .streamlit/
│   └── config.toml               # Dark theme + Streamlit server config
│
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration & Customisation

### Add more cities
Edit `CITIES` and `VIIRS_RADIANCE` dictionaries in `data/fetch_realtime.py`.  
VIIRS values can be extracted from the public [EOG GeoTIFF downloads](https://eogdata.mines.edu/products/vnl/) or via NASA LAADS DAAC.

### Force data refresh
Click the **⟳** button in the sidebar, or run:
```python
from data.fetch_realtime import clear_cache
clear_cache()
```

### Change cluster count
Use the **K-Means Clusters** slider in the sidebar (3–7).

---

## 📐 Key Metrics Explained

| Metric | Definition |
|--------|-----------|
| **Radiance (nW/cm²/sr)** | VIIRS DNB median annual radiance; proxy for artificial light intensity |
| **Radiance Growth** | (radiance₂₀₂₃ − radiance₂₀₁₃) / radiance₂₀₁₃ × 100% |
| **Light per Capita** | radiance / (population / 1 million) |
| **Efficiency Score** | GDP per capita / radiance, normalised 0-100; higher = more output per unit of light |
| **Anomaly Score** | Isolation Forest output; lower = more anomalous |

---

## 📜 Data Citations

- Román, M.O. et al. (2018). *NASA's Black Marble nighttime lights product suite*. Remote Sensing of Environment, 210, 113–143.
- Elvidge, C.D. et al. (2021). *Annual time series of global VIIRS nighttime lights*. Remote Sensing, 13(5), 922.
- World Bank (2024). *World Development Indicators*. https://data.worldbank.org

---

## 📄 License

MIT © 2024
