# 🔥 NYC Fire Risk Prediction Dashboard

A multi-resolution fire risk prediction system for New York City, built with Python, scikit-learn, and Streamlit. Uses public NYC Open Data to train a RandomForest model that predicts structural fire risk at **zip code**, **PUMA (Public Use Microdata Area)**, and **borough** granularity levels.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project ingests FDNY incident data from the NYC Open Data SODA API, engineers features from multiple public datasets, and trains a RandomForest regressor to predict structural fire counts across NYC geographies. The interactive Streamlit dashboard lets users explore risk at three spatial resolutions.

### What Makes This Different

| Feature | This Project | Prior Work (e.g., Christiansen 2019) |
|---|---|---|
| Spatial resolution | Zip → PUMA → Borough (toggle) | Census tract only |
| PUMA-level analysis | ✅ First to use PUMAs for fire risk | ❌ |
| Feature sources | NYFIRS + DOB violations + 311 + building age | NYFIRS + PLUTO + ACS |
| Temporal features | Year-over-year trend, seasonal concentration, rolling averages | Cross-sectional only |
| Deployment | Live Streamlit app with real-time API ingestion | Static notebook |
| Model explainability | SHAP values + permutation importance | Feature importance only |

### Data Sources

All data is sourced from public APIs — no API key required:

| Dataset | Source | API Endpoint |
|---|---|---|
| FDNY Fire Incidents (NYFIRS) | NYC Open Data | `tm6d-hbzd` |
| DOB Violations | NYC Open Data | `3h2n-5cm9` |
| 311 Complaints | NYC Open Data | `erm2-nwe9` |
| PUMA Boundaries | NYC Dept. of City Planning | GeoJSON |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/nyc-fire-risk.git
cd nyc-fire-risk
pip install -r requirements.txt
```

## Usage

### Run locally
```bash
streamlit run app.py
```

### Data pipeline only
```python
from data.fetch_data import FireDataPipeline

pipeline = FireDataPipeline()
df = pipeline.fetch_and_process()
```

## Project Structure

```
nyc-fire-risk/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml           # Streamlit theme config
├── data/
│   ├── fetch_data.py          # NYC Open Data API ingestion
│   ├── feature_engineering.py # Feature construction at all granularities
│   └── puma_mapping.py        # Zip-to-PUMA crosswalk
├── models/
│   └── fire_model.py          # RandomForest training, evaluation, SHAP
├── utils/
│   └── visualization.py       # Map rendering, charts, color scales
└── pages/
    ├── 1_🗺️_Risk_Map.py      # Interactive map view
    ├── 2_📊_Model_Performance.py  # Metrics, feature importance, SHAP
    └── 3_🔍_Zone_Explorer.py  # Deep dive into individual zones
```

## Model Details

- **Algorithm**: RandomForest Regressor (100 trees, max depth 12)
- **Target**: Structural fire count per geographic unit per year
- **Features**: 15+ engineered features including incident history, building characteristics, seasonal patterns, and complaint density
- **Evaluation**: Temporal train/test split (train on 2013–2021, test on 2022+)
- **Explainability**: SHAP TreeExplainer for global and local interpretability

## License

MIT
