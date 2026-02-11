"""
🔥 NYC Fire Risk Prediction Dashboard

Multi-resolution fire risk prediction for New York City using
RandomForest on public FDNY incident data. Supports zip code,
PUMA, and borough granularity.
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fetch_data import FireDataPipeline
from data.feature_engineering import (
    engineer_features_by_zip,
    aggregate_to_puma,
    aggregate_to_borough,
    get_feature_matrix,
)
from models.fire_model import FireRiskModel
from utils.visualization import (
    COLORS, risk_color, risk_label,
    make_feature_importance_chart,
    make_actual_vs_predicted_chart,
    make_risk_distribution_chart,
    make_borough_comparison_chart,
    make_monthly_chart,
    create_folium_map,
)
from data.building_data import (
    fetch_pluto_buildings,
    process_pluto_buildings,
    score_buildings_with_neighborhood,
)

# ─── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Fire Risk Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State ───────────────────────────────────────────────────────
if "selected_zone" not in st.session_state:
    st.session_state.selected_zone = None

# ─── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&display=swap');

    .stApp { font-family: 'JetBrains Mono', monospace; }
    .risk-critical { background: #FF3B4E20; color: #FF3B4E; padding: 3px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; border: 1px solid #FF3B4E40; }
    .risk-high { background: #FFAA2B20; color: #FFAA2B; padding: 3px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; border: 1px solid #FFAA2B40; }
    .risk-moderate { background: #FF6B3520; color: #FF6B35; padding: 3px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; border: 1px solid #FF6B3540; }
    .risk-low { background: #2ECC7120; color: #2ECC71; padding: 3px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; border: 1px solid #2ECC7140; }
    div[data-testid="stSidebar"] { background: #0D1117; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading & Model Training (with disk cache) ────────────────────
import pickle
import os
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_FILE = CACHE_DIR / "model_cache.pkl"
CACHE_MAX_AGE_HOURS = 24  # Retrain after this many hours


def _cache_is_fresh():
    """Check if the disk cache exists and is recent enough."""
    if not CACHE_FILE.exists():
        return False
    import time
    age_hours = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
    return age_hours < CACHE_MAX_AGE_HOURS


def _save_cache(data):
    """Save model results to disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)


def _load_cache():
    """Load model results from disk."""
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_and_train(force_retrain=False):
    """
    Full pipeline: fetch data, engineer features, train model.

    Results are cached to disk so the model doesn't retrain on every
    app restart. Cache expires after CACHE_MAX_AGE_HOURS hours.
    Pass force_retrain=True to ignore the cache.
    """
    # Try loading from disk cache first
    if not force_retrain and _cache_is_fresh():
        try:
            cached = _load_cache()
            # Verify it has the expected keys
            if all(k in cached for k in ["zip_features", "model", "results"]):
                return cached
        except Exception:
            pass  # Cache corrupted, retrain

    # Full pipeline
    pipeline = FireDataPipeline()
    df = pipeline.fetch_and_process(limit=50000)

    zip_features = engineer_features_by_zip(df)
    puma_features = aggregate_to_puma(zip_features)
    boro_features = aggregate_to_borough(zip_features)

    # Train on zip-level data
    X, y, feature_names = get_feature_matrix(zip_features)
    model = FireRiskModel(n_estimators=100, max_depth=12)
    results = model.fit(X, y, feature_names)

    # Zip predictions
    zip_preds, zip_risk = model.predict_with_risk(X)
    zip_features["predicted_fires"] = zip_preds
    zip_features["risk_score"] = zip_risk
    zip_features["risk_label"] = zip_features["risk_score"].map(risk_label)

    # PUMA-level prediction
    X_puma, y_puma, fn_puma = get_feature_matrix(puma_features)
    if len(X_puma) > 5:
        puma_model = FireRiskModel(n_estimators=80, max_depth=8)
        puma_model.fit(X_puma, y_puma, fn_puma)
        puma_preds, puma_risk = puma_model.predict_with_risk(X_puma)
    else:
        puma_preds = y_puma
        puma_risk = np.clip(y_puma / (y_puma.max() or 1), 0, 1)
    puma_features["predicted_fires"] = puma_preds
    puma_features["risk_score"] = puma_risk
    puma_features["risk_label"] = puma_features["risk_score"].map(risk_label)

    # Borough-level
    max_sf = boro_features["structural_fires"].max() or 1
    boro_features["risk_score"] = np.clip(boro_features["structural_fires"] / max_sf, 0, 1)
    boro_features["risk_label"] = boro_features["risk_score"].map(risk_label)
    boro_features["predicted_fires"] = boro_features["structural_fires"]

    data = {
        "raw_df": df,
        "zip_features": zip_features,
        "puma_features": puma_features,
        "boro_features": boro_features,
        "model": model,
        "results": results,
        "X": X, "y": y,
        "zip_risk": zip_risk,
    }

    # Save to disk
    _save_cache(data)

    return data


# ─── Main App ────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔥 NYC Fire Risk")
        st.caption("PREDICTIVE INTELLIGENCE")
        st.divider()

        granularity = st.radio(
            "**Spatial Granularity**",
            ["Zip Code", "PUMA", "Borough"],
            help="Zip code (~180 zones), PUMA/neighborhood (~55), Borough (5)",
        )

        st.divider()
        st.markdown(
            '<div style="font-size:10px;color:#7B8DA4;line-height:1.6">'
            '<b style="color:#FF6B35">DATA</b><br>'
            'FDNY NYFIRS via NYC Open Data<br>'
            'SODA endpoint: tm6d-hbzd<br><br>'
            '<b style="color:#FF6B35">MODEL</b><br>'
            'RandomForest (n=100, depth=12)<br>'
            'scikit-learn + SHAP</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("🔄 Retrain Model", width="stretch",
                      help="Force re-fetch data and retrain. Use if data is stale."):
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
            st.cache_resource.clear()
            st.rerun()

    # Load data
    with st.spinner("Fetching FDNY data & training model..."):
        data = load_and_train()

    zip_features = data["zip_features"]
    puma_features = data["puma_features"]
    boro_features = data["boro_features"]
    model = data["model"]
    results = data["results"]

    # Active dataset
    if granularity == "Zip Code":
        active_df = zip_features.copy()
        label_col = "zip_code"
    elif granularity == "PUMA":
        active_df = puma_features.copy()
        label_col = "puma_name"
    else:
        active_df = boro_features.copy()
        label_col = "borough"

    # ─── Header ──────────────────────────────────────────────────────────
    st.markdown(f"## NYC Fire Risk — {granularity} Level")
    st.caption(
        f"{len(active_df)} zones · RandomForest (n=100) · "
        f"{results['train']['n_samples']} training samples"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zones Analyzed", len(active_df))
    c2.metric("Structural Fires", f"{int(active_df['structural_fires'].sum()):,}")
    c3.metric("Critical Zones", len(active_df[active_df["risk_score"] >= 0.75]))
    c4.metric("Model R²", f"{results['train']['r2']:.3f}")

    # ─── Tabs ────────────────────────────────────────────────────────────
    tab_map, tab_rankings, tab_model, tab_explorer, tab_buildings = st.tabs(
        ["🗺️ Risk Map", "📊 Rankings", "🧠 Model", "🔍 Explorer", "🏢 Buildings"]
    )

    # ── TAB: Map ─────────────────────────────────────────────────────────
    with tab_map:
        col_map, col_side = st.columns([3, 1])

        # Determine map center/zoom from selected zone
        selected = st.session_state.selected_zone
        map_center = None
        map_zoom = 11 if granularity == "Zip Code" else (11 if granularity == "PUMA" else 10)

        if selected is not None:
            sel_row = active_df[active_df[label_col] == selected]
            if not sel_row.empty:
                lat = sel_row.iloc[0].get("latitude")
                lng = sel_row.iloc[0].get("longitude")
                if pd.notna(lat) and pd.notna(lng):
                    map_center = [lat, lng]
                    map_zoom = 14 if granularity == "Zip Code" else (13 if granularity == "PUMA" else 11)

        with col_map:
            try:
                from streamlit_folium import st_folium
                fmap = create_folium_map(
                    active_df.reset_index(drop=True),
                    active_df["risk_score"].values,
                    label_col,
                    center=map_center,
                    zoom=map_zoom,
                    highlight_zone=selected,
                )
                st_folium(fmap, width=None, height=520, returned_objects=[])
            except ImportError:
                st.info("Install `streamlit-folium` for interactive maps. Showing table view instead.")
                st.dataframe(
                    active_df[[label_col, "structural_fires", "risk_score", "risk_label"]]
                    .sort_values("risk_score", ascending=False),
                    width="stretch",
                )

        with col_side:
            st.markdown("**Highest Risk** *(click to focus)*")
            top_zones = active_df.nlargest(8, "risk_score")
            for _, row in top_zones.iterrows():
                zone_id = row[label_col]
                rl = row["risk_label"]
                fires = int(row["structural_fires"])
                is_selected = (zone_id == selected)
                icon = "→ " if is_selected else ""

                if st.button(
                    f"{icon}{zone_id}  ·  {rl}  ·  {fires} fires",
                    key=f"risk_btn_{zone_id}",
                    width="stretch",
                ):
                    # Toggle: click again to deselect
                    st.session_state.selected_zone = zone_id if zone_id != selected else None
                    st.rerun()

            # Clear selection
            if selected is not None:
                if st.button("✕ Clear selection", key="clear_sel", width="stretch"):
                    st.session_state.selected_zone = None
                    st.rerun()

            st.markdown("---")

            # Detail card for selected zone
            if selected is not None:
                sel_data = active_df[active_df[label_col] == selected]
                if not sel_data.empty:
                    s = sel_data.iloc[0]
                    rl = s["risk_label"].lower()
                    st.markdown(
                        f'<div style="background:#131820;border:1px solid #2A3548;'
                        f'border-radius:8px;padding:12px;margin-bottom:12px">'
                        f'<div style="font-size:16px;font-weight:700">{selected}</div>'
                        f'<span class="risk-{rl}">{s["risk_label"]} Risk '
                        f'({s["risk_score"]:.2f})</span><br>'
                        f'<span style="color:#7B8DA4;font-size:11px">'
                        f'{int(s["structural_fires"])} structural fires · '
                        f'{int(s["total_incidents"])} total incidents · '
                        f'Fire rate: {s["structural_fire_rate"]:.1%}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Citywide monthly
            month_cols = [f"month_{m}" for m in range(1, 13)]
            existing = [c for c in month_cols if c in active_df.columns]
            if existing:
                monthly_total = active_df[existing].sum().values
                fig = make_monthly_chart(monthly_total, "Citywide Monthly Pattern")
                st.plotly_chart(fig, width="stretch")

    # ── TAB: Rankings ────────────────────────────────────────────────────
    with tab_rankings:
        ranked = active_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
        ranked.index += 1  # 1-based rank

        display_cols = [
            label_col, "risk_label", "structural_fires",
            "predicted_fires", "total_incidents",
            "structural_fire_rate", "risk_score",
        ]
        display_cols = [c for c in display_cols if c in ranked.columns]

        display_df = ranked[display_cols].copy()
        display_df["structural_fire_rate"] = display_df["structural_fire_rate"].map("{:.1%}".format)
        display_df["risk_score"] = display_df["risk_score"].map("{:.3f}".format)
        display_df["predicted_fires"] = display_df["predicted_fires"].map("{:.0f}".format)
        st.dataframe(display_df, width="stretch", height=500)

        # Borough comparison
        if granularity != "Borough":
            st.markdown("### Borough Comparison")
            fig = make_borough_comparison_chart(boro_features)
            st.plotly_chart(fig, width="stretch")

    # ── TAB: Model ───────────────────────────────────────────────────────
    with tab_model:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("### Performance Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("R² (Train)", f"{results['train']['r2']:.3f}")
            m2.metric("RMSE", f"{results['train']['rmse']:.1f}")
            m3.metric("MAE", f"{results['train']['mae']:.1f}")

            cv = results["cv"]
            st.metric(
                "R² (5-Fold CV)",
                f"{cv['cv_r2_mean']:.3f} ± {cv['cv_r2_std']:.3f}",
            )

            # Actual vs Predicted
            X, y = data["X"], data["y"]
            preds = model.predict(X)
            fig = make_actual_vs_predicted_chart(y, preds)
            st.plotly_chart(fig, width="stretch")

        with col_m2:
            st.markdown("### Feature Importance")
            fig = make_feature_importance_chart(results["importance"])
            st.plotly_chart(fig, width="stretch")

            # Risk distribution
            st.markdown("### Risk Distribution")
            fig = make_risk_distribution_chart(active_df["risk_score"].values)
            st.plotly_chart(fig, width="stretch")

        # Model description
        with st.expander("Model Architecture Details"):
            st.markdown("""
            **Algorithm**: RandomForest Regressor (scikit-learn)
            - 100 decision trees, max depth 12, min 3 samples per leaf
            - Bootstrap sampling with replacement
            - All CPU cores used for parallel training

            **Target Variable**: Count of structural fires (NFIRS codes 100–199)
            per geographic unit across the dataset time period.

            **Features** (15 engineered):
            - *Incident counts*: total incidents, structural fires, false alarms
            - *Rate features*: structural fire rate, false alarm rate, medical rate
            - *Response*: average units on scene
            - *Temporal*: year-over-year trend slope, incident volatility (std dev)
            - *Seasonal*: winter concentration (Dec–Feb), summer concentration (Jun–Aug)
            - *Peak load*: max monthly incidents
            - *Property mix*: residential %, commercial %
            - *Data coverage*: unique years, avg yearly incidents

            **Evaluation**: 5-fold cross-validation with R² scoring.
            Permutation importance computed with 10 repeats.

            **Spatial Levels**:
            - *Zip Code*: Finest granularity (~180 zones). Trained directly.
            - *PUMA*: Neighborhood-level (~55 zones). Features aggregated from zip level.
            - *Borough*: Coarsest (5 zones). Descriptive only.
            """)

    # ── TAB: Explorer ────────────────────────────────────────────────────
    with tab_explorer:
        st.markdown("### Zone Deep Dive")

        zone_options = active_df[label_col].tolist()
        selected_explorer = st.selectbox(f"Select {granularity}", zone_options)

        if selected_explorer:
            row = active_df[active_df[label_col] == selected_explorer].iloc[0]

            col_e1, col_e2 = st.columns([1, 2])

            with col_e1:
                rl = row["risk_label"].lower()
                st.markdown(
                    f'<div style="text-align:center;padding:20px">'
                    f'<div style="font-size:28px;font-weight:700">{selected_explorer}</div>'
                    f'<div style="margin-top:8px">'
                    f'<span class="risk-{rl}" style="font-size:14px;padding:6px 16px">'
                    f'{row["risk_label"]} Risk ({row["risk_score"]:.2f})</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.metric("Structural Fires", int(row["structural_fires"]))
                st.metric("Total Incidents", int(row["total_incidents"]))
                st.metric("Structural Fire Rate", f"{row['structural_fire_rate']:.1%}")
                st.metric("Predicted Fires", f"{row['predicted_fires']:.0f}")
                st.metric("Avg Units on Scene", f"{row['avg_units_onscene']:.1f}")

                if "trend_slope" in row.index:
                    trend = row["trend_slope"]
                    st.metric(
                        "Year-over-Year Trend",
                        f"{trend:+.1f} fires/yr",
                        delta=f"{'Increasing' if trend > 0 else 'Decreasing'}",
                        delta_color="inverse",
                    )

            with col_e2:
                # Monthly chart for this zone
                month_cols = [f"month_{m}" for m in range(1, 13)]
                existing = [c for c in month_cols if c in row.index]
                if existing:
                    monthly_vals = [row[c] for c in existing]
                    fig = make_monthly_chart(monthly_vals, f"Monthly Distribution — {selected_explorer}")
                    st.plotly_chart(fig, width="stretch")

                # Show constituent zips for PUMA view
                if granularity == "PUMA" and "puma_code" in row.index:
                    puma_code = row["puma_code"]
                    constituent_zips = zip_features[zip_features["puma_code"] == puma_code]
                    if not constituent_zips.empty:
                        st.markdown(f"**Constituent Zip Codes** ({len(constituent_zips)})")
                        cz = constituent_zips[["zip_code", "structural_fires", "risk_score", "risk_label"]].sort_values("risk_score", ascending=False).copy()
                        cz["risk_score"] = cz["risk_score"].map("{:.3f}".format)
                        st.dataframe(cz, width="stretch", height=250)

    # ── TAB: Buildings ───────────────────────────────────────────────────
    with tab_buildings:
        st.markdown("### 🏢 Building-Level Risk Explorer")
        st.caption(
            "Pulls building data from NYC PLUTO (870K+ tax lots) and scores each building "
            "using physical characteristics + neighborhood fire history."
        )

        bldg_col1, bldg_col2 = st.columns([1, 1])
        with bldg_col1:
            bldg_search_mode = st.radio(
                "Search by",
                ["Zip Code", "Borough"],
                horizontal=True,
                key="bldg_search_mode",
            )
        with bldg_col2:
            if bldg_search_mode == "Zip Code":
                available_zips = sorted(zip_features["zip_code"].tolist())
                bldg_zip = st.selectbox("Select Zip Code", available_zips, key="bldg_zip")
                bldg_borough = None
            else:
                bldg_borough = st.selectbox(
                    "Select Borough",
                    ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"],
                    key="bldg_boro",
                )
                bldg_zip = None

        bldg_limit = st.slider(
            "Max buildings to load",
            min_value=100, max_value=10000, value=2000, step=500,
            help="Higher = more comprehensive but slower to load",
            key="bldg_limit",
        )

        if st.button("🔍 Load & Score Buildings", key="load_buildings", width="stretch"):
            with st.spinner("Fetching building data from PLUTO API..."):
                raw_buildings = fetch_pluto_buildings(
                    zip_code=bldg_zip,
                    borough=bldg_borough,
                    limit=bldg_limit,
                )

            if raw_buildings.empty:
                st.warning("No building data returned. Try a different search.")
            else:
                with st.spinner(f"Processing {len(raw_buildings)} buildings..."):
                    buildings = process_pluto_buildings(raw_buildings)

                with st.spinner("Scoring buildings with neighborhood risk..."):
                    scored = score_buildings_with_neighborhood(buildings, zip_features)

                st.session_state["scored_buildings"] = scored
                st.success(f"Scored {len(scored)} buildings.")

        # Display results if we have them
        if "scored_buildings" in st.session_state:
            scored = st.session_state["scored_buildings"]

            # Summary metrics
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Buildings Scored", f"{len(scored):,}")
            bc2.metric("Critical Risk", len(scored[scored["risk_score"] >= 0.75]))
            bc3.metric("Avg Building Age", f"{scored['building_age'].mean():.0f} yrs")
            bc4.metric("Pre-1968 Code", f"{scored['is_pre_code'].mean():.0%}")

            st.markdown("---")

            # Map + Table layout
            bldg_map_col, bldg_info_col = st.columns([3, 2])

            with bldg_map_col:
                st.markdown("**Building Risk Map**")
                try:
                    from streamlit_folium import st_folium
                    import folium
                    from branca.colormap import LinearColormap

                    center_lat = scored["latitude"].mean()
                    center_lng = scored["longitude"].mean()
                    bldg_map = folium.Map(
                        location=[center_lat, center_lng],
                        zoom_start=14,
                        tiles="CartoDB dark_matter",
                    )

                    colormap = LinearColormap(
                        colors=["#2ECC71", "#FF6B35", "#FFAA2B", "#FF3B4E"],
                        vmin=0, vmax=1,
                        caption="Building Fire Risk",
                    )

                    # Sample if too many for performance
                    display_df = scored if len(scored) <= 2000 else scored.sample(2000, random_state=42)

                    for _, bldg in display_df.iterrows():
                        lat = bldg.get("latitude", 0)
                        lng = bldg.get("longitude", 0)
                        if lat == 0 or lng == 0:
                            continue

                        risk = bldg["risk_score"]
                        rc = risk_color(risk)
                        radius = 3 + risk * 8

                        popup_html = (
                            f'<div style="font-family:monospace;font-size:11px;min-width:180px">'
                            f'<b>{bldg.get("address", "N/A")}</b><br>'
                            f'BBL: {bldg.get("bbl", "N/A")}<br>'
                            f'Risk: <span style="color:{rc}">{bldg["risk_label"]} ({risk:.2f})</span><br>'
                            f'Built: {int(bldg["yearbuilt"])} ({int(bldg["building_age"])} yrs)<br>'
                            f'Floors: {int(bldg["numfloors"])} · Units: {int(bldg["unitsres"])}<br>'
                            f'Area: {int(bldg["bldgarea"]):,} sqft<br>'
                            f'Class: {bldg.get("bldgclass", "N/A")}'
                            f'</div>'
                        )

                        folium.CircleMarker(
                            location=[lat, lng],
                            radius=radius,
                            color=rc,
                            fill=True,
                            fill_color=rc,
                            fill_opacity=0.6,
                            popup=folium.Popup(popup_html, max_width=250),
                            tooltip=f'{bldg.get("address", "N/A")} — {bldg["risk_label"]}',
                        ).add_to(bldg_map)

                    colormap.add_to(bldg_map)
                    st_folium(bldg_map, width=None, height=450, returned_objects=[])

                except ImportError:
                    st.info("Install `streamlit-folium` for map view.")

            with bldg_info_col:
                st.markdown("**Highest Risk Buildings**")
                top_bldgs = scored.nlargest(15, "risk_score")
                display_bldg_cols = ["address", "risk_label", "risk_score", "yearbuilt",
                                      "numfloors", "unitsres", "bldgclass"]
                display_bldg_cols = [c for c in display_bldg_cols if c in top_bldgs.columns]
                top_display = top_bldgs[display_bldg_cols].copy()
                top_display["risk_score"] = top_display["risk_score"].map("{:.3f}".format)
                top_display["yearbuilt"] = top_display["yearbuilt"].astype(int)
                st.dataframe(top_display, width="stretch", height=350)

                # Risk breakdown
                st.markdown("**Risk Distribution**")
                risk_counts = scored["risk_label"].value_counts()
                for label in ["Critical", "High", "Moderate", "Low"]:
                    count = risk_counts.get(label, 0)
                    pct = count / len(scored) * 100
                    color = {"Critical": "#FF3B4E", "High": "#FFAA2B",
                             "Moderate": "#FF6B35", "Low": "#2ECC71"}[label]
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:4px 0">'
                        f'<span style="color:{color};font-weight:600">{label}</span>'
                        f'<span style="color:#7B8DA4">{count:,} ({pct:.1f}%)</span></div>',
                        unsafe_allow_html=True,
                    )

            # Building search
            st.markdown("---")
            st.markdown("**Search Building by Address**")
            addr_search = st.text_input("Enter address (partial match)", key="addr_search")
            if addr_search and len(addr_search) >= 3:
                matches = scored[scored["address"].str.contains(addr_search.upper(), na=False)]
                if matches.empty:
                    st.info("No matches found.")
                else:
                    st.write(f"Found {len(matches)} match(es):")
                    for _, bldg in matches.head(5).iterrows():
                        rl = bldg["risk_label"].lower()
                        st.markdown(
                            f'<div style="background:#131820;border:1px solid #2A3548;'
                            f'border-radius:8px;padding:12px;margin-bottom:8px">'
                            f'<b>{bldg.get("address", "N/A")}</b> '
                            f'<span class="risk-{rl}">{bldg["risk_label"]}</span><br>'
                            f'<span style="color:#7B8DA4;font-size:11px">'
                            f'BBL: {bldg.get("bbl", "N/A")} · '
                            f'Built {int(bldg["yearbuilt"])} ({int(bldg["building_age"])} yrs) · '
                            f'{int(bldg["numfloors"])} floors · '
                            f'{int(bldg["unitsres"])} units · '
                            f'{int(bldg["bldgarea"]):,} sqft · '
                            f'Class {bldg.get("bldgclass", "N/A")}'
                            f'</span></div>',
                            unsafe_allow_html=True,
                        )

    # ─── Footer ──────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Data: NYC Open Data — Incidents Responded to by Fire Companies (NYFIRS) · "
        "PLUTO (Primary Land Use Tax Lot Output) · "
        "SODA API endpoints tm6d-hbzd, 64uk-42ks · "
        "Model: scikit-learn RandomForest · "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()
