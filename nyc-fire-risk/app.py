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

# ─── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Fire Risk Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# ─── Data Loading & Model Training (cached) ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_train():
    """Full pipeline: fetch data, engineer features, train model."""
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

    return {
        "raw_df": df,
        "zip_features": zip_features,
        "puma_features": puma_features,
        "boro_features": boro_features,
        "model": model,
        "results": results,
        "X": X, "y": y,
        "zip_risk": zip_risk,
    }


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
    tab_map, tab_rankings, tab_model, tab_explorer = st.tabs(
        ["🗺️ Risk Map", "📊 Rankings", "🧠 Model", "🔍 Explorer"]
    )

    # ── TAB: Map ─────────────────────────────────────────────────────────
    with tab_map:
        col_map, col_side = st.columns([3, 1])

        with col_map:
            try:
                from streamlit_folium import st_folium
                fmap = create_folium_map(
                    active_df.reset_index(drop=True),
                    active_df["risk_score"].values,
                    label_col,
                    zoom=11 if granularity == "Zip Code" else (11 if granularity == "PUMA" else 10),
                )
                st_folium(fmap, width=None, height=520, returned_objects=[])
            except ImportError:
                st.info("Install `streamlit-folium` for interactive maps. Showing table view instead.")
                st.dataframe(
                    active_df[[label_col, "structural_fires", "risk_score", "risk_label"]]
                    .sort_values("risk_score", ascending=False),
                    use_container_width=True,
                )

        with col_side:
            st.markdown("**Highest Risk**")
            top5 = active_df.nlargest(5, "risk_score")
            for _, row in top5.iterrows():
                rl = row["risk_label"].lower()
                st.markdown(
                    f'<div style="padding:8px 0;border-bottom:1px solid #2A354822">'
                    f'<b>{row[label_col]}</b><br>'
                    f'<span class="risk-{rl}">{row["risk_label"]}</span> '
                    f'<span style="color:#7B8DA4;font-size:11px">'
                    f'{int(row["structural_fires"])} fires</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            # Citywide monthly
            month_cols = [f"month_{m}" for m in range(1, 13)]
            existing = [c for c in month_cols if c in active_df.columns]
            if existing:
                monthly_total = active_df[existing].sum().values
                fig = make_monthly_chart(monthly_total, "Citywide Monthly Pattern")
                st.plotly_chart(fig, use_container_width=True)

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

        st.dataframe(
            ranked[display_cols],
            use_container_width=True,
            height=500,
        )

        # Borough comparison
        if granularity != "Borough":
            st.markdown("### Borough Comparison")
            fig = make_borough_comparison_chart(boro_features)
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

        with col_m2:
            st.markdown("### Feature Importance")
            fig = make_feature_importance_chart(results["importance"])
            st.plotly_chart(fig, use_container_width=True)

            # Risk distribution
            st.markdown("### Risk Distribution")
            fig = make_risk_distribution_chart(active_df["risk_score"].values)
            st.plotly_chart(fig, use_container_width=True)

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
        selected = st.selectbox(f"Select {granularity}", zone_options)

        if selected:
            row = active_df[active_df[label_col] == selected].iloc[0]

            col_e1, col_e2 = st.columns([1, 2])

            with col_e1:
                rl = row["risk_label"].lower()
                st.markdown(
                    f'<div style="text-align:center;padding:20px">'
                    f'<div style="font-size:28px;font-weight:700">{selected}</div>'
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
                    fig = make_monthly_chart(monthly_vals, f"Monthly Distribution — {selected}")
                    st.plotly_chart(fig, use_container_width=True)

                # Show constituent zips for PUMA view
                if granularity == "PUMA" and "puma_code" in row.index:
                    puma_code = row["puma_code"]
                    constituent_zips = zip_features[zip_features["puma_code"] == puma_code]
                    if not constituent_zips.empty:
                        st.markdown(f"**Constituent Zip Codes** ({len(constituent_zips)})")
                        st.dataframe(
                            constituent_zips[["zip_code", "structural_fires", "risk_score", "risk_label"]]
                            .sort_values("risk_score", ascending=False),
                            use_container_width=True,
                            height=250,
                        )

    # ─── Footer ──────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Data: NYC Open Data — Incidents Responded to by Fire Companies (NYFIRS) · "
        "SODA API endpoint tm6d-hbzd · "
        "Model: scikit-learn RandomForest · "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()
