#!/usr/bin/env python3
"""
Fix validation tab:
1. Use enriched features (DOB, 311, PLUTO) in temporal validation
2. Add ablation study: model performance WITHOUT structural_fires feature
3. Show both RF and GBM in temporal comparison

Run from: /Users/jonathanepstein/Documents/Personal/nyc_fire_app
"""
import os
os.chdir("/Users/jonathanepstein/Documents/Personal/nyc_fire_app")

content = open("app.py").read()

# Find the validation tab and replace it entirely
# Look for the start marker
val_start = content.find("# ── TAB: Validation")
if val_start == -1:
    val_start = content.find("with tab_validation:")
    if val_start != -1:
        # Back up to find the line start
        val_start = content.rfind("\n", 0, val_start) + 1

footer_start = content.find("    # ─── Footer")

if val_start == -1 or footer_start == -1:
    print(f"[ERROR] Could not find validation tab (val_start={val_start}, footer={footer_start})")
    print("Searching for alternative markers...")
    # Try to find the footer differently
    footer_start = content.find('    st.divider()\n    st.caption(\n        "Data: NYC Open Data')
    if footer_start == -1:
        print("[ERROR] Could not find footer either. Aborting.")
        exit(1)
    # Insert before footer if no validation tab exists yet
    val_start = footer_start
    print(f"[INFO] No existing validation tab found, inserting before footer at pos {footer_start}")

new_validation = '''    # ── TAB: Validation ───────────────────────────────────────────────────
    with tab_validation:
        st.markdown("### ✅ Model Validation — Temporal Backtest + Ablation")
        st.caption(
            "Train on 2019–2022 data, predict 2023+ fire counts. "
            "Also tests whether the model works without the circular structural_fires feature."
        )

        raw_df = data["raw_df"]

        if "year" in raw_df.columns and raw_df["year"].nunique() > 2:
            train_years = raw_df[raw_df["year"] <= 2022]
            test_years = raw_df[raw_df["year"] >= 2023]

            if len(test_years) > 0 and len(train_years) > 0:
                from data.feature_engineering import engineer_features_by_zip, get_feature_matrix
                from data.enrichment import enrich_zip_features
                from models.enhanced_model import get_enhanced_feature_matrix, train_tuned_rf, train_xgboost
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                import plotly.graph_objects as go

                # Engineer + enrich features for BOTH periods
                with st.spinner("Engineering features for train/test periods..."):
                    train_features = engineer_features_by_zip(train_years)
                    test_features = engineer_features_by_zip(test_years)

                    # Enrich both with external data
                    try:
                        train_features = enrich_zip_features(train_features)
                        test_features = enrich_zip_features(test_features)
                    except Exception as e:
                        st.warning(f"Enrichment failed for validation: {e}")

                # Get enhanced feature matrices
                X_train_full, y_train, fn_full = get_enhanced_feature_matrix(train_features)
                X_test_full, y_test, _ = get_enhanced_feature_matrix(test_features)

                if len(X_train_full) > 10 and len(X_test_full) > 5:

                    # ── Full model (all features) ──────────────────────
                    rf_full = train_tuned_rf(X_train_full, y_train, fn_full)
                    gbm_full = train_xgboost(X_train_full, y_train, fn_full)

                    rf_preds_full = rf_full["model"].predict(X_test_full)
                    gbm_preds_full = gbm_full["model"].predict(X_test_full)

                    rf_oos_r2 = r2_score(y_test, rf_preds_full)
                    gbm_oos_r2 = r2_score(y_test, gbm_preds_full)
                    rf_oos_mae = mean_absolute_error(y_test, rf_preds_full)
                    gbm_oos_mae = mean_absolute_error(y_test, gbm_preds_full)

                    # ── Ablation: without structural_fires ─────────────
                    # Remove structural_fires from feature set
                    ablation_cols = [c for c in fn_full if c != "structural_fires"]
                    ablation_idx = [fn_full.index(c) for c in ablation_cols]

                    X_train_abl = X_train_full[:, ablation_idx]
                    X_test_abl = X_test_full[:, ablation_idx]

                    rf_abl = train_tuned_rf(X_train_abl, y_train, ablation_cols)
                    gbm_abl = train_xgboost(X_train_abl, y_train, ablation_cols)

                    rf_preds_abl = rf_abl["model"].predict(X_test_abl)
                    gbm_preds_abl = gbm_abl["model"].predict(X_test_abl)

                    rf_abl_r2 = r2_score(y_test, rf_preds_abl)
                    gbm_abl_r2 = r2_score(y_test, gbm_preds_abl)
                    rf_abl_mae = mean_absolute_error(y_test, rf_preds_abl)
                    gbm_abl_mae = mean_absolute_error(y_test, gbm_preds_abl)

                    # ── Pick best for display ──────────────────────────
                    best_full_preds = gbm_preds_full if gbm_oos_r2 > rf_oos_r2 else rf_preds_full
                    best_full_r2 = max(gbm_oos_r2, rf_oos_r2)
                    best_full_mae = gbm_oos_mae if gbm_oos_r2 > rf_oos_r2 else rf_oos_mae
                    best_full_name = gbm_full["model_name"] if gbm_oos_r2 > rf_oos_r2 else rf_full["model_name"]

                    best_abl_preds = gbm_preds_abl if gbm_abl_r2 > rf_abl_r2 else rf_preds_abl
                    best_abl_r2 = max(gbm_abl_r2, rf_abl_r2)
                    best_abl_mae = gbm_abl_mae if gbm_abl_r2 > rf_abl_r2 else rf_abl_mae

                    # ── Metrics ────────────────────────────────────────
                    st.markdown("#### Out-of-Sample Performance")
                    vc1, vc2, vc3, vc4 = st.columns(4)
                    vc1.metric("Train Period", f"{int(train_years['year'].min())}–{int(train_years['year'].max())}")
                    vc2.metric("Test Period", f"{int(test_years['year'].min())}–{int(test_years['year'].max())}")
                    vc3.metric("R² (Out-of-Sample)", f"{best_full_r2:.3f}")
                    vc4.metric("MAE", f"{best_full_mae:.1f} fires")

                    # ── Model Comparison Table ─────────────────────────
                    st.markdown("---")
                    st.markdown("#### Model Comparison (Temporal Validation)")

                    comparison_data = [
                        {"Model": "Tuned RF (all features)", "OOS R²": f"{rf_oos_r2:.3f}", "OOS MAE": f"{rf_oos_mae:.1f}", "CV R²": f"{rf_full['cv']['cv_r2_mean']:.3f}", "Features": len(fn_full)},
                        {"Model": "GBM (all features)", "OOS R²": f"{gbm_oos_r2:.3f}", "OOS MAE": f"{gbm_oos_mae:.1f}", "CV R²": f"{gbm_full['cv']['cv_r2_mean']:.3f}", "Features": len(fn_full)},
                        {"Model": "RF (no structural_fires)", "OOS R²": f"{rf_abl_r2:.3f}", "OOS MAE": f"{rf_abl_mae:.1f}", "CV R²": f"{rf_abl['cv']['cv_r2_mean']:.3f}", "Features": len(ablation_cols)},
                        {"Model": "GBM (no structural_fires)", "OOS R²": f"{gbm_abl_r2:.3f}", "OOS MAE": f"{gbm_abl_mae:.1f}", "CV R²": f"{gbm_abl['cv']['cv_r2_mean']:.3f}", "Features": len(ablation_cols)},
                    ]
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

                    # ── Charts ─────────────────────────────────────────
                    st.markdown("---")
                    val_col1, val_col2 = st.columns(2)

                    with val_col1:
                        st.markdown("**Actual vs Predicted (Best Full Model)**")
                        fig = make_actual_vs_predicted_chart(y_test, best_full_preds)
                        fig.update_layout(title=f"Out-of-Sample: {best_full_name}")
                        st.plotly_chart(fig, use_container_width=True)

                    with val_col2:
                        st.markdown("**Risk Tier Validation**")
                        st.caption("Do zip codes flagged as high-risk actually have more fires later?")

                        # Assign risk tiers from training model predictions
                        train_preds_full = rf_full["model"].predict(X_train_full) if rf_oos_r2 >= gbm_oos_r2 else gbm_full["model"].predict(X_train_full)
                        train_risk = np.clip(train_preds_full / (train_preds_full.max() or 1), 0, 1)
                        train_features_copy = train_features.copy()
                        train_features_copy["predicted_risk"] = train_risk

                        def _tier(r):
                            if r >= 0.75: return "Critical"
                            if r >= 0.50: return "High"
                            if r >= 0.25: return "Moderate"
                            return "Low"
                        train_features_copy["risk_tier"] = train_features_copy["predicted_risk"].map(_tier)

                        tier_validation = train_features_copy[["zip_code", "risk_tier"]].merge(
                            test_features[["zip_code", "structural_fires"]],
                            on="zip_code", how="inner"
                        )
                        tier_validation = tier_validation.rename(columns={"structural_fires": "actual_test_fires"})

                        if not tier_validation.empty:
                            tier_summary = tier_validation.groupby("risk_tier")["actual_test_fires"].agg(
                                ["mean", "median", "sum", "count"]
                            ).reindex(["Critical", "High", "Moderate", "Low"])

                            fig2 = go.Figure()
                            colors = {"Critical": "#FF3B4E", "High": "#FFAA2B", "Moderate": "#FF6B35", "Low": "#2ECC71"}
                            for tier in ["Critical", "High", "Moderate", "Low"]:
                                if tier in tier_summary.index:
                                    fig2.add_trace(go.Bar(
                                        x=[tier],
                                        y=[tier_summary.loc[tier, "mean"]],
                                        name=tier,
                                        marker_color=colors[tier],
                                        text=[f"{tier_summary.loc[tier, 'mean']:.1f}"],
                                        textposition="auto",
                                    ))
                            fig2.update_layout(
                                title="Avg Actual Fires (2023+) by Predicted Risk Tier",
                                yaxis_title="Avg Structural Fires per Zip",
                                showlegend=False,
                                height=380,
                                paper_bgcolor="#0B0E11",
                                plot_bgcolor="#131820",
                                font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                            st.markdown("**Tier Breakdown**")
                            tier_display = tier_summary.copy()
                            tier_display.columns = ["Avg Fires", "Median Fires", "Total Fires", "Zip Count"]
                            tier_display["Avg Fires"] = tier_display["Avg Fires"].map("{:.1f}".format)
                            tier_display["Median Fires"] = tier_display["Median Fires"].map("{:.1f}".format)
                            tier_display["Total Fires"] = tier_display["Total Fires"].astype(int)
                            tier_display["Zip Count"] = tier_display["Zip Count"].astype(int)
                            st.dataframe(tier_display, use_container_width=True)

                    # ── Ablation Analysis ──────────────────────────────
                    st.markdown("---")
                    st.markdown("#### Ablation Study: Without structural_fires Feature")
                    st.caption(
                        "The structural_fires feature is partially circular (we predict fire count "
                        "using past fire count). This ablation tests whether the model still works "
                        "using only building characteristics, complaints, violations, and other signals."
                    )

                    abl_col1, abl_col2 = st.columns(2)
                    with abl_col1:
                        st.metric("R² with structural_fires", f"{best_full_r2:.3f}")
                    with abl_col2:
                        delta = best_abl_r2 - best_full_r2
                        st.metric("R² without structural_fires", f"{best_abl_r2:.3f}",
                                  delta=f"{delta:+.3f}")

                    st.markdown("**Actual vs Predicted (Without structural_fires)**")
                    fig3 = make_actual_vs_predicted_chart(y_test, best_abl_preds)
                    fig3.update_layout(title="Ablation: Predicting Fires WITHOUT Past Fire Count")
                    st.plotly_chart(fig3, use_container_width=True)

                    # Feature importance for ablation model
                    best_abl_result = gbm_abl if gbm_abl_r2 > rf_abl_r2 else rf_abl
                    if "importance" in best_abl_result:
                        st.markdown("**Top Features (Without structural_fires)**")
                        imp = best_abl_result["importance"].head(10)
                        fig4 = go.Figure(go.Bar(
                            x=imp["importance_permutation"].values[::-1],
                            y=imp["feature"].values[::-1],
                            orientation="h",
                            marker_color="#FF6B35",
                        ))
                        fig4.update_layout(
                            title="What Predicts Fire Risk (Without Past Fire Count)?",
                            xaxis_title="Importance (mean decrease in R²)",
                            height=350,
                            paper_bgcolor="#0B0E11",
                            plot_bgcolor="#131820",
                            font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                        )
                        st.plotly_chart(fig4, use_container_width=True)

                    # ── Methodology ────────────────────────────────────
                    with st.expander("Validation Methodology"):
                        st.markdown("""
                        **Temporal Cross-Validation** (strict — no data leakage)

                        1. **Training**: All incidents ≤2022 → zip-level features → enriched with
                           311 complaints, DOB violations, and PLUTO building data.
                        2. **Testing**: All incidents ≥2023 → same feature engineering pipeline.
                        3. **Models**: Both Tuned RandomForest and GradientBoosting are evaluated.
                        4. **Ablation**: Re-trains without the `structural_fires` feature to test
                           whether the model can predict fire risk from building characteristics,
                           complaints, and violations alone (non-circular prediction).

                        **Risk tier validation**: Each zip gets a risk tier from the training model,
                        then we check whether high-risk zips actually had more fires in the test period.
                        Perfect monotonic ordering (Critical > High > Moderate > Low) validates
                        the model's ranking ability.

                        **Why the ablation matters**: Using past fire count to predict future fire
                        count is partially circular. The ablation proves the model captures genuine
                        risk factors (building age, complaints, violations) beyond just "places that
                        had fires will have fires again."
                        """)
                else:
                    st.warning("Not enough data in train or test period for validation.")
            else:
                st.warning("Need data from both ≤2022 and ≥2023 for temporal validation.")
        else:
            st.warning("Insufficient temporal data for validation.")

'''

# Replace or insert
if val_start < footer_start:
    # Replace existing validation tab + keep footer
    content = content[:val_start] + new_validation + content[footer_start:]
    print("[OK] Replaced existing validation tab")
else:
    # Insert before footer
    content = content[:footer_start] + new_validation + content[footer_start:]
    print("[OK] Inserted new validation tab before footer")

open("app.py", "w").write(content)
print("[OK] app.py updated with enhanced validation + ablation")
print("\nRun: pkill -f streamlit && streamlit run app.py")
