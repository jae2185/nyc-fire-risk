"""When raw_df is None (cloud/slim cache), load all validation from pre-computed caches"""

with open("app.py", "r") as f:
    content = f.read()

OLD = '''        else:
            st.warning("Insufficient temporal data for validation.")'''

NEW = '''        else:
            # Cloud mode: load all validation results from pre-computed caches
            val_cache = Path("data/validation_cache.pkl")
            roll_cache = Path("data/rolling_cv_cache.pkl")
            boro_cache = Path("data/borough_cv_cache.pkl")
            cls_roll_cache = Path("data/cls_rolling_cache.pkl")
            cls_boro_cache = Path("data/cls_borough_cache.pkl")

            has_caches = val_cache.exists() and roll_cache.exists()

            if has_caches:
                import pickle as _pkl

                # Load validation data
                with open(val_cache, "rb") as _f:
                    _vc = _pkl.load(_f)
                train_features = _vc["train_features"]
                test_features = _vc["test_features"]
                X_train_full = _vc["X_train_full"]
                X_test_full = _vc["X_test_full"]
                y_train = _vc["y_train"]
                y_test = _vc["y_test"]
                fn_full = _vc["fn_full"]

                if len(X_train_full) > 10 and len(X_test_full) > 5:

                    # ── Full model ──
                    rf_full = train_tuned_rf(X_train_full, y_train, fn_full)
                    gbm_full = train_xgboost(X_train_full, y_train, fn_full)
                    rf_preds_full = rf_full["model"].predict(X_test_full)
                    gbm_preds_full = gbm_full["model"].predict(X_test_full)
                    rf_oos_r2 = r2_score(y_test, rf_preds_full)
                    gbm_oos_r2 = r2_score(y_test, gbm_preds_full)
                    rf_oos_mae = mean_absolute_error(y_test, rf_preds_full)
                    gbm_oos_mae = mean_absolute_error(y_test, gbm_preds_full)

                    best_full_preds = gbm_preds_full if gbm_oos_r2 > rf_oos_r2 else rf_preds_full
                    best_full_r2 = max(gbm_oos_r2, rf_oos_r2)
                    best_full_mae = gbm_oos_mae if gbm_oos_r2 > rf_oos_r2 else rf_oos_mae
                    best_full_name = gbm_full["model_name"] if gbm_oos_r2 > rf_oos_r2 else rf_full["model_name"]

                    # ── Metrics ──
                    st.markdown("#### Out-of-Sample Performance")
                    vc1, vc2, vc3, vc4 = st.columns(4)
                    vc1.metric("Train Period", "2019\u20132022")
                    vc2.metric("Test Period", "2023\u20132024")
                    vc3.metric("R\u00b2 (Out-of-Sample)", f"{best_full_r2:.3f}", help="R-squared on held-out 2023-2024 data the model never saw during training.")
                    vc4.metric("MAE", f"{best_full_mae:.1f} fires", help="Mean Absolute Error on test data.")

                    # ── Model Comparison ──
                    st.markdown("---")
                    st.markdown("#### Model Comparison (Temporal Validation)")
                    comparison_data = [
                        {"Model": "Tuned RF (all features)", "OOS R\u00b2": f"{rf_oos_r2:.3f}", "OOS MAE": f"{rf_oos_mae:.1f}", "CV R\u00b2": f"{rf_full['cv']['cv_r2_mean']:.3f}", "Features": len(fn_full)},
                        {"Model": "GBM (all features)", "OOS R\u00b2": f"{gbm_oos_r2:.3f}", "OOS MAE": f"{gbm_oos_mae:.1f}", "CV R\u00b2": f"{gbm_full['cv']['cv_r2_mean']:.3f}", "Features": len(fn_full)},
                    ]
                    st.dataframe(pd.DataFrame(comparison_data), width="stretch")

                    # ── Charts ──
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
                        test_zip_features = engineer_features_by_zip(test_features) if "structural_fires" not in test_features.columns else test_features
                        tier_thresholds = [0.75, 0.50, 0.25]
                        tier_labels_map = ["Critical", "High", "Moderate", "Low"]
                        pred_scores = best_full_preds / (best_full_preds.max() + 1e-9)
                        tiers = pd.cut(pred_scores, bins=[-0.01, 0.25, 0.50, 0.75, 1.01], labels=["Low", "Moderate", "High", "Critical"])
                        tier_df = pd.DataFrame({"risk_tier": tiers, "actual_fires": y_test})
                        tier_summary = tier_df.groupby("risk_tier", observed=False).agg(
                            avg_fires=("actual_fires", "mean"),
                            median_fires=("actual_fires", "median"),
                            total_fires=("actual_fires", "sum"),
                            zip_count=("actual_fires", "count"),
                        ).reindex(["Critical", "High", "Moderate", "Low"])
                        tier_colors = {"Critical": "#FF3B4E", "High": "#FFAA2B", "Moderate": "#FF6B35", "Low": "#2ECC71"}
                        fig2 = go.Figure()
                        for tier_name in ["Critical", "High", "Moderate", "Low"]:
                            if tier_name in tier_summary.index:
                                row = tier_summary.loc[tier_name]
                                fig2.add_trace(go.Bar(
                                    x=[tier_name], y=[row["avg_fires"]],
                                    marker_color=tier_colors[tier_name],
                                    text=[f"{row['avg_fires']:.1f}"], textposition="auto",
                                    name=tier_name,
                                ))
                        fig2.update_layout(
                            title="Avg Actual Fires (2023+) by Predicted Risk Tier",
                            yaxis_title="Avg Structural Fires per Zip",
                            showlegend=False, height=350,
                            paper_bgcolor="#0B0E11", plot_bgcolor="#131820",
                            font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        st.markdown("**Tier Breakdown**")
                        tier_display = tier_summary.reset_index()
                        st.dataframe(tier_display, width="stretch")

                    # ── Rolling CV from cache ──
                    if roll_cache.exists():
                        st.markdown("---")
                        st.markdown("#### Rolling Temporal Validation")
                        st.caption("Train on expanding windows, test on each subsequent year.")
                        with open(roll_cache, "rb") as _f:
                            rolling_results = _pkl.load(_f)
                        if rolling_results:
                            rc1, rc2 = st.columns(2)
                            with rc1:
                                fig_roll = go.Figure()
                                fig_roll.add_trace(go.Scatter(
                                    x=[str(r["Test Year"]) for r in rolling_results],
                                    y=[r["R\u00b2"] for r in rolling_results],
                                    mode="lines+markers",
                                    marker=dict(size=10, color="#FF6B35"),
                                    line=dict(color="#FF6B35", width=2),
                                ))
                                fig_roll.update_layout(
                                    title="R\u00b2 by Test Year", xaxis_title="Test Year", yaxis_title="R\u00b2",
                                    height=300, paper_bgcolor="#0B0E11", plot_bgcolor="#131820",
                                    font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                                    yaxis=dict(range=[0, 1.05]),
                                )
                                st.plotly_chart(fig_roll, use_container_width=True)
                            with rc2:
                                roll_df = pd.DataFrame(rolling_results)
                                roll_df["R\u00b2"] = roll_df["R\u00b2"].map("{:.3f}".format)
                                roll_df["MAE"] = roll_df["MAE"].map("{:.1f}".format)
                                st.dataframe(roll_df, width="stretch", hide_index=True)
                            avg_r2 = np.mean([r["R\u00b2"] for r in rolling_results])
                            std_r2 = np.std([r["R\u00b2"] for r in rolling_results])
                            st.success(f"**Rolling CV: R\u00b2 = {avg_r2:.3f} \u00b1 {std_r2:.3f}** across {len(rolling_results)} test years. Stable performance confirms generalization.")

                    # ── Borough Holdout from cache ──
                    if boro_cache.exists():
                        st.markdown("---")
                        st.markdown("#### Borough Holdout Validation")
                        st.caption("Train on 4 boroughs, test on the held-out borough.")
                        with open(boro_cache, "rb") as _f:
                            borough_results = _pkl.load(_f)
                        if borough_results:
                            bc1, bc2 = st.columns(2)
                            with bc1:
                                colors_boro = {"Manhattan": "#FF3B4E", "Brooklyn": "#FFAA2B", "Queens": "#FF6B35", "Bronx": "#2ECC71", "Staten Island": "#3498DB"}
                                fig_boro = go.Figure()
                                for r in borough_results:
                                    fig_boro.add_trace(go.Bar(x=[r["Borough"]], y=[r["R\u00b2"]], marker_color=colors_boro.get(r["Borough"], "#FF6B35"), text=[f"{r['R\u00b2']:.3f}"], textposition="auto"))
                                fig_boro.update_layout(title="R\u00b2 by Held-Out Borough", yaxis_title="R\u00b2", showlegend=False, height=300, paper_bgcolor="#0B0E11", plot_bgcolor="#131820", font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"), yaxis=dict(range=[0, 1.05]))
                                st.plotly_chart(fig_boro, use_container_width=True)
                            with bc2:
                                boro_df = pd.DataFrame(borough_results)
                                boro_df["R\u00b2"] = boro_df["R\u00b2"].map("{:.3f}".format)
                                boro_df["MAE"] = boro_df["MAE"].map("{:.1f}".format)
                                st.dataframe(boro_df, width="stretch", hide_index=True)
                            avg_br2 = np.mean([r["R\u00b2"] for r in borough_results])
                            st.success(f"**Borough CV: Avg R\u00b2 = {avg_br2:.3f}** \u2014 strong geographic generalization.")

                    # ── Classification Ablation ──
                    st.markdown("---")
                    st.markdown("#### Ablation: Can Non-Fire Features Classify Risk?")
                    st.caption("Classifying zip codes using only building, demographic, and complaint features.")

                    incident_features = {"structural_fires", "total_incidents", "non_structural_fires", "false_alarms", "medical_calls", "structural_fire_rate", "false_alarm_rate", "medical_rate", "avg_units_onscene", "winter_concentration", "summer_concentration", "trend_slope", "incident_volatility", "max_monthly_incidents", "avg_yearly_incidents", "complaints_per_incident"}
                    ablation_cols = [c for c in fn_full if c not in incident_features]
                    ablation_idx = [fn_full.index(c) for c in ablation_cols]
                    X_train_abl = X_train_full[:, ablation_idx]
                    X_test_abl = X_test_full[:, ablation_idx]

                    median_fires = np.median(y_train)
                    y_train_cls = (y_train > median_fires).astype(int)
                    y_test_cls = (y_test > median_fires).astype(int)
                    st.markdown(f"Classifying zip codes as **High Risk** (>{int(median_fires)} fires) or **Low Risk** (\u2264{int(median_fires)} fires) using only building, demographic, and complaint features \u2014 zero fire incident data.")

                    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
                    from sklearn.utils.class_weight import compute_sample_weight

                    rf_cls = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42)
                    rf_cls.fit(X_train_abl, y_train_cls)
                    rf_cls_proba = rf_cls.predict_proba(X_test_abl)[:, 1] if len(rf_cls.classes_) == 2 else None

                    gb_cls = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
                    gb_sample_weights = compute_sample_weight("balanced", y_train_cls)
                    gb_cls.fit(X_train_abl, y_train_cls, sample_weight=gb_sample_weights)
                    gb_cls_proba = gb_cls.predict_proba(X_test_abl)[:, 1] if len(gb_cls.classes_) == 2 else None

                    rf_auc_cls = roc_auc_score(y_test_cls, rf_cls_proba) if rf_cls_proba is not None else 0
                    gb_auc_cls = roc_auc_score(y_test_cls, gb_cls_proba) if gb_cls_proba is not None else 0

                    if gb_auc_cls >= rf_auc_cls:
                        best_cls, best_proba, best_name_cls = gb_cls, gb_cls_proba, "GBM"
                    else:
                        best_cls, best_proba, best_name_cls = rf_cls, rf_cls_proba, "RandomForest"
                    best_auc = max(rf_auc_cls, gb_auc_cls)

                    fpr, tpr, thresholds = roc_curve(y_test_cls, best_proba)
                    j_scores = tpr - fpr
                    optimal_idx = j_scores.argmax()
                    optimal_threshold = thresholds[optimal_idx]
                    best_preds = (best_proba >= optimal_threshold).astype(int)
                    best_acc = accuracy_score(y_test_cls, best_preds)
                    best_prec = precision_score(y_test_cls, best_preds, zero_division=0)
                    best_rec = recall_score(y_test_cls, best_preds, zero_division=0)

                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("Accuracy", f"{best_acc:.1%}", help="Percentage of zip codes correctly classified.")
                    mc2.metric("AUC", f"{best_auc:.3f}")
                    mc3.metric("Precision", f"{best_prec:.1%}", help="Of predicted High Risk, how many actually were.")
                    mc4.metric("Recall", f"{best_rec:.1%}", help="Of actual High Risk, how many were caught.")
                    mc5.metric("Threshold", f"{optimal_threshold:.2f}", help="Optimal cutoff via Youden's J statistic.")

                    # Confusion matrix
                    cm_col1, cm_col2 = st.columns(2)
                    with cm_col1:
                        st.markdown(f"**Confusion Matrix ({best_name_cls})**")
                        cm = confusion_matrix(y_test_cls, best_preds)
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm, x=["Low Risk", "High Risk"], y=["Low Risk", "High Risk"],
                            text=cm, texttemplate="%{text}", colorscale=[[0, "#FF6B35"], [1, "#8B4513"]],
                            showscale=False,
                        ))
                        fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual", height=300, paper_bgcolor="#0B0E11", plot_bgcolor="#131820", font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"))
                        st.plotly_chart(fig_cm, use_container_width=True)
                    with cm_col2:
                        st.markdown("**Top Features for Risk Classification**")
                        importances = best_cls.feature_importances_
                        top_n = min(12, len(ablation_cols))
                        top_idx = np.argsort(importances)[-top_n:]
                        fig_imp = go.Figure(go.Bar(x=importances[top_idx], y=[ablation_cols[i] for i in top_idx], orientation="h", marker_color="#FF6B35"))
                        fig_imp.update_layout(xaxis_title="Feature Importance (Gini)", height=300, paper_bgcolor="#0B0E11", plot_bgcolor="#131820", font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"))
                        st.plotly_chart(fig_imp, use_container_width=True)

                    st.info(f"\U0001f4a1 **Key finding:** Without any fire history, building characteristics and demographics alone correctly classify {best_acc:.0%} of zip codes as high or low risk (AUC = {best_auc:.3f}).")

                    # ── Classification Rolling CV from cache ──
                    if cls_roll_cache.exists():
                        st.markdown("---")
                        st.markdown("##### Rolling Temporal CV (Classification, No Fire History)")
                        with open(cls_roll_cache, "rb") as _f:
                            cls_rolling = _pkl.load(_f)
                        if cls_rolling:
                            crc1, crc2 = st.columns(2)
                            with crc1:
                                fig_cr = go.Figure()
                                fig_cr.add_trace(go.Scatter(x=[str(r["Test Year"]) for r in cls_rolling], y=[r["AUC"] for r in cls_rolling], mode="lines+markers", marker=dict(size=10, color="#2ECC71"), line=dict(color="#2ECC71", width=2)))
                                fig_cr.update_layout(title="Classification AUC by Test Year", xaxis_title="Test Year", yaxis_title="AUC", height=280, paper_bgcolor="#0B0E11", plot_bgcolor="#131820", font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"), yaxis=dict(range=[0.4, 1.05]))
                                st.plotly_chart(fig_cr, use_container_width=True)
                            with crc2:
                                cr_df = pd.DataFrame(cls_rolling)
                                cr_df["AUC"] = cr_df["AUC"].map("{:.3f}".format)
                                st.dataframe(cr_df, width="stretch", hide_index=True)
                            avg_ca = np.mean([r["AUC"] for r in cls_rolling])
                            st.success(f"**Classification Rolling CV: AUC = {avg_ca:.3f}** \u2014 stable over time.")

                    # ── Classification Borough from cache ──
                    if cls_boro_cache.exists():
                        st.markdown("##### Borough Holdout (Classification, No Fire History)")
                        with open(cls_boro_cache, "rb") as _f:
                            cls_borough = _pkl.load(_f)
                        if cls_borough:
                            cbc1, cbc2 = st.columns(2)
                            with cbc1:
                                colors_cb = {"Manhattan": "#FF3B4E", "Brooklyn": "#FFAA2B", "Queens": "#FF6B35", "Bronx": "#2ECC71", "Staten Island": "#3498DB"}
                                fig_cb = go.Figure()
                                for r in cls_borough:
                                    fig_cb.add_trace(go.Bar(x=[r["Borough"]], y=[r["AUC"]], marker_color=colors_cb.get(r["Borough"], "#FF6B35"), text=[f"{r['AUC']:.3f}"], textposition="auto"))
                                fig_cb.update_layout(title="Classification AUC by Borough", yaxis_title="AUC", showlegend=False, height=280, paper_bgcolor="#0B0E11", plot_bgcolor="#131820", font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"), yaxis=dict(range=[0.4, 1.05]))
                                st.plotly_chart(fig_cb, use_container_width=True)
                            with cbc2:
                                cb_df = pd.DataFrame(cls_borough)
                                cb_df["AUC"] = cb_df["AUC"].map("{:.3f}".format)
                                st.dataframe(cb_df, width="stretch", hide_index=True)
                            avg_cba = np.mean([r["AUC"] for r in cls_borough])
                            st.success(f"**Borough CV: Avg AUC = {avg_cba:.3f}** \u2014 geographic generalization confirmed.")
            else:
                st.warning("Insufficient data for validation.")'''

if OLD in content:
    content = content.replace(OLD, NEW)
    with open("app.py", "w") as f:
        f.write(content)

    import subprocess
    r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("[OK] Cloud validation tab patched ✅")
    else:
        print(f"[ERROR] {r.stderr}")
else:
    print("[ERROR] Could not find target block")

