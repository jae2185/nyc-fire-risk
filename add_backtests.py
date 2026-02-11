"""Add rolling temporal CV and borough spatial CV to validation tab"""

with open("app.py", "r") as f:
    content = f.read()

MARKER = "                    # ── Classification Ablation ────────────────────────"

BACKTEST_BLOCK = '''                    # ── Rolling Temporal Cross-Validation ──────────────
                    st.markdown("---")
                    st.markdown("#### Rolling Temporal Validation")
                    st.caption(
                        "Train on expanding windows, test on each subsequent year. "
                        "Confirms the model generalizes across different time periods, not just one split."
                    )

                    with st.spinner("Running rolling temporal CV..."):
                        rolling_results = []
                        years_available = sorted(raw_df["year"].unique())
                        # Need at least 2 years to train, 1 to test
                        for test_yr in years_available[2:]:
                            train_slice = raw_df[raw_df["year"] < test_yr]
                            test_slice = raw_df[raw_df["year"] == test_yr]
                            if len(train_slice) < 50 or len(test_slice) < 10:
                                continue
                            try:
                                tr_feat = engineer_features_by_zip(train_slice)
                                te_feat = engineer_features_by_zip(test_slice)
                                try:
                                    tr_feat = enrich_zip_features(tr_feat)
                                    te_feat = enrich_zip_features(te_feat)
                                except Exception:
                                    pass
                                X_tr, y_tr, fn_r = get_enhanced_feature_matrix(tr_feat)
                                X_te, y_te, _ = get_enhanced_feature_matrix(te_feat)
                                if len(X_tr) < 10 or len(X_te) < 5:
                                    continue
                                from sklearn.ensemble import GradientBoostingRegressor
                                gbm_r = GradientBoostingRegressor(
                                    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
                                )
                                gbm_r.fit(X_tr, y_tr)
                                preds_r = gbm_r.predict(X_te)
                                r2_r = r2_score(y_te, preds_r)
                                mae_r = mean_absolute_error(y_te, preds_r)
                                train_yrs = f"{int(train_slice['year'].min())}-{int(train_slice['year'].max())}"
                                rolling_results.append({
                                    "Train Window": train_yrs,
                                    "Test Year": int(test_yr),
                                    "R²": r2_r,
                                    "MAE": mae_r,
                                    "Train Zips": len(X_tr),
                                    "Test Zips": len(X_te),
                                })
                            except Exception:
                                continue

                    if rolling_results:
                        roll_df = pd.DataFrame(rolling_results)
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            fig_roll = go.Figure()
                            fig_roll.add_trace(go.Scatter(
                                x=[str(r["Test Year"]) for r in rolling_results],
                                y=[r["R²"] for r in rolling_results],
                                mode="lines+markers",
                                marker=dict(size=10, color="#FF6B35"),
                                line=dict(color="#FF6B35", width=2),
                                hovertemplate="Test %{x}<br>R² = %{y:.3f}<extra></extra>",
                            ))
                            fig_roll.update_layout(
                                title="R² by Test Year (Rolling Window)",
                                xaxis_title="Test Year",
                                yaxis_title="R² (Out-of-Sample)",
                                height=300,
                                paper_bgcolor="#0B0E11",
                                plot_bgcolor="#131820",
                                font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                                yaxis=dict(range=[0, 1.05]),
                            )
                            st.plotly_chart(fig_roll, use_container_width=True)
                        with rc2:
                            roll_display = roll_df.copy()
                            roll_display["R²"] = roll_display["R²"].map("{:.3f}".format)
                            roll_display["MAE"] = roll_display["MAE"].map("{:.1f}".format)
                            st.dataframe(roll_display, width="stretch", hide_index=True)

                        avg_r2 = np.mean([r["R²"] for r in rolling_results])
                        std_r2 = np.std([r["R²"] for r in rolling_results])
                        st.success(
                            f"**Rolling CV: R² = {avg_r2:.3f} ± {std_r2:.3f}** across {len(rolling_results)} test years. "
                            f"{'Stable performance confirms the model generalizes well over time.' if std_r2 < 0.1 else 'Some variance across years — model performance depends on the time period.'}"
                        )
                    else:
                        st.warning("Not enough yearly data for rolling validation.")

                    # ── Borough Spatial Cross-Validation ───────────────
                    st.markdown("---")
                    st.markdown("#### Borough Holdout Validation")
                    st.caption(
                        "Train on 4 boroughs, test on the held-out borough. "
                        "Tests whether the model generalizes geographically or just memorizes borough-level patterns."
                    )

                    with st.spinner("Running borough holdout CV..."):
                        borough_results = []
                        if "borough" in train_features.columns:
                            borough_col = "borough"
                        elif "zip_code" in train_features.columns:
                            # Map zip codes to boroughs
                            zip_borough = {}
                            bx = range(10451, 10476)
                            mn = list(range(10001, 10041)) + list(range(10101, 10200))
                            bk = list(range(11201, 11257))
                            qn = list(range(11001, 11010)) + list(range(11351, 11698))
                            si = list(range(10301, 10315))
                            for z in bx: zip_borough[str(z)] = "Bronx"
                            for z in mn: zip_borough[str(z)] = "Manhattan"
                            for z in bk: zip_borough[str(z)] = "Brooklyn"
                            for z in qn: zip_borough[str(z)] = "Queens"
                            for z in si: zip_borough[str(z)] = "Staten Island"
                            train_features["_borough"] = train_features["zip_code"].astype(str).map(zip_borough)
                            test_features["_borough"] = test_features["zip_code"].astype(str).map(zip_borough)
                            borough_col = "_borough"
                        else:
                            borough_col = None

                        if borough_col and train_features[borough_col].notna().sum() > 0:
                            all_features_combined = pd.concat([train_features, test_features], ignore_index=True)
                            X_all, y_all, fn_all = get_enhanced_feature_matrix(all_features_combined)

                            for boro in sorted(all_features_combined[borough_col].dropna().unique()):
                                mask = all_features_combined[borough_col] == boro
                                train_mask = ~mask.values
                                test_mask = mask.values

                                if test_mask.sum() < 3 or train_mask.sum() < 10:
                                    continue

                                X_tr_b = X_all[train_mask]
                                y_tr_b = y_all[train_mask]
                                X_te_b = X_all[test_mask]
                                y_te_b = y_all[test_mask]

                                try:
                                    from sklearn.ensemble import GradientBoostingRegressor
                                    gbm_b = GradientBoostingRegressor(
                                        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
                                    )
                                    gbm_b.fit(X_tr_b, y_tr_b)
                                    preds_b = gbm_b.predict(X_te_b)
                                    r2_b = r2_score(y_te_b, preds_b)
                                    mae_b = mean_absolute_error(y_te_b, preds_b)
                                    borough_results.append({
                                        "Borough": boro,
                                        "R²": r2_b,
                                        "MAE": mae_b,
                                        "Zips": int(test_mask.sum()),
                                    })
                                except Exception:
                                    continue

                    if borough_results:
                        boro_df = pd.DataFrame(borough_results)
                        bc1, bc2 = st.columns(2)
                        with bc1:
                            colors_boro = {"Manhattan": "#FF3B4E", "Brooklyn": "#FFAA2B", "Queens": "#FF6B35", "Bronx": "#2ECC71", "Staten Island": "#3498DB"}
                            fig_boro = go.Figure()
                            for _, row in boro_df.iterrows():
                                fig_boro.add_trace(go.Bar(
                                    x=[row["Borough"]],
                                    y=[row["R²"]],
                                    name=row["Borough"],
                                    marker_color=colors_boro.get(row["Borough"], "#FF6B35"),
                                    text=[f"{row['R²']:.3f}"],
                                    textposition="auto",
                                ))
                            fig_boro.update_layout(
                                title="R² by Held-Out Borough",
                                yaxis_title="R² (Out-of-Sample)",
                                showlegend=False,
                                height=300,
                                paper_bgcolor="#0B0E11",
                                plot_bgcolor="#131820",
                                font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                                yaxis=dict(range=[min(0, min(r["R²"] for r in borough_results) - 0.1), 1.05]),
                            )
                            st.plotly_chart(fig_boro, use_container_width=True)
                        with bc2:
                            boro_display = boro_df.copy()
                            boro_display["R²"] = boro_display["R²"].map("{:.3f}".format)
                            boro_display["MAE"] = boro_display["MAE"].map("{:.1f}".format)
                            st.dataframe(boro_display, width="stretch", hide_index=True)

                        avg_boro_r2 = np.mean([r["R²"] for r in borough_results])
                        min_boro = min(borough_results, key=lambda x: x["R²"])
                        max_boro = max(borough_results, key=lambda x: x["R²"])
                        st.success(
                            f"**Borough CV: Avg R² = {avg_boro_r2:.3f}** | "
                            f"Best: {max_boro['Borough']} ({max_boro['R²']:.3f}) | "
                            f"Weakest: {min_boro['Borough']} ({min_boro['R²']:.3f}). "
                            f"{'Strong geographic generalization.' if avg_boro_r2 > 0.7 else 'Some boroughs are harder to predict — consider borough-specific features.'}"
                        )
                    else:
                        st.warning("Borough information not available for spatial validation.")

'''

if MARKER in content:
    content = content.replace(MARKER, BACKTEST_BLOCK + MARKER)
    with open("app.py", "w") as f:
        f.write(content)

    import subprocess
    r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("[OK] Rolling temporal CV + Borough holdout added ✅")
        print("     Rolling CV: trains on expanding windows, tests each year")
        print("     Borough CV: holds out one borough at a time")
    else:
        print(f"[ERROR] {r.stderr}")
else:
    print("[ERROR] Could not find insertion marker")

