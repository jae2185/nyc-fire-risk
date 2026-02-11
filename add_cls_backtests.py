"""Add rolling temporal and borough holdout for classification ablation"""

with open("app.py", "r") as f:
    content = f.read()

MARKER = "                    # ── Methodology ────────────────────────────────────"

CLS_BACKTEST = '''                    # ── Classification: Rolling Temporal CV ─────────────
                    st.markdown("---")
                    st.markdown("##### Rolling Temporal CV (Classification, No Fire History)")
                    st.caption("Same expanding-window approach, but testing the non-fire classification model.")

                    with st.spinner("Running classification rolling CV..."):
                        cls_rolling = []
                        for test_yr in years_available[2:]:
                            train_slice = raw_df[raw_df["year"] < test_yr]
                            test_slice = raw_df[raw_df["year"] == test_yr]
                            if len(train_slice) < 50 or len(test_slice) < 10:
                                continue
                            try:
                                tr_f = engineer_features_by_zip(train_slice)
                                te_f = engineer_features_by_zip(test_slice)
                                try:
                                    tr_f = enrich_zip_features(tr_f)
                                    te_f = enrich_zip_features(te_f)
                                except Exception:
                                    pass
                                X_tr_r, y_tr_r, fn_r = get_enhanced_feature_matrix(tr_f)
                                X_te_r, y_te_r, _ = get_enhanced_feature_matrix(te_f)
                                if len(X_tr_r) < 10 or len(X_te_r) < 5:
                                    continue

                                # Ablation: remove fire features
                                abl_idx_r = [fn_r.index(c) for c in fn_r if c not in incident_features]
                                X_tr_abl_r = X_tr_r[:, abl_idx_r]
                                X_te_abl_r = X_te_r[:, abl_idx_r]

                                med_r = np.median(y_tr_r)
                                y_tr_cls_r = (y_tr_r > med_r).astype(int)
                                y_te_cls_r = (y_te_r > med_r).astype(int)

                                if len(np.unique(y_te_cls_r)) < 2:
                                    continue

                                from sklearn.ensemble import RandomForestClassifier as RFC_r
                                clf_r = RFC_r(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42)
                                clf_r.fit(X_tr_abl_r, y_tr_cls_r)
                                proba_r = clf_r.predict_proba(X_te_abl_r)[:, 1] if len(clf_r.classes_) == 2 else None

                                if proba_r is not None:
                                    auc_r = roc_auc_score(y_te_cls_r, proba_r)
                                    cls_rolling.append({
                                        "Train Window": f"{int(train_slice['year'].min())}-{int(train_slice['year'].max())}",
                                        "Test Year": int(test_yr),
                                        "AUC": auc_r,
                                        "Test Zips": len(X_te_r),
                                    })
                            except Exception:
                                continue

                    if cls_rolling:
                        crc1, crc2 = st.columns(2)
                        with crc1:
                            fig_cr = go.Figure()
                            fig_cr.add_trace(go.Scatter(
                                x=[str(r["Test Year"]) for r in cls_rolling],
                                y=[r["AUC"] for r in cls_rolling],
                                mode="lines+markers",
                                marker=dict(size=10, color="#2ECC71"),
                                line=dict(color="#2ECC71", width=2),
                                hovertemplate="Test %{x}<br>AUC = %{y:.3f}<extra></extra>",
                            ))
                            fig_cr.update_layout(
                                title="Classification AUC by Test Year",
                                xaxis_title="Test Year",
                                yaxis_title="AUC",
                                height=280,
                                paper_bgcolor="#0B0E11",
                                plot_bgcolor="#131820",
                                font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                                yaxis=dict(range=[0.4, 1.05]),
                            )
                            st.plotly_chart(fig_cr, use_container_width=True)
                        with crc2:
                            cr_df = pd.DataFrame(cls_rolling)
                            cr_display = cr_df.copy()
                            cr_display["AUC"] = cr_display["AUC"].map("{:.3f}".format)
                            st.dataframe(cr_display, width="stretch", hide_index=True)

                        avg_cls_auc = np.mean([r["AUC"] for r in cls_rolling])
                        std_cls_auc = np.std([r["AUC"] for r in cls_rolling])
                        st.success(
                            f"**Classification Rolling CV: AUC = {avg_cls_auc:.3f} ± {std_cls_auc:.3f}** across {len(cls_rolling)} years. "
                            f"{'Non-fire features provide stable risk classification over time.' if std_cls_auc < 0.1 else 'Some variation — classification performance depends on the year.'}"
                        )

                    # ── Classification: Borough Holdout ─────────────────
                    st.markdown("##### Borough Holdout (Classification, No Fire History)")
                    st.caption("Hold out one borough at a time. Tests geographic generalization of non-fire risk classification.")

                    with st.spinner("Running classification borough holdout..."):
                        cls_borough = []
                        if borough_col and all_features_combined[borough_col].notna().sum() > 0:
                            abl_cols_all = [c for c in fn_all if c not in incident_features]
                            abl_idx_all = [fn_all.index(c) for c in abl_cols_all]
                            X_all_abl = X_all[:, abl_idx_all]

                            for boro in sorted(all_features_combined[borough_col].dropna().unique()):
                                mask_b = (all_features_combined[borough_col] == boro).values
                                if mask_b.sum() < 3 or (~mask_b).sum() < 10:
                                    continue

                                X_tr_cb = X_all_abl[~mask_b]
                                y_tr_cb = y_all[~mask_b]
                                X_te_cb = X_all_abl[mask_b]
                                y_te_cb = y_all[mask_b]

                                med_cb = np.median(y_tr_cb)
                                y_tr_cls_cb = (y_tr_cb > med_cb).astype(int)
                                y_te_cls_cb = (y_te_cb > med_cb).astype(int)

                                if len(np.unique(y_te_cls_cb)) < 2:
                                    continue

                                try:
                                    from sklearn.ensemble import RandomForestClassifier as RFC_b
                                    clf_b = RFC_b(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42)
                                    clf_b.fit(X_tr_cb, y_tr_cls_cb)
                                    proba_cb = clf_b.predict_proba(X_te_cb)[:, 1] if len(clf_b.classes_) == 2 else None

                                    if proba_cb is not None:
                                        auc_cb = roc_auc_score(y_te_cls_cb, proba_cb)
                                        cls_borough.append({
                                            "Borough": boro,
                                            "AUC": auc_cb,
                                            "Zips": int(mask_b.sum()),
                                        })
                                except Exception:
                                    continue

                    if cls_borough:
                        cbc1, cbc2 = st.columns(2)
                        with cbc1:
                            fig_cb = go.Figure()
                            colors_cb = {"Manhattan": "#FF3B4E", "Brooklyn": "#FFAA2B", "Queens": "#FF6B35", "Bronx": "#2ECC71", "Staten Island": "#3498DB"}
                            for _, row in pd.DataFrame(cls_borough).iterrows():
                                fig_cb.add_trace(go.Bar(
                                    x=[row["Borough"]],
                                    y=[row["AUC"]],
                                    marker_color=colors_cb.get(row["Borough"], "#FF6B35"),
                                    text=[f"{row['AUC']:.3f}"],
                                    textposition="auto",
                                ))
                            fig_cb.update_layout(
                                title="Classification AUC by Held-Out Borough",
                                yaxis_title="AUC",
                                showlegend=False,
                                height=280,
                                paper_bgcolor="#0B0E11",
                                plot_bgcolor="#131820",
                                font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                                yaxis=dict(range=[0.4, 1.05]),
                            )
                            st.plotly_chart(fig_cb, use_container_width=True)
                        with cbc2:
                            cb_df = pd.DataFrame(cls_borough)
                            cb_display = cb_df.copy()
                            cb_display["AUC"] = cb_display["AUC"].map("{:.3f}".format)
                            st.dataframe(cb_display, width="stretch", hide_index=True)

                        avg_cb_auc = np.mean([r["AUC"] for r in cls_borough])
                        min_cb = min(cls_borough, key=lambda x: x["AUC"])
                        max_cb = max(cls_borough, key=lambda x: x["AUC"])
                        st.success(
                            f"**Borough CV: Avg AUC = {avg_cb_auc:.3f}** | "
                            f"Best: {max_cb['Borough']} ({max_cb['AUC']:.3f}) | "
                            f"Weakest: {min_cb['Borough']} ({min_cb['AUC']:.3f}). "
                            f"{'Non-fire features generalize well across boroughs.' if avg_cb_auc > 0.7 else 'Geographic variation — some boroughs need different risk indicators.'}"
                        )

'''

if MARKER in content:
    content = content.replace(MARKER, CLS_BACKTEST + MARKER)
    with open("app.py", "w") as f:
        f.write(content)

    import subprocess
    r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("[OK] Classification backtests added ✅")
    else:
        print(f"[ERROR] {r.stderr}")
else:
    print("[ERROR] Marker not found")

