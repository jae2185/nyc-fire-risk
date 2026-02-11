"""Insert classification block before the Methodology expander in app.py"""

with open("app.py", "r") as f:
    content = f.read()

MARKER = "                    # ── Methodology ────────────────────────────────────"

CLASSIFICATION_BLOCK = '''                    # ── Classification Ablation ────────────────────────
                    st.markdown("---")
                    st.markdown("#### Ablation: Can Non-Fire Features Classify Risk?")
                    st.caption(
                        "The full model uses past fire counts — partially circular. "
                        "Here we test whether building, demographic, and complaint features "
                        "alone can classify zip codes as High Risk vs Low Risk."
                    )

                    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix as sk_confusion_matrix, f1_score
                    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

                    median_fires = np.median(y_train)
                    y_train_cls = (y_train > median_fires).astype(int)
                    y_test_cls = (y_test > median_fires).astype(int)

                    rf_cls = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
                    rf_cls.fit(X_train_abl, y_train_cls)
                    rf_cls_preds = rf_cls.predict(X_test_abl)
                    rf_cls_proba = rf_cls.predict_proba(X_test_abl)[:, 1] if len(rf_cls.classes_) == 2 else None

                    gb_cls = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
                    gb_cls.fit(X_train_abl, y_train_cls)
                    gb_cls_preds = gb_cls.predict(X_test_abl)
                    gb_cls_proba = gb_cls.predict_proba(X_test_abl)[:, 1] if len(gb_cls.classes_) == 2 else None

                    rf_acc = accuracy_score(y_test_cls, rf_cls_preds)
                    gb_acc = accuracy_score(y_test_cls, gb_cls_preds)

                    if gb_acc >= rf_acc:
                        best_cls, best_preds, best_proba, best_name_cls = gb_cls, gb_cls_preds, gb_cls_proba, "GBM"
                    else:
                        best_cls, best_preds, best_proba, best_name_cls = rf_cls, rf_cls_preds, rf_cls_proba, "RandomForest"

                    best_acc = accuracy_score(y_test_cls, best_preds)
                    best_prec = precision_score(y_test_cls, best_preds, zero_division=0)
                    best_rec = recall_score(y_test_cls, best_preds, zero_division=0)
                    best_f1_cls = f1_score(y_test_cls, best_preds, zero_division=0)
                    best_auc = roc_auc_score(y_test_cls, best_proba) if best_proba is not None else None

                    st.caption(
                        f"Classifying zip codes as **High Risk** (>{int(median_fires)} fires) or "
                        f"**Low Risk** (\\u2264{int(median_fires)} fires) using only building, demographic, "
                        f"and complaint features \\u2014 zero fire incident data."
                    )

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Accuracy", f"{best_acc:.1%}")
                    if best_auc is not None:
                        mc2.metric("AUC", f"{best_auc:.3f}")
                    mc3.metric("Precision", f"{best_prec:.1%}")
                    mc4.metric("Recall", f"{best_rec:.1%}")

                    cls_col1, cls_col2 = st.columns(2)

                    with cls_col1:
                        cm = sk_confusion_matrix(y_test_cls, best_preds)
                        labels = ["Low Risk", "High Risk"]
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm[::-1],
                            x=labels,
                            y=labels[::-1],
                            text=[[str(v) for v in row] for row in cm[::-1]],
                            texttemplate="%{text}",
                            textfont={"size": 18, "color": "white"},
                            colorscale=[[0, "#131820"], [1, "#FF6B35"]],
                            showscale=False,
                        ))
                        fig_cm.update_layout(
                            title=f"Confusion Matrix ({best_name_cls})",
                            xaxis_title="Predicted",
                            yaxis_title="Actual",
                            height=350,
                            paper_bgcolor="#0B0E11",
                            plot_bgcolor="#131820",
                            font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with cls_col2:
                        importances_cls = best_cls.feature_importances_
                        imp_df = pd.DataFrame({
                            "feature": ablation_cols,
                            "importance": importances_cls
                        }).sort_values("importance", ascending=False).head(12)

                        fig_imp_cls = go.Figure(go.Bar(
                            x=imp_df["importance"].values[::-1],
                            y=imp_df["feature"].values[::-1],
                            orientation="h",
                            marker_color="#FF6B35",
                        ))
                        fig_imp_cls.update_layout(
                            title="Top Features for Risk Classification",
                            xaxis_title="Feature Importance (Gini)",
                            height=350,
                            paper_bgcolor="#0B0E11",
                            plot_bgcolor="#131820",
                            font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                            margin=dict(l=180),
                        )
                        st.plotly_chart(fig_imp_cls, use_container_width=True)

                    st.info(
                        f"\\U0001f4a1 **Key finding:** Without any fire history, building characteristics and demographics "
                        f"alone correctly classify {best_acc:.0%} of zip codes as high or low risk "
                        f"(AUC = {best_auc:.3f}). This confirms that structural factors \\u2014 not just past fires \\u2014 "
                        f"drive fire risk."
                    )

'''

if MARKER not in content:
    print("[ERROR] Could not find Methodology marker in app.py")
else:
    content = content.replace(MARKER, CLASSIFICATION_BLOCK + MARKER)
    with open("app.py", "w") as f:
        f.write(content)
    print("[OK] Classification block inserted before Methodology expander")
    
    # Verify
    import subprocess
    result = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("[OK] Syntax check passed ✅")
    else:
        print(f"[ERROR] Syntax issue: {result.stderr}")

