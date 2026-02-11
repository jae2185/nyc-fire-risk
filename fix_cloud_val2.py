with open("app.py", "r") as f:
    content = f.read()

OLD = """                if len(X_train_full) > 10 and len(X_test_full) > 5:

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
                    st.dataframe(pd.DataFrame(comparison_data), width="stretch")"""

NEW = """                if len(X_train_full) > 10 and len(X_test_full) > 5:

                    # ── Full model (load from cache if available) ──
                    if "rf_preds" in _vc:
                        rf_preds_full = _vc["rf_preds"]
                        gbm_preds_full = _vc["gbm_preds"]
                        rf_oos_r2 = _vc["rf_r2"]
                        gbm_oos_r2 = _vc["gbm_r2"]
                        rf_oos_mae = _vc["rf_mae"]
                        gbm_oos_mae = _vc["gbm_mae"]
                        rf_cv = _vc["rf_cv"]
                        gbm_cv = _vc["gbm_cv"]
                        rf_name = _vc.get("rf_name", "RandomForest")
                        gbm_name = _vc.get("gbm_name", "GradientBoosting")
                    else:
                        rf_full = train_tuned_rf(X_train_full, y_train, fn_full)
                        gbm_full = train_xgboost(X_train_full, y_train, fn_full)
                        rf_preds_full = rf_full["model"].predict(X_test_full)
                        gbm_preds_full = gbm_full["model"].predict(X_test_full)
                        rf_oos_r2 = r2_score(y_test, rf_preds_full)
                        gbm_oos_r2 = r2_score(y_test, gbm_preds_full)
                        rf_oos_mae = mean_absolute_error(y_test, rf_preds_full)
                        gbm_oos_mae = mean_absolute_error(y_test, gbm_preds_full)
                        rf_cv = rf_full["cv"]
                        gbm_cv = gbm_full["cv"]
                        rf_name = rf_full["model_name"]
                        gbm_name = gbm_full["model_name"]

                    best_full_preds = gbm_preds_full if gbm_oos_r2 > rf_oos_r2 else rf_preds_full
                    best_full_r2 = max(gbm_oos_r2, rf_oos_r2)
                    best_full_mae = gbm_oos_mae if gbm_oos_r2 > rf_oos_r2 else rf_oos_mae
                    best_full_name = gbm_name if gbm_oos_r2 > rf_oos_r2 else rf_name

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
                        {"Model": "Tuned RF (all features)", "OOS R\u00b2": f"{rf_oos_r2:.3f}", "OOS MAE": f"{rf_oos_mae:.1f}", "CV R\u00b2": f"{rf_cv['cv_r2_mean']:.3f}", "Features": len(fn_full)},
                        {"Model": "GBM (all features)", "OOS R\u00b2": f"{gbm_oos_r2:.3f}", "OOS MAE": f"{gbm_oos_mae:.1f}", "CV R\u00b2": f"{gbm_cv['cv_r2_mean']:.3f}", "Features": len(fn_full)},
                    ]
                    st.dataframe(pd.DataFrame(comparison_data), width="stretch")"""

# This replacement targets the CLOUD block (the second occurrence)
# Find position of both occurrences
first_idx = content.find(OLD)
if first_idx >= 0:
    second_idx = content.find(OLD, first_idx + 1)
    if second_idx >= 0:
        # Replace only the second occurrence (cloud block)
        content = content[:second_idx] + NEW + content[second_idx + len(OLD):]
        print("[OK] Patched cloud validation block (2nd occurrence)")
    else:
        # Only one occurrence, replace it
        content = content.replace(OLD, NEW)
        print("[OK] Patched validation block (single occurrence)")
else:
    print("[ERROR] Target block not found")

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] Syntax check passed ✅")
else:
    print(f"[ERROR] {r.stderr}")

