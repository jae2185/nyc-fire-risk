"""
1. Remove dead ablation regression code
2. Add optimal threshold to display
3. Save cached data locally for demo mode
"""

with open("app.py", "r") as f:
    content = f.read()

# ═══════════════════════════════════════════════════════════════
# TASK 1: Remove dead ablation regression code
# ═══════════════════════════════════════════════════════════════

# Remove the old regression ablation training (keep ablation_cols setup for classification)
OLD_ABL_REGRESSION = """                    rf_abl = train_tuned_rf(X_train_abl, y_train, ablation_cols)
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
                    best_abl_mae = gbm_abl_mae if gbm_abl_r2 > rf_abl_r2 else rf_abl_mae"""

NEW_PICK_BEST = """                    # ── Pick best full model for display ──────────────
                    best_full_preds = gbm_preds_full if gbm_oos_r2 > rf_oos_r2 else rf_preds_full
                    best_full_r2 = max(gbm_oos_r2, rf_oos_r2)
                    best_full_mae = gbm_oos_mae if gbm_oos_r2 > rf_oos_r2 else rf_oos_mae
                    best_full_name = gbm_full["model_name"] if gbm_oos_r2 > rf_oos_r2 else rf_full["model_name"]"""

if OLD_ABL_REGRESSION in content:
    content = content.replace(OLD_ABL_REGRESSION, NEW_PICK_BEST)
    print("[OK] Removed dead ablation regression code")
else:
    print("[SKIP] Dead ablation code not found (may already be cleaned)")

# ═══════════════════════════════════════════════════════════════
# TASK 2: Add optimal threshold to classification display
# ═══════════════════════════════════════════════════════════════

OLD_METRICS = """                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Accuracy", f"{best_acc:.1%}")
                    if best_auc is not None:
                        mc2.metric("AUC", f"{best_auc:.3f}")
                    mc3.metric("Precision", f"{best_prec:.1%}")
                    mc4.metric("Recall", f"{best_rec:.1%}")"""

NEW_METRICS = """                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("Accuracy", f"{best_acc:.1%}")
                    if best_auc is not None:
                        mc2.metric("AUC", f"{best_auc:.3f}")
                    mc3.metric("Precision", f"{best_prec:.1%}")
                    mc4.metric("Recall", f"{best_rec:.1%}")
                    mc5.metric("Threshold", f"{optimal_threshold:.2f}")"""

if OLD_METRICS in content:
    content = content.replace(OLD_METRICS, NEW_METRICS)
    print("[OK] Added optimal threshold to display")
else:
    print("[SKIP] Metrics block not found")

# ═══════════════════════════════════════════════════════════════
# TASK 3: Add local data caching for demo mode
# ═══════════════════════════════════════════════════════════════

# Add save-to-disk after data fetch, and load-from-disk as fallback
OLD_DATA_CACHE = '''                data_key = "val_data_cache"
                if data_key not in st.session_state:
                    with st.spinner("Fetching & engineering features (cached after first run)..."):
                        train_features = engineer_features_by_zip(train_years)
                        test_features = engineer_features_by_zip(test_years)
                        try:
                            train_features = enrich_zip_features(train_features)
                            test_features = enrich_zip_features(test_features)
                        except Exception as e:
                            st.warning(f"Enrichment failed for validation: {e}")
                        X_train_full, y_train, fn_full = get_enhanced_feature_matrix(train_features)
                        X_test_full, y_test, _ = get_enhanced_feature_matrix(test_features)
                        st.session_state[data_key] = {
                            "train_features": train_features,
                            "test_features": test_features,
                            "X_train_full": X_train_full,
                            "X_test_full": X_test_full,
                            "y_train": y_train,
                            "y_test": y_test,
                            "fn_full": fn_full,
                        }'''

NEW_DATA_CACHE = '''                data_key = "val_data_cache"
                demo_cache = Path("data/validation_cache.pkl")
                if data_key not in st.session_state:
                    if demo_cache.exists():
                        import pickle
                        with open(demo_cache, "rb") as f:
                            st.session_state[data_key] = pickle.load(f)
                        st.toast("Loaded cached validation data from disk")
                    else:
                        with st.spinner("Fetching & engineering features (cached after first run)..."):
                            train_features = engineer_features_by_zip(train_years)
                            test_features = engineer_features_by_zip(test_years)
                            try:
                                train_features = enrich_zip_features(train_features)
                                test_features = enrich_zip_features(test_features)
                            except Exception as e:
                                st.warning(f"Enrichment failed for validation: {e}")
                            X_train_full, y_train, fn_full = get_enhanced_feature_matrix(train_features)
                            X_test_full, y_test, _ = get_enhanced_feature_matrix(test_features)
                            st.session_state[data_key] = {
                                "train_features": train_features,
                                "test_features": test_features,
                                "X_train_full": X_train_full,
                                "X_test_full": X_test_full,
                                "y_train": y_train,
                                "y_test": y_test,
                                "fn_full": fn_full,
                            }
                            # Save to disk for future demo mode
                            import pickle
                            with open(demo_cache, "wb") as f:
                                pickle.dump(st.session_state[data_key], f)'''

if OLD_DATA_CACHE in content:
    content = content.replace(OLD_DATA_CACHE, NEW_DATA_CACHE)
    print("[OK] Added disk-backed demo cache")
    print("     First run: fetches from API, saves to data/validation_cache.pkl")
    print("     Future runs: loads from disk instantly")
else:
    print("[SKIP] Data cache block not found")

# Write and verify
with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("\n[OK] Syntax check passed ✅")
else:
    print(f"\n[ERROR] {r.stderr}")

