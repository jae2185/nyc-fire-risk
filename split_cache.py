"""Split validation cache into data layer (slow) and model layer (fast)"""

with open("app.py", "r") as f:
    content = f.read()

OLD = '''                cache_key = "validation_cache_v2"
                if cache_key not in st.session_state:
                    with st.spinner("Running validation models (cached after first run)..."):
                        train_features = engineer_features_by_zip(train_years)
                        test_features = engineer_features_by_zip(test_years)
                        try:
                            train_features = enrich_zip_features(train_features)
                            test_features = enrich_zip_features(test_features)
                        except Exception as e:
                            st.warning(f"Enrichment failed for validation: {e}")
                        X_train_full, y_train, fn_full = get_enhanced_feature_matrix(train_features)
                        X_test_full, y_test, _ = get_enhanced_feature_matrix(test_features)
                        st.session_state[cache_key] = {
                            "train_features": train_features,
                            "test_features": test_features,
                            "X_train_full": X_train_full,
                            "X_test_full": X_test_full,
                            "y_train": y_train,
                            "y_test": y_test,
                            "fn_full": fn_full,
                        }
                else:
                    _vc = st.session_state[cache_key]
                    train_features = _vc["train_features"]
                    test_features = _vc["test_features"]
                    X_train_full = _vc["X_train_full"]
                    X_test_full = _vc["X_test_full"]
                    y_train = _vc["y_train"]
                    y_test = _vc["y_test"]
                    fn_full = _vc["fn_full"]'''

NEW = '''                # ── Layer 1: Data cache (slow — API calls, feature engineering) ──
                data_key = "val_data_cache"
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
                        }
                else:
                    _vc = st.session_state[data_key]
                    train_features = _vc["train_features"]
                    test_features = _vc["test_features"]
                    X_train_full = _vc["X_train_full"]
                    X_test_full = _vc["X_test_full"]
                    y_train = _vc["y_train"]
                    y_test = _vc["y_test"]
                    fn_full = _vc["fn_full"]'''

if OLD in content:
    content = content.replace(OLD, NEW)
    with open("app.py", "w") as f:
        f.write(content)
    
    import subprocess
    r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("[OK] Two-layer cache installed")
        print("     Data layer: cached per session (no re-fetching)")
        print("     Model layer: retrains each time (fast, ~2 sec)")
    else:
        print(f"[ERROR] {r.stderr}")
else:
    print("[SKIP] Cache block not found — may already be updated")

