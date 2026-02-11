"""Simple caching: use st.session_state to skip recomputation"""

with open("app.py", "r") as f:
    content = f.read()

# Wrap the heavy computation in a session_state check
OLD = """                with st.spinner("Engineering features for train/test periods..."):
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
                X_test_full, y_test, _ = get_enhanced_feature_matrix(test_features)"""

NEW = """                cache_key = "validation_cache"
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
                    fn_full = _vc["fn_full"]"""

if OLD in content:
    content = content.replace(OLD, NEW)
    with open("app.py", "w") as f:
        f.write(content)
    
    import subprocess
    r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("[OK] Session-state caching added ✅")
        print("     First visit to Validation tab: full computation")
        print("     Subsequent visits: instant (uses cached data)")
        print("     Retrain button or page refresh: recomputes")
    else:
        print(f"[ERROR] {r.stderr}")
else:
    print("[ERROR] Could not find target block — may have already been modified")

