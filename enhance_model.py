#!/usr/bin/env python3
"""
Adds: DOB + 311 + PLUTO enrichment, XGBoost, model comparison tab.
Run from: /Users/jonathanepstein/Documents/Personal/nyc_fire_app
Usage: python3 enhance_model.py
"""
import os, shutil
os.chdir("/Users/jonathanepstein/Documents/Personal/nyc_fire_app")

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Write enrichment.py
# ═══════════════════════════════════════════════════════════════════════
# (This is the external data fetching module)
# Already created as a separate file — will be copied from downloads

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Write enhanced_model.py
# ═══════════════════════════════════════════════════════════════════════
# Already created as a separate file — will be copied from downloads

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Patch app.py — add enrichment to pipeline + model comparison
# ═══════════════════════════════════════════════════════════════════════
content = open("app.py").read()

# 3a. Add imports at the top
old_import = """from data.building_data import (
    fetch_pluto_buildings,
    process_pluto_buildings,
    score_buildings_with_neighborhood,
)"""

new_import = """from data.building_data import (
    fetch_pluto_buildings,
    process_pluto_buildings,
    score_buildings_with_neighborhood,
)
from data.enrichment import enrich_zip_features
from models.enhanced_model import (
    get_enhanced_feature_matrix,
    compare_models,
    train_tuned_rf,
    train_xgboost,
)"""

if old_import in content:
    content = content.replace(old_import, new_import)
    print("[OK] Added enrichment + enhanced model imports")
else:
    print("[SKIP] Import block not found — may already be patched")

# 3b. Add enrichment to the load_and_train pipeline
# Find where zip_features are created and add enrichment after
old_pipeline = """    zip_features = engineer_features_by_zip(df)
    puma_features = aggregate_to_puma(zip_features)
    boro_features = aggregate_to_borough(zip_features)

    # Train on zip-level data
    X, y, feature_names = get_feature_matrix(zip_features)
    model = FireRiskModel(n_estimators=100, max_depth=12)
    results = model.fit(X, y, feature_names)"""

new_pipeline = """    zip_features = engineer_features_by_zip(df)

    # Enrich with external data (DOB, 311, PLUTO)
    try:
        zip_features = enrich_zip_features(zip_features)
        print(f"[PIPELINE] Enriched features: {len(zip_features.columns)} columns")
    except Exception as e:
        print(f"[PIPELINE] Enrichment failed (continuing with base features): {e}")

    puma_features = aggregate_to_puma(zip_features)
    boro_features = aggregate_to_borough(zip_features)

    # Train enhanced model (tries both RF and XGBoost)
    X, y, feature_names = get_enhanced_feature_matrix(zip_features)
    if X.shape[1] == 0:
        # Fallback to original features if enhanced matrix is empty
        from data.feature_engineering import get_feature_matrix as get_orig_features
        X, y, feature_names = get_orig_features(zip_features)

    model_comparison = compare_models(X, y, feature_names)
    best = model_comparison["best"]

    # Use the best model
    model = FireRiskModel.__new__(FireRiskModel)
    model.model = best["model"]
    model.feature_names = feature_names
    model.importance = best["importance"]
    results = best"""

if old_pipeline in content:
    content = content.replace(old_pipeline, new_pipeline)
    print("[OK] Added enrichment pipeline + model comparison")
else:
    print("[SKIP] Pipeline block not found")

# 3c. Fix the model.predict calls to use the sklearn model directly
# The existing code calls model.predict() and model.predict_with_risk()
# We need to make these work with the raw sklearn model

old_predict = """    # Zip predictions
    zip_preds, zip_risk = model.predict_with_risk(X)"""

new_predict = """    # Zip predictions
    zip_preds = best["model"].predict(X)
    zip_risk = np.clip(zip_preds / (zip_preds.max() or 1), 0, 1)"""

if old_predict in content:
    content = content.replace(old_predict, new_predict)
    print("[OK] Fixed zip prediction calls")

# Fix PUMA prediction
old_puma_pred = """    X_puma, y_puma, fn_puma = get_feature_matrix(puma_features)
    if len(X_puma) > 5:
        puma_model = FireRiskModel(n_estimators=80, max_depth=8)
        puma_model.fit(X_puma, y_puma, fn_puma)
        puma_preds, puma_risk = puma_model.predict_with_risk(X_puma)
    else:
        puma_preds = y_puma
        puma_risk = np.clip(y_puma / (y_puma.max() or 1), 0, 1)"""

new_puma_pred = """    from data.feature_engineering import get_feature_matrix as get_orig_features
    X_puma, y_puma, fn_puma = get_orig_features(puma_features)
    if len(X_puma) > 5:
        from sklearn.ensemble import RandomForestRegressor
        puma_rf = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)
        puma_rf.fit(X_puma, y_puma)
        puma_preds = puma_rf.predict(X_puma)
        puma_risk = np.clip(puma_preds / (puma_preds.max() or 1), 0, 1)
    else:
        puma_preds = y_puma
        puma_risk = np.clip(y_puma / (y_puma.max() or 1), 0, 1)"""

if old_puma_pred in content:
    content = content.replace(old_puma_pred, new_puma_pred)
    print("[OK] Fixed PUMA prediction")

# 3d. Store model_comparison in the returned data
old_return = """    return {
        "raw_df": df,
        "zip_features": zip_features,
        "puma_features": puma_features,
        "boro_features": boro_features,
        "model": model,
        "results": results,
        "X": X, "y": y,
        "zip_risk": zip_risk,
    }"""

new_return = """    return {
        "raw_df": df,
        "zip_features": zip_features,
        "puma_features": puma_features,
        "boro_features": boro_features,
        "model": model,
        "results": results,
        "model_comparison": model_comparison,
        "X": X, "y": y,
        "zip_risk": zip_risk,
    }"""

if old_return in content:
    content = content.replace(old_return, new_return)
    print("[OK] Added model_comparison to return dict")

# 3e. Fix model.predict in the Model tab
old_model_pred = """            X, y = data["X"], data["y"]
            preds = model.predict(X)"""

new_model_pred = """            X, y = data["X"], data["y"]
            preds = results["model"].predict(X)"""

if old_model_pred in content:
    content = content.replace(old_model_pred, new_model_pred)
    print("[OK] Fixed Model tab prediction")

# 3f. Add model comparison display to Model tab
old_model_desc = """        # Model description
        with st.expander("Model Architecture Details"):"""

new_model_desc = """        # Model comparison table
        if "model_comparison" in data:
            st.markdown("### Model Comparison")
            mc = data["model_comparison"]["comparison"]
            mc_display = mc.copy()
            for col in ["Train R²", "CV R² (mean)", "CV R² (std)", "Train RMSE", "Train MAE"]:
                if col in mc_display.columns:
                    mc_display[col] = mc_display[col].map("{:.3f}".format)
            st.dataframe(mc_display, use_container_width=True)

            best_name = data["model_comparison"]["best"]["model_name"]
            best_cv = data["model_comparison"]["best"]["cv"]["cv_r2_mean"]
            st.success(f"**Best model: {best_name}** (CV R² = {best_cv:.3f})")

        # Model description
        with st.expander("Model Architecture Details"):"""

if old_model_desc in content:
    content = content.replace(old_model_desc, new_model_desc)
    print("[OK] Added model comparison display")

# 3g. Update the header to show feature count and model name
old_header_caption = """    st.caption(
        f"{len(active_df)} zones · RandomForest (n=100) · "
        f"{results['train']['n_samples']} training samples"
    )"""

new_header_caption = """    model_name = results.get("model_name", "RandomForest")
    n_feats = results["train"].get("n_features", "?")
    st.caption(
        f"{len(active_df)} zones · {model_name} · "
        f"{n_feats} features · {results['train']['n_samples']} training samples"
    )"""

if old_header_caption in content:
    content = content.replace(old_header_caption, new_header_caption)
    print("[OK] Updated header caption")

open("app.py", "w").write(content)
print("[OK] app.py fully patched")

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Add xgboost to requirements.txt
# ═══════════════════════════════════════════════════════════════════════
req = open("requirements.txt").read()
if "xgboost" not in req:
    req = req.strip() + "\nxgboost>=2.0.0\n"
    open("requirements.txt", "w").write(req)
    print("[OK] Added xgboost to requirements.txt")

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Clear cache
# ═══════════════════════════════════════════════════════════════════════
if os.path.exists(".cache"):
    shutil.rmtree(".cache")
    print("[OK] Cleared .cache")

print("\n=== Enhancement complete. ===")
print("Make sure enrichment.py and enhanced_model.py are in the right folders,")
print("then run: streamlit run app.py")
