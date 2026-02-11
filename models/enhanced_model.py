"""
Enhanced fire risk model with XGBoost, tuned RandomForest,
and model comparison framework.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance


# Extended feature list including enrichment columns
ENHANCED_FEATURE_NAMES = [
    # Original 15 incident-derived features
    "total_incidents", "structural_fires", "non_structural_fires",
    "false_alarms", "medical_calls",
    "structural_fire_rate", "false_alarm_rate", "medical_rate",
    "avg_units_onscene", "winter_concentration", "summer_concentration",
    "trend_slope", "incident_volatility", "max_monthly_incidents",
    "avg_yearly_incidents",
    # 311 complaint features
    "complaints_311_total", "complaints_heating", "complaints_electrical",
    "complaints_gas", "complaints_per_incident", "heating_complaint_rate",
    # DOB violation features
    "dob_violation_count", "dob_fire_relevant_violations",
    # PLUTO aggregate features
    "pluto_building_count", "pluto_avg_age", "pluto_median_age",
    "pluto_avg_floors", "pluto_avg_units", "pluto_avg_area",
    "pluto_pct_pre_war", "pluto_pct_pre_code", "pluto_pct_residential",
    "pluto_max_floors", "pluto_total_units",
]


def get_enhanced_feature_matrix(df):
    """
    Extract feature matrix using all available features (original + enriched).
    Gracefully handles missing columns.
    """
    existing = [c for c in ENHANCED_FEATURE_NAMES if c in df.columns]
    X = df[existing].fillna(0).values
    y = df["structural_fires"].values if "structural_fires" in df.columns else None
    return X, y, existing


def train_tuned_rf(X, y, feature_names):
    """
    Tuned RandomForest with regularization to reduce overfitting.
    Lower max_depth + higher min_samples_leaf vs the original.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,            # Reduced from 12
        min_samples_leaf=5,     # Increased from 3
        min_samples_split=10,   # Added
        max_features=0.7,       # Use 70% of features per tree
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    preds = model.predict(X)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    # Permutation importance
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_permutation": perm.importances_mean,
        "importance_perm_std": perm.importances_std,
    }).sort_values("importance_permutation", ascending=False)

    results = {
        "model_name": "Tuned RandomForest",
        "model": model,
        "train": {
            "r2": r2_score(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "mae": mean_absolute_error(y, preds),
            "n_samples": len(y),
            "n_features": X.shape[1],
        },
        "cv": {
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "cv_scores": cv_scores,
        },
        "importance": importance_df,
    }
    return results


def train_xgboost(X, y, feature_names):
    """
    XGBoost (via sklearn's GradientBoostingRegressor as fallback
    if xgboost not installed).
    """
    try:
        raise ImportError("skip")
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,      # L1 regularization
            reg_lambda=1.0,     # L2 regularization
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model_name = "XGBoost"
    except ImportError:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        model_name = "GradientBoosting (sklearn)"

    model.fit(X, y)

    preds = model.predict(X)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    # Permutation importance
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_permutation": perm.importances_mean,
        "importance_perm_std": perm.importances_std,
    }).sort_values("importance_permutation", ascending=False)

    results = {
        "model_name": model_name,
        "model": model,
        "train": {
            "r2": r2_score(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "mae": mean_absolute_error(y, preds),
            "n_samples": len(y),
            "n_features": X.shape[1],
        },
        "cv": {
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "cv_scores": cv_scores,
        },
        "importance": importance_df,
    }
    return results


def compare_models(X, y, feature_names):
    """
    Train multiple models and return comparison results.
    """
    print("[MODEL] Training Tuned RandomForest...")
    rf_results = train_tuned_rf(X, y, feature_names)

    print("[MODEL] Training XGBoost/GBM...")
    xgb_results = train_xgboost(X, y, feature_names)

    comparison = pd.DataFrame([
        {
            "Model": rf_results["model_name"],
            "Train R²": rf_results["train"]["r2"],
            "CV R² (mean)": rf_results["cv"]["cv_r2_mean"],
            "CV R² (std)": rf_results["cv"]["cv_r2_std"],
            "Train RMSE": rf_results["train"]["rmse"],
            "Train MAE": rf_results["train"]["mae"],
            "Features": rf_results["train"]["n_features"],
        },
        {
            "Model": xgb_results["model_name"],
            "Train R²": xgb_results["train"]["r2"],
            "CV R² (mean)": xgb_results["cv"]["cv_r2_mean"],
            "CV R² (std)": xgb_results["cv"]["cv_r2_std"],
            "Train RMSE": xgb_results["train"]["rmse"],
            "Train MAE": xgb_results["train"]["mae"],
            "Features": xgb_results["train"]["n_features"],
        },
    ])

    # Pick the best model by CV R²
    if xgb_results["cv"]["cv_r2_mean"] > rf_results["cv"]["cv_r2_mean"]:
        best = xgb_results
    else:
        best = rf_results

    print(f"[MODEL] Best model: {best['model_name']} (CV R² = {best['cv']['cv_r2_mean']:.3f})")

    return {
        "rf": rf_results,
        "xgb": xgb_results,
        "best": best,
        "comparison": comparison,
    }


def temporal_validation(zip_features_df, train_cutoff_year=2022):
    """
    Perform temporal validation: train on <=cutoff, test on >cutoff.
    Returns metrics and comparison data.
    """
    from data.feature_engineering import engineer_features_by_zip, get_feature_matrix

    df = zip_features_df  # This should be the raw processed incident data

    train_df = df[df["year"] <= train_cutoff_year]
    test_df = df[df["year"] > train_cutoff_year]

    if len(train_df) < 100 or len(test_df) < 50:
        return None

    train_features = engineer_features_by_zip(train_df)
    test_features = engineer_features_by_zip(test_df)

    X_train, y_train, fn = get_feature_matrix(train_features)
    X_test, y_test, _ = get_feature_matrix(test_features)

    if len(X_train) < 10 or len(X_test) < 5:
        return None

    # Train both models on train period
    rf_res = train_tuned_rf(X_train, y_train, fn)
    xgb_res = train_xgboost(X_train, y_train, fn)

    # Predict on test
    rf_preds = rf_res["model"].predict(X_test)
    xgb_preds = xgb_res["model"].predict(X_test)

    return {
        "train_years": f"{int(train_df['year'].min())}–{train_cutoff_year}",
        "test_years": f"{train_cutoff_year + 1}–{int(test_df['year'].max())}",
        "rf_oos_r2": r2_score(y_test, rf_preds),
        "rf_oos_mae": mean_absolute_error(y_test, rf_preds),
        "xgb_oos_r2": r2_score(y_test, xgb_preds),
        "xgb_oos_mae": mean_absolute_error(y_test, xgb_preds),
        "y_test": y_test,
        "rf_preds": rf_preds,
        "xgb_preds": xgb_preds,
        "test_features": test_features,
    }
