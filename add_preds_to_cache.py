"""Add pre-computed validation predictions to slim cache"""
import pickle
import numpy as np
from pathlib import Path

with open(".cache/model_cache.pkl", "rb") as f:
    full_data = pickle.load(f)

with open("data/model_cache_slim.pkl", "rb") as f:
    slim = pickle.load(f)

with open("data/validation_cache.pkl", "rb") as f:
    val = pickle.load(f)

# Import model training functions
import sys
sys.path.insert(0, ".")
from models.enhanced_model import get_enhanced_feature_matrix, train_tuned_rf, train_xgboost
from sklearn.metrics import r2_score, mean_absolute_error

X_train = val["X_train_full"]
X_test = val["X_test_full"]
y_train = val["y_train"]
y_test = val["y_test"]
fn_full = val["fn_full"]

print("Training RF and GBM for validation cache...")
rf_full = train_tuned_rf(X_train, y_train, fn_full)
gbm_full = train_xgboost(X_train, y_train, fn_full)

rf_preds = rf_full["model"].predict(X_test)
gbm_preds = gbm_full["model"].predict(X_test)

rf_r2 = r2_score(y_test, rf_preds)
gbm_r2 = r2_score(y_test, gbm_preds)

val["rf_preds"] = rf_preds
val["gbm_preds"] = gbm_preds
val["rf_r2"] = rf_r2
val["gbm_r2"] = gbm_r2
val["rf_mae"] = mean_absolute_error(y_test, rf_preds)
val["gbm_mae"] = mean_absolute_error(y_test, gbm_preds)
val["rf_cv"] = rf_full["cv"]
val["gbm_cv"] = gbm_full["cv"]
val["rf_name"] = rf_full["model_name"]
val["gbm_name"] = gbm_full["model_name"]

with open("data/validation_cache.pkl", "wb") as f:
    pickle.dump(val, f)

print(f"Updated validation cache with predictions")
print(f"  RF R²={rf_r2:.3f}, GBM R²={gbm_r2:.3f}")
print(f"  Cache size: {Path('data/validation_cache.pkl').stat().st_size / 1024:.1f} KB")

