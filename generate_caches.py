"""Run all backtests locally and save results to pickle files."""
import sys, os
sys.path.insert(0, ".")
os.environ["STREAMLIT_RUNTIME"] = "false"

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from data.fetch_data import FireDataPipeline
from data.feature_engineering import engineer_features_by_zip
from data.enrichment import enrich_zip_features
from models.enhanced_model import get_enhanced_feature_matrix

print("Fetching data...")
pipeline = FireDataPipeline()
raw_df = pipeline.fetch_and_process(limit=50000)
years = sorted(raw_df["year"].unique())
print(f"Years available: {years}")

train_years = raw_df[raw_df["year"] <= 2022]
test_years = raw_df[raw_df["year"] >= 2023]

print("Engineering features...")
train_features = engineer_features_by_zip(train_years)
test_features = engineer_features_by_zip(test_years)
try:
    train_features = enrich_zip_features(train_features)
    test_features = enrich_zip_features(test_features)
except Exception as e:
    print(f"Enrichment warning: {e}")

X_train_full, y_train, fn_full = get_enhanced_feature_matrix(train_features)
X_test_full, y_test, _ = get_enhanced_feature_matrix(test_features)

incident_features = {"structural_fires", "total_incidents", "non_structural_fires", "false_alarms", "medical_calls", "structural_fire_rate", "false_alarm_rate", "medical_rate", "avg_units_onscene", "winter_concentration", "summer_concentration", "trend_slope", "incident_volatility", "max_monthly_incidents", "avg_yearly_incidents", "complaints_per_incident"}

# ═══ 1. Rolling Temporal CV (Full Model) ═══
print("\n=== Rolling Temporal CV (Full Model) ===")
rolling_results = []
for test_yr in years[2:]:
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
        except: pass
        X_tr, y_tr, fn_r = get_enhanced_feature_matrix(tr_feat)
        X_te, y_te, _ = get_enhanced_feature_matrix(te_feat)
        if len(X_tr) < 10 or len(X_te) < 5:
            continue
        gbm = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        gbm.fit(X_tr, y_tr)
        preds = gbm.predict(X_te)
        r2 = r2_score(y_te, preds)
        mae = mean_absolute_error(y_te, preds)
        rolling_results.append({"Train Window": f"{int(train_slice['year'].min())}-{int(train_slice['year'].max())}", "Test Year": int(test_yr), "R²": r2, "MAE": mae, "Train Zips": len(X_tr), "Test Zips": len(X_te)})
        print(f"  Test {test_yr}: R²={r2:.3f}, MAE={mae:.1f}")
    except Exception as e:
        print(f"  Test {test_yr}: FAILED - {e}")

with open("data/rolling_cv_cache.pkl", "wb") as f:
    pickle.dump(rolling_results, f)
print(f"Saved {len(rolling_results)} rolling results")

# ═══ 2. Borough Holdout (Full Model) ═══
print("\n=== Borough Holdout (Full Model) ===")
borough_results = []
zip_borough = {}
for z in range(10451, 10476): zip_borough[str(z)] = "Bronx"
for z in list(range(10001, 10041)) + list(range(10101, 10200)): zip_borough[str(z)] = "Manhattan"
for z in list(range(11201, 11257)): zip_borough[str(z)] = "Brooklyn"
for z in list(range(11001, 11010)) + list(range(11351, 11698)): zip_borough[str(z)] = "Queens"
for z in range(10301, 10315): zip_borough[str(z)] = "Staten Island"

all_features = pd.concat([train_features, test_features], ignore_index=True)
all_features["_borough"] = all_features["zip_code"].astype(str).map(zip_borough)
X_all, y_all, fn_all = get_enhanced_feature_matrix(all_features)

for boro in sorted(all_features["_borough"].dropna().unique()):
    mask = (all_features["_borough"] == boro).values
    if mask.sum() < 3 or (~mask).sum() < 10:
        continue
    try:
        gbm = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        gbm.fit(X_all[~mask], y_all[~mask])
        preds = gbm.predict(X_all[mask])
        r2 = r2_score(y_all[mask], preds)
        mae = mean_absolute_error(y_all[mask], preds)
        borough_results.append({"Borough": boro, "R²": r2, "MAE": mae, "Zips": int(mask.sum())})
        print(f"  {boro}: R²={r2:.3f}, MAE={mae:.1f}")
    except Exception as e:
        print(f"  {boro}: FAILED - {e}")

with open("data/borough_cv_cache.pkl", "wb") as f:
    pickle.dump(borough_results, f)
print(f"Saved {len(borough_results)} borough results")

# ═══ 3. Classification Rolling CV ═══
print("\n=== Classification Rolling CV ===")
cls_rolling = []
for test_yr in years[2:]:
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
        except: pass
        X_tr, y_tr, fn_r = get_enhanced_feature_matrix(tr_feat)
        X_te, y_te, _ = get_enhanced_feature_matrix(te_feat)
        if len(X_tr) < 10 or len(X_te) < 5:
            continue
        abl_idx = [fn_r.index(c) for c in fn_r if c not in incident_features]
        X_tr_a, X_te_a = X_tr[:, abl_idx], X_te[:, abl_idx]
        med = np.median(y_tr)
        y_tr_c, y_te_c = (y_tr > med).astype(int), (y_te > med).astype(int)
        if len(np.unique(y_te_c)) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42)
        clf.fit(X_tr_a, y_tr_c)
        proba = clf.predict_proba(X_te_a)[:, 1]
        auc = roc_auc_score(y_te_c, proba)
        cls_rolling.append({"Train Window": f"{int(train_slice['year'].min())}-{int(train_slice['year'].max())}", "Test Year": int(test_yr), "AUC": auc, "Test Zips": len(X_te)})
        print(f"  Test {test_yr}: AUC={auc:.3f}")
    except Exception as e:
        print(f"  Test {test_yr}: FAILED - {e}")

with open("data/cls_rolling_cache.pkl", "wb") as f:
    pickle.dump(cls_rolling, f)
print(f"Saved {len(cls_rolling)} classification rolling results")

# ═══ 4. Classification Borough Holdout ═══
print("\n=== Classification Borough Holdout ===")
cls_borough = []
abl_idx_all = [fn_all.index(c) for c in fn_all if c not in incident_features]
X_all_abl = X_all[:, abl_idx_all]

for boro in sorted(all_features["_borough"].dropna().unique()):
    mask = (all_features["_borough"] == boro).values
    if mask.sum() < 3 or (~mask).sum() < 10:
        continue
    try:
        y_tr_b, y_te_b = y_all[~mask], y_all[mask]
        med = np.median(y_tr_b)
        y_tr_c, y_te_c = (y_tr_b > med).astype(int), (y_te_b > med).astype(int)
        if len(np.unique(y_te_c)) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42)
        clf.fit(X_all_abl[~mask], y_tr_c)
        proba = clf.predict_proba(X_all_abl[mask])[:, 1]
        auc = roc_auc_score(y_te_c, proba)
        cls_borough.append({"Borough": boro, "AUC": auc, "Zips": int(mask.sum())})
        print(f"  {boro}: AUC={auc:.3f}")
    except Exception as e:
        print(f"  {boro}: FAILED - {e}")

with open("data/cls_borough_cache.pkl", "wb") as f:
    pickle.dump(cls_borough, f)
print(f"Saved {len(cls_borough)} classification borough results")

print("\n✅ All caches generated!")
for p in Path("data").glob("*cache*.pkl"):
    print(f"  {p} ({p.stat().st_size / 1024:.1f} KB)")

