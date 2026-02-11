"""Add help tooltips to all metrics"""

with open("app.py", "r") as f:
    content = f.read()

# ═══ Header metrics ═══
content = content.replace(
    'c1.metric("Zones Analyzed", len(active_df))',
    'c1.metric("Zones Analyzed", len(active_df), help="Number of geographic zones (zip codes, PUMAs, or boroughs) included in the analysis.")'
)
content = content.replace(
    'c2.metric("Structural Fires", f"{int(active_df[\'structural_fires\'].sum()):,}")',
    'c2.metric("Structural Fires", f"{int(active_df[\'structural_fires\'].sum()):,}", help="Total structural fire incidents recorded across all zones in the training period (2019–2022).")'
)
content = content.replace(
    'c3.metric("Critical Zones", len(active_df[active_df["risk_score"] >= 0.75]))',
    'c3.metric("Critical Zones", len(active_df[active_df["risk_score"] >= 0.75]), help="Zones with a predicted risk score ≥ 0.75, indicating the highest fire risk based on the model.")'
)
content = content.replace(
    'c4.metric("Model R²", f"{results[\'train\'][\'r2\']:.3f}")',
    'c4.metric("Model R²", f"{results[\'train\'][\'r2\']:.3f}", help="R² (coefficient of determination) on training data. 1.0 = perfect fit. Measures how well the model explains variance in fire counts.")'
)

# ═══ Model tab metrics ═══
content = content.replace(
    'm1.metric("R² (Train)", f"{results[\'train\'][\'r2\']:.3f}")',
    'm1.metric("R² (Train)", f"{results[\'train\'][\'r2\']:.3f}", help="Proportion of variance in fire counts explained by the model on training data. Higher is better; 1.0 is perfect.")'
)
content = content.replace(
    'm2.metric("RMSE", f"{results[\'train\'][\'rmse\']:.1f}")',
    'm2.metric("RMSE", f"{results[\'train\'][\'rmse\']:.1f}", help="Root Mean Squared Error — average prediction error in fire counts. Lower is better. Penalizes large errors more than MAE.")'
)
content = content.replace(
    'm3.metric("MAE", f"{results[\'train\'][\'mae\']:.1f}")',
    'm3.metric("MAE", f"{results[\'train\'][\'mae\']:.1f}", help="Mean Absolute Error — average absolute difference between predicted and actual fire counts. Lower is better.")'
)
content = content.replace(
    '''            st.metric(
                "R² (5-Fold CV)",
                f"{cv['cv_r2_mean']:.3f} ± {cv['cv_r2_std']:.3f}",
            )''',
    '''            st.metric(
                "R² (5-Fold CV)",
                f"{cv['cv_r2_mean']:.3f} ± {cv['cv_r2_std']:.3f}",
                help="Cross-validated R² using 5 random train/test splits. Tests generalization — a score close to training R² means the model isn't overfitting.",
            )'''
)

# ═══ Explorer tab metrics ═══
content = content.replace(
    'st.metric("Structural Fires", int(row["structural_fires"]))',
    'st.metric("Structural Fires", int(row["structural_fires"]), help="Number of structural fire incidents in this zone during the training period.")'
)
content = content.replace(
    'st.metric("Total Incidents", int(row["total_incidents"]))',
    'st.metric("Total Incidents", int(row["total_incidents"]), help="All FDNY responses in this zone, including medical calls, false alarms, and non-structural fires.")'
)
content = content.replace(
    'st.metric("Structural Fire Rate", f"{row[\'structural_fire_rate\']:.1%}")',
    'st.metric("Structural Fire Rate", f"{row[\'structural_fire_rate\']:.1%}", help="Percentage of all incidents that were structural fires. Higher rates may indicate aging infrastructure or code violations.")'
)
content = content.replace(
    'st.metric("Predicted Fires", f"{row[\'predicted_fires\']:.0f}")',
    'st.metric("Predicted Fires", f"{row[\'predicted_fires\']:.0f}", help="Model prediction of expected structural fire count based on building characteristics, complaints, violations, and historical patterns.")'
)
content = content.replace(
    'st.metric("Avg Units on Scene", f"{row[\'avg_units_onscene\']:.1f}")',
    'st.metric("Avg Units on Scene", f"{row[\'avg_units_onscene\']:.1f}", help="Average number of FDNY units dispatched per incident. Higher values suggest more severe incidents requiring greater response.")'
)

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] Help tooltips added to all metrics ✅")
else:
    print(f"[ERROR] {r.stderr}")

