"""Fix deprecation warnings + add metric help tooltips"""

with open("app.py", "r") as f:
    content = f.read()

# ═══════════════════════════════════════════════════════════════
# TASK 1: Replace use_container_width with width="stretch"
# ═══════════════════════════════════════════════════════════════
content = content.replace('use_container_width=True', 'width="stretch"')
print("[OK] Replaced use_container_width → width='stretch'")

# ═══════════════════════════════════════════════════════════════
# TASK 2: Add help tooltips to metrics
# ═══════════════════════════════════════════════════════════════

# Header metrics (line ~162-165)
content = content.replace(
    'c1.metric("Zones Analyzed", len(active_df))',
    'c1.metric("Zones Analyzed", len(active_df), help="Number of geographic zones included in the analysis at the selected spatial granularity.")'
)
content = content.replace(
    """c2.metric("Structural Fires", f"{int(active_df['structural_fires'].sum()):,}")""",
    """c2.metric("Structural Fires", f"{int(active_df['structural_fires'].sum()):,}", help="Total structural fire incidents across all zones in the training period (2019-2022). Source: FDNY NYFIRS via NYC Open Data.")"""
)
content = content.replace(
    'c3.metric("Critical Zones", len(active_df[active_df["risk_score"] >= 0.75]))',
    'c3.metric("Critical Zones", len(active_df[active_df["risk_score"] >= 0.75]), help="Zones with a model-predicted risk score >= 0.75 (top quartile). These areas have the highest concentration of fire risk factors.")'
)
content = content.replace(
    """c4.metric("Model R\\u00b2", f"{results['train']['r2']:.3f}")""",
    """c4.metric("Model R\\u00b2", f"{results['train']['r2']:.3f}", help="R-squared on training data. Measures how well the model explains variance in fire counts. 1.0 = perfect fit.")"""
)

# Try alternate quote style for c4 if above didn't match
content = content.replace(
    'c4.metric("Model R²", f"{results[\'train\'][\'r2\']:.3f}")',
    'c4.metric("Model R²", f"{results[\'train\'][\'r2\']:.3f}", help="R-squared on training data. Measures how well the model explains variance in fire counts. 1.0 = perfect fit.")'
)

# Model tab metrics
content = content.replace(
    'm1.metric("R² (Train)", f"{results[\'train\'][\'r2\']:.3f}")',
    'm1.metric("R² (Train)", f"{results[\'train\'][\'r2\']:.3f}", help="Proportion of variance in fire counts explained by the model. Higher is better; 1.0 is perfect.")'
)
content = content.replace(
    'm2.metric("RMSE", f"{results[\'train\'][\'rmse\']:.1f}")',
    'm2.metric("RMSE", f"{results[\'train\'][\'rmse\']:.1f}", help="Root Mean Squared Error — average prediction error in fire counts. Lower is better. Penalizes large errors more heavily than MAE.")'
)
content = content.replace(
    'm3.metric("MAE", f"{results[\'train\'][\'mae\']:.1f}")',
    'm3.metric("MAE", f"{results[\'train\'][\'mae\']:.1f}", help="Mean Absolute Error — average absolute difference between predicted and actual fire counts. Lower is better.")'
)

# Validation tab metrics
content = content.replace(
    'vc3.metric("R² (Out-of-Sample)", f"{best_full_r2:.3f}")',
    'vc3.metric("R² (Out-of-Sample)", f"{best_full_r2:.3f}", help="R-squared on held-out 2023-2024 data the model never saw during training. Tests whether the model generalizes to new data.")'
)
content = content.replace(
    'vc4.metric("MAE", f"{best_full_mae:.1f} fires")',
    'vc4.metric("MAE", f"{best_full_mae:.1f} fires", help="Mean Absolute Error on test data — on average, predictions are off by this many fires per zip code.")'
)

# Classification ablation metrics
content = content.replace(
    'mc1.metric("Accuracy", f"{best_acc:.1%}")',
    'mc1.metric("Accuracy", f"{best_acc:.1%}", help="Percentage of zip codes correctly classified as High Risk or Low Risk using only non-fire features.")'
)
content = content.replace(
    'mc3.metric("Precision", f"{best_prec:.1%}")',
    'mc3.metric("Precision", f"{best_prec:.1%}", help="Of zip codes predicted as High Risk, what percentage actually were. Low precision means many false alarms.")'
)
content = content.replace(
    'mc4.metric("Recall", f"{best_rec:.1%}")',
    'mc4.metric("Recall", f"{best_rec:.1%}", help="Of actual High Risk zip codes, what percentage did the model catch. High recall means few missed dangers.")'
)
content = content.replace(
    'mc5.metric("Threshold", f"{optimal_threshold:.2f}")',
    'mc5.metric("Threshold", f"{optimal_threshold:.2f}", help="Optimal probability cutoff (via Youden\\\'s J statistic) for classifying High vs Low Risk. Values above this threshold are flagged High Risk.")'
)

print("[OK] Added help tooltips to metrics")

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] Syntax check passed ✅")
else:
    print(f"[ERROR] {r.stderr}")

