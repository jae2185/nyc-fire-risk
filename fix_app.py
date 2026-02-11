"""Fix app.py: remove duplicate ablation, keep classification, restore main guard"""

with open("app.py", "r") as f:
    content = f.read()

# Step 1: Find and remove everything after 'if __name__ == "__main__":'
# that isn't just main()
main_guard = 'if __name__ == "__main__":'
idx = content.index(main_guard)

# Everything before the main guard
before_main = content[:idx]

# Step 2: Remove the OLD ablation section (R² regression) 
# It starts at "# ── Ablation Analysis ──"
# and ends before "# ── Methodology ──"
old_abl_start = before_main.index("                    # ── Ablation Analysis ──")
old_abl_end = before_main.index("                    # ── Methodology ──")
before_main = before_main[:old_abl_start] + before_main[old_abl_end:]

# Step 3: Also remove old ablation model comparison rows that reference rf_abl_r2
# The old comparison_data has 4 rows; we only want the first 2
old_comp_start = before_main.index('                    comparison_data = [')
old_comp_end = before_main.index('                    st.dataframe(pd.DataFrame(comparison_data)')
# Replace with just the full-model rows
new_comp = '''                    comparison_data = [
                        {"Model": "Tuned RF (all features)", "OOS R²": f"{rf_oos_r2:.3f}", "OOS MAE": f"{rf_oos_mae:.1f}", "CV R²": f"{rf_full['cv']['cv_r2_mean']:.3f}", "Features": len(fn_full)},
                        {"Model": "GBM (all features)", "OOS R²": f"{gbm_oos_r2:.3f}", "OOS MAE": f"{gbm_oos_mae:.1f}", "CV R²": f"{gbm_full['cv']['cv_r2_mean']:.3f}", "Features": len(fn_full)},
                    ]
'''
before_main = before_main[:old_comp_start] + new_comp + before_main[old_comp_end:]

# Step 4: Restore clean ending
final = before_main.rstrip() + "\n\n\nif __name__ == \"__main__\":\n    main()\n"

with open("app.py", "w") as f:
    f.write(final)

print("[OK] Fixed app.py")
print("  - Removed old ablation regression section")  
print("  - Removed duplicate code after if __name__")
print("  - Classification block preserved")
print("  - Clean main() guard restored")

# Verify
lines = final.split('\n')
print(f"  - Total lines: {len(lines)}")
