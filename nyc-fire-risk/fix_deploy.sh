#!/bin/bash
# Run from: /Users/jonathanepstein/Documents/Personal/nyc_fire_app
cd /Users/jonathanepstein/Documents/Personal/nyc_fire_app

# 1. Add matplotlib to requirements.txt
cat > requirements.txt << 'EOF'
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
requests>=2.31.0
shap>=0.43.0
folium>=0.15.0
streamlit-folium>=0.15.0
branca>=0.7.0
matplotlib>=3.7.0
EOF

# 2. Fix the ranked dataframe styling in app.py (around line 220)
python3 << 'PYEOF'
import re

with open("app.py", "r") as f:
    content = f.read()

# Fix 1: Replace the ranked dataframe with background_gradient
old_ranked = '''        st.dataframe(
            ranked[display_cols].style.format({
                "structural_fire_rate": "{:.1%}",
                "risk_score": "{:.3f}",
                "predicted_fires": "{:.0f}",
            }).background_gradient(
                subset=["risk_score"], cmap="YlOrRd"
            ),
            use_container_width=True,
            height=500,
        )'''

new_ranked = '''        display_df = ranked[display_cols].copy()
        display_df["structural_fire_rate"] = display_df["structural_fire_rate"].map("{:.1%}".format)
        display_df["risk_score"] = display_df["risk_score"].map("{:.3f}".format)
        display_df["predicted_fires"] = display_df["predicted_fires"].map("{:.0f}".format)
        st.dataframe(display_df, use_container_width=True, height=500)'''

content = content.replace(old_ranked, new_ranked)

# Fix 2: Replace the constituent zips dataframe with background_gradient
old_czips = '''                        st.dataframe(
                            constituent_zips[["zip_code", "structural_fires", "risk_score", "risk_label"]]
                            .sort_values("risk_score", ascending=False)
                            .style.format({"risk_score": "{:.3f}"})
                            .background_gradient(subset=["risk_score"], cmap="YlOrRd"),
                            use_container_width=True,
                            height=250,
                        )'''

new_czips = '''                        cz = constituent_zips[["zip_code", "structural_fires", "risk_score", "risk_label"]].sort_values("risk_score", ascending=False).copy()
                        cz["risk_score"] = cz["risk_score"].map("{:.3f}".format)
                        st.dataframe(cz, use_container_width=True, height=250)'''

content = content.replace(old_czips, new_czips)

with open("app.py", "w") as f:
    f.write(content)

print("app.py patched successfully")
PYEOF

# 3. Commit and push
git add .
git commit -m "Fix: add matplotlib dep, simplify styled dataframes"
git push

echo ""
echo "Done! Streamlit Cloud will auto-redeploy."
