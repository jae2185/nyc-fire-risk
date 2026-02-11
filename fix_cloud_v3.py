with open("app.py", "r") as f:
    content = f.read()

# Wrap the tier validation chart in try/except in the cloud block
OLD = """                    with val_col2:
                        st.markdown("**Risk Tier Validation**")
                        st.caption("Do zip codes flagged as high-risk actually have more fires later?")
                        tier_thresholds = [0.75, 0.50, 0.25]
                        tier_labels_map = ["Critical", "High", "Moderate", "Low"]
                        pred_scores = best_full_preds / (best_full_preds.max() + 1e-9)
                        tiers = pd.cut(pred_scores, bins=[-0.01, 0.25, 0.50, 0.75, 1.01], labels=["Low", "Moderate", "High", "Critical"])
                        tier_df = pd.DataFrame({"risk_tier": tiers, "actual_fires": y_test})
                        tier_summary = tier_df.groupby("risk_tier", observed=False).agg(
                            avg_fires=("actual_fires", "mean"),
                            median_fires=("actual_fires", "median"),
                            total_fires=("actual_fires", "sum"),
                            zip_count=("actual_fires", "count"),
                        ).reindex(["Critical", "High", "Moderate", "Low"])
                        tier_colors = {"Critical": "#FF3B4E", "High": "#FFAA2B", "Moderate": "#FF6B35", "Low": "#2ECC71"}
                        fig2 = go.Figure()
                        for tier_name in ["Critical", "High", "Moderate", "Low"]:
                            if tier_name in tier_summary.index:
                                row = tier_summary.loc[tier_name]
                                fig2.add_trace(go.Bar(
                                    x=[tier_name], y=[row["avg_fires"]],
                                    marker_color=tier_colors[tier_name],
                                    text=[f"{row['avg_fires']:.1f}"], textposition="auto",
                                    name=tier_name,
                                ))
                        fig2.update_layout(
                            title="Avg Actual Fires (2023+) by Predicted Risk Tier",
                            yaxis_title="Avg Structural Fires per Zip",
                            showlegend=False, height=350,
                            paper_bgcolor="#0B0E11", plot_bgcolor="#131820",
                            font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        st.markdown("**Tier Breakdown**")
                        tier_display = tier_summary.reset_index()
                        st.dataframe(tier_display, width="stretch")"""

# Find which occurrence is in the cloud block (second one)
first_idx = content.find(OLD)
second_idx = content.find(OLD, first_idx + 1) if first_idx >= 0 else -1

if second_idx >= 0:
    NEW = """                    with val_col2:
                        st.markdown("**Risk Tier Validation**")
                        st.caption("Do zip codes flagged as high-risk actually have more fires later?")
                        try:
                            pred_scores = best_full_preds / (best_full_preds.max() + 1e-9)
                            tiers = pd.cut(pred_scores, bins=[-0.01, 0.25, 0.50, 0.75, 1.01], labels=["Low", "Moderate", "High", "Critical"])
                            tier_df = pd.DataFrame({"risk_tier": tiers, "actual_fires": y_test})
                            tier_summary = tier_df.groupby("risk_tier", observed=False).agg(
                                avg_fires=("actual_fires", "mean"),
                                median_fires=("actual_fires", "median"),
                                total_fires=("actual_fires", "sum"),
                                zip_count=("actual_fires", "count"),
                            ).reindex(["Critical", "High", "Moderate", "Low"])
                            tier_colors = {"Critical": "#FF3B4E", "High": "#FFAA2B", "Moderate": "#FF6B35", "Low": "#2ECC71"}
                            fig2 = go.Figure()
                            for tier_name in ["Critical", "High", "Moderate", "Low"]:
                                if tier_name in tier_summary.index:
                                    row = tier_summary.loc[tier_name]
                                    fig2.add_trace(go.Bar(
                                        x=[tier_name], y=[row["avg_fires"]],
                                        marker_color=tier_colors[tier_name],
                                        text=[f"{row['avg_fires']:.1f}"], textposition="auto",
                                        name=tier_name,
                                    ))
                            fig2.update_layout(
                                title="Avg Actual Fires (2023+) by Predicted Risk Tier",
                                yaxis_title="Avg Structural Fires per Zip",
                                showlegend=False, height=350,
                                paper_bgcolor="#0B0E11", plot_bgcolor="#131820",
                                font=dict(color="#E8ECF1", family="JetBrains Mono, monospace"),
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                            st.markdown("**Tier Breakdown**")
                            tier_display = tier_summary.reset_index()
                            st.dataframe(tier_display, width="stretch")
                        except Exception as e:
                            st.warning(f"Tier validation chart unavailable: {e}")"""

    content = content[:second_idx] + NEW + content[second_idx + len(OLD):]

    with open("app.py", "w") as f:
        f.write(content)

    import subprocess
    r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("[OK] Cloud tier chart wrapped in try/except ✅")
    else:
        print(f"[ERROR] {r.stderr}")
else:
    print(f"[INFO] first_idx={first_idx}, second_idx={second_idx}")
    print("[ERROR] Could not find second occurrence of target block")

