"""Cache all backtest results to disk to reduce memory usage on Streamlit Cloud"""

with open("app.py", "r") as f:
    content = f.read()

# ═══════════════════════════════════════════════════════════════
# Wrap Rolling Temporal CV in disk cache
# ═══════════════════════════════════════════════════════════════

OLD_ROLLING = '''                    with st.spinner("Running rolling temporal CV..."):
                        rolling_results = []
                        years_available = sorted(raw_df["year"].unique())'''

NEW_ROLLING = '''                    rolling_cache = Path("data/rolling_cv_cache.pkl")
                    if rolling_cache.exists():
                        import pickle
                        with open(rolling_cache, "rb") as _f:
                            rolling_results = pickle.load(_f)
                        years_available = sorted(raw_df["year"].unique())
                    else:
                      with st.spinner("Running rolling temporal CV..."):
                        rolling_results = []
                        years_available = sorted(raw_df["year"].unique())'''

content = content.replace(OLD_ROLLING, NEW_ROLLING)

# Close the else block and save after rolling results
OLD_ROLLING_END = '''                        avg_r2 = np.mean([r["R²"] for r in rolling_results])'''

NEW_ROLLING_END = '''                      if not rolling_cache.exists() and rolling_results:
                            import pickle
                            with open(rolling_cache, "wb") as _f:
                                pickle.dump(rolling_results, _f)

                        avg_r2 = np.mean([r["R²"] for r in rolling_results])'''

content = content.replace(OLD_ROLLING_END, NEW_ROLLING_END)

# ═══════════════════════════════════════════════════════════════
# Wrap Borough Holdout in disk cache
# ═══════════════════════════════════════════════════════════════

OLD_BORO = '''                    with st.spinner("Running borough holdout CV..."):
                        borough_results = []'''

NEW_BORO = '''                    boro_cache = Path("data/borough_cv_cache.pkl")
                    if boro_cache.exists():
                        import pickle
                        with open(boro_cache, "rb") as _f:
                            borough_results = pickle.load(_f)
                    else:
                      with st.spinner("Running borough holdout CV..."):
                        borough_results = []'''

content = content.replace(OLD_BORO, NEW_BORO)

OLD_BORO_END = '''                        avg_boro_r2 = np.mean([r["R²"] for r in borough_results])'''

NEW_BORO_END = '''                      if not boro_cache.exists() and borough_results:
                            import pickle
                            with open(boro_cache, "wb") as _f:
                                pickle.dump(borough_results, _f)

                        avg_boro_r2 = np.mean([r["R²"] for r in borough_results])'''

content = content.replace(OLD_BORO_END, NEW_BORO_END)

# ═══════════════════════════════════════════════════════════════
# Wrap Classification Rolling CV in disk cache
# ═══════════════════════════════════════════════════════════════

OLD_CLS_ROLL = '''                    with st.spinner("Running classification rolling CV..."):
                        cls_rolling = []'''

NEW_CLS_ROLL = '''                    cls_roll_cache = Path("data/cls_rolling_cache.pkl")
                    if cls_roll_cache.exists():
                        import pickle
                        with open(cls_roll_cache, "rb") as _f:
                            cls_rolling = pickle.load(_f)
                    else:
                      with st.spinner("Running classification rolling CV..."):
                        cls_rolling = []'''

content = content.replace(OLD_CLS_ROLL, NEW_CLS_ROLL)

OLD_CLS_ROLL_END = '''                        avg_cls_auc = np.mean([r["AUC"] for r in cls_rolling])'''

NEW_CLS_ROLL_END = '''                      if not cls_roll_cache.exists() and cls_rolling:
                            import pickle
                            with open(cls_roll_cache, "wb") as _f:
                                pickle.dump(cls_rolling, _f)

                        avg_cls_auc = np.mean([r["AUC"] for r in cls_rolling])'''

content = content.replace(OLD_CLS_ROLL_END, NEW_CLS_ROLL_END)

# ═══════════════════════════════════════════════════════════════
# Wrap Classification Borough Holdout in disk cache
# ═══════════════════════════════════════════════════════════════

OLD_CLS_BORO = '''                    with st.spinner("Running classification borough holdout..."):
                        cls_borough = []'''

NEW_CLS_BORO = '''                    cls_boro_cache = Path("data/cls_borough_cache.pkl")
                    if cls_boro_cache.exists():
                        import pickle
                        with open(cls_boro_cache, "rb") as _f:
                            cls_borough = pickle.load(_f)
                    else:
                      with st.spinner("Running classification borough holdout..."):
                        cls_borough = []'''

content = content.replace(OLD_CLS_BORO, NEW_CLS_BORO)

OLD_CLS_BORO_END = '''                        avg_cb_auc = np.mean([r["AUC"] for r in cls_borough])'''

NEW_CLS_BORO_END = '''                      if not cls_boro_cache.exists() and cls_borough:
                            import pickle
                            with open(cls_boro_cache, "wb") as _f:
                                pickle.dump(cls_borough, _f)

                        avg_cb_auc = np.mean([r["AUC"] for r in cls_borough])'''

content = content.replace(OLD_CLS_BORO_END, NEW_CLS_BORO_END)

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] All backtests now disk-cached ✅")
else:
    print(f"[ERROR] {r.stderr}")

