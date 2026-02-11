"""Patch app.py to load backtest results from disk cache instead of computing"""

with open("app.py", "r") as f:
    content = f.read()

# ═══ Rolling Temporal: wrap spinner block in cache check ═══
OLD = """                    with st.spinner("Running rolling temporal CV..."):
                        rolling_results = []
                        years_available = sorted(raw_df["year"].unique())"""

NEW = """                    rolling_cache = Path("data/rolling_cv_cache.pkl")
                    years_available = sorted(raw_df["year"].unique())
                    if rolling_cache.exists():
                        import pickle as _pkl
                        with open(rolling_cache, "rb") as _f:
                            rolling_results = _pkl.load(_f)
                    else:
                     with st.spinner("Running rolling temporal CV..."):
                        rolling_results = []"""

content = content.replace(OLD, NEW)

# Close the else for rolling
OLD2 = """                            except Exception:
                                continue

                    if rolling_results:"""

# Find the right one - it's the first occurrence
idx = content.find(OLD2)
if idx > 0:
    # Check if this is in the rolling section (before borough)
    before = content[:idx]
    if "rolling_results" in before[-500:]:
        content = content[:idx] + """                            except Exception:
                                continue

                     if not rolling_cache.exists() and rolling_results:
                        import pickle as _pkl
                        with open(rolling_cache, "wb") as _f:
                            _pkl.dump(rolling_results, _f)

                    if rolling_results:""" + content[idx + len(OLD2):]

# ═══ Borough Holdout: wrap in cache check ═══
OLD_B = """                    with st.spinner("Running borough holdout CV..."):
                        borough_results = []"""

NEW_B = """                    boro_cache = Path("data/borough_cv_cache.pkl")
                    if boro_cache.exists():
                        import pickle as _pkl
                        with open(boro_cache, "rb") as _f:
                            borough_results = _pkl.load(_f)
                    else:
                     with st.spinner("Running borough holdout CV..."):
                        borough_results = []"""

content = content.replace(OLD_B, NEW_B)

# ═══ Classification Rolling: wrap in cache check ═══
OLD_CR = """                    with st.spinner("Running classification rolling CV..."):
                        cls_rolling = []"""

NEW_CR = """                    cls_roll_cache = Path("data/cls_rolling_cache.pkl")
                    if cls_roll_cache.exists():
                        import pickle as _pkl
                        with open(cls_roll_cache, "rb") as _f:
                            cls_rolling = _pkl.load(_f)
                    else:
                     with st.spinner("Running classification rolling CV..."):
                        cls_rolling = []"""

content = content.replace(OLD_CR, NEW_CR)

# ═══ Classification Borough: wrap in cache check ═══
OLD_CB = """                    with st.spinner("Running classification borough holdout..."):
                        cls_borough = []"""

NEW_CB = """                    cls_boro_cache = Path("data/cls_borough_cache.pkl")
                    if cls_boro_cache.exists():
                        import pickle as _pkl
                        with open(cls_boro_cache, "rb") as _f:
                            cls_borough = _pkl.load(_f)
                    else:
                     with st.spinner("Running classification borough holdout..."):
                        cls_borough = []"""

content = content.replace(OLD_CB, NEW_CB)

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] App patched to load from cache ✅")
else:
    print(f"[ERROR] {r.stderr}")

