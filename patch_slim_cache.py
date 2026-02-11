"""Patch app.py to load slim cache for cloud deployment"""

with open("app.py", "r") as f:
    content = f.read()

# Make _load_cache try slim cache first
OLD_LOAD = '''def _load_cache():
    """Load model results from disk."""
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)'''

NEW_LOAD = '''def _load_cache():
    """Load model results from disk. Tries slim cache first (for cloud deployment)."""
    slim_cache = Path(__file__).parent / "data" / "model_cache_slim.pkl"
    if slim_cache.exists() and not CACHE_FILE.exists():
        with open(slim_cache, "rb") as f:
            return pickle.load(f)
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)'''

content = content.replace(OLD_LOAD, NEW_LOAD)

# Make _cache_is_fresh also check slim cache
OLD_FRESH = '''def _cache_is_fresh():
    """Check if the disk cache exists and is recent enough."""
    if not CACHE_FILE.exists():
        return False
    import time
    age_hours = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
    return age_hours < CACHE_MAX_AGE_HOURS'''

NEW_FRESH = '''def _cache_is_fresh():
    """Check if the disk cache exists and is recent enough."""
    slim_cache = Path(__file__).parent / "data" / "model_cache_slim.pkl"
    if slim_cache.exists() and not CACHE_FILE.exists():
        return True  # Slim cache is always fresh (pre-built for deployment)
    if not CACHE_FILE.exists():
        return False
    import time
    age_hours = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
    return age_hours < CACHE_MAX_AGE_HOURS'''

content = content.replace(OLD_FRESH, NEW_FRESH)

# Guard raw_df usage in validation tab
OLD_RAW = '''        raw_df = data["raw_df"]'''

NEW_RAW = '''        raw_df = data.get("raw_df", None)'''

content = content.replace(OLD_RAW, NEW_RAW)

# Guard the validation section that needs raw_df
OLD_YEAR_CHECK = '''        if "year" in raw_df.columns and raw_df["year"].nunique() > 2:'''

NEW_YEAR_CHECK = '''        if raw_df is not None and "year" in raw_df.columns and raw_df["year"].nunique() > 2:'''

content = content.replace(OLD_YEAR_CHECK, NEW_YEAR_CHECK)

with open("app.py", "w") as f:
    f.write(content)

import subprocess
r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("[OK] Slim cache support added ✅")
    print("     Cloud: loads data/model_cache_slim.pkl (0.5 MB)")
    print("     Local: still uses .cache/model_cache.pkl when available")
    print("     Validation backtests: loaded from pre-computed pkl files")
else:
    print(f"[ERROR] {r.stderr}")

