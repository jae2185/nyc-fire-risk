#!/usr/bin/env python3
"""
Run this from: /Users/jonathanepstein/Documents/Personal/nyc_fire_app
Usage: python3 fix_both.py
"""
import os
os.chdir("/Users/jonathanepstein/Documents/Personal/nyc_fire_app")

# ═══════════════════════════════════════════════════════════════════════
# FIX 1: fetch_data.py — pull 200K records across all years
# ═══════════════════════════════════════════════════════════════════════
content = open("data/fetch_data.py").read()

old_fetch = '''    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fire_incidents(_self, limit: int = 50000) -> pd.DataFrame:
        """
        Fetch fire incident data from the SODA API.
        Falls back to synthetic data if the API is unavailable.
        """
        all_records = []
        batch_size = 10000
        offsets = range(0, limit, batch_size)

        for offset in offsets:
            url = (
                f"{ENDPOINTS['fire_incidents']['base']}"
                f"?$limit={batch_size}&$offset={offset}"
                f"&$order=incident_date_time DESC"
            )
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    if not data:
                        break
                    all_records.extend(data)
                else:
                    break
            except Exception as e:
                print(f"API request failed at offset {offset}: {e}")
                break
            time.sleep(0.25)  # Be polite to the API

        if all_records:
            df = pd.DataFrame(all_records)
            return df
        else:
            return _self._generate_synthetic_data()'''

new_fetch = '''    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fire_incidents(_self, limit: int = 200000) -> pd.DataFrame:
        """
        Fetch fire incident data from the SODA API.
        Samples across multiple years for representative coverage.
        Falls back to synthetic data if the API is unavailable.
        """
        all_records = []
        batch_size = 10000

        # Fetch data year by year for balanced temporal coverage
        years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
        per_year_limit = limit // len(years)

        for year in years:
            offset = 0
            year_records = 0
            while year_records < per_year_limit:
                fetch_size = min(batch_size, per_year_limit - year_records)
                url = ENDPOINTS["fire_incidents"]["base"]
                params = {
                    "$limit": fetch_size,
                    "$offset": offset,
                    "$where": f"incident_date_time >= '{year}-01-01T00:00:00' AND incident_date_time < '{year + 1}-01-01T00:00:00'",
                    "$order": "incident_date_time",
                }
                try:
                    resp = requests.get(url, params=params, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        if not data:
                            break
                        all_records.extend(data)
                        year_records += len(data)
                        offset += fetch_size
                        if len(data) < fetch_size:
                            break
                    else:
                        break
                except Exception as e:
                    print(f"API request failed for year {year} at offset {offset}: {e}")
                    break
                time.sleep(0.2)

        if all_records:
            df = pd.DataFrame(all_records)
            return df
        else:
            return _self._generate_synthetic_data()'''

if old_fetch in content:
    content = content.replace(old_fetch, new_fetch)
    print("[OK] fetch_data.py: Patched fetch method (year-by-year, 200K)")
else:
    print("[SKIP] fetch_data.py: Old fetch method not found (may already be patched)")

# Also fix default limit in fetch_and_process
content = content.replace(
    "def fetch_and_process(self, limit: int = 50000)",
    "def fetch_and_process(self, limit: int = 200000)"
)

open("data/fetch_data.py", "w").write(content)

# ═══════════════════════════════════════════════════════════════════════
# FIX 2: building_data.py — better API error handling + debug output
# ═══════════════════════════════════════════════════════════════════════
content = open("data/building_data.py").read()

# Add print statements to debug PLUTO API issues
old_pluto_fetch = '''    all_records = []
    offset = 0
    batch_size = min(limit, 5000)

    while offset < limit:
        params["$limit"] = batch_size
        params["$offset"] = offset
        try:
            resp = requests.get(PLUTO_ENDPOINT, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    break
                all_records.extend(data)
                if len(data) < batch_size:
                    break
                offset += batch_size
            else:
                break
        except Exception as e:
            print(f"PLUTO API error at offset {offset}: {e}")
            break
        time.sleep(0.2)

    if all_records:
        return pd.DataFrame(all_records)
    else:
        return _generate_synthetic_buildings(zip_code, borough)'''

new_pluto_fetch = '''    all_records = []
    offset = 0
    batch_size = min(limit, 5000)

    while offset < limit:
        params["$limit"] = batch_size
        params["$offset"] = offset
        try:
            print(f"[PLUTO] Fetching offset={offset}, batch={batch_size}")
            resp = requests.get(PLUTO_ENDPOINT, params=params, timeout=45)
            print(f"[PLUTO] Status: {resp.status_code}, length: {len(resp.content)}")
            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    print("[PLUTO] Empty response, done.")
                    break
                all_records.extend(data)
                print(f"[PLUTO] Got {len(data)} records (total: {len(all_records)})")
                if len(data) < batch_size:
                    break
                offset += batch_size
            else:
                print(f"[PLUTO] Bad status: {resp.status_code} - {resp.text[:200]}")
                break
        except Exception as e:
            print(f"[PLUTO] API error at offset {offset}: {e}")
            break
        time.sleep(0.2)

    if all_records:
        print(f"[PLUTO] Success: {len(all_records)} total buildings from API")
        return pd.DataFrame(all_records)
    else:
        print("[PLUTO] No API data, using synthetic fallback")
        return _generate_synthetic_buildings(zip_code, borough)'''

if old_pluto_fetch in content:
    content = content.replace(old_pluto_fetch, new_pluto_fetch)
    print("[OK] building_data.py: Added PLUTO API debug logging")
else:
    print("[SKIP] building_data.py: Fetch block not found (may differ)")

open("data/building_data.py", "w").write(content)

# ═══════════════════════════════════════════════════════════════════════
# Clear caches
# ═══════════════════════════════════════════════════════════════════════
import shutil
if os.path.exists(".cache"):
    shutil.rmtree(".cache")
    print("[OK] Cleared .cache directory")

print("\n=== All patches applied. Run: streamlit run app.py ===")
