#!/usr/bin/env python3
"""
Fix: SODA API without app token throttles at ~6250 records per query.
Solution: Query by year+month (96 queries) to get representative coverage.

Run from: /Users/jonathanepstein/Documents/Personal/nyc_fire_app
"""
import os
os.chdir("/Users/jonathanepstein/Documents/Personal/nyc_fire_app")

content = open("data/fetch_data.py").read()

# Find and replace the entire fetch_fire_incidents method
# We'll locate it by the def line and the next def line
lines = content.split("\n")

start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if "def fetch_fire_incidents" in line:
        # Find the decorator line above it
        start_idx = i - 1 if i > 0 and "cache" in lines[i-1] else i
        # Also check for blank line with decorator
        if i >= 2 and "cache" in lines[i-2]:
            start_idx = i - 2
    if start_idx is not None and i > start_idx + 5 and line.strip().startswith("def ") and "fetch_fire" not in line:
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print("[ERROR] Could not find fetch_fire_incidents method boundaries")
    print(f"  start_idx={start_idx}, end_idx={end_idx}")
    exit(1)

print(f"[OK] Found method at lines {start_idx+1}-{end_idx}")

new_method = '''    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fire_incidents(_self, limit: int = 200000) -> pd.DataFrame:
        """
        Fetch fire incident data from the SODA API.
        Queries by year+month to ensure full 12-month coverage
        despite SODA API throttling (~6250 records per unauthenticated query).
        """
        all_records = []

        # Query each year+month separately to ensure full temporal coverage
        years = [2019, 2020, 2021, 2022, 2023, 2024]
        months = list(range(1, 13))
        per_month_limit = max(1000, limit // (len(years) * 12))

        for year in years:
            year_count = 0
            for month in months:
                # Build date range for this month
                if month == 12:
                    next_year, next_month = year + 1, 1
                else:
                    next_year, next_month = year, month + 1

                date_start = f"{year}-{month:02d}-01T00:00:00"
                date_end = f"{next_year}-{next_month:02d}-01T00:00:00"

                params = {
                    "$limit": per_month_limit,
                    "$where": f"incident_date_time >= '{date_start}' AND incident_date_time < '{date_end}'",
                    "$order": "incident_date_time",
                }
                try:
                    resp = requests.get(
                        ENDPOINTS["fire_incidents"]["base"],
                        params=params,
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data:
                            all_records.extend(data)
                            year_count += len(data)
                except Exception as e:
                    print(f"[FETCH] Error {year}-{month:02d}: {e}")
                time.sleep(0.15)

            print(f"[FETCH] Year {year}: {year_count} records")

        if all_records:
            print(f"[FETCH] Total: {len(all_records)} records across {len(years)} years")
            df = pd.DataFrame(all_records)
            return df
        else:
            return _self._generate_synthetic_data()

'''

# Replace the method
new_lines = lines[:start_idx] + new_method.split("\n") + lines[end_idx:]
content = "\n".join(new_lines)

open("data/fetch_data.py", "w").write(content)
print("[OK] Replaced fetch method with year+month strategy")
print(f"[INFO] Will query {6 * 12} = 72 month-buckets")
print("[INFO] Run: pkill -f streamlit && rm -rf .cache && streamlit run app.py")
