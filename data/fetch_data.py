"""
Data ingestion from NYC Open Data SODA API.

Fetches fire incident data from the FDNY NYFIRS dataset and optionally
enriches with DOB violations and 311 complaints. All endpoints are public
and require no API key (though requests may be throttled).
"""

import pandas as pd
import numpy as np
import requests
import time
import streamlit as st
from typing import Optional

# NYC Open Data SODA API endpoints
ENDPOINTS = {
    "fire_incidents": {
        "base": "https://data.cityofnewyork.us/resource/tm6d-hbzd.json",
        "description": "Incidents Responded to by Fire Companies (NYFIRS)",
    },
}

# Known high-risk zip codes (from FDNY historical data) for synthetic fallback
HIGH_RISK_ZIPS = [
    "10029", "10035", "10037", "10039", "10451", "10452", "10453", "10455",
    "10456", "10457", "10459", "10460", "10472", "10473", "11207", "11208",
    "11212", "11233", "11236",
]
MED_RISK_ZIPS = [
    "10002", "10003", "10009", "10025", "10026", "10027", "10030", "10031",
    "10032", "10033", "10458", "10462", "10467", "10468", "11201", "11203",
    "11206", "11211", "11216", "11221", "11226", "11237",
]

# Approximate NYC zip code centroids
NYC_ZIP_COORDS = {
    "10001": [40.7506, -73.9971], "10002": [40.7157, -73.9863], "10003": [40.7317, -73.9893],
    "10004": [40.6990, -74.0384], "10005": [40.7069, -74.0089], "10006": [40.7094, -74.0131],
    "10007": [40.7134, -74.0076], "10009": [40.7265, -73.9797], "10010": [40.7390, -73.9826],
    "10011": [40.7418, -74.0002], "10012": [40.7258, -73.9981], "10013": [40.7207, -74.0049],
    "10014": [40.7340, -74.0054], "10016": [40.7459, -73.9781], "10017": [40.7524, -73.9727],
    "10018": [40.7549, -73.9929], "10019": [40.7654, -73.9858], "10020": [40.7590, -73.9800],
    "10021": [40.7692, -73.9587], "10022": [40.7583, -73.9680], "10023": [40.7764, -73.9827],
    "10024": [40.7897, -73.9712], "10025": [40.7988, -73.9662], "10026": [40.8027, -73.9533],
    "10027": [40.8113, -73.9534], "10028": [40.7766, -73.9534], "10029": [40.7917, -73.9438],
    "10030": [40.8190, -73.9423], "10031": [40.8243, -73.9497], "10032": [40.8380, -73.9428],
    "10033": [40.8507, -73.9346], "10034": [40.8677, -73.9265], "10035": [40.7980, -73.9303],
    "10036": [40.7593, -73.9903], "10037": [40.8127, -73.9369], "10038": [40.7090, -74.0026],
    "10039": [40.8253, -73.9382], "10040": [40.8583, -73.9299], "10044": [40.7617, -73.9500],
    "10065": [40.7649, -73.9634], "10069": [40.7755, -73.9890], "10075": [40.7706, -73.9554],
    "10128": [40.7810, -73.9500], "10280": [40.7085, -74.0159], "10282": [40.7167, -74.0143],
    "10301": [40.6424, -74.0773], "10302": [40.6324, -74.1374], "10303": [40.6320, -74.1512],
    "10304": [40.6073, -74.0920], "10305": [40.5975, -74.0754], "10306": [40.5719, -74.1142],
    "10307": [40.5124, -74.2483], "10308": [40.5519, -74.1503], "10309": [40.5293, -74.2183],
    "10310": [40.6325, -74.1162], "10312": [40.5448, -74.1793], "10314": [40.5981, -74.1636],
    "10451": [40.8204, -73.9239], "10452": [40.8378, -73.9233], "10453": [40.8530, -73.9124],
    "10454": [40.8073, -73.9192], "10455": [40.8137, -73.9087], "10456": [40.8316, -73.9081],
    "10457": [40.8468, -73.8988], "10458": [40.8632, -73.8882], "10459": [40.8243, -73.8926],
    "10460": [40.8425, -73.8794], "10461": [40.8457, -73.8420], "10462": [40.8438, -73.8569],
    "10463": [40.8799, -73.9065], "10464": [40.8677, -73.8045], "10465": [40.8229, -73.8218],
    "10466": [40.8903, -73.8463], "10467": [40.8735, -73.8712], "10468": [40.8684, -73.8996],
    "10469": [40.8709, -73.8534], "10470": [40.8960, -73.8672], "10471": [40.8982, -73.8955],
    "10472": [40.8292, -73.8683], "10473": [40.8190, -73.8588], "10474": [40.8127, -73.8858],
    "10475": [40.8777, -73.8268],
    "11001": [40.7277, -73.7115], "11004": [40.7444, -73.7113], "11005": [40.7551, -73.7150],
    "11101": [40.7479, -73.9395], "11102": [40.7719, -73.9235], "11103": [40.7629, -73.9124],
    "11104": [40.7443, -73.9206], "11105": [40.7788, -73.9068], "11106": [40.7623, -73.9304],
    "11201": [40.6934, -73.9897], "11203": [40.6494, -73.9346], "11204": [40.6189, -73.9849],
    "11205": [40.6944, -73.9665], "11206": [40.7013, -73.9420], "11207": [40.6713, -73.8930],
    "11208": [40.6685, -73.8713], "11209": [40.6216, -74.0305], "11210": [40.6268, -73.9466],
    "11211": [40.7133, -73.9517], "11212": [40.6633, -73.9128], "11213": [40.6711, -73.9337],
    "11214": [40.5987, -73.9961], "11215": [40.6633, -73.9857], "11216": [40.6813, -73.9493],
    "11217": [40.6817, -73.9779], "11218": [40.6430, -73.9769], "11219": [40.6325, -73.9963],
    "11220": [40.6391, -74.0178], "11221": [40.6917, -73.9294], "11222": [40.7275, -73.9484],
    "11223": [40.5979, -73.9733], "11224": [40.5769, -73.9890], "11225": [40.6632, -73.9537],
    "11226": [40.6462, -73.9574], "11228": [40.6144, -74.0132], "11229": [40.6015, -73.9431],
    "11230": [40.6228, -73.9664], "11231": [40.6800, -74.0003], "11232": [40.6563, -74.0061],
    "11233": [40.6784, -73.9200], "11234": [40.6049, -73.9115], "11235": [40.5843, -73.9489],
    "11236": [40.6396, -73.9010], "11237": [40.7044, -73.9208], "11238": [40.6796, -73.9631],
    "11239": [40.6491, -73.8794],
    "11354": [40.7670, -73.8290], "11355": [40.7536, -73.8197], "11356": [40.7849, -73.8426],
    "11357": [40.7862, -73.8103], "11358": [40.7612, -73.7955], "11360": [40.7816, -73.7815],
    "11361": [40.7653, -73.7724], "11362": [40.7592, -73.7340], "11363": [40.7710, -73.7457],
    "11364": [40.7457, -73.7569], "11365": [40.7396, -73.7928], "11366": [40.7282, -73.7886],
    "11367": [40.7281, -73.8185], "11368": [40.7498, -73.8515], "11369": [40.7634, -73.8739],
    "11370": [40.7657, -73.8910], "11372": [40.7516, -73.8828], "11373": [40.7390, -73.8783],
    "11374": [40.7263, -73.8618], "11375": [40.7210, -73.8447], "11377": [40.7453, -73.9028],
    "11378": [40.7251, -73.9083], "11379": [40.7168, -73.8795], "11385": [40.7002, -73.8892],
    "11411": [40.6942, -73.7356], "11412": [40.6983, -73.7590], "11413": [40.6724, -73.7485],
    "11414": [40.6584, -73.8449], "11415": [40.7076, -73.8270], "11416": [40.6841, -73.8505],
    "11417": [40.6760, -73.8441], "11418": [40.7009, -73.8355], "11419": [40.6882, -73.8232],
    "11420": [40.6731, -73.8171], "11421": [40.6932, -73.8576], "11422": [40.6606, -73.7360],
    "11423": [40.7156, -73.7680], "11426": [40.7356, -73.7223], "11427": [40.7300, -73.7445],
    "11428": [40.7208, -73.7415], "11429": [40.7098, -73.7392], "11430": [40.6518, -73.7906],
    "11432": [40.7154, -73.7920], "11433": [40.6975, -73.7871], "11434": [40.6773, -73.7753],
    "11435": [40.7022, -73.8090], "11436": [40.6759, -73.7952],
}


class FireDataPipeline:
    """Fetches and processes FDNY incident data from NYC Open Data."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.raw_data = None
        self.processed_data = None

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fire_incidents(_self, limit: int = 200000) -> pd.DataFrame:
        """
        Fetch fire incident data from the SODA API.
        Queries by year+month to ensure full 12-month coverage
        despite SODA API throttling (~6250 records per unauthenticated query).
        """
        # Check disk cache first
        import os
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        cache_file = os.path.join(cache_dir, 'fire_incidents.parquet')
        if os.path.exists(cache_file):
            print('[FETCH] Loading from disk cache...')
            df = pd.read_parquet(cache_file)
            print(f"[FETCH] Loaded {len(df)} records from cache")
            return df

        all_records = []

        # Query each year+month separately to ensure full temporal coverage
        years = [2019, 2020, 2021, 2022, 2023, 2024]
        months = list(range(1, 13))
        per_month_limit = 50000

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

        # Save to disk cache
        import os
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'fire_incidents.parquet')

        if all_records:
            _df = pd.DataFrame(all_records)
            _df.to_parquet(cache_file, index=False)
            print(f'[FETCH] Saved {len(_df)} records to disk cache')
            print(f"[FETCH] Total: {len(all_records)} records across {len(years)} years")
            df = pd.DataFrame(all_records)
            return df
        else:
            return _self._generate_synthetic_data()


    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate realistic synthetic fire incident data calibrated against
        known FDNY historical patterns when the API is unavailable.
        """
        np.random.seed(42)
        records = []

        incident_types = [
            "111 - Building fire", "113 - Cooking fire, confined",
            "114 - Chimney fire", "116 - Fuel burner fire",
            "131 - Passenger vehicle fire", "142 - Brush fire",
            "211 - Overpressure rupture - steam", "300 - Rescue call",
            "400 - Hazardous condition", "500 - Service call",
            "600 - Good intent call", "700 - False alarm",
            "900 - Severe weather",
        ]

        property_types = [
            "419 - 1 or 2 family dwelling",
            "429 - Multifamily dwelling",
            "500 - Mercantile, business",
            "213 - Church, temple",
            "161 - Restaurant or cafeteria",
            "000 - Property use not classified",
        ]

        for zip_code, coords in NYC_ZIP_COORDS.items():
            if zip_code in HIGH_RISK_ZIPS:
                factor = 3.0 + np.random.uniform(0, 2)
            elif zip_code in MED_RISK_ZIPS:
                factor = 1.5 + np.random.uniform(0, 1)
            else:
                factor = 0.3 + np.random.uniform(0, 0.7)

            n_incidents = int(80 * factor + np.random.normal(0, 15) * factor)
            n_incidents = max(5, n_incidents)

            for _ in range(n_incidents):
                year = np.random.choice([2018, 2019, 2020, 2021, 2022, 2023],
                                        p=[0.12, 0.14, 0.16, 0.18, 0.20, 0.20])
                month = np.random.choice(range(1, 13),
                                         p=[0.10, 0.09, 0.08, 0.07, 0.07, 0.06,
                                            0.07, 0.07, 0.08, 0.09, 0.10, 0.12])
                day = np.random.randint(1, 29)

                # Higher risk areas get more structural fires
                structural_prob = 0.35 if factor > 2 else (0.20 if factor > 1 else 0.10)
                if np.random.random() < structural_prob:
                    type_idx = np.random.choice([0, 1, 2, 3])
                else:
                    type_idx = np.random.choice(range(4, len(incident_types)))

                units = max(1, int(np.random.exponential(2 * factor) + 1))

                records.append({
                    "im_incident_key": f"{year}{np.random.randint(100000, 999999)}",
                    "incident_type_desc": incident_types[type_idx],
                    "incident_date_time": f"{year}-{month:02d}-{day:02d}T{np.random.randint(0,24):02d}:00:00.000",
                    "zip_code": zip_code,
                    "borough_desc": _zip_to_borough(zip_code),
                    "property_use_desc": np.random.choice(property_types),
                    "units_onscene": str(units),
                    "floor_of_origin": str(np.random.randint(1, 6)),
                })

        return pd.DataFrame(records)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and parse raw incident data."""
        df = df.copy()

        # Normalize zip code field
        if "zip_code" not in df.columns and "zipcode" in df.columns:
            df["zip_code"] = df["zipcode"]

        # Filter to valid NYC zips
        df["zip_code"] = df["zip_code"].astype(str).str.strip()
        df = df[df["zip_code"].str.match(r"^1[01]\d{3}$", na=False)]

        # Parse incident type code (first 3 chars)
        if "incident_type_desc" in df.columns:
            df["incident_code"] = pd.to_numeric(
                df["incident_type_desc"].str[:3], errors="coerce"
            ).fillna(0).astype(int)
        else:
            df["incident_code"] = 0

        # Classify incident type
        df["is_structural_fire"] = df["incident_code"].between(100, 199).astype(int)
        df["is_non_structural_fire"] = df["incident_code"].between(200, 299).astype(int)
        df["is_false_alarm"] = df["incident_code"].between(700, 799).astype(int)
        df["is_medical"] = df["incident_code"].between(300, 399).astype(int)

        # Parse datetime
        if "incident_date_time" in df.columns:
            df["incident_dt"] = pd.to_datetime(df["incident_date_time"], errors="coerce")
            df["year"] = df["incident_dt"].dt.year
            df["month"] = df["incident_dt"].dt.month
        else:
            df["year"] = 2022
            df["month"] = 6

        # Units on scene
        df["units_onscene"] = pd.to_numeric(
            df.get("units_onscene", pd.Series(dtype=float)), errors="coerce"
        ).fillna(1)

        # Add coordinates
        df["latitude"] = df["zip_code"].map(lambda z: NYC_ZIP_COORDS.get(z, [None, None])[0])
        df["longitude"] = df["zip_code"].map(lambda z: NYC_ZIP_COORDS.get(z, [None, None])[1])

        self.processed_data = df
        return df

    def fetch_and_process(self, limit: int = 200000) -> pd.DataFrame:
        """Full pipeline: fetch → clean → return."""
        raw = self.fetch_fire_incidents(limit=limit)
        return self.process(raw)


def _zip_to_borough(zip_code: str) -> str:
    """Quick zip-to-borough lookup."""
    z = int(zip_code)
    if 10001 <= z <= 10282:
        return "MANHATTAN"
    elif 10301 <= z <= 10314:
        return "STATEN ISLAND"
    elif 10451 <= z <= 10475:
        return "BRONX"
    elif 11201 <= z <= 11256:
        return "BROOKLYN"
    elif 11001 <= z <= 11109 or 11351 <= z <= 11697:
        return "QUEENS"
    return "UNKNOWN"
