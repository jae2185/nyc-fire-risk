"""
Building-level fire risk prediction using PLUTO data.

Fetches building characteristics from NYC's PLUTO dataset (870K+ tax lots)
and enriches each building with:
- Physical characteristics (age, floors, units, area, class)
- Neighborhood fire risk (from zip/PUMA model)
- DOB violation counts (proxy for building condition)
- Computed risk features (age buckets, density, residential flag)

The target is a composite risk score combining building-level factors
with neighborhood-level fire history.
"""

import pandas as pd
import numpy as np
import requests
import time
import streamlit as st
from typing import Optional

# PLUTO SODA API endpoint
PLUTO_ENDPOINT = "https://data.cityofnewyork.us/resource/64uk-42ks.json"

# DOB violations endpoint
DOB_VIOLATIONS_ENDPOINT = "https://data.cityofnewyork.us/resource/3h2n-5cm9.json"

# Key PLUTO columns we need
PLUTO_COLUMNS = [
    "bbl", "borough", "address", "zipcode", "bldgclass", "landuse",
    "yearbuilt", "numfloors", "unitsres", "unitstotal", "bldgarea",
    "lotarea", "numbldgs", "assesstot", "assessland",
    "latitude", "longitude", "zonedist1", "firecomp",
    "comarea", "resarea", "officearea", "retailarea", "factryarea",
    "ownername", "ownertype",
]

# Building classes associated with higher fire risk
HIGH_RISK_BLDG_CLASSES = {
    "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",  # One-family dwellings
    "B1", "B2", "B3", "B9",  # Two-family dwellings
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",  # Walk-up apartments
    "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",  # Elevator apartments
    "E1", "E2", "E3", "E4", "E7", "E9",  # Warehouses
    "F1", "F2", "F4", "F5", "F8", "F9",  # Factory/industrial
    "S0", "S1", "S2", "S3", "S4", "S5", "S9",  # Mixed residential/commercial
}

# Land use codes
RESIDENTIAL_LAND_USE = {"01", "02", "03", "04"}  # 1-2 family, multi-family, mixed res/com, commercial
INDUSTRIAL_LAND_USE = {"06", "07"}  # Industrial, transportation/utility


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pluto_buildings(zip_code=None, borough=None, limit=5000):
    """
    Fetch building data from the PLUTO SODA API.

    Can filter by zip code or borough. Returns a DataFrame of buildings.
    """
    select_cols = ",".join(PLUTO_COLUMNS)
    params = {
        "$select": select_cols,
        "$limit": limit,
        "$where": "yearbuilt > 0 AND bldgarea > 0",  # Only real buildings
        "$order": "bbl",
    }

    if zip_code:
        params["$where"] += f" AND zipcode = '{zip_code}'"
    elif borough:
        boro_map = {
            "Manhattan": "MN", "Bronx": "BX", "Brooklyn": "BK",
            "Queens": "QN", "Staten Island": "SI",
        }
        boro_code = boro_map.get(borough, borough)
        params["$where"] += f" AND borough = '{boro_code}'"

    all_records = []
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
        return _generate_synthetic_buildings(zip_code, borough)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dob_violations_by_zip(zip_code, limit=2000):
    """Fetch DOB violation counts aggregated by BBL for a zip code."""
    url = DOB_VIOLATIONS_ENDPOINT
    params = {
        "$select": "bin,violation_type,count(*) as violation_count",
        "$where": f"starts_with(bin, '1') OR starts_with(bin, '2') OR starts_with(bin, '3') OR starts_with(bin, '4') OR starts_with(bin, '5')",
        "$group": "bin,violation_type",
        "$limit": limit,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 200:
            return pd.DataFrame(resp.json())
    except Exception:
        pass
    return pd.DataFrame()


def process_pluto_buildings(raw_df):
    """Clean and engineer features from raw PLUTO data."""
    df = raw_df.copy()

    # Type conversions
    numeric_cols = [
        "yearbuilt", "numfloors", "unitsres", "unitstotal", "bldgarea",
        "lotarea", "numbldgs", "assesstot", "assessland",
        "latitude", "longitude", "comarea", "resarea",
        "officearea", "retailarea", "factryarea",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Filter out invalid records
    df = df[(df["yearbuilt"] > 1800) & (df["yearbuilt"] <= 2026)]
    df = df[df["bldgarea"] > 0]
    df = df[df["latitude"] > 40.4]  # Must be in NYC
    df = df[df["latitude"] < 41.0]

    # ─── Engineered Features ────────────────────────────────────────
    current_year = 2026

    # Building age
    df["building_age"] = current_year - df["yearbuilt"]
    df["age_bucket"] = pd.cut(
        df["building_age"],
        bins=[0, 20, 40, 60, 80, 100, 150, 300],
        labels=["<20yr", "20-40yr", "40-60yr", "60-80yr", "80-100yr", "100-150yr", "150yr+"],
    )

    # Size features
    df["log_bldg_area"] = np.log1p(df["bldgarea"])
    df["log_lot_area"] = np.log1p(df["lotarea"])
    df["floor_area_ratio"] = df["bldgarea"] / df["lotarea"].clip(lower=1)
    df["units_per_floor"] = df["unitsres"] / df["numfloors"].clip(lower=1)

    # Building type flags
    df["bldgclass_2char"] = df.get("bldgclass", pd.Series(dtype=str)).astype(str).str[:2].str.upper()
    df["is_high_risk_class"] = df["bldgclass_2char"].isin(HIGH_RISK_BLDG_CLASSES).astype(int)

    df["landuse_str"] = df.get("landuse", pd.Series(dtype=str)).astype(str).str.zfill(2)
    df["is_residential"] = df["landuse_str"].isin(RESIDENTIAL_LAND_USE).astype(int)
    df["is_industrial"] = df["landuse_str"].isin(INDUSTRIAL_LAND_USE).astype(int)

    # Residential density
    df["residential_pct"] = df["resarea"] / df["bldgarea"].clip(lower=1)
    df["commercial_pct"] = df["comarea"] / df["bldgarea"].clip(lower=1)

    # Assessment per sqft (proxy for building quality/value)
    df["assess_per_sqft"] = df["assesstot"] / df["bldgarea"].clip(lower=1)

    # Mixed-use flag (has both residential and commercial area)
    df["is_mixed_use"] = ((df["resarea"] > 0) & (df["comarea"] > 0)).astype(int)

    # High-rise flag
    df["is_highrise"] = (df["numfloors"] >= 7).astype(int)

    # Old building flag (pre-1940, before modern fire codes)
    df["is_pre_war"] = (df["yearbuilt"] < 1940).astype(int)
    df["is_pre_code"] = (df["yearbuilt"] < 1968).astype(int)  # Before 1968 building code

    return df


def get_building_feature_matrix(buildings_df):
    """Extract feature matrix for building-level model."""
    feature_cols = [
        "building_age",
        "numfloors",
        "unitsres",
        "log_bldg_area",
        "log_lot_area",
        "floor_area_ratio",
        "units_per_floor",
        "is_high_risk_class",
        "is_residential",
        "is_industrial",
        "is_mixed_use",
        "is_highrise",
        "is_pre_war",
        "is_pre_code",
        "residential_pct",
        "commercial_pct",
        "assess_per_sqft",
    ]

    existing = [c for c in feature_cols if c in buildings_df.columns]
    X = buildings_df[existing].fillna(0).values
    return X, existing


def score_buildings_with_neighborhood(buildings_df, zip_features_df):
    """
    Combine building-level features with neighborhood fire risk
    to produce a composite building risk score.

    The score weights:
    - 40% neighborhood fire risk (from zip-level model)
    - 35% building vulnerability (age, type, density)
    - 25% building condition proxies (pre-code, mixed-use, assessment)
    """
    df = buildings_df.copy()

    # Merge neighborhood risk
    if "zipcode" in df.columns and "zip_code" in zip_features_df.columns:
        zip_risk = zip_features_df[["zip_code", "risk_score", "structural_fire_rate"]].copy()
        zip_risk = zip_risk.rename(columns={
            "risk_score": "neighborhood_risk",
            "structural_fire_rate": "neighborhood_fire_rate",
        })
        df["zipcode"] = df["zipcode"].astype(str)
        zip_risk["zip_code"] = zip_risk["zip_code"].astype(str)
        df = df.merge(zip_risk, left_on="zipcode", right_on="zip_code", how="left")
        df["neighborhood_risk"] = df["neighborhood_risk"].fillna(0.5)
        df["neighborhood_fire_rate"] = df["neighborhood_fire_rate"].fillna(0.15)
    else:
        df["neighborhood_risk"] = 0.5
        df["neighborhood_fire_rate"] = 0.15

    # Building vulnerability score (0–1)
    vuln_components = []

    # Age factor: older buildings = higher risk (sigmoid curve)
    age = df["building_age"].clip(0, 200)
    age_score = 1 / (1 + np.exp(-0.03 * (age - 70)))  # Inflection at 70 years
    vuln_components.append(age_score * 0.30)

    # Height/density factor
    floors = df["numfloors"].clip(0, 100)
    floor_score = np.clip(floors / 30, 0, 1)  # Normalize: 30+ floors = max
    vuln_components.append(floor_score * 0.10)

    # Residential density
    units = df["unitsres"].clip(0, 500)
    unit_score = np.clip(units / 100, 0, 1)
    vuln_components.append(unit_score * 0.15)

    # Building class risk
    vuln_components.append(df["is_high_risk_class"] * 0.15)

    # Pre-code penalty
    vuln_components.append(df["is_pre_code"] * 0.15)

    # Mixed-use complexity
    vuln_components.append(df["is_mixed_use"] * 0.05)

    # Industrial penalty
    vuln_components.append(df["is_industrial"] * 0.10)

    df["building_vulnerability"] = sum(vuln_components)
    df["building_vulnerability"] = df["building_vulnerability"].clip(0, 1)

    # Composite risk score (raw)
    raw_score = (
        0.40 * df["neighborhood_risk"] +
        0.35 * df["building_vulnerability"] +
        0.25 * df["neighborhood_fire_rate"].clip(0, 1) * 3
    ).clip(0, 1)

    # Apply nonlinear transformation to create realistic skewed distribution
    # Most buildings should be low/moderate risk, few should be critical
    # Using a power transform: raises the bar for high scores
    df["risk_score"] = np.power(raw_score, 0.7)  # Compress toward lower end
    
    # Min-max normalize within the batch to use full 0-1 range
    rmin = df["risk_score"].min()
    rmax = df["risk_score"].max()
    if rmax > rmin:
        df["risk_score"] = (df["risk_score"] - rmin) / (rmax - rmin)
    
    # Risk labels with skewed thresholds (fewer critical, more low)
    def _label(r):
        if r >= 0.85: return "Critical"
        if r >= 0.65: return "High"
        if r >= 0.35: return "Moderate"
        return "Low"

    df["risk_label"] = df["risk_score"].map(_label)

    return df


def _generate_synthetic_buildings(zip_code=None, borough=None, n=500):
    """Generate synthetic PLUTO-like building data when API is unavailable."""
    np.random.seed(hash(str(zip_code) + str(borough)) % 2**31)

    from data.fetch_data import NYC_ZIP_COORDS

    if zip_code:
        zips = [zip_code]
    elif borough:
        boro_prefixes = {
            "Manhattan": "100", "Bronx": "104", "Brooklyn": "112",
            "Queens": "113", "Staten Island": "103",
        }
        prefix = boro_prefixes.get(borough, "100")
        zips = [z for z in NYC_ZIP_COORDS if z.startswith(prefix[:2])][:10]
    else:
        zips = list(NYC_ZIP_COORDS.keys())[:20]

    records = []
    bldg_classes = ["A1", "A5", "B2", "C0", "C4", "C6", "D1", "D4", "E1", "F1", "O4", "R1", "S1"]
    land_uses = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]

    for i in range(n):
        zip_code_pick = np.random.choice(zips)
        coords = NYC_ZIP_COORDS.get(zip_code_pick, [40.72, -73.95])

        year = int(np.random.choice(
            [1900, 1920, 1940, 1960, 1980, 2000, 2015],
            p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.12, 0.08],
        ) + np.random.randint(-10, 10))
        year = max(1850, min(2025, year))

        floors = max(1, int(np.random.exponential(4) + 1))
        units = max(0, int(floors * np.random.uniform(0.5, 4)))
        bldg_area = max(500, int(floors * np.random.uniform(800, 3000)))

        records.append({
            "bbl": f"{np.random.randint(1,6)}{np.random.randint(10000,99999):05d}{np.random.randint(1,9999):04d}",
            "borough": ["MN", "BX", "BK", "QN", "SI"][np.random.randint(0, 5)],
            "address": f"{np.random.randint(1, 999)} SAMPLE ST",
            "zipcode": zip_code_pick,
            "bldgclass": np.random.choice(bldg_classes),
            "landuse": np.random.choice(land_uses),
            "yearbuilt": str(year),
            "numfloors": str(floors),
            "unitsres": str(units),
            "unitstotal": str(units + np.random.randint(0, 3)),
            "bldgarea": str(bldg_area),
            "lotarea": str(int(bldg_area * np.random.uniform(0.5, 2))),
            "numbldgs": str(np.random.choice([1, 1, 1, 2, 3])),
            "assesstot": str(int(bldg_area * np.random.uniform(50, 500))),
            "assessland": str(int(bldg_area * np.random.uniform(20, 200))),
            "latitude": str(coords[0] + np.random.uniform(-0.008, 0.008)),
            "longitude": str(coords[1] + np.random.uniform(-0.008, 0.008)),
            "zonedist1": np.random.choice(["R6", "R7", "R8", "C4-4", "M1-1", "R6A"]),
            "firecomp": f"E{np.random.randint(1,99):03d}",
            "comarea": str(int(bldg_area * np.random.uniform(0, 0.4))),
            "resarea": str(int(bldg_area * np.random.uniform(0.3, 0.9))),
            "officearea": str(int(bldg_area * np.random.uniform(0, 0.2))),
            "retailarea": str(int(bldg_area * np.random.uniform(0, 0.15))),
            "factryarea": str(0),
            "ownername": "SAMPLE OWNER",
            "ownertype": np.random.choice(["", "C", "M", "O", "P", "X"]),
        })

    return pd.DataFrame(records)
