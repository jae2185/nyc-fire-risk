"""
Feature engineering for fire risk prediction.

Constructs features at three spatial resolutions:
- Zip code (finest)
- PUMA / Public Use Microdata Area (neighborhood-level)
- Borough (coarsest)

Features include incident history, fire type ratios, temporal patterns,
response intensity, and trend indicators.
"""

import pandas as pd
import numpy as np
from data.puma_mapping import get_puma_for_zip, get_puma_name, get_puma_borough, get_puma_centroid
from data.fetch_data import NYC_ZIP_COORDS


FEATURE_NAMES = [
    "total_incidents",
    "structural_fire_rate",
    "non_structural_fire_rate",
    "false_alarm_rate",
    "medical_rate",
    "avg_units_onscene",
    "winter_concentration",
    "summer_concentration",
    "trend_slope",
    "incident_volatility",
    "max_monthly_incidents",
    "pct_residential",
    "pct_commercial",
    "unique_years",
    "avg_yearly_incidents",
]


def engineer_features_by_zip(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate incident data to zip-code level and compute features."""
    grouped = df.groupby("zip_code")

    features = pd.DataFrame({
        "zip_code": grouped["zip_code"].first(),
        "total_incidents": grouped.size(),
        "structural_fires": grouped["is_structural_fire"].sum(),
        "non_structural_fires": grouped["is_non_structural_fire"].sum(),
        "false_alarms": grouped["is_false_alarm"].sum(),
        "medical_incidents": grouped["is_medical"].sum(),
        "avg_units_onscene": grouped["units_onscene"].mean(),
    })

    # Rates
    features["structural_fire_rate"] = features["structural_fires"] / features["total_incidents"]
    features["non_structural_fire_rate"] = features["non_structural_fires"] / features["total_incidents"]
    features["false_alarm_rate"] = features["false_alarms"] / features["total_incidents"]
    features["medical_rate"] = features["medical_incidents"] / features["total_incidents"]

    # Seasonal concentration
    monthly = df.groupby(["zip_code", "month"]).size().unstack(fill_value=0)
    winter_months = [12, 1, 2]
    summer_months = [6, 7, 8]
    existing_winter = [m for m in winter_months if m in monthly.columns]
    existing_summer = [m for m in summer_months if m in monthly.columns]

    if existing_winter:
        winter_counts = monthly[existing_winter].sum(axis=1)
        features["winter_concentration"] = winter_counts / features["total_incidents"]
    else:
        features["winter_concentration"] = 0.25

    if existing_summer:
        summer_counts = monthly[existing_summer].sum(axis=1)
        features["summer_concentration"] = summer_counts / features["total_incidents"]
    else:
        features["summer_concentration"] = 0.25

    # Monthly max (peak load)
    if not monthly.empty:
        features["max_monthly_incidents"] = monthly.max(axis=1)
    else:
        features["max_monthly_incidents"] = 0

    # Year-over-year trend
    yearly = df.groupby(["zip_code", "year"]).size().unstack(fill_value=0)
    if yearly.shape[1] >= 2:
        years = sorted(yearly.columns)
        x = np.arange(len(years)).astype(float)
        slopes = []
        volatilities = []
        for zip_code in features.index:
            if zip_code in yearly.index:
                y = yearly.loc[zip_code].values.astype(float)
                if len(y) >= 2:
                    slope = np.polyfit(x[:len(y)], y, 1)[0]
                    slopes.append(slope)
                    volatilities.append(np.std(y))
                else:
                    slopes.append(0)
                    volatilities.append(0)
            else:
                slopes.append(0)
                volatilities.append(0)
        features["trend_slope"] = slopes
        features["incident_volatility"] = volatilities
    else:
        features["trend_slope"] = 0
        features["incident_volatility"] = 0

    # Unique years and average
    year_counts = df.groupby("zip_code")["year"].nunique()
    features["unique_years"] = year_counts
    features["avg_yearly_incidents"] = features["total_incidents"] / features["unique_years"].clip(lower=1)

    # Property type ratios (if available)
    if "property_use_desc" in df.columns:
        prop = df.groupby(["zip_code", "property_use_desc"]).size().unstack(fill_value=0)
        residential_cols = [c for c in prop.columns if any(kw in c.lower() for kw in ["dwelling", "residential", "apartment", "1 or 2 family"])]
        commercial_cols = [c for c in prop.columns if any(kw in c.lower() for kw in ["mercantile", "business", "office", "restaurant"])]

        features["pct_residential"] = prop[residential_cols].sum(axis=1) / features["total_incidents"] if residential_cols else 0
        features["pct_commercial"] = prop[commercial_cols].sum(axis=1) / features["total_incidents"] if commercial_cols else 0
    else:
        features["pct_residential"] = 0.5
        features["pct_commercial"] = 0.2

    features = features.fillna(0)

    # Add coordinates
    features["latitude"] = features["zip_code"].map(lambda z: NYC_ZIP_COORDS.get(z, [None, None])[0])
    features["longitude"] = features["zip_code"].map(lambda z: NYC_ZIP_COORDS.get(z, [None, None])[1])

    # Add PUMA and borough
    features["puma_code"] = features["zip_code"].map(get_puma_for_zip)
    features["puma_name"] = features["puma_code"].map(lambda p: get_puma_name(p) if p else "Unknown")
    features["borough"] = features["puma_code"].map(lambda p: get_puma_borough(p) if p else "Unknown")

    # Monthly distribution (for charts)
    for m in range(1, 13):
        if m in monthly.columns:
            features[f"month_{m}"] = monthly[m]
        else:
            features[f"month_{m}"] = 0

    return features.reset_index(drop=True)


def aggregate_to_puma(zip_features: pd.DataFrame) -> pd.DataFrame:
    """Roll up zip-level features to PUMA level."""
    puma_df = zip_features.dropna(subset=["puma_code"]).copy()

    # Weight by total incidents for rate features
    agg_dict = {
        "total_incidents": "sum",
        "structural_fires": "sum",
        "non_structural_fires": "sum",
        "false_alarms": "sum",
        "medical_incidents": "sum",
        "avg_units_onscene": "mean",
        "trend_slope": "mean",
        "incident_volatility": "mean",
        "max_monthly_incidents": "max",
        "unique_years": "max",
        "zip_code": "count",  # Number of zips in PUMA
    }

    # Add monthly columns
    for m in range(1, 13):
        col = f"month_{m}"
        if col in puma_df.columns:
            agg_dict[col] = "sum"

    puma_agg = puma_df.groupby("puma_code").agg(agg_dict).reset_index()
    puma_agg = puma_agg.rename(columns={"zip_code": "n_zip_codes"})

    # Recompute rates from sums
    puma_agg["structural_fire_rate"] = puma_agg["structural_fires"] / puma_agg["total_incidents"].clip(lower=1)
    puma_agg["non_structural_fire_rate"] = puma_agg["non_structural_fires"] / puma_agg["total_incidents"].clip(lower=1)
    puma_agg["false_alarm_rate"] = puma_agg["false_alarms"] / puma_agg["total_incidents"].clip(lower=1)
    puma_agg["medical_rate"] = puma_agg["medical_incidents"] / puma_agg["total_incidents"].clip(lower=1)
    puma_agg["avg_yearly_incidents"] = puma_agg["total_incidents"] / puma_agg["unique_years"].clip(lower=1)

    # Seasonal concentration from monthly sums
    winter_cols = [f"month_{m}" for m in [12, 1, 2] if f"month_{m}" in puma_agg.columns]
    summer_cols = [f"month_{m}" for m in [6, 7, 8] if f"month_{m}" in puma_agg.columns]
    puma_agg["winter_concentration"] = puma_agg[winter_cols].sum(axis=1) / puma_agg["total_incidents"].clip(lower=1) if winter_cols else 0.25
    puma_agg["summer_concentration"] = puma_agg[summer_cols].sum(axis=1) / puma_agg["total_incidents"].clip(lower=1) if summer_cols else 0.25

    # Property ratios (approximate)
    if "pct_residential" in zip_features.columns:
        res_agg = puma_df.groupby("puma_code").apply(
            lambda x: np.average(x["pct_residential"], weights=x["total_incidents"]) if x["total_incidents"].sum() > 0 else 0.5,
            include_groups=False,
        ).reset_index(name="pct_residential")
        com_agg = puma_df.groupby("puma_code").apply(
            lambda x: np.average(x["pct_commercial"], weights=x["total_incidents"]) if x["total_incidents"].sum() > 0 else 0.2,
            include_groups=False,
        ).reset_index(name="pct_commercial")
        puma_agg = puma_agg.merge(res_agg, on="puma_code", how="left")
        puma_agg = puma_agg.merge(com_agg, on="puma_code", how="left")
    else:
        puma_agg["pct_residential"] = 0.5
        puma_agg["pct_commercial"] = 0.2

    # Add metadata
    puma_agg["puma_name"] = puma_agg["puma_code"].map(get_puma_name)
    puma_agg["borough"] = puma_agg["puma_code"].map(get_puma_borough)
    puma_agg["latitude"] = puma_agg["puma_code"].map(lambda p: (get_puma_centroid(p) or [None])[0])
    puma_agg["longitude"] = puma_agg["puma_code"].map(lambda p: (get_puma_centroid(p) or [None, None])[1])

    return puma_agg


def aggregate_to_borough(zip_features: pd.DataFrame) -> pd.DataFrame:
    """Roll up zip-level features to borough level."""
    boro_df = zip_features.dropna(subset=["borough"]).copy()
    boro_df = boro_df[boro_df["borough"] != "Unknown"]

    agg_dict = {
        "total_incidents": "sum",
        "structural_fires": "sum",
        "non_structural_fires": "sum",
        "false_alarms": "sum",
        "medical_incidents": "sum",
        "avg_units_onscene": "mean",
        "trend_slope": "mean",
        "incident_volatility": "mean",
        "max_monthly_incidents": "max",
        "unique_years": "max",
        "zip_code": "count",
    }

    for m in range(1, 13):
        col = f"month_{m}"
        if col in boro_df.columns:
            agg_dict[col] = "sum"

    boro_agg = boro_df.groupby("borough").agg(agg_dict).reset_index()
    boro_agg = boro_agg.rename(columns={"zip_code": "n_zip_codes"})

    # Recompute rates
    boro_agg["structural_fire_rate"] = boro_agg["structural_fires"] / boro_agg["total_incidents"].clip(lower=1)
    boro_agg["non_structural_fire_rate"] = boro_agg["non_structural_fires"] / boro_agg["total_incidents"].clip(lower=1)
    boro_agg["false_alarm_rate"] = boro_agg["false_alarms"] / boro_agg["total_incidents"].clip(lower=1)
    boro_agg["medical_rate"] = boro_agg["medical_incidents"] / boro_agg["total_incidents"].clip(lower=1)
    boro_agg["avg_yearly_incidents"] = boro_agg["total_incidents"] / boro_agg["unique_years"].clip(lower=1)

    winter_cols = [f"month_{m}" for m in [12, 1, 2] if f"month_{m}" in boro_agg.columns]
    summer_cols = [f"month_{m}" for m in [6, 7, 8] if f"month_{m}" in boro_agg.columns]
    boro_agg["winter_concentration"] = boro_agg[winter_cols].sum(axis=1) / boro_agg["total_incidents"].clip(lower=1) if winter_cols else 0.25
    boro_agg["summer_concentration"] = boro_agg[summer_cols].sum(axis=1) / boro_agg["total_incidents"].clip(lower=1) if summer_cols else 0.25
    boro_agg["pct_residential"] = 0.5
    boro_agg["pct_commercial"] = 0.2

    # Borough centroids
    BORO_CENTROIDS = {
        "Manhattan": [40.758, -73.985],
        "Bronx": [40.845, -73.880],
        "Brooklyn": [40.650, -73.950],
        "Queens": [40.730, -73.830],
        "Staten Island": [40.580, -74.150],
    }
    boro_agg["latitude"] = boro_agg["borough"].map(lambda b: BORO_CENTROIDS.get(b, [None])[0])
    boro_agg["longitude"] = boro_agg["borough"].map(lambda b: BORO_CENTROIDS.get(b, [None, None])[1])

    return boro_agg


def get_feature_matrix(features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract X (features) and y (target: structural fires) from a features DataFrame.
    Returns (X, y, feature_names).
    """
    feature_cols = [c for c in FEATURE_NAMES if c in features_df.columns]
    X = features_df[feature_cols].fillna(0).values
    y = features_df["structural_fires"].fillna(0).values
    return X, y, feature_cols
