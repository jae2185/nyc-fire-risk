"""
Zip Code to PUMA (Public Use Microdata Area) crosswalk for NYC.

PUMAs are ~100k-population geographic units used by the Census Bureau for the
American Community Survey Public Use Microdata Sample (PUMS). They sit between
zip codes and boroughs in granularity, making them ideal for neighborhood-level
analysis that can be enriched with ACS microdata.

NYC has 55 PUMAs (2020 vintage). This module provides the crosswalk from zip
codes to PUMAs and PUMA metadata.
"""

import pandas as pd

# NYC PUMA definitions (2020 vintage) with approximate neighborhood names
# Source: NYC Dept of City Planning
PUMA_METADATA = {
    # Manhattan (36061)
    "3801": {"name": "Washington Heights & Inwood", "borough": "Manhattan", "boro_code": "1"},
    "3802": {"name": "Hamilton Heights & Manhattanville", "borough": "Manhattan", "boro_code": "1"},
    "3803": {"name": "Central Harlem", "borough": "Manhattan", "boro_code": "1"},
    "3804": {"name": "East Harlem", "borough": "Manhattan", "boro_code": "1"},
    "3805": {"name": "Upper West Side & West Side", "borough": "Manhattan", "boro_code": "1"},
    "3806": {"name": "Upper East Side", "borough": "Manhattan", "boro_code": "1"},
    "3807": {"name": "Midtown West & Chelsea", "borough": "Manhattan", "boro_code": "1"},
    "3808": {"name": "Midtown East & Turtle Bay", "borough": "Manhattan", "boro_code": "1"},
    "3809": {"name": "Gramercy & Murray Hill", "borough": "Manhattan", "boro_code": "1"},
    "3810": {"name": "Greenwich Village & SoHo", "borough": "Manhattan", "boro_code": "1"},
    "3811": {"name": "Lower East Side & Chinatown", "borough": "Manhattan", "boro_code": "1"},
    "3812": {"name": "Lower Manhattan & Financial District", "borough": "Manhattan", "boro_code": "1"},
    # Bronx (36005)
    "3701": {"name": "Mott Haven & Hunts Point", "borough": "Bronx", "boro_code": "2"},
    "3702": {"name": "Morrisania & Crotona", "borough": "Bronx", "boro_code": "2"},
    "3703": {"name": "Highbridge & Concourse", "borough": "Bronx", "boro_code": "2"},
    "3704": {"name": "University Heights & Fordham", "borough": "Bronx", "boro_code": "2"},
    "3705": {"name": "Kingsbridge & Riverdale", "borough": "Bronx", "boro_code": "2"},
    "3706": {"name": "Belmont & East Tremont", "borough": "Bronx", "boro_code": "2"},
    "3707": {"name": "Parkchester & Soundview", "borough": "Bronx", "boro_code": "2"},
    "3708": {"name": "Throgs Neck & Co-op City", "borough": "Bronx", "boro_code": "2"},
    "3709": {"name": "Williamsbridge & Baychester", "borough": "Bronx", "boro_code": "2"},
    "3710": {"name": "Pelham Parkway & Morris Park", "borough": "Bronx", "boro_code": "2"},
    # Brooklyn (36047)
    "4001": {"name": "Greenpoint & Williamsburg", "borough": "Brooklyn", "boro_code": "3"},
    "4002": {"name": "Downtown Brooklyn & DUMBO", "borough": "Brooklyn", "boro_code": "3"},
    "4003": {"name": "Fort Greene & Brooklyn Heights", "borough": "Brooklyn", "boro_code": "3"},
    "4004": {"name": "Bushwick", "borough": "Brooklyn", "boro_code": "3"},
    "4005": {"name": "Bedford-Stuyvesant", "borough": "Brooklyn", "boro_code": "3"},
    "4006": {"name": "Park Slope & Carroll Gardens", "borough": "Brooklyn", "boro_code": "3"},
    "4007": {"name": "Sunset Park & Windsor Terrace", "borough": "Brooklyn", "boro_code": "3"},
    "4008": {"name": "Crown Heights North", "borough": "Brooklyn", "boro_code": "3"},
    "4009": {"name": "Crown Heights South & Prospect Lefferts", "borough": "Brooklyn", "boro_code": "3"},
    "4010": {"name": "Bay Ridge & Dyker Heights", "borough": "Brooklyn", "boro_code": "3"},
    "4011": {"name": "Bensonhurst & Bath Beach", "borough": "Brooklyn", "boro_code": "3"},
    "4012": {"name": "Borough Park & Kensington", "borough": "Brooklyn", "boro_code": "3"},
    "4013": {"name": "Flatbush & Midwood", "borough": "Brooklyn", "boro_code": "3"},
    "4014": {"name": "East Flatbush", "borough": "Brooklyn", "boro_code": "3"},
    "4015": {"name": "Brownsville & Ocean Hill", "borough": "Brooklyn", "boro_code": "3"},
    "4016": {"name": "East New York & Cypress Hills", "borough": "Brooklyn", "boro_code": "3"},
    "4017": {"name": "Canarsie & Flatlands", "borough": "Brooklyn", "boro_code": "3"},
    "4018": {"name": "Sheepshead Bay & Gravesend", "borough": "Brooklyn", "boro_code": "3"},
    # Queens (36081)
    "4101": {"name": "Astoria & Long Island City", "borough": "Queens", "boro_code": "4"},
    "4102": {"name": "Sunnyside & Woodside", "borough": "Queens", "boro_code": "4"},
    "4103": {"name": "Jackson Heights & Elmhurst", "borough": "Queens", "boro_code": "4"},
    "4104": {"name": "Flushing & Whitestone", "borough": "Queens", "boro_code": "4"},
    "4105": {"name": "Bayside & Little Neck", "borough": "Queens", "boro_code": "4"},
    "4106": {"name": "Fresh Meadows & Hillcrest", "borough": "Queens", "boro_code": "4"},
    "4107": {"name": "Ridgewood & Middle Village", "borough": "Queens", "boro_code": "4"},
    "4108": {"name": "Forest Hills & Rego Park", "borough": "Queens", "boro_code": "4"},
    "4109": {"name": "Woodhaven & Richmond Hill", "borough": "Queens", "boro_code": "4"},
    "4110": {"name": "South Ozone Park & Howard Beach", "borough": "Queens", "boro_code": "4"},
    "4111": {"name": "Jamaica & Hollis", "borough": "Queens", "boro_code": "4"},
    "4112": {"name": "Queens Village & Cambria Heights", "borough": "Queens", "boro_code": "4"},
    "4113": {"name": "Far Rockaway & Broad Channel", "borough": "Queens", "boro_code": "4"},
    "4114": {"name": "Corona & North Corona", "borough": "Queens", "boro_code": "4"},
    # Staten Island (36085)
    "3901": {"name": "St. George & Stapleton", "borough": "Staten Island", "boro_code": "5"},
    "3902": {"name": "Port Richmond & Mariners Harbor", "borough": "Staten Island", "boro_code": "5"},
    "3903": {"name": "Tottenville & Great Kills", "borough": "Staten Island", "boro_code": "5"},
}

# Zip code to PUMA crosswalk (approximate, based on geographic overlap)
# A zip code maps to its primary PUMA (the one with greatest area overlap)
ZIP_TO_PUMA = {
    # Manhattan
    "10001": "3807", "10002": "3811", "10003": "3810", "10004": "3812",
    "10005": "3812", "10006": "3812", "10007": "3812", "10009": "3811",
    "10010": "3809", "10011": "3807", "10012": "3810", "10013": "3810",
    "10014": "3810", "10016": "3809", "10017": "3808", "10018": "3807",
    "10019": "3807", "10020": "3808", "10021": "3806", "10022": "3808",
    "10023": "3805", "10024": "3805", "10025": "3805", "10026": "3803",
    "10027": "3803", "10028": "3806", "10029": "3804", "10030": "3802",
    "10031": "3802", "10032": "3801", "10033": "3801", "10034": "3801",
    "10035": "3804", "10036": "3807", "10037": "3803", "10038": "3812",
    "10039": "3802", "10040": "3801", "10044": "3806", "10065": "3806",
    "10069": "3805", "10075": "3806", "10128": "3806", "10280": "3812",
    "10282": "3812",
    # Bronx
    "10451": "3701", "10452": "3703", "10453": "3704", "10454": "3701",
    "10455": "3701", "10456": "3702", "10457": "3706", "10458": "3704",
    "10459": "3702", "10460": "3706", "10461": "3710", "10462": "3710",
    "10463": "3705", "10464": "3708", "10465": "3708", "10466": "3709",
    "10467": "3709", "10468": "3704", "10469": "3709", "10470": "3709",
    "10471": "3705", "10472": "3707", "10473": "3707", "10474": "3701",
    "10475": "3708",
    # Brooklyn
    "11201": "4002", "11203": "4014", "11204": "4012", "11205": "4003",
    "11206": "4004", "11207": "4016", "11208": "4016", "11209": "4010",
    "11210": "4013", "11211": "4001", "11212": "4015", "11213": "4008",
    "11214": "4011", "11215": "4006", "11216": "4005", "11217": "4003",
    "11218": "4012", "11219": "4012", "11220": "4007", "11221": "4005",
    "11222": "4001", "11223": "4018", "11224": "4018", "11225": "4009",
    "11226": "4013", "11228": "4010", "11229": "4018", "11230": "4013",
    "11231": "4006", "11232": "4007", "11233": "4015", "11234": "4017",
    "11235": "4018", "11236": "4017", "11237": "4004", "11238": "4008",
    "11239": "4016",
    # Queens
    "11101": "4101", "11102": "4101", "11103": "4101", "11104": "4102",
    "11105": "4101", "11106": "4101", "11354": "4104", "11355": "4104",
    "11356": "4104", "11357": "4104", "11358": "4105", "11360": "4105",
    "11361": "4105", "11362": "4105", "11363": "4105", "11364": "4106",
    "11365": "4106", "11366": "4106", "11367": "4106", "11368": "4114",
    "11369": "4114", "11370": "4114", "11372": "4103", "11373": "4103",
    "11374": "4108", "11375": "4108", "11377": "4102", "11378": "4107",
    "11379": "4107", "11385": "4107", "11411": "4112", "11412": "4111",
    "11413": "4112", "11414": "4110", "11415": "4109", "11416": "4109",
    "11417": "4110", "11418": "4109", "11419": "4109", "11420": "4110",
    "11421": "4109", "11422": "4112", "11423": "4111", "11426": "4112",
    "11427": "4112", "11428": "4112", "11429": "4112", "11430": "4110",
    "11432": "4111", "11433": "4111", "11434": "4111", "11435": "4111",
    "11436": "4110",
    # Staten Island
    "10301": "3901", "10302": "3902", "10303": "3902", "10304": "3901",
    "10305": "3901", "10306": "3903", "10307": "3903", "10308": "3903",
    "10309": "3903", "10310": "3902", "10312": "3903", "10314": "3902",
}

# PUMA approximate centroids for map rendering
PUMA_CENTROIDS = {
    "3801": [40.860, -73.930], "3802": [40.822, -73.947], "3803": [40.810, -73.945],
    "3804": [40.795, -73.938], "3805": [40.790, -73.972], "3806": [40.774, -73.957],
    "3807": [40.752, -73.994], "3808": [40.756, -73.972], "3809": [40.742, -73.982],
    "3810": [40.728, -73.998], "3811": [40.718, -73.984], "3812": [40.710, -74.008],
    "3701": [40.813, -73.916], "3702": [40.832, -73.904], "3703": [40.838, -73.922],
    "3704": [40.858, -73.899], "3705": [40.886, -73.905], "3706": [40.851, -73.885],
    "3707": [40.828, -73.862], "3708": [40.845, -73.822], "3709": [40.878, -73.855],
    "3710": [40.850, -73.848],
    "4001": [40.720, -73.950], "4002": [40.693, -73.988], "4003": [40.690, -73.970],
    "4004": [40.700, -73.925], "4005": [40.685, -73.940], "4006": [40.672, -73.988],
    "4007": [40.645, -74.005], "4008": [40.672, -73.948], "4009": [40.658, -73.948],
    "4010": [40.618, -74.022], "4011": [40.608, -73.993], "4012": [40.632, -73.988],
    "4013": [40.635, -73.962], "4014": [40.648, -73.928], "4015": [40.665, -73.910],
    "4016": [40.662, -73.882], "4017": [40.625, -73.902], "4018": [40.592, -73.958],
    "4101": [40.760, -73.925], "4102": [40.748, -73.908], "4103": [40.748, -73.878],
    "4104": [40.770, -73.825], "4105": [40.770, -73.775], "4106": [40.738, -73.798],
    "4107": [40.712, -73.892], "4108": [40.722, -73.852], "4109": [40.695, -73.838],
    "4110": [40.668, -73.828], "4111": [40.705, -73.792], "4112": [40.725, -73.738],
    "4113": [40.598, -73.785], "4114": [40.755, -73.862],
    "3901": [40.625, -74.080], "3902": [40.628, -74.138], "3903": [40.545, -74.175],
}


def get_puma_for_zip(zip_code: str) -> str | None:
    """Return the primary PUMA code for a given NYC zip code."""
    return ZIP_TO_PUMA.get(str(zip_code))


def get_puma_name(puma_code: str) -> str:
    """Return the human-readable neighborhood name for a PUMA."""
    meta = PUMA_METADATA.get(str(puma_code), {})
    return meta.get("name", f"PUMA {puma_code}")


def get_puma_borough(puma_code: str) -> str:
    """Return the borough for a PUMA."""
    meta = PUMA_METADATA.get(str(puma_code), {})
    return meta.get("borough", "Unknown")


def get_puma_centroid(puma_code: str) -> list | None:
    """Return [lat, lng] centroid for a PUMA."""
    return PUMA_CENTROIDS.get(str(puma_code))


def build_crosswalk_df() -> pd.DataFrame:
    """Build a full crosswalk DataFrame: zip -> puma -> borough."""
    rows = []
    for zip_code, puma_code in ZIP_TO_PUMA.items():
        meta = PUMA_METADATA.get(puma_code, {})
        rows.append({
            "zip_code": zip_code,
            "puma_code": puma_code,
            "puma_name": meta.get("name", f"PUMA {puma_code}"),
            "borough": meta.get("borough", "Unknown"),
            "boro_code": meta.get("boro_code", "0"),
        })
    return pd.DataFrame(rows)
