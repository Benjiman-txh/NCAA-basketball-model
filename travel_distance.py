"""
Travel Distance Calculator for NCAA Basketball Games.

Produces three cached CSVs:
  - cities_geocoded.csv: lat/lon for all 509 cities
  - team_home_cities.csv: each team's home CityID
  - game_distances.csv: per-game travel distances (miles) for each team
"""

import math
import os
import time

import polars as pl
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Hardcoded fallback coordinates for cities that Nominatim struggles with ──
FALLBACK_COORDS: dict[int, tuple[float, float]] = {
    # International venues
    4025: (18.3985, -66.1536),    # Bayamon, PR
    4037: (18.4655, -67.1518),    # Borinquen, PR
    4053: (21.1619, -86.8515),    # Cancun, MX
    4263: (25.0820, -77.3230),    # Paradise Island, BA
    4271: (20.6296, -87.0739),    # Playa del Carmen, MX
    4283: (20.6534, -105.2253),   # Puerto Vallarta, MX
    4288: (49.4400, 7.5886),      # Ramstein-Miesenbach, GY (Germany)
    4307: (18.4655, -66.1057),    # San Juan, PR
    4314: (31.2304, 121.4737),    # Shanghai, CH
    4319: (37.5665, 126.9780),    # South Korea, SK
    4330: (18.3358, -64.9307),    # St. Thomas, VI
    4347: (18.4311, -64.6231),    # Tortola, VI
    4358: (49.2827, -123.1207),   # Vancouver, BC
    4382: (26.5285, -78.6968),    # Freeport, BA
    4383: (18.3685, -66.1121),    # Guaynabo, PR
    4399: (18.3550, -66.0078),    # Trujillo Alto, PR
    4410: (54.5973, -5.9301),     # Belfast, IR (Ireland)
    4412: (18.4712, -77.9218),    # Montego Bay, JA
    4413: (25.0480, -77.3554),    # Nassau, BA
    4415: (19.3222, -81.2409),    # Grand Cayman, CI
    4422: (25.7267, -79.2690),    # Bimini, BA
    4424: (18.3985, -65.9911),    # Carolina, PR
    4437: (26.5333, -78.7500),    # Lucaya, BA
    4442: (43.6532, -79.3832),    # Toronto, ON
    4451: (42.4048, -82.1910),    # Chatham, ON
    4472: (20.5800, -87.1200),    # Riviera Maya, MX
    4473: (18.4449, -77.9468),    # St. James, JA
    4474: (48.4284, -123.3656),   # Victoria, BC
    4489: (53.3498, -6.2603),     # Dublin, IR (Ireland)
    4491: (45.5833, -73.7500),    # Laval, QC
    4492: (51.5074, -0.1278),     # London, GB
    4496: (46.0878, -64.7782),    # Moncton, NB
    4505: (48.8566, 2.3522),      # Paris, FR
    4511: (49.6942, -112.8328),   # Lethbridge, ON (Alberta actually)
    4516: (46.2382, -63.1311),    # Charlottetown, PE
    4518: (18.4286, -66.4849),    # Manati, PR
    4529: (18.3228, -64.9671),    # Charlotte Amalie West, VI
    4530: (20.5034, -87.2285),    # Puerto Aventuras, MX

    # Tricky US locations
    4034: (35.2848, -81.7120),    # Boiling Springs, NC
    4048: (40.6952, -73.9488),    # Brooklyn Heights, NY (near Brooklyn)
    4050: (35.4049, -78.8386),    # Buies Creek, NC
    4087: (35.3052, -83.1767),    # Cullowhee, NC
    4100: (33.9627, -84.1463),    # Duluth, GA
    4110: (39.6945, -77.3270),    # Emmitsburg, MD
    4165: (33.4951, -90.3210),    # Itta Bena, MS
    4182: (20.8783, -156.6825),   # Lahaina, HI
    4200: (31.6908, -91.0527),    # Lorman, MS
    4245: (34.7640, -86.5718),    # Normal, AL
    4252: (38.7387, -87.3420),    # Oakland City, IN
    4264: (21.3269, -157.9488),   # Pearl Harbor-Hickam, HI
    4277: (34.5040, -81.8732),    # Presbyterian, SC (Clinton, SC area)
    4356: (38.9983, -104.8614),   # USAF Academy, CO
    4385: (21.6472, -157.9225),   # Laie, HI
    4403: (20.8911, -156.5047),   # Wailuku, HI
    4439: (33.4518, -90.4566),    # Moorehead, MS
    4441: (43.1376, -79.0376),    # Niagara University, NY
    4452: (20.8830, -156.4700),   # Kahului, HI
    4454: (20.8634, -156.4432),   # Maui, HI
    4459: (34.4133, -119.8610),   # Isla Vista, CA (near UCSB)
    4460: (41.8059, -72.2279),    # Mansfield, CT
    4461: (41.4387, -72.1153),    # Montville, CT (Mohegan Sun area)
    4479: (31.8770, -91.1361),    # Alcorn State, MS (Lorman area)
    4494: (20.7644, -156.4450),   # Kihei, HI
    4517: (20.9260, -156.3310),   # Makawao, HI
}

# ── State/country code → geocoding suffix mapping ──
STATE_SUFFIX: dict[str, str] = {
    "PR": "Puerto Rico",
    "VI": "US Virgin Islands",
    "BA": "Bahamas",
    "MX": "Mexico",
    "CH": "China",
    "SK": "South Korea",
    "IR": "Ireland",
    "GB": "United Kingdom",
    "JA": "Jamaica",
    "CI": "Cayman Islands",
    "BC": "British Columbia, Canada",
    "ON": "Ontario, Canada",
    "QC": "Quebec, Canada",
    "NB": "New Brunswick, Canada",
    "PE": "Prince Edward Island, Canada",
    "FR": "France",
    "GY": "Germany",
}

# US state abbreviations
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

# ── Manual overrides for team → home city ──
# Teams where the most-frequent-game-city heuristic picks the wrong city
TEAM_HOME_OVERRIDES: dict[int, int] = {
    1193: 4116,  # Fairfield → Fairfield, CT
    1263: 4259,  # Maine → Orono, ME
    1281: 4463,  # Missouri → Columbia, MO
}


def geocode_cities(cities_path: str, output_path: str) -> pl.DataFrame:
    """Step 1: Geocode all cities, with caching and fallbacks."""
    if os.path.exists(output_path):
        print(f"  Loading cached {output_path}")
        return pl.read_csv(output_path)

    cities = pl.read_csv(cities_path)
    geolocator = Nominatim(user_agent="ncaa_march_mania_2026", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

    results = []
    total = len(cities)

    for i, row in enumerate(cities.iter_rows(named=True)):
        cid = row["CityID"]
        city = row["City"]
        state = row["State"]

        # Use hardcoded fallback if available
        if cid in FALLBACK_COORDS:
            lat, lon = FALLBACK_COORDS[cid]
            results.append((cid, city, state, lat, lon))
            print(f"  [{i+1}/{total}] {city}, {state} → ({lat}, {lon}) [fallback]")
            continue

        # Build geocoding query
        if state in US_STATES:
            query = f"{city}, {state}, USA"
        elif state in STATE_SUFFIX:
            query = f"{city}, {STATE_SUFFIX[state]}"
        else:
            query = f"{city}, {state}, USA"

        try:
            location = geocode(query)
            if location:
                lat, lon = location.latitude, location.longitude
                results.append((cid, city, state, lat, lon))
                print(f"  [{i+1}/{total}] {city}, {state} → ({lat:.4f}, {lon:.4f})")
            else:
                print(f"  [{i+1}/{total}] WARNING: Could not geocode {city}, {state}")
                results.append((cid, city, state, None, None))
        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR geocoding {city}, {state}: {e}")
            results.append((cid, city, state, None, None))

    df = pl.DataFrame(
        results,
        schema={"CityID": pl.Int64, "City": pl.Utf8, "State": pl.Utf8,
                "Latitude": pl.Float64, "Longitude": pl.Float64},
        orient="row",
    )

    # Check for any missing geocodes
    missing = df.filter(pl.col("Latitude").is_null())
    if len(missing) > 0:
        print(f"\n  WARNING: {len(missing)} cities could not be geocoded:")
        for row in missing.iter_rows(named=True):
            print(f"    CityID={row['CityID']} {row['City']}, {row['State']}")

    df.write_csv(output_path)
    print(f"  Saved {output_path} ({len(df)} cities)")
    return df


def map_team_home_cities(
    game_cities_path: str, teams_path: str, output_path: str
) -> pl.DataFrame:
    """Step 2: Map each team to its home city (most frequent game city)."""
    if os.path.exists(output_path):
        print(f"  Loading cached {output_path}")
        return pl.read_csv(output_path)

    gc = pl.read_csv(game_cities_path)
    teams = pl.read_csv(teams_path)

    # Count how often each team plays in each city (as winner or loser)
    w_counts = (
        gc.select(pl.col("WTeamID").alias("TeamID"), "CityID")
        .group_by("TeamID", "CityID")
        .len()
    )
    l_counts = (
        gc.select(pl.col("LTeamID").alias("TeamID"), "CityID")
        .group_by("TeamID", "CityID")
        .len()
    )

    # Combine and find most frequent city per team
    all_counts = pl.concat([w_counts, l_counts]).group_by("TeamID", "CityID").agg(
        pl.col("len").sum()
    )
    home_cities = all_counts.sort("len", descending=True).group_by("TeamID").first()
    home_cities = home_cities.select("TeamID", pl.col("CityID").alias("HomeCityID"))

    # Join with team names
    result = teams.select("TeamID", "TeamName").join(home_cities, on="TeamID", how="left")

    # Apply manual overrides
    for team_id, city_id in TEAM_HOME_OVERRIDES.items():
        result = result.with_columns(
            pl.when(pl.col("TeamID") == team_id)
            .then(pl.lit(city_id))
            .otherwise(pl.col("HomeCityID"))
            .alias("HomeCityID")
        )

    # Teams with no games in MGameCities won't have a home city
    missing = result.filter(pl.col("HomeCityID").is_null())
    if len(missing) > 0:
        print(f"  Note: {len(missing)} teams have no games in MGameCities (no home city assigned)")

    result.write_csv(output_path)
    print(f"  Saved {output_path} ({result.filter(pl.col('HomeCityID').is_not_null()).height} teams with home cities)")
    return result


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in miles."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def calculate_distances(
    game_cities_path: str,
    geocoded_cities: pl.DataFrame,
    team_homes: pl.DataFrame,
    output_path: str,
) -> pl.DataFrame:
    """Step 3: Calculate per-game travel distances."""
    if os.path.exists(output_path):
        print(f"  Loading cached {output_path}")
        return pl.read_csv(output_path)

    gc = pl.read_csv(game_cities_path)

    # Build lookup dicts for speed
    city_coords: dict[int, tuple[float, float]] = {}
    for row in geocoded_cities.filter(pl.col("Latitude").is_not_null()).iter_rows(named=True):
        city_coords[row["CityID"]] = (row["Latitude"], row["Longitude"])

    team_home: dict[int, int] = {}
    for row in team_homes.filter(pl.col("HomeCityID").is_not_null()).iter_rows(named=True):
        team_home[row["TeamID"]] = row["HomeCityID"]

    seasons = []
    daynums = []
    w_teams = []
    l_teams = []
    cr_types = []
    w_dists = []
    l_dists = []

    for row in gc.iter_rows(named=True):
        game_city_id = row["CityID"]
        w_id = row["WTeamID"]
        l_id = row["LTeamID"]

        game_coords = city_coords.get(game_city_id)

        # Winning team distance
        w_home_id = team_home.get(w_id)
        if w_home_id and game_coords:
            w_home_coords = city_coords.get(w_home_id)
            w_dist = haversine_miles(*w_home_coords, *game_coords) if w_home_coords else None
        else:
            w_dist = None

        # Losing team distance
        l_home_id = team_home.get(l_id)
        if l_home_id and game_coords:
            l_home_coords = city_coords.get(l_home_id)
            l_dist = haversine_miles(*l_home_coords, *game_coords) if l_home_coords else None
        else:
            l_dist = None

        seasons.append(row["Season"])
        daynums.append(row["DayNum"])
        w_teams.append(w_id)
        l_teams.append(l_id)
        cr_types.append(row["CRType"])
        w_dists.append(round(w_dist, 1) if w_dist is not None else None)
        l_dists.append(round(l_dist, 1) if l_dist is not None else None)

    result = pl.DataFrame({
        "Season": seasons,
        "DayNum": daynums,
        "WTeamID": w_teams,
        "LTeamID": l_teams,
        "CRType": cr_types,
        "WTeamDist": w_dists,
        "LTeamDist": l_dists,
    })

    result.write_csv(output_path)
    print(f"  Saved {output_path} ({len(result)} games)")
    return result


def main():
    cities_csv = os.path.join(BASE_DIR, "Cities.csv")
    game_cities_csv = os.path.join(BASE_DIR, "MGameCities.csv")
    teams_csv = os.path.join(BASE_DIR, "MTeams.csv")

    geocoded_path = os.path.join(BASE_DIR, "cities_geocoded.csv")
    homes_path = os.path.join(BASE_DIR, "team_home_cities.csv")
    distances_path = os.path.join(BASE_DIR, "game_distances.csv")

    print("Step 1: Geocoding cities...")
    geocoded = geocode_cities(cities_csv, geocoded_path)

    print("\nStep 2: Mapping teams to home cities...")
    homes = map_team_home_cities(game_cities_csv, teams_csv, homes_path)

    print("\nStep 3: Calculating game distances...")
    distances = calculate_distances(game_cities_csv, geocoded, homes, distances_path)

    # Summary stats
    print("\n── Summary ──")
    print(f"  Cities geocoded: {geocoded.filter(pl.col('Latitude').is_not_null()).height}/{len(geocoded)}")
    print(f"  Teams with home cities: {homes.filter(pl.col('HomeCityID').is_not_null()).height}/{len(homes)}")

    valid = distances.filter(pl.col("WTeamDist").is_not_null() & pl.col("LTeamDist").is_not_null())
    print(f"  Games with distances: {len(valid)}/{len(distances)}")
    if len(valid) > 0:
        all_dists = pl.concat([
            valid.select(pl.col("WTeamDist").alias("dist")),
            valid.select(pl.col("LTeamDist").alias("dist")),
        ])
        print(f"  Distance stats (miles): mean={all_dists['dist'].mean():.0f}, "
              f"median={all_dists['dist'].median():.0f}, "
              f"max={all_dists['dist'].max():.0f}")


if __name__ == "__main__":
    main()
