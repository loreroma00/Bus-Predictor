import os
import sys
import pandas as pd
import numpy as np
import h3

from config import Traffic
from persistence.database import get_sync_engine_for_pipeline

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PARQUET_DIR = os.path.join(PROJECT_ROOT, "parquets")


def compute_traffic_averages(output_path=None):
    """
    Fetches traffic data from DB and computes averages by h3_index, day_type, and hour.

    Returns a DataFrame with columns:
    - h3_index (hexagon id)
    - day_type (0=normal, 1=saturday, 2=sunday/holiday)
    - hour (0-23)
    - avg_speed_ratio
    - avg_current_traffic_speed

    Optionally saves to parquet if output_path provided.
    """
    engine = get_sync_engine_for_pipeline("traffic")

    if engine is None:
        print(
            "Warning: Traffic database not configured. Skipping traffic averages computation."
        )
        return None

    query = f"""
    SELECT
        v.hexagon_id AS h3_index,
        v.day_type,
        v.ts,
        l.speed_ratio,
        l.current_traffic_speed
    FROM {Traffic.VECTOR_TABLE} v
    JOIN {Traffic.LABEL_TABLE} l ON v.id::text = l.id::text AND v.ts = l.ts
    """

    print("Fetching traffic data from database...")
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error querying database: {e}")
        return None

    if df.empty:
        print("No traffic data found.")
        return None

    print(f"Loaded {len(df)} traffic records.")

    df["ts"] = pd.to_datetime(df["ts"])
    df["hour"] = df["ts"].dt.hour

    print("Computing averages by h3_index, day_type, and hour...")
    averages = (
        df.groupby(["h3_index", "day_type", "hour"])
        .agg(
            avg_speed_ratio=("speed_ratio", "mean"),
            avg_current_traffic_speed=("current_traffic_speed", "mean"),
            sample_count=("speed_ratio", "count"),
        )
        .reset_index()
    )

    print(f"Computed averages for {len(averages)} h3_index/day_type/hour combinations.")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        averages.to_parquet(output_path, index=False)
        print(f"Saved traffic averages to {output_path}")

    return averages


# Constants
H3_RESOLUTION = 9
R_EARTH = 6371000  # Meters

# Ensure we can import from application.domain
current_dir = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from application.domain.static_data_fetcher import StaticDataFetcher


def get_h3_index(lat: float, lng: float, resolution: int = H3_RESOLUTION) -> str:
    return h3.latlng_to_cell(lat, lng, resolution)


def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * R_EARTH


def process_stop_route_map(
    trips_path: str,
    stop_times_path: str,
    stops_path: str,
    output_path: str,
    config_output_path: str,
):
    """
    Build a stop-based route map from GTFS data.
    No shapes.txt required — only stop_times.txt, stops.txt, and trips.txt.

    For each (route_id, direction_id), identifies the canonical stop pattern
    (most common number of stops), then builds a map with one row per stop
    including H3 hex indices and inter-stop distances.

    Output columns:
        route_id, direction_id, stop_idx, stop_id, stop_sequence,
        h3_index, stop_lat, stop_lon, shape_dist_at_stop,
        distance_to_next_stop, num_stops, route_len_m
    """
    print(f"Loading GTFS data from {trips_path}, {stop_times_path}, {stops_path}...")

    try:
        trips = pd.read_csv(trips_path, dtype={"route_id": str}, low_memory=False)
        stop_times = pd.read_csv(stop_times_path, low_memory=False)
        stops = pd.read_csv(stops_path, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # =========================================================
    # Phase A: Identify canonical stop pattern per route+direction
    # =========================================================
    print("Phase A: Identifying canonical stop patterns...")

    # Link stop_times to route_id + direction_id via trips
    trip_route = trips[["trip_id", "route_id", "direction_id"]].dropna(
        subset=["route_id", "direction_id"]
    )
    st = stop_times.merge(trip_route, on="trip_id", how="inner")

    # Count stops per trip
    stops_per_trip = (
        st.groupby(["route_id", "direction_id", "trip_id"])
        .size()
        .reset_index(name="n_stops")
    )

    # For each (route_id, direction_id), find the most common stop count
    most_common_count = (
        stops_per_trip.groupby(["route_id", "direction_id", "n_stops"])
        .size()
        .reset_index(name="freq")
        .sort_values(
            ["route_id", "direction_id", "freq"], ascending=[True, True, False]
        )
        .drop_duplicates(subset=["route_id", "direction_id"], keep="first")
    )

    # Pick one representative trip per (route_id, direction_id) with that stop count
    canonical_trips = most_common_count.merge(
        stops_per_trip, on=["route_id", "direction_id", "n_stops"], how="inner"
    )
    canonical_trips = canonical_trips.drop_duplicates(
        subset=["route_id", "direction_id"], keep="first"
    )

    print(f"Identified {len(canonical_trips)} canonical route+direction pairs.")

    # =========================================================
    # Phase B: Extract stop sequences with coordinates
    # =========================================================
    print("Phase B: Extracting stop sequences with coordinates...")

    canonical_trip_ids = set(canonical_trips["trip_id"].values)
    canonical_st = st[st["trip_id"].isin(canonical_trip_ids)].copy()

    # Attach stop coordinates
    canonical_st = canonical_st.merge(
        stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left"
    )

    # Drop stops without coordinates
    before = len(canonical_st)
    canonical_st = canonical_st.dropna(subset=["stop_lat", "stop_lon"])
    if len(canonical_st) < before:
        print(f"  Dropped {before - len(canonical_st)} stops without coordinates.")

    canonical_st = canonical_st.sort_values(
        ["route_id", "direction_id", "trip_id", "stop_sequence"]
    )

    # =========================================================
    # Phase C: Compute distances and H3 indices
    # =========================================================
    print("Phase C: Computing inter-stop distances and H3 indices...")

    results = []
    max_stops = 0

    for (route_id, direction_id, trip_id), group in canonical_st.groupby(
        ["route_id", "direction_id", "trip_id"]
    ):
        group = group.sort_values("stop_sequence").reset_index(drop=True)
        n_stops = len(group)

        if n_stops < 2:
            continue

        lats = group["stop_lat"].values
        lons = group["stop_lon"].values
        stop_ids = group["stop_id"].values
        stop_seqs = group["stop_sequence"].values

        # Cumulative Haversine distances
        cum_dists = [0.0]
        for i in range(1, n_stops):
            d = float(haversine_np(lats[i - 1], lons[i - 1], lats[i], lons[i]))
            cum_dists.append(cum_dists[-1] + d)

        route_len = cum_dists[-1]
        if route_len == 0:
            continue

        if n_stops > max_stops:
            max_stops = n_stops

        for i in range(n_stops):
            dist_to_next = cum_dists[i + 1] - cum_dists[i] if i < n_stops - 1 else 0.0
            h3_idx = get_h3_index(lats[i], lons[i])

            results.append(
                {
                    "route_id": route_id,
                    "direction_id": int(direction_id),
                    "stop_idx": i,
                    "stop_id": str(stop_ids[i]),
                    "stop_sequence": int(stop_seqs[i]),
                    "h3_index": h3_idx,
                    "stop_lat": lats[i],
                    "stop_lon": lons[i],
                    "shape_dist_at_stop": cum_dists[i],
                    "distance_to_next_stop": dist_to_next,
                    "num_stops": n_stops,
                    "route_len_m": route_len,
                }
            )

    if not results:
        print("Error: No valid routes produced. Check GTFS data.")
        return

    final_df = pd.DataFrame(results)

    n_routes = final_df.groupby(["route_id", "direction_id"]).ngroups
    print(f"\nBuilt stop map for {n_routes} route+direction pairs.")
    print(f"MAX_STOPS = {max_stops}")
    print(f"Stop count distribution:")
    stop_counts = final_df.drop_duplicates(
        subset=["route_id", "direction_id"]
    )["num_stops"]
    print(f"  Min: {stop_counts.min()}, Median: {stop_counts.median():.0f}, "
          f"Max: {stop_counts.max()}, Mean: {stop_counts.mean():.1f}")

    # Save parquet
    print(f"\nSaving {len(final_df)} rows to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    # Save config with MAX_STOPS
    import json
    config = {"max_stops": max_stops}
    with open(config_output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved stop route config to {config_output_path}")
    print("Done.")


def main():
    print("Starting Stop-Based Route Mapping...")

    # 1. Setup & Fetch Data
    try:
        zip_path = "rome_static_gtfs.zip"
        if os.path.exists(zip_path):
            print(f"Removing stale {zip_path}...")
            os.remove(zip_path)

        fetcher = StaticDataFetcher()
        fetcher.fetch()
    except Exception as e:
        print(f"Warning: StaticDataFetcher failed: {e}")
        print("Checking if files exist locally...")

    trips_file = "trips.txt"
    stop_times_file = "stop_times.txt"
    stops_file = "stops.txt"

    for f in [trips_file, stop_times_file, stops_file]:
        if not os.path.exists(f):
            print(f"Error: {f} not found. Please ensure GTFS data is available.")
            sys.exit(1)

    output_file = os.path.join(PROJECT_ROOT, "parquets", "stop_route_map.parquet")
    config_file = os.path.join(PROJECT_ROOT, "parquets", "stop_route_config.json")

    process_stop_route_map(
        trips_file, stop_times_file, stops_file, output_file, config_file
    )

    print("\n--- Computing Traffic Averages ---")
    traffic_output = os.path.join(PARQUET_DIR, "traffic_averages.parquet")
    compute_traffic_averages(output_path=traffic_output)


if __name__ == "__main__":
    main()
