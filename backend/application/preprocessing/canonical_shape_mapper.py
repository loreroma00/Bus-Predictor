import os
import sys
import pandas as pd
import numpy as np
import h3
import sqlalchemy
import configparser

# --- Configuration ---
CONFIG_FILE = "config.ini"
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PARQUET_DIR = os.path.join(PROJECT_ROOT, "parquets")


def load_config():
    """Loads database configuration from config.ini."""
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"Configuration file '{CONFIG_FILE}' not found. Please create it based on config.ini.example."
        )

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    if "traffic_tables" not in config:
        config["traffic_tables"] = {
            "vector_table": "traffic_vector",
            "label_table": "traffic_label",
        }

    return config["database"], config["traffic_tables"]


def get_db_engine(db_config):
    """Creates a SQLAlchemy engine from the configuration."""
    conn_str = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    return sqlalchemy.create_engine(conn_str)


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
    try:
        db_config, table_config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

    engine = get_db_engine(db_config)

    query = f"""
    SELECT
        v.hexagon_id AS h3_index,
        v.day_type,
        v.ts,
        l.speed_ratio,
        l.current_traffic_speed
    FROM {table_config.get("vector_table", "traffic_vector")} v
    JOIN {table_config.get("label_table", "traffic_label")} l ON v.id::text = l.id::text AND v.ts = l.ts
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
INTERPOLATION_POINTS = 100
H3_RESOLUTION = 9
R_EARTH = 6371000  # Meters

# Ensure we can import from application.domain
current_dir = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from application.domain.static_data_fetcher import StaticDataFetcher


def get_h3_index(lat: float, lng: float, resolution: int = H3_RESOLUTION) -> str:
    """
    Get H3 index for a given lat/lng.
    Using local implementation to avoid dependency issues with osmnx in h3_utils.
    """
    return h3.latlng_to_cell(lat, lng, resolution)


def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Vectorized version using numpy.
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * R_EARTH


def process_canonical_shapes(
    trips_path: str,
    shapes_path: str,
    stop_times_path: str,
    stops_path: str,
    output_path: str,
):
    print(
        f"Loading data from {trips_path}, {shapes_path}, {stop_times_path}, and {stops_path}..."
    )

    try:
        trips = pd.read_csv(
            trips_path, dtype={"route_id": str, "shape_id": str}, low_memory=False
        )
        shapes = pd.read_csv(shapes_path, dtype={"shape_id": str}, low_memory=False)
        stop_times = pd.read_csv(stop_times_path, low_memory=False)
        stops = pd.read_csv(stops_path, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("Phase A: Identifying Canonical Shapes...")

    trips_subset = trips[["route_id", "direction_id", "shape_id"]].dropna()

    shape_counts = (
        trips_subset.groupby(["route_id", "direction_id", "shape_id"])
        .size()
        .reset_index(name="count")
    )

    canonical_shapes = shape_counts.sort_values(
        ["route_id", "direction_id", "count"], ascending=[True, True, False]
    )
    canonical_shapes = canonical_shapes.drop_duplicates(
        subset=["route_id", "direction_id"], keep="first"
    )

    print(f"Identified {len(canonical_shapes)} canonical shapes.")

    print("Phase A.5: Mapping stops to canonical shapes...")

    trips_with_shape = trips[
        ["trip_id", "route_id", "direction_id", "shape_id"]
    ].dropna()
    stop_times_with_shape = stop_times.merge(
        trips_with_shape, on="trip_id", how="inner"
    )

    canonical_trips = canonical_shapes[["route_id", "direction_id", "shape_id"]].merge(
        stop_times_with_shape, on=["route_id", "direction_id", "shape_id"], how="inner"
    )

    if "shape_dist_traveled" not in canonical_trips.columns:
        print("Warning: shape_dist_traveled not in stop_times.")
        canonical_trips["shape_dist_traveled"] = np.nan

    stop_map = canonical_trips[
        ["shape_id", "stop_sequence", "stop_id", "shape_dist_traveled"]
    ].drop_duplicates()

    missing_dist = stop_map["shape_dist_traveled"].isna()
    if missing_dist.any():
        print(
            f"Calculating distances for {missing_dist.sum()} stops missing shape_dist_traveled..."
        )
        stop_map = stop_map.merge(
            stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left"
        )

        missing_dist = stop_map["shape_dist_traveled"].isna()

        for shape_id in stop_map.loc[missing_dist, "shape_id"].unique():
            shape_mask = (stop_map["shape_id"] == shape_id) & missing_dist
            shape_stops = stop_map[shape_mask].sort_values("stop_sequence")

            if shape_stops.empty:
                continue

            lats = shape_stops["stop_lat"].values
            lons = shape_stops["stop_lon"].values

            if np.any(np.isnan(lats)) or np.any(np.isnan(lons)):
                print(f"  Warning: Missing coordinates for shape {shape_id}")
                continue

            distances = [0.0]
            for i in range(1, len(lats)):
                dist = haversine_np(lats[i - 1], lons[i - 1], lats[i], lons[i])
                distances.append(distances[-1] + dist)

            stop_map.loc[shape_stops.index, "shape_dist_traveled"] = distances

        stop_map = stop_map.drop(columns=["stop_lat", "stop_lon"])

    stop_map = stop_map.dropna(subset=["shape_dist_traveled"])
    stop_map = stop_map.sort_values(["shape_id", "stop_sequence"])

    print("Phase B: Building Map (Normalization & H3)...")

    # Filter shapes DataFrame to include only the canonical shape_ids
    canonical_shape_ids = set(canonical_shapes["shape_id"].unique())
    shapes_filtered = shapes[shapes["shape_id"].isin(canonical_shape_ids)].copy()

    # Ensure sorted by sequence
    shapes_filtered = shapes_filtered.sort_values(["shape_id", "shape_pt_sequence"])

    # Check if we need to calculate distances
    cols = shapes_filtered.columns
    if (
        "shape_dist_traveled" not in cols
        or shapes_filtered["shape_dist_traveled"].isnull().any()
    ):
        print("Calculating shape distances...")
        # Calculate distance from previous point
        # Shift lat/lon to get previous point
        shapes_filtered["prev_lat"] = shapes_filtered.groupby("shape_id")[
            "shape_pt_lat"
        ].shift(1)
        shapes_filtered["prev_lon"] = shapes_filtered.groupby("shape_id")[
            "shape_pt_lon"
        ].shift(1)

        # Calculate distance using Haversine
        # Fill NaN (first point) with 0 distance
        dists = haversine_np(
            shapes_filtered["prev_lat"],
            shapes_filtered["prev_lon"],
            shapes_filtered["shape_pt_lat"],
            shapes_filtered["shape_pt_lon"],
        )
        shapes_filtered["dist_seg"] = dists.fillna(0)

        # Cumulative sum
        shapes_filtered["shape_dist_traveled"] = shapes_filtered.groupby("shape_id")[
            "dist_seg"
        ].cumsum()

        # Cleanup
        shapes_filtered.drop(columns=["prev_lat", "prev_lon", "dist_seg"], inplace=True)

    # Now, for each canonical shape, we interpolate 100 points
    results = []

    # Iterate over each unique shape_id in our filtered shapes
    grouped = shapes_filtered.groupby("shape_id")

    for shape_id, group in grouped:
        distances = group["shape_dist_traveled"].values
        lats = group["shape_pt_lat"].values
        lons = group["shape_pt_lon"].values

        total_dist = distances[-1]

        shape_stops = stop_map[stop_map["shape_id"] == shape_id]
        stop_distances = (
            shape_stops["shape_dist_traveled"].values
            if len(shape_stops) > 0
            else np.array([])
        )
        stop_sequences = (
            shape_stops["stop_sequence"].values
            if len(shape_stops) > 0
            else np.array([])
        )

        if total_dist == 0:
            interp_lats = np.full(INTERPOLATION_POINTS, lats[0])
            interp_lons = np.full(INTERPOLATION_POINTS, lons[0])
        else:
            target_dists = np.linspace(0, total_dist, INTERPOLATION_POINTS)
            interp_lats = np.interp(target_dists, distances, lats)
            interp_lons = np.interp(target_dists, distances, lons)

        for i in range(INTERPOLATION_POINTS):
            lat = interp_lats[i]
            lon = interp_lons[i]
            h3_idx = get_h3_index(lat, lon)
            segment_dist = (
                (i / (INTERPOLATION_POINTS - 1)) * total_dist if total_dist > 0 else 0
            )

            stop_sequence = 1
            distance_to_next_stop = 0
            if len(stop_distances) > 0:
                passed_stops = stop_distances <= segment_dist
                if np.any(passed_stops):
                    last_passed_idx = np.where(passed_stops)[0][-1]
                    stop_sequence = int(stop_sequences[last_passed_idx])

                future_stops = stop_distances > segment_dist
                if np.any(future_stops):
                    next_stop_idx = np.where(future_stops)[0][0]
                    distance_to_next_stop = stop_distances[next_stop_idx] - segment_dist

            results.append(
                {
                    "shape_id": shape_id,
                    "segment_idx": i,
                    "h3_index": h3_idx,
                    "canonical_len_m": total_dist,
                    "can_shape_dist_travelled": segment_dist,
                    "stop_sequence": stop_sequence,
                    "can_distance_to_next_stop": distance_to_next_stop,
                }
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Join with canonical_shapes to map shape_id back to route_id, direction_id
    final_df = pd.merge(
        canonical_shapes[["route_id", "direction_id", "shape_id"]],
        results_df,
        on="shape_id",
        how="inner",
    )

    final_df = final_df[
        [
            "route_id",
            "direction_id",
            "segment_idx",
            "h3_index",
            "canonical_len_m",
            "can_shape_dist_travelled",
            "stop_sequence",
            "can_distance_to_next_stop",
        ]
    ]

    # Save to Parquet
    print(f"Saving {len(final_df)} rows to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print("Done.")


def main():
    print("Starting Canonical Shape Mapping...")

    # 1. Setup & Fetch Data
    try:
        # To ensure we don't use stale data, we remove the local zip if it exists
        # to force StaticDataFetcher to download the latest version.
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
    shapes_file = "shapes.txt"
    stop_times_file = "stop_times.txt"
    stops_file = "stops.txt"

    if (
        not os.path.exists(trips_file)
        or not os.path.exists(shapes_file)
        or not os.path.exists(stop_times_file)
        or not os.path.exists(stops_file)
    ):
        print(
            "Error: trips.txt, shapes.txt, stop_times.txt, or stops.txt not found. Please ensure GTFS data is available."
        )
        sys.exit(1)

    output_file = os.path.join(PROJECT_ROOT, "parquets", "canonical_route_map.parquet")

    process_canonical_shapes(
        trips_file, shapes_file, stop_times_file, stops_file, output_file
    )

    print("\n--- Computing Traffic Averages ---")
    traffic_output = os.path.join(PARQUET_DIR, "traffic_averages.parquet")
    compute_traffic_averages(output_path=traffic_output)


if __name__ == "__main__":
    main()
