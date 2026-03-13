import re
import os
import glob
import random
import pandas as pd
from datetime import datetime

from config import Prediction, Ledger
from persistence.database import get_sync_engine_for_pipeline
from persistence.ledger_db import read_historical, read_vehicle_trips

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PARQUET_DIR = os.path.join(PROJECT_ROOT, "parquets")


def get_processed_dates(parquet_dir):
    """
    Scans the output directory for existing parquet files and returns a set of processed dates.
    Format of files: dataset_YYYY-MM-DD.parquet
    """
    processed_dates = set()
    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)
        return processed_dates

    # Use glob to find files matching the pattern
    files = glob.glob(os.path.join(parquet_dir, "dataset_*.parquet"))
    for f in files:
        filename = os.path.basename(f)
        # Extract date string: dataset_YYYY-MM-DD.parquet -> YYYY-MM-DD
        try:
            date_str = filename.replace("dataset_", "").replace(".parquet", "")
            # Validate date format simply
            datetime.strptime(date_str, "%Y-%m-%d")
            processed_dates.add(date_str)
        except ValueError:
            print(f"Warning: Skipping file with unexpected name format: {filename}")
            continue

    return processed_dates


def generate_synthetic_trip_id(df):
    """
    Generates a synthetic 'trip_id' based on sequential logic.
    Logic:
    1. Sort by route_id, direction_id, ts.
    2. Detect breaks where stop_sequence decreases OR route/direction changes.
    3. Use cumulative sum to assign unique IDs.
    """
    print("Generating Synthetic Trip IDs...")

    # 1. Sorting (Crucial)
    # Ensure strict ordering for the logic to work
    df.sort_values(by=["route_id", "direction_id", "ts"], inplace=True)

    # 2. Vectorized Break Detection
    # Calculate shifts to compare current row with previous row
    prev_route = df["route_id"].shift(1)
    prev_dir = df["direction_id"].shift(1)
    prev_seq = df["stop_sequence"].shift(1)

    # Condition 1: Change in Route ID
    route_change = df["route_id"] != prev_route

    # Condition 2: Change in Direction ID
    dir_change = df["direction_id"] != prev_dir

    # Condition 3: Stop Sequence Reset (current < previous)
    # We handle the first row (NaN in prev_seq) by filling with a value that triggers the condition or defaults correctly.
    # Actually, simpler logic: if sequence drops, it's a new trip.
    # Note: If route or direction changes, sequence comparison is irrelevant, it's a new trip anyway.
    seq_reset = df["stop_sequence"] < prev_seq

    # Condition 4: Large Time Gap (within same route/direction)
    # If time_seconds jumps by more than 600 seconds (10 min), it's likely a new trip
    # This catches abandoned trips that only have stop_sequence=1
    prev_time = df["time_seconds"].shift(1)
    large_time_gap = (df["time_seconds"] - prev_time) > 600

    # Condition 5: Time going backward (not a midnight crossing) = new trip
    # This catches cases where a new trip starts at an earlier time of day
    # Midnight crossing is ~86400 -> ~0, so we exclude gaps > 40000
    time_backward = (df["time_seconds"] < prev_time) & (
        (prev_time - df["time_seconds"]) < 40000
    )

    # Combine conditions: A new trip starts if ANY of these are true
    is_new_trip = route_change | dir_change | seq_reset | large_time_gap | time_backward

    # 3. Generate ID
    # cumsum() treats True as 1 and False as 0.
    # This increments the ID every time a new trip is detected.
    df["trip_id"] = is_new_trip.cumsum()

    # Clean up: First row will always be True for route_change/dir_change because shift produces NaN (and NaN != value is True or needs handling).
    # Let's double check pandas behavior. NaN != val is True. So first row gets a new ID. Correct.

    print(f"Generated {df['trip_id'].max()} unique trips.")
    return df


def main(start_date: str = None):
    """
    Extract data from database to daily parquet files.

    Args:
        start_date: Optional start date (YYYY-MM-DD). Only process data from this date onwards.
    """
    engine = get_sync_engine_for_pipeline("prediction")

    if engine is None:
        print(
            "Error: Prediction database not configured. Check config.ini [prediction] section."
        )
        return

    processed_dates = get_processed_dates(PARQUET_DIR)
    print(
        f"Found {len(processed_dates)} already processed days: {sorted(list(processed_dates))}"
    )

    query = f"""
    SELECT
        v.id,
        v.ts,
        v.trip_id,
        v.route_id,
        v.direction_id,
        v.stop_sequence,
        v.shape_dist_travelled,
        v.distance_to_next_stop,
        v.far_status,
        v.day_type,
        v.rush_hour_status,
        v.time_feat,
        v.time_sin,
        v.time_cos,
        v.schedule_adherence,
        v.speed_ratio,
        v.current_traffic_speed,
        v.current_speed,
        v.precipitation,
        v.weather_code,
        v.bus_type,
        v.door_number,
        v.deposit_grottarossa,
        v.deposit_magliana,
        v.deposit_tor_sapienza,
        v.deposit_portonaccio,
        v.deposit_monte_sacro,
        v.deposit_tor_pagnotta,
        v.deposit_tor_cervara,
        v.deposit_maglianella,
        v.deposit_costi,
        v.deposit_trastevere,
        v.deposit_acilia,
        v.deposit_tor_vergata,
        v.deposit_porta_maggiore,
        v.served_ratio,
        v.starting_time_sin,
        v.starting_time_cos,
        v.sch_starting_time_sin,
        v.sch_starting_time_cos,
        l.time_seconds,
        l.occupancy_status
    FROM {Prediction.VECTOR_TABLE} v
    JOIN {Prediction.LABEL_TABLE} l ON v.id = l.id
    """

    if start_date:
        query += f"\n    WHERE v.ts >= '{start_date} 00:00:00'"
        print(f"Filtering data from {start_date} onwards...")

    print("Executing query and loading data (this may take a while)...")
    print(f"Query tables: {Prediction.VECTOR_TABLE} + {Prediction.LABEL_TABLE}")
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error querying database: {e}")
        return

    print(f"Loaded {len(df)} rows from database.")

    if df.empty:
        print("No data found in database.")
        return

    if start_date and "ts" in df.columns:
        min_ts = df["ts"].min()
        max_ts = df["ts"].max()
        print(f"Data range: {min_ts} to {max_ts}")

    # Ensure ts is datetime
    df["ts"] = pd.to_datetime(df["ts"])

    # Convert UUID to string for parquet compatibility
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)

    if "trip_id" in df.columns:
        has_trip_id = df["trip_id"].notna()
        if has_trip_id.any():
            print(f"Found {has_trip_id.sum()} rows with existing trip_id.")

            df["trip_id"] = df["trip_id"].astype(object)
            existing_ids = df.loc[has_trip_id, "trip_id"]
            df.loc[has_trip_id, "trip_id"] = existing_ids.apply(
                lambda x: int(re.sub(r"\D", "", str(x)))
                if re.sub(r"\D", "", str(x))
                else 0
            )

            if (~has_trip_id).any():
                print(
                    f"Generating synthetic trip_id for {(~has_trip_id).sum()} rows without it..."
                )
                missing_df = df[~has_trip_id].copy()
                missing_df = generate_synthetic_trip_id(missing_df)
                random_padding = random.randint(10000, 99999)
                max_existing = df.loc[has_trip_id, "trip_id"].max()
                missing_df["trip_id"] = (
                    missing_df["trip_id"] * 100000 + max_existing + random_padding
                )
                df.loc[~has_trip_id, "trip_id"] = missing_df["trip_id"]
        else:
            df = generate_synthetic_trip_id(df)
    else:
        df = generate_synthetic_trip_id(df)

    # Extract Date for Partitioning
    df["date_str"] = df["ts"].dt.date.astype(str)

    unique_days = df["date_str"].unique()
    print(f"Processing data for {len(unique_days)} unique days found in dataset...")

    for day in unique_days:
        if day in processed_dates:
            print(f"Skipping {day} (already processed).")
            continue

        print(f"Saving data for {day}...")
        day_df = df[df["date_str"] == day].copy()

        # Drop the helper column used for partitioning
        day_df.drop(columns=["date_str"], inplace=True)

        output_path = os.path.join(PARQUET_DIR, f"dataset_{day}.parquet")

        # Save to Parquet
        # index=False to not save the pandas index
        day_df.to_parquet(output_path, engine="pyarrow", index=False)
        print(f"Saved {output_path}")

    print("Pipeline completed successfully.")


def extract_historical(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Extract stop-level historical arrival data from the ledger database.

    This provides the ground truth for schedule adherence analysis and
    model validation, complementing the raw vector pipeline used for ML
    training.

    Args:
        start_date: Optional start date (YYYY-MM-DD). Converted to Unix timestamp.
        end_date:   Optional end date (YYYY-MM-DD). Converted to Unix timestamp.

    Returns:
        DataFrame with columns: trip_id, stop_id, stop_sequence,
        actual_arrival_time, schedule_adherence, occupancy_status, vehicle_id
    """
    date_start = None
    date_end = None
    if start_date:
        date_start = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
    if end_date:
        date_end = datetime.strptime(end_date, "%Y-%m-%d").timestamp()

    df = read_historical(
        Ledger.DB_CONNECTION, Ledger.HISTORICAL_TABLE,
        date_start=date_start, date_end=date_end,
    )
    print(f"Loaded {len(df)} historical arrival records from ledger.")
    return df


def extract_vehicle_trips(
    start_date: str = None, end_date: str = None, route_id: str = None
) -> pd.DataFrame:
    """
    Extract vehicle-level trip performance data from the ledger database.

    Provides per-trip delay/occupancy summaries with vehicle characteristics
    (fuel type, euro class, capacity) for fleet analytics.

    Args:
        start_date: Optional start date (YYYY-MM-DD).
        end_date:   Optional end date (YYYY-MM-DD).
        route_id:   Optional route filter.

    Returns:
        DataFrame with all VehicleTripRecord fields.
    """
    df = read_vehicle_trips(
        Ledger.DB_CONNECTION, Ledger.VEHICLE_TABLE,
        route_id=route_id, date_start=start_date, date_end=end_date,
    )
    print(f"Loaded {len(df)} vehicle trip records from ledger.")
    return df


if __name__ == "__main__":
    main()
