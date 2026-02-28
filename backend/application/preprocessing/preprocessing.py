import re
import os
import glob
import pandas as pd
import sqlalchemy
import configparser
import random
from datetime import datetime

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

    # Default table names if not specified
    if "tables" not in config:
        config["tables"] = {
            "vector_table": "prediction_vector",
            "label_table": "prediction_label",
        }

    return config["database"], config["tables"]


def get_db_engine(db_config):
    """Creates a SQLAlchemy engine from the configuration."""
    # Construct connection string: postgresql+psycopg2://user:password@host:port/dbname
    conn_str = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    return sqlalchemy.create_engine(conn_str)


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


def main():
    try:
        db_config, table_config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    processed_dates = get_processed_dates(PARQUET_DIR)
    print(
        f"Found {len(processed_dates)} already processed days: {sorted(list(processed_dates))}"
    )

    # Connect to DB
    engine = get_db_engine(db_config)

    # Define Query
    # We fetch ALL data that corresponds to dates NOT in processed_dates?
    # Actually, fetching everything and filtering in Pandas is safer for the Trip ID logic
    # because a trip might span across midnight, though standard bus trips usually reset or have distinct IDs if we had them.
    # However, the task says "Partitioning & Saving: Iterate through days... Skip if existing".
    # But to generate consistent Trip IDs globally, we technically need the whole history or at least complete sequences.
    # Given the instructions "Incremental processing (skip days already present)", strict global Trip ID consistency
    # across runs might be tricky if we don't load previous state.
    # BUT, assuming we process the whole DB once or big chunks, we can just process what we fetch.
    # For now, we will fetch everything and filter at save time as per instructions.

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
    FROM {table_config.get("vector_table", "prediction_vector")} v
    JOIN {table_config.get("label_table", "prediction_label")} l ON v.id = l.id
    """

    print("Executing query and loading data (this may take a while)...")
    try:
        # Use a chunksize if data is massive, but for this task logic (global sort/cumsum),
        # loading into memory is required unless we use Dask/Spark. Assuming memory fits.
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error querying database: {e}")
        return

    if df.empty:
        print("No data found in database.")
        return

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


if __name__ == "__main__":
    main()
