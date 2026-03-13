"""Deprecated"""

"""import pandas as pd
from .. import domain
from persistence import persistence as p
import os
import glob
from datetime import datetime, timedelta

INPUT_PATTERN = "diaries/diaries_*.parquet"
OUTPUT_FILE = "normalized_diaries.parquet"


def load_all_diaries():
    search_path = os.path.join(p.DIARIES_PATH, "diaries_*.parquet")
    files = glob.glob(search_path)

    # Fallback to current directory for robust path handling
    if not files:
        files = glob.glob("diaries_*.parquet")

    if not files:
        print("Error: No 'diaries_*.parquet' files found.")
        return None

    print(f"Found {len(files)} diary files.")
    dfs = []
    for f in files:
        try:
            df = p.readParquet(f)
            if not df.empty:
                # Normalize column names if necessary (handle 'actual_time' vs 'actual_arrival_time')
                if (
                    "actual_time" in df.columns
                    and "actual_arrival_time" not in df.columns
                ):
                    df["actual_arrival_time"] = df["actual_time"]
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def get_service_date(timestamp):
    if pd.isna(timestamp) or timestamp < 0:
        return None
    # -4 hours offset to handle late-night trips (e.g., 25:00) belonging to previous day
    dt = datetime.fromtimestamp(timestamp) - timedelta(hours=4)
    return dt.strftime("%Y%m%d")


def normalize_trip_stops(trip_df, static_trip):
    # Sort by sequence
    trip_df = trip_df.sort_values("stop_sequence")

    # Static info map
    static_info = {}
    for st in static_trip.stop_times:
        try:
            seq = int(st["stop_sequence"])
            dist = float(st.get("shape_dist_traveled", 0))
            static_info[seq] = {
                "stop_id": st["stop_id"],
                "dist": dist,
                "scheduled_time": st["arrival_time"],
            }
        except ValueError:
            continue

    records = trip_df.to_dict("records")
    new_rows = []

    for i in range(len(records) - 1):
        curr_rec = records[i]
        next_rec = records[i + 1]

        curr_seq = int(curr_rec["stop_sequence"])
        next_seq = int(next_rec["stop_sequence"])

        # Check for gap > 1
        if next_seq > curr_seq + 1:
            start_time = curr_rec["actual_arrival_time"]
            end_time = next_rec["actual_arrival_time"]

            start_dist = static_info.get(curr_seq, {}).get("dist", 0)
            end_dist = static_info.get(next_seq, {}).get("dist", 0)

            delta_dist = end_dist - start_dist
            delta_time = end_time - start_time

            if delta_time <= 0:
                continue

            speed = (delta_dist / delta_time) if delta_dist > 0 else 0

            for seq in range(curr_seq + 1, next_seq):
                if seq in static_info:
                    missing_stop = static_info[seq]

                    if speed > 0:
                        dist_from_start = missing_stop["dist"] - start_dist
                        time_offset = dist_from_start / speed
                    else:
                        # Linear by index
                        ratio = (seq - curr_seq) / (next_seq - curr_seq)
                        time_offset = delta_time * ratio

                    interpolated_time = int(start_time + time_offset)

                    new_row = curr_rec.copy()
                    new_row.update(
                        {
                            "stop_id": missing_stop["stop_id"],
                            "stop_sequence": seq,
                            "scheduled_time": missing_stop["scheduled_time"],
                            "actual_arrival_time": interpolated_time,
                            "measurement_timestamp": -1,  # Flag as Interpolated
                            "formatted_time": domain.to_readable_time(
                                interpolated_time
                            ),
                        }
                    )
                    new_rows.append(new_row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        return pd.concat([trip_df, new_df], ignore_index=True).sort_values(
            "stop_sequence"
        )
    return trip_df


def main():
    print("Starting Normalization Process...")

    # 1. Load Diaries
    diaries_df = load_all_diaries()
    if diaries_df is None:
        print("No diaries found. Exiting.")
        return

    # 2. Load Topology
    print("Loading Topology...")
    obs = domain.Observatory()
    topology = obs.get_topology()
    trips_map = topology.trips

    # 3. Identify Dates from Diaries
    print("Identifying Dates from Diaries...")
    valid_ts = diaries_df[diaries_df["actual_arrival_time"] > 0]["actual_arrival_time"]
    # Create a temporary series for dates
    service_dates = valid_ts.apply(get_service_date)
    unique_dates = service_dates.dropna().unique()
    print(f"Dates covered in diaries: {unique_dates}")

    # Add service_date to diaries_df for grouping
    # We apply the same function to the whole column
    diaries_df["service_date"] = diaries_df["actual_arrival_time"].apply(
        get_service_date
    )

    # 4. Build Expected Schedule
    print("Building Expected Schedule (this may take a moment)...")
    expected_trips = []

    for t_id, trip in trips_map.items():
        for d in unique_dates:
            if d in trip.dates:
                expected_trips.append({"trip_id": t_id, "service_date": d})

    expected_df = pd.DataFrame(expected_trips)
    print(f"Found {len(expected_df)} expected trip-runs.")

    # 5. Determine Served Status
    # Extract unique trips that actually ran
    served_keys = diaries_df[["trip_id", "service_date"]].drop_duplicates()
    served_keys["served"] = True

    # Left Merge: Keep all Expected, mark Served if found
    merged = pd.merge(
        expected_df, served_keys, on=["trip_id", "service_date"], how="left"
    )
    merged["served"] = merged["served"].fillna(False)

    # 6. Process DataFrames
    final_dfs = []

    # --- Process Served Trips ---
    print("Processing Served Trips (Interpolating)...")
    # Optimize by grouping existing diaries
    # We only care about trips that are in 'merged' (Expected).
    # Whatever is in 'diaries_df' but NOT in 'expected_df' is technically "Unexpected" (extra).
    # We will process ALL diaries_df to be safe, but mark them as Served=True.

    # Better: Iterate diary groups
    diary_groups = diaries_df.groupby(["trip_id", "service_date"])

    count_served = 0
    for (t_id, s_date), group in diary_groups:
        trip = trips_map.get(t_id)
        if not trip:
            # Trip exists in diary but not in static ledger?
            # normalize_trip_stops requires static ledger info.
            # We skip normalization but keep the data? Or drop?
            # Let's keep data as is.
            group["served"] = True
            final_dfs.append(group)
            continue

        norm = normalize_trip_stops(group, trip)
        norm["served"] = True
        norm["service_date"] = s_date
        final_dfs.append(norm)
        count_served += 1

    print(f"Processed {count_served} served trip clusters.")

    # --- Process Missing Trips ---
    print("Generating Phantom Entries for Missing Trips...")
    missing_trips = merged[~merged["served"]]
    print(f"Found {len(missing_trips)} missing trips.")

    count_missing = 0
    for _, row in missing_trips.iterrows():
        t_id = row["trip_id"]
        s_date = row["service_date"]
        trip = trips_map.get(t_id)
        if not trip:
            continue

        # Create phantom stops
        phantom_rows = []
        for st in trip.stop_times:
            phantom_rows.append(
                {
                    "trip_id": t_id,
                    "stop_id": st["stop_id"],
                    "stop_sequence": int(st["stop_sequence"]),
                    "scheduled_time": st["arrival_time"],
                    "actual_arrival_time": None,
                    "measurement_timestamp": None,
                    "formatted_time": None,
                    "service_date": s_date,
                    "served": False,
                }
            )
        if phantom_rows:
            final_dfs.append(pd.DataFrame(phantom_rows))
            count_missing += 1

    print(f"Generated phantom data for {count_missing} trips.")

    # 7. Save
    print("Saving normalized_diaries.parquet...")
    if final_dfs:
        full_df = pd.concat(final_dfs, ignore_index=True)
        # Ensure correct types for parquet
        full_df["served"] = full_df["served"].astype(bool)

        full_df.to_parquet(OUTPUT_FILE, engine="pyarrow")
        print(f"Saved {len(full_df)} rows to {OUTPUT_FILE}")
    else:
        print("Result Empty.")


if __name__ == "__main__":
    main()
"""
