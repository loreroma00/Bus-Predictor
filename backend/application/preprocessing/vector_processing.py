import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
import json

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PARQUET_DIR = os.path.join(PROJECT_ROOT, "parquets")
STATIC_MAP_FILE = os.path.join(PARQUET_DIR, "canonical_route_map.parquet")
TRAFFIC_AVG_FILE = os.path.join(PARQUET_DIR, "traffic_averages.parquet")
OUTPUT_FILE = os.path.join(PARQUET_DIR, "dataset_lstm_unscaled.parquet")
ENCODING_MAP_FILE = os.path.join(PARQUET_DIR, "route_encoding.json")

# Feature Groups
DYNAMIC_FEATURES = [
    "shape_dist_travelled",
    "distance_to_next_stop",
    "far_status",
    "rush_hour_status",
    "schedule_adherence",
    "speed_ratio",
    "current_traffic_speed",
    "current_speed",
    "precipitation",
    "time_seconds",
    "occupancy_status",
]

# Static Trip Features (Constant for the whole trip)
STATIC_TRIP_FEATURES = [
    "route_id",
    "direction_id",
    "day_type",
    "weather_code",
    "bus_type",
    "door_number",
    "deposit_grottarossa",
    "deposit_magliana",
    "deposit_tor_sapienza",
    "deposit_portonaccio",
    "deposit_monte_sacro",
    "deposit_tor_pagnotta",
    "deposit_tor_cervara",
    "deposit_maglianella",
    "deposit_costi",
    "deposit_trastevere",
    "deposit_acilia",
    "deposit_tor_vergata",
    "deposit_porta_maggiore",
    "served_ratio",
]


def load_static_map():
    """Loads the canonical route map."""
    if not os.path.exists(STATIC_MAP_FILE):
        raise FileNotFoundError(f"Static map file '{STATIC_MAP_FILE}' not found.")
    return pd.read_parquet(STATIC_MAP_FILE)


def load_traffic_avg():
    """Loads traffic averages by h3_index, day_type, and hour."""
    if not os.path.exists(TRAFFIC_AVG_FILE):
        print(f"Warning: Traffic averages file '{TRAFFIC_AVG_FILE}' not found.")
        return pd.DataFrame()
    return pd.read_parquet(TRAFFIC_AVG_FILE)


def load_dynamic_data():
    """Loads all dynamic dataset files."""
    files = glob.glob(os.path.join(PARQUET_DIR, "dataset_*.parquet"))
    files = [f for f in files if "dataset_lstm_unscaled" not in f]

    if not files:
        print("No dynamic dataset files found in ./parquets/")
        return pd.DataFrame()

    print(f"Loading {len(files)} dynamic dataset files...")
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def unroll_time(df, trip_col="trip_id_synthetic"):
    """Unrolls time to handle day crossings (midnight)."""
    df = df.sort_values(by=[trip_col, "ts"])

    # Calculate diff
    # We use the raw 'time_seconds' from the DB (seconds from midnight)
    # If it jumps from ~86400 to ~0, the diff is large negative.
    df["time_diff"] = df.groupby(trip_col)["time_seconds"].diff()

    # Detect jumps (e.g. < -40000 to be safe)
    mask = df["time_diff"] < -40000

    # Cumulative sum of jumps per trip
    df["day_cycle"] = df.groupby(trip_col)["time_diff"].transform(
        lambda x: (x < -40000).cumsum()
    )

    # Apply offset
    df["time_seconds_unrolled"] = df["time_seconds"] + (df["day_cycle"] * 86400)

    return df


def find_nearest_segment(shape_dist, can_shape_dists):
    """
    Finds the nearest segment index for a given shape_dist_travelled.

    Logic:
    - Find i where can_shape_dists[i] < shape_dist < can_shape_dists[i+1]
    - Assign to segment_i if abs(shape_dist - can_shape_dists[i]) < abs(shape_dist - can_shape_dists[i+1])
    - Edge cases: before first -> 0, after last -> len-1
    """
    if shape_dist <= can_shape_dists[0]:
        return 0
    if shape_dist >= can_shape_dists[-1]:
        return len(can_shape_dists) - 1

    # Find the interval using searchsorted
    # searchsorted returns the index where shape_dist would be inserted
    # This gives us i+1 where can_shape_dists[i] < shape_dist < can_shape_dists[i+1]
    idx = np.searchsorted(can_shape_dists, shape_dist)

    # idx is now the right boundary (i+1)
    # Compare distances to left and right boundaries
    dist_left = abs(shape_dist - can_shape_dists[idx - 1])
    dist_right = abs(shape_dist - can_shape_dists[idx])

    if dist_left < dist_right:
        return idx - 1
    else:
        return idx


def compute_dynamic(
    trip_df, static_map, traffic_avg, day_type, x_col="progress", grid_size=100
):
    """
    Interpolates dynamic features onto the fixed grid (0..99).
    Returns a DataFrame with 100 rows.
    """
    route_id = trip_df["route_id"].iloc[0]
    direction_id = trip_df["direction_id"].iloc[0]

    # Get static segments for this trip early (needed for distance-based mapping)
    trip_static = static_map[
        (static_map["route_id"] == route_id)
        & (static_map["direction_id"] == direction_id)
    ].sort_values("segment_idx")

    # Check if we can use distance-based segment assignment
    use_distance_based = (
        not trip_static.empty
        and "can_shape_dist_travelled" in trip_static.columns
        and "shape_dist_travelled" in trip_df.columns
    )

    if use_distance_based:
        can_shape_dists = trip_static["can_shape_dist_travelled"].values
        x_col_actual = "shape_dist_travelled"
    else:
        can_shape_dists = None
        x_col_actual = x_col

    # 1. Prepare X and Y
    # Sort by the appropriate column and drop duplicates
    trip_df = trip_df.sort_values(x_col_actual)
    trip_df = trip_df.drop_duplicates(subset=[x_col_actual], keep="last")

    if len(trip_df) < 2:
        return None

    x = trip_df[x_col_actual].values
    x_new = np.linspace(0, 1, grid_size) if can_shape_dists is None else can_shape_dists

    # Initialize result dictionary
    result = {"segment_idx": np.arange(grid_size)}

    # Mark segments that have genuine measurements
    segment_has_measurement = np.zeros(grid_size, dtype=bool)

    if use_distance_based:
        for shape_dist in x:
            seg_idx = find_nearest_segment(shape_dist, can_shape_dists)
            segment_has_measurement[seg_idx] = True
    else:
        for progress_val in x:
            seg_idx = min(int(progress_val * grid_size), grid_size - 1)
            segment_has_measurement[seg_idx] = True
    result["is_genuine"] = segment_has_measurement.astype(int)

    # 2. Interpolate Features
    # We use PCHIP for continuous features

    for feat in DYNAMIC_FEATURES:
        if feat not in trip_df.columns:
            continue

        y = trip_df[feat].values

        # Determine method
        if feat == "time_seconds":
            y = trip_df["time_seconds_unrolled"].values
            try:
                interpolator = PchipInterpolator(x, y, extrapolate=False)
                y_new = interpolator(x_new)
                nan_mask = np.isnan(y_new)
                if nan_mask.any():
                    coeffs = np.polyfit(x, y, deg=1)
                    y_new[nan_mask] = np.polyval(coeffs, x_new[nan_mask])
                result["time_seconds"] = y_new
            except:
                f = interp1d(x, y, kind="linear", fill_value="extrapolate")
                result["time_seconds"] = f(x_new)

        elif feat == "current_speed":
            try:
                interpolator = PchipInterpolator(x, y, extrapolate=False)
                y_new = interpolator(x_new)
                nan_mask = np.isnan(y_new)
                if nan_mask.any():
                    left_mask = x_new < x[0]
                    right_mask = x_new > x[-1]
                    y_new[left_mask] = y[0]
                    y_new[right_mask] = y[-1]
                result[feat] = y_new
            except:
                y_new = np.full_like(x_new, y[0])
                result[feat] = y_new

        elif feat == "occupancy_status":
            f_zoh = interp1d(
                x, y, kind="nearest", bounds_error=False, fill_value=(y[0], y[-1])
            )
            y_new = f_zoh(x_new)
            y_new = np.clip(y_new, 0, 7)
            y_new = np.round(y_new).astype(int)
            result[feat] = y_new
            result["occupancy_status_available"] = ((y_new >= 0) & (y_new <= 6)).astype(
                int
            )

        elif feat in ["far_status", "rush_hour_status"]:
            continue

        else:
            try:
                interpolator = PchipInterpolator(x, y, extrapolate=True)
                result[feat] = interpolator(x_new)
            except:
                f = interp1d(x, y, kind="linear", fill_value="extrapolate")
                result[feat] = f(x_new)

    # 3. Specific Logic for Status Flags

    # A. Rush Hour Status
    if "time_seconds" in result:
        day_seconds = result["time_seconds"] % 86400
        # 7-9 AM (25200 to 32400) OR 5-7.30 PM (17:00-19:30 -> 61200 to 70200)
        is_morning = (day_seconds >= 25200) & (day_seconds <= 32400)
        is_evening = (day_seconds >= 61200) & (day_seconds <= 70200)
        result["rush_hour_status"] = (is_morning | is_evening).astype(int)

    # B. Far Status & Static Enrichment
    # trip_static, route_id, direction_id already loaded above

    if not trip_static.empty:
        # 1. Far Status
        if "can_distance_to_next_stop" in trip_static.columns:
            if "can_shape_dist_travelled" in trip_static.columns:
                dist_diff = (
                    trip_static["can_distance_to_next_stop"]
                    - trip_static["can_shape_dist_travelled"]
                )
            else:
                canonical_len = trip_static["canonical_len_m"].iloc[0]
                pos = (trip_static["segment_idx"] / 100) * canonical_len
                dist_diff = trip_static["can_distance_to_next_stop"] - pos

            is_near = (dist_diff > 0) & (dist_diff < 250)
            far_status_map = dict(
                zip(trip_static["segment_idx"], (~is_near).astype(int))
            )
            result["far_status"] = [
                far_status_map.get(idx, 1) for idx in result["segment_idx"]
            ]
        else:
            # Fallback if column missing
            result["far_status"] = np.ones(grid_size, dtype=int)

        # 2. H3 Index
        if "h3_index" in trip_static.columns:
            h3_map = dict(zip(trip_static["segment_idx"], trip_static["h3_index"]))
            result["h3_index"] = [
                h3_map.get(idx, None) for idx in result["segment_idx"]
            ]

        # 3. Stop Sequence
        if "stop_sequence" in trip_static.columns:
            stop_seq_map = dict(
                zip(trip_static["segment_idx"], trip_static["stop_sequence"])
            )
            result["stop_sequence"] = [
                stop_seq_map.get(idx, -1) for idx in result["segment_idx"]
            ]

        # 4. Canonical Distance Features
        if "can_shape_dist_travelled" in trip_static.columns:
            can_dist_map = dict(
                zip(trip_static["segment_idx"], trip_static["can_shape_dist_travelled"])
            )
            result["can_shape_dist_travelled"] = [
                can_dist_map.get(idx, 0) for idx in result["segment_idx"]
            ]

        if "can_distance_to_next_stop" in trip_static.columns:
            can_dist_next_map = dict(
                zip(
                    trip_static["segment_idx"], trip_static["can_distance_to_next_stop"]
                )
            )
            result["can_distance_to_next_stop"] = [
                can_dist_next_map.get(idx, 0) for idx in result["segment_idx"]
            ]

    else:
        # Fallback if static map is missing this route
        result["far_status"] = np.ones(grid_size, dtype=int)
        result["h3_index"] = [None] * grid_size
        result["stop_sequence"] = [-1] * grid_size
        result["can_shape_dist_travelled"] = [0] * grid_size
        result["can_distance_to_next_stop"] = [0] * grid_size

    # Add route_id for encoding
    result["route_id"] = [route_id] * grid_size

    # 4. Derived Time Features (Sin/Cos)
    if "time_seconds" in result:
        day_seconds = result["time_seconds"] % 86400
        result["time_sin"] = np.sin(2 * np.pi * day_seconds / 86400)
        result["time_cos"] = np.cos(2 * np.pi * day_seconds / 86400)

    # 5. Traffic Averages Enrichment
    # Enrich each segment with canonical traffic pattern based on h3_index, day_type, and hour
    if not traffic_avg.empty and "h3_index" in result and "time_seconds" in result:
        global_avg_speed_ratio = traffic_avg["avg_speed_ratio"].mean()
        global_avg_traffic_speed = traffic_avg["avg_current_traffic_speed"].mean()

        hours = (result["time_seconds"] % 86400).astype(int) // 3600
        h3_indices = result["h3_index"]

        avg_speed_ratios = []
        avg_traffic_speeds = []

        for i in range(grid_size):
            h3_idx = h3_indices[i]
            hour = hours[i]

            match = traffic_avg[
                (traffic_avg["h3_index"] == h3_idx)
                & (traffic_avg["day_type"] == day_type)
                & (traffic_avg["hour"] == hour)
            ]

            if len(match) == 1:
                avg_speed_ratios.append(match["avg_speed_ratio"].iloc[0])
                avg_traffic_speeds.append(match["avg_current_traffic_speed"].iloc[0])
            elif len(match) > 1:
                avg_speed_ratios.append(match["avg_speed_ratio"].mean())
                avg_traffic_speeds.append(match["avg_current_traffic_speed"].mean())
            else:
                avg_speed_ratios.append(global_avg_speed_ratio)
                avg_traffic_speeds.append(global_avg_traffic_speed)

        result["can_avg_speed_ratio"] = avg_speed_ratios
        result["can_avg_traffic_speed"] = avg_traffic_speeds
    else:
        result["can_avg_speed_ratio"] = [0.0] * grid_size
        result["can_avg_traffic_speed"] = [0.0] * grid_size

    return encode_numerical_values(pd.DataFrame(result), static_map)


def encode_numerical_values(df: pd.DataFrame, static_map: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical numerical values.
    1. route_id -> Kept as-is (will be encoded by ML model embeddings).
    2. h3_index -> Label encoding sorted by alphabetical order (0..N).
    """
    df = df.reset_index(drop=True)

    if "h3_index" in df.columns:
        all_h3 = sorted(static_map["h3_index"].dropna().unique())
        h3_to_id = {h3: i for i, h3 in enumerate(all_h3)}
        df["h3_index_encoded"] = df["h3_index"].map(h3_to_id).fillna(-1).astype(int)

    return df.reset_index(drop=True)


def compute_static(trip_df, grid_size=100):
    """
    Broadcasts static trip features to the grid.
    Also computes scheduled and actual start times if not already present.
    """
    first_row = trip_df.iloc[0]

    result = {}
    for feat in STATIC_TRIP_FEATURES:
        if feat in trip_df.columns:
            result[feat] = np.repeat(first_row[feat], grid_size)

    # Check if start time features are already provided and valid (in any row)
    has_start_times = False
    valid_start_row = None

    start_time_cols = [
        "starting_time_sin",
        "starting_time_cos",
        "sch_starting_time_sin",
        "sch_starting_time_cos",
    ]
    if all(col in trip_df.columns for col in start_time_cols):
        # Find any row with valid values
        valid_mask = (
            trip_df["starting_time_sin"].notna()
            & trip_df["starting_time_cos"].notna()
            & trip_df["sch_starting_time_sin"].notna()
            & trip_df["sch_starting_time_cos"].notna()
        )
        if valid_mask.any():
            has_start_times = True
            valid_start_row = trip_df[valid_mask].iloc[0]

    if has_start_times:
        # Use provided values
        result["actual_start_time_sin"] = np.repeat(
            valid_start_row["starting_time_sin"], grid_size
        )
        result["actual_start_time_cos"] = np.repeat(
            valid_start_row["starting_time_cos"], grid_size
        )
        result["scheduled_start_time_sin"] = np.repeat(
            valid_start_row["sch_starting_time_sin"], grid_size
        )
        result["scheduled_start_time_cos"] = np.repeat(
            valid_start_row["sch_starting_time_cos"], grid_size
        )
    else:
        # Compute scheduled and actual start times
        # Find where the trip actually starts (distance_to_next_stop decreases significantly)
        if (
            "distance_to_next_stop" in trip_df.columns
            and "time_seconds_unrolled" in trip_df.columns
        ):
            trip_sorted = trip_df.sort_values("ts")
            dist_to_next = trip_sorted["distance_to_next_stop"].values
            time_unrolled = trip_sorted["time_seconds_unrolled"].values

            # Find first point where distance_to_next_stop decreases by > 100m
            # This indicates the bus has started moving from the first stop
            start_idx = None
            for i in range(1, len(dist_to_next)):
                if not pd.isna(dist_to_next[i]) and not pd.isna(dist_to_next[i - 1]):
                    if dist_to_next[i - 1] - dist_to_next[i] > 100:
                        start_idx = i
                        break

            if start_idx is not None:
                actual_start_time = time_unrolled[start_idx]

                # Calculate scheduled start time
                # schedule_adherence = actual_time - expected_time
                # scheduled_start = expected_time_at_start = actual_time - schedule_adherence
                if "schedule_adherence" in trip_sorted.columns:
                    sched_adh = trip_sorted["schedule_adherence"].iloc[start_idx]
                    if not pd.isna(sched_adh):
                        scheduled_start_time = actual_start_time - sched_adh
                    else:
                        scheduled_start_time = actual_start_time
                else:
                    scheduled_start_time = actual_start_time
            else:
                # Fallback: use first observation
                actual_start_time = time_unrolled[0] if len(time_unrolled) > 0 else 0
                if "schedule_adherence" in trip_sorted.columns and len(trip_sorted) > 0:
                    sched_adh = trip_sorted["schedule_adherence"].iloc[0]
                    if not pd.isna(sched_adh):
                        scheduled_start_time = actual_start_time - sched_adh
                    else:
                        scheduled_start_time = actual_start_time
                else:
                    scheduled_start_time = actual_start_time

            # Convert to seconds from midnight (mod 86400) and create sin/cos features
            actual_start_seconds = actual_start_time % 86400
            scheduled_start_seconds = scheduled_start_time % 86400

            result["actual_start_time_sin"] = np.repeat(
                np.sin(2 * np.pi * actual_start_seconds / 86400), grid_size
            )
            result["actual_start_time_cos"] = np.repeat(
                np.cos(2 * np.pi * actual_start_seconds / 86400), grid_size
            )
            result["scheduled_start_time_sin"] = np.repeat(
                np.sin(2 * np.pi * scheduled_start_seconds / 86400), grid_size
            )
            result["scheduled_start_time_cos"] = np.repeat(
                np.cos(2 * np.pi * scheduled_start_seconds / 86400), grid_size
            )

    return pd.DataFrame(result).reset_index(drop=True)


def join_vector(dynamic_df, static_df):
    """Joins dynamic and static dataframes."""
    dynamic_df = dynamic_df.reset_index(drop=True).copy()
    static_df = static_df.reset_index(drop=True).copy()
    for col in static_df.columns:
        dynamic_df[col] = static_df[col].values
    return dynamic_df


def process_data():
    print("Loading static map...")
    static_map = load_static_map()

    print("Loading traffic averages...")
    traffic_avg = load_traffic_avg()

    print("Loading dynamic data...")
    df = load_dynamic_data()

    if df.empty:
        print("No dynamic data to process.")
        return

    # Drop deprecated 'time_feat' if present
    if "time_feat" in df.columns:
        print("Dropping deprecated 'time_feat' column.")
        df.drop(columns=["time_feat"], inplace=True)

    # --- 5σ Outlier Filter (before interpolation) ---
    trip_col = "trip_id_synthetic" if "trip_id_synthetic" in df.columns else "trip_id"
    exclude_cols = {"delay", "occupancy_status", trip_col}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    outlier_stats = {}
    for col in numeric_cols:
        if df[col].isna().all():
            continue
        col_std = df[col].std()
        if col_std == 0 or pd.isna(col_std):
            continue
        col_mean = df[col].mean()
        outlier_stats[col] = (col_mean, col_std)

    if outlier_stats:
        valid_trips_outlier = []
        for tid, group in df.groupby(trip_col):
            has_outlier = False
            for col, (mean, std) in outlier_stats.items():
                col_vals = group[col].dropna()
                if len(col_vals) > 0:
                    if (np.abs(col_vals - mean) > 5 * std).any():
                        has_outlier = True
                        break
            if not has_outlier:
                valid_trips_outlier.append(tid)

        before_count = df[trip_col].nunique()
        df = df[df[trip_col].isin(valid_trips_outlier)]
        after_count = df[trip_col].nunique()
        print(
            f"Dropped {before_count - after_count} trips with outliers >5σ ({after_count} remaining)"
        )

    # --- Trip Duration Filter (before stop_sequence is dropped) ---
    trip_col = "trip_id_synthetic" if "trip_id_synthetic" in df.columns else "trip_id"
    print(f"Using trip column: {trip_col}")

    print("Unrolling time...")
    df = unroll_time(df, trip_col)

    print("Filtering trips over 2h duration...")
    if "stop_sequence" in df.columns:
        valid_trips = []
        for tid, group in df.groupby(trip_col):
            if len(group) < 2:
                valid_trips.append(tid)
                continue
            min_stop = group["stop_sequence"].min()
            max_stop = group["stop_sequence"].max()
            if pd.isna(min_stop) or pd.isna(max_stop):
                valid_trips.append(tid)
                continue
            time_at_min = group.loc[
                group["stop_sequence"] == min_stop, "time_seconds_unrolled"
            ]
            time_at_max = group.loc[
                group["stop_sequence"] == max_stop, "time_seconds_unrolled"
            ]
            if len(time_at_min) == 0 or len(time_at_max) == 0:
                valid_trips.append(tid)
                continue
            duration = time_at_max.iloc[0] - time_at_min.iloc[0]
            if duration <= 7200:
                valid_trips.append(tid)

        before_count = df[trip_col].nunique()
        df = df[df[trip_col].isin(valid_trips)]
        after_count = df[trip_col].nunique()
        print(
            f"Dropped {before_count - after_count} trips over 2h duration ({after_count} remaining)"
        )

    # Also drop 'stop_sequence' from dynamic data as it will be statically deduced
    if "stop_sequence" in df.columns:
        print(
            "Dropping dynamic 'stop_sequence' (will be replaced by static map value)."
        )
        df.drop(columns=["stop_sequence"], inplace=True)

    # --- NaN Handling for distance_to_next_stop ---
    if "distance_to_next_stop" in df.columns:
        df["distance_to_next_stop"] = df["distance_to_next_stop"].fillna(-1000)

    # --- schedule_adherence: Filter trips with < 70% real data, interpolate rest ---
    print("Checking schedule_adherence availability per trip...")
    if "schedule_adherence" in df.columns:
        sched_availability = df.groupby(trip_col)["schedule_adherence"].apply(
            lambda x: x.notna().sum() / len(x)
        )
        valid_sched_trips = sched_availability[sched_availability >= 0.7].index
        before_count = df[trip_col].nunique()
        df = df[df[trip_col].isin(valid_sched_trips)]
        after_count = df[trip_col].nunique()
        print(
            f"Dropped {before_count - after_count} trips with < 70% schedule_adherence data ({after_count} remaining)"
        )

        # Interpolate missing schedule_adherence values per trip
        print("Interpolating missing schedule_adherence values...")
        df = df.sort_values([trip_col, "ts"])

        # Save trip_col values before groupby (which may drop the column)
        trip_col_values = df[trip_col].values.copy()

        def interpolate_sched(group):
            if group["schedule_adherence"].isna().all():
                return group
            mask = group["schedule_adherence"].isna()
            if mask.any():
                valid_idx = group.loc[~mask, "ts"].values
                valid_vals = group.loc[~mask, "schedule_adherence"].values
                if len(valid_idx) >= 2:
                    interp_vals = np.interp(
                        group.loc[mask, "ts"].astype(np.int64).values,
                        valid_idx.astype(np.int64).values,
                        valid_vals,
                    )
                    group.loc[mask, "schedule_adherence"] = interp_vals
                elif len(valid_idx) == 1:
                    group["schedule_adherence"] = group["schedule_adherence"].fillna(
                        valid_vals[0]
                    )
            return group

        df = df.groupby(trip_col, group_keys=False).apply(interpolate_sched)

        # Restore trip_col if it was dropped
        if trip_col not in df.columns:
            df[trip_col] = trip_col_values

    # --- Pre-computation Logic ---

    # 1. Attach Canonical Length (Needed for Progress)
    print("Attaching canonical length...")
    if "canonical_len_m" not in static_map.columns:
        # Assuming canonical_len_m exists based on previous checks
        pass

    lookup = static_map[
        ["route_id", "direction_id", "canonical_len_m"]
    ].drop_duplicates()
    df = df.merge(lookup, on=["route_id", "direction_id"], how="left")
    df = df.dropna(subset=["canonical_len_m"])

    # 2. Calculate Progress
    df["progress"] = df["shape_dist_travelled"] / df["canonical_len_m"]
    df["progress"] = df["progress"].clip(0.0, 1.0)

    # 3. Filter trips by progress coverage (drop trips covering < 20% of route)
    print("Filtering trips by progress coverage...")
    progress_coverage = df.groupby(trip_col)["progress"].agg(["min", "max"])
    progress_coverage["coverage"] = progress_coverage["max"] - progress_coverage["min"]
    valid_coverage_trips = progress_coverage[progress_coverage["coverage"] >= 0.2].index
    before_count = df[trip_col].nunique()
    df = df[df[trip_col].isin(valid_coverage_trips)]
    after_count = df[trip_col].nunique()
    print(
        f"Dropped {before_count - after_count} trips with low progress coverage ({after_count} remaining)"
    )

    # --- Main Loop ---
    print("Processing trips (Dynamic + Static)...")

    final_dfs = []

    for trip_id, trip_df in df.groupby(trip_col):
        day_type = trip_df["day_type"].iloc[0] if "day_type" in trip_df.columns else 0
        dyn_df = compute_dynamic(trip_df, static_map, traffic_avg, day_type)
        if dyn_df is None:
            continue

        # Post-interpolation check: drop trips with extreme time_seconds values
        if "time_seconds" in dyn_df.columns:
            time_min = dyn_df["time_seconds"].min()
            time_max = dyn_df["time_seconds"].max()
            # Reasonable bounds: 0 to 3 days in seconds (259200)
            if time_min < 0 or time_max > 259200:
                continue

        # Post-interpolation check: drop trips with extreme schedule_adherence (> 1 hour)
        if "schedule_adherence" in dyn_df.columns:
            sched_abs_max = dyn_df["schedule_adherence"].abs().max()
            if sched_abs_max > 3600:
                continue

        # B. Compute Static
        stat_df = compute_static(trip_df)

        # C. Join
        full_df = join_vector(dyn_df, stat_df)

        # Add Trip ID
        full_df = full_df.assign(**{trip_col: trip_id})

        final_dfs.append(full_df)

    if not final_dfs:
        print("No valid trips generated.")
        return

    combined_df = pd.concat(final_dfs, ignore_index=True)

    # --- Post-Processing Enrichment ---

    # 5. Encoding reference
    print("Saving encoding mappings...")
    unique_routes = sorted(static_map["route_id"].unique())
    route_map = {route: idx for idx, route in enumerate(unique_routes)}

    # Save Map
    with open(ENCODING_MAP_FILE, "w") as f:
        # Convert keys/values to standard types
        json.dump({str(k): int(v) for k, v in route_map.items()}, f)

    # Save H3 map for reference as well
    h3_map_file = os.path.join(PARQUET_DIR, "h3_encoding.json")
    unique_h3 = sorted(static_map["h3_index"].dropna().unique())
    h3_map = {h3: idx for idx, h3 in enumerate(unique_h3)}
    with open(h3_map_file, "w") as f:
        json.dump({str(k): int(v) for k, v in h3_map.items()}, f)

    # Final Filter
    combined_df = combined_df.dropna(subset=["h3_index"])

    # Check 100 rows per trip
    trip_counts = combined_df[trip_col].value_counts()
    valid_trips = trip_counts[trip_counts == 100].index
    combined_df = combined_df[combined_df[trip_col].isin(valid_trips)]

    print(f"Final dataset: {len(valid_trips)} trips, {len(combined_df)} rows.")

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    combined_df.to_parquet(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    process_data()
