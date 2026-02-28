"""
Vector Processing - Optimized for performance.

Interpolates dynamic trip data onto fixed 100-segment grid.
Uses:
- Pre-indexed lookups for static_map and traffic_avg
- Parallel processing with joblib
- Vectorized numpy operations
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d
from joblib import Parallel, delayed
from typing import Optional, Tuple, Dict, Any

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PARQUET_DIR = os.path.join(PROJECT_ROOT, "parquets")
STATIC_MAP_FILE = os.path.join(PARQUET_DIR, "canonical_route_map.parquet")
TRAFFIC_AVG_FILE = os.path.join(PARQUET_DIR, "traffic_averages.parquet")
OUTPUT_FILE = os.path.join(PARQUET_DIR, "dataset_lstm_unscaled.parquet")
ENCODING_MAP_FILE = os.path.join(PARQUET_DIR, "route_encoding.json")

GRID_SIZE = 100

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


def load_static_map() -> pd.DataFrame:
    if not os.path.exists(STATIC_MAP_FILE):
        raise FileNotFoundError(f"Static map file '{STATIC_MAP_FILE}' not found.")
    return pd.read_parquet(STATIC_MAP_FILE)


def load_traffic_avg() -> pd.DataFrame:
    if not os.path.exists(TRAFFIC_AVG_FILE):
        print(f"Warning: Traffic averages file '{TRAFFIC_AVG_FILE}' not found.")
        return pd.DataFrame()
    return pd.read_parquet(TRAFFIC_AVG_FILE)


def load_dynamic_data(start_date: str = None) -> pd.DataFrame:
    """
    Load all dynamic dataset files.

    Args:
        start_date: Optional start date (YYYY-MM-DD). Only load files from this date onwards.
    """
    files = glob.glob(os.path.join(PARQUET_DIR, "dataset_*.parquet"))
    files = [
        f
        for f in files
        if "dataset_lstm_unscaled" not in f and "dataset_lstm_final" not in f
    ]

    if not files:
        print("No dynamic dataset files found in ./parquets/")
        return pd.DataFrame()

    # Filter files by start_date if provided
    if start_date:
        from datetime import datetime

        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        filtered_files = []
        for f in files:
            try:
                filename = os.path.basename(f)
                date_str = filename.replace("dataset_", "").replace(".parquet", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if file_date >= start_dt:
                    filtered_files.append(f)
            except ValueError:
                continue
        files = filtered_files
        print(f"Filtering to {len(files)} files from {start_date} onwards")
    else:
        print(f"Loading {len(files)} dynamic dataset files...")

    dfs = [pd.read_parquet(f) for f in sorted(files)]

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def build_static_map_index(
    static_map: pd.DataFrame,
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Pre-index static_map by (route_id, direction_id).
    Returns dict mapping (route_id, direction_id) -> sorted DataFrame of segments.
    """
    print("Building static map index...")
    index = {}
    for (route_id, direction_id), group in static_map.groupby(
        ["route_id", "direction_id"]
    ):
        index[(str(route_id), int(direction_id))] = group.sort_values(
            "segment_idx"
        ).reset_index(drop=True)
    print(f"  Indexed {len(index)} route+direction combinations")
    return index


def build_traffic_avg_index(
    traffic_avg: pd.DataFrame,
) -> Dict[Tuple[str, int, int], Tuple[float, float]]:
    """
    Pre-index traffic_avg by (h3_index, day_type, hour).
    Returns dict mapping (h3_index, day_type, hour) -> (avg_speed_ratio, avg_current_traffic_speed).
    """
    if traffic_avg.empty:
        return {}

    print("Building traffic average index...")
    index = {}
    for _, row in traffic_avg.iterrows():
        key = (str(row["h3_index"]), int(row["day_type"]), int(row["hour"]))
        index[key] = (
            float(row["avg_speed_ratio"]),
            float(row["avg_current_traffic_speed"]),
        )

    print(f"  Indexed {len(index)} traffic average entries")
    return index


def build_h3_encoding(static_map: pd.DataFrame) -> Dict[str, int]:
    """Pre-build h3_index to integer encoding."""
    unique_h3 = sorted(static_map["h3_index"].dropna().unique())
    return {str(h3): i for i, h3 in enumerate(unique_h3)}


def unroll_time(df: pd.DataFrame, trip_col: str = "trip_id") -> pd.DataFrame:
    """Unrolls time to handle day crossings (midnight)."""
    df = df.sort_values(by=[trip_col, "ts"])
    df["time_diff"] = df.groupby(trip_col)["time_seconds"].diff()
    df["day_cycle"] = df.groupby(trip_col)["time_diff"].transform(
        lambda x: (x < -40000).cumsum()
    )
    df["time_seconds_unrolled"] = df["time_seconds"] + (df["day_cycle"] * 86400)
    return df


def compute_outlier_stats(
    df: pd.DataFrame, trip_col: str
) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std for outlier detection."""
    exclude_cols = {"delay", "occupancy_status", trip_col, "id", "ts"}
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

    return outlier_stats


def filter_trips_combined(
    df: pd.DataFrame, trip_col: str, outlier_stats: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Combined filter pass: outlier, duration, and schedule_adherence in one loop.
    """
    print("Filtering trips (combined pass)...")

    valid_trips = []
    total_trips = df[trip_col].nunique()
    dropped_outlier = 0
    dropped_duration = 0
    dropped_sched = 0
    dropped_coverage = 0

    for tid, group in df.groupby(trip_col):
        # 1. Outlier check
        has_outlier = False
        for col, (mean, std) in outlier_stats.items():
            col_vals = group[col].dropna()
            if len(col_vals) > 0 and (np.abs(col_vals - mean) > 5 * std).any():
                has_outlier = True
                dropped_outlier += 1
                break
        if has_outlier:
            continue

        # 2. Duration check (over 2h)
        if (
            "stop_sequence" in group.columns
            and "time_seconds_unrolled" in group.columns
        ):
            min_stop = group["stop_sequence"].min()
            max_stop = group["stop_sequence"].max()
            if not pd.isna(min_stop) and not pd.isna(max_stop):
                time_min = group.loc[
                    group["stop_sequence"] == min_stop, "time_seconds_unrolled"
                ]
                time_max = group.loc[
                    group["stop_sequence"] == max_stop, "time_seconds_unrolled"
                ]
                if len(time_min) > 0 and len(time_max) > 0:
                    duration = time_max.iloc[0] - time_min.iloc[0]
                    if duration > 7200:
                        dropped_duration += 1
                        continue

        # 3. Schedule adherence check (>= 70% data)
        if "schedule_adherence" in group.columns:
            sched_avail = group["schedule_adherence"].notna().sum() / len(group)
            if sched_avail < 0.7:
                dropped_sched += 1
                continue

        # 4. Progress coverage check (>= 20%)
        if "progress" in group.columns:
            coverage = group["progress"].max() - group["progress"].min()
            if coverage < 0.2:
                dropped_coverage += 1
                continue

        valid_trips.append(tid)

    print(f"  Total trips: {total_trips}")
    print(f"  Dropped by outlier: {dropped_outlier}")
    print(f"  Dropped by duration: {dropped_duration}")
    print(f"  Dropped by schedule: {dropped_sched}")
    print(f"  Dropped by coverage: {dropped_coverage}")
    print(f"  Remaining: {len(valid_trips)}")

    return df[df[trip_col].isin(valid_trips)]


def interpolate_schedule_adherence(df: pd.DataFrame, trip_col: str) -> pd.DataFrame:
    """Interpolate missing schedule_adherence values per trip."""
    if "schedule_adherence" not in df.columns:
        return df

    print("Interpolating missing schedule_adherence values...")
    df = df.sort_values([trip_col, "ts"])

    def interp_group(group):
        mask = group["schedule_adherence"].isna()
        if not mask.any():
            return group

        valid_ts = group.loc[~mask, "ts"].values
        valid_vals = group.loc[~mask, "schedule_adherence"].values

        if len(valid_ts) >= 2:
            interp_vals = np.interp(
                group.loc[mask, "ts"].astype(np.int64).values,
                valid_ts.astype(np.int64).values,
                valid_vals,
            )
            group.loc[mask, "schedule_adherence"] = interp_vals
        elif len(valid_ts) == 1:
            group["schedule_adherence"] = group["schedule_adherence"].fillna(
                valid_vals[0]
            )

        return group

    df = df.groupby(trip_col, group_keys=False).apply(interp_group)
    return df


def process_single_trip(
    trip_id: Any,
    trip_df: pd.DataFrame,
    static_map_index: Dict[Tuple[str, int], pd.DataFrame],
    traffic_avg_index: Dict[Tuple[str, int, int], Tuple[float, float]],
    h3_encoding: Dict[str, int],
    global_avg_speed_ratio: float,
    global_avg_traffic_speed: float,
    trip_col: str,
) -> Optional[pd.DataFrame]:
    """
    Process a single trip - designed for parallelization.
    Returns DataFrame with 100 rows (one per segment) or None if invalid.
    """
    route_id = str(trip_df["route_id"].iloc[0])
    direction_id = int(trip_df["direction_id"].iloc[0])
    day_type = int(trip_df["day_type"].iloc[0]) if "day_type" in trip_df.columns else 0

    # O(1) lookup for static map
    trip_static = static_map_index.get((route_id, direction_id))
    if trip_static is None or trip_static.empty:
        return None

    # Check for distance-based mapping
    use_distance = (
        "can_shape_dist_travelled" in trip_static.columns
        and "shape_dist_travelled" in trip_df.columns
    )

    if use_distance:
        can_shape_dists = trip_static["can_shape_dist_travelled"].values
        x_col = "shape_dist_travelled"
    else:
        can_shape_dists = None
        x_col = "progress"

    # Sort and dedupe
    trip_df = trip_df.sort_values(x_col).drop_duplicates(subset=[x_col], keep="last")

    if len(trip_df) < 2:
        return None

    x = trip_df[x_col].values
    x_new = can_shape_dists if use_distance else np.linspace(0, 1, GRID_SIZE)

    # Initialize result arrays
    result = {"segment_idx": np.arange(GRID_SIZE)}

    # Mark genuine measurements
    if use_distance:
        segment_has_measurement = np.zeros(GRID_SIZE, dtype=bool)
        for shape_dist in x:
            idx = np.searchsorted(can_shape_dists, shape_dist)
            if idx == 0:
                seg_idx = 0
            elif idx >= len(can_shape_dists):
                seg_idx = len(can_shape_dists) - 1
            else:
                dist_left = abs(shape_dist - can_shape_dists[idx - 1])
                dist_right = abs(shape_dist - can_shape_dists[idx])
                seg_idx = idx - 1 if dist_left < dist_right else idx
            segment_has_measurement[seg_idx] = True
        result["is_genuine"] = segment_has_measurement.astype(int)
    else:
        seg_indices = np.minimum((x * GRID_SIZE).astype(int), GRID_SIZE - 1)
        segment_has_measurement = np.zeros(GRID_SIZE, dtype=bool)
        segment_has_measurement[seg_indices] = True
        result["is_genuine"] = segment_has_measurement.astype(int)

    # Interpolate dynamic features
    for feat in DYNAMIC_FEATURES:
        if feat not in trip_df.columns:
            continue

        y = trip_df[feat].values

        if feat == "time_seconds":
            y = trip_df["time_seconds_unrolled"].values
            try:
                interp = PchipInterpolator(x, y, extrapolate=False)
                y_new = interp(x_new)
                nan_mask = np.isnan(y_new)
                if nan_mask.any():
                    coeffs = np.polyfit(x, y, deg=1)
                    y_new[nan_mask] = np.polyval(coeffs, x_new[nan_mask])
                result["time_seconds"] = y_new
            except Exception:
                f = interp1d(x, y, kind="linear", fill_value="extrapolate")
                result["time_seconds"] = f(x_new)

        elif feat == "current_speed":
            try:
                interp = PchipInterpolator(x, y, extrapolate=False)
                y_new = interp(x_new)
                nan_mask = np.isnan(y_new)
                if nan_mask.any():
                    y_new[x_new < x[0]] = y[0]
                    y_new[x_new > x[-1]] = y[-1]
                result[feat] = y_new
            except Exception:
                result[feat] = np.full(GRID_SIZE, y[0])

        elif feat == "occupancy_status":
            f_zoh = interp1d(
                x, y, kind="nearest", bounds_error=False, fill_value=(y[0], y[-1])
            )
            y_new = np.clip(f_zoh(x_new), 0, 7)
            result[feat] = np.round(y_new).astype(int)
            result["occupancy_status_available"] = ((y_new >= 0) & (y_new <= 6)).astype(
                int
            )

        elif feat in ["far_status", "rush_hour_status"]:
            continue

        else:
            try:
                interp = PchipInterpolator(x, y, extrapolate=True)
                result[feat] = interp(x_new)
            except Exception:
                f = interp1d(x, y, kind="linear", fill_value="extrapolate")
                result[feat] = f(x_new)

    # Rush hour status from time_seconds
    if "time_seconds" in result:
        day_seconds = result["time_seconds"] % 86400
        is_morning = (day_seconds >= 25200) & (day_seconds <= 32400)
        is_evening = (day_seconds >= 61200) & (day_seconds <= 70200)
        result["rush_hour_status"] = (is_morning | is_evening).astype(int)

    # Static enrichment from trip_static (vectorized)
    segment_idx_arr = trip_static["segment_idx"].values

    # Far status
    if "can_distance_to_next_stop" in trip_static.columns:
        if "can_shape_dist_travelled" in trip_static.columns:
            dist_diff = (
                trip_static["can_distance_to_next_stop"].values
                - trip_static["can_shape_dist_travelled"].values
            )
        else:
            canonical_len = trip_static["canonical_len_m"].iloc[0]
            pos = (segment_idx_arr / 100) * canonical_len
            dist_diff = trip_static["can_distance_to_next_stop"].values - pos

        is_near = (dist_diff > 0) & (dist_diff < 250)
        far_status = np.full(GRID_SIZE, 1, dtype=int)
        far_status[segment_idx_arr] = (~is_near).astype(int)
        result["far_status"] = far_status
    else:
        result["far_status"] = np.ones(GRID_SIZE, dtype=int)

    # H3 Index
    if "h3_index" in trip_static.columns:
        h3_arr = np.full(GRID_SIZE, None, dtype=object)
        h3_arr[segment_idx_arr] = trip_static["h3_index"].values
        result["h3_index"] = h3_arr

        # H3 encoding
        h3_encoded = np.full(GRID_SIZE, -1, dtype=int)
        for i in range(GRID_SIZE):
            if h3_arr[i] is not None and str(h3_arr[i]) in h3_encoding:
                h3_encoded[i] = h3_encoding[str(h3_arr[i])]
        result["h3_index_encoded"] = h3_encoded
    else:
        result["h3_index"] = np.full(GRID_SIZE, None, dtype=object)
        result["h3_index_encoded"] = np.full(GRID_SIZE, -1, dtype=int)

    # Stop sequence
    if "stop_sequence" in trip_static.columns:
        stop_seq = np.full(GRID_SIZE, -1, dtype=int)
        stop_seq[segment_idx_arr] = trip_static["stop_sequence"].values
        result["stop_sequence"] = stop_seq
    else:
        result["stop_sequence"] = np.full(GRID_SIZE, -1, dtype=int)

    # Canonical distance features
    if "can_shape_dist_travelled" in trip_static.columns:
        can_dist = np.full(GRID_SIZE, 0.0)
        can_dist[segment_idx_arr] = trip_static["can_shape_dist_travelled"].values
        result["can_shape_dist_travelled"] = can_dist

    if "can_distance_to_next_stop" in trip_static.columns:
        can_dist_next = np.full(GRID_SIZE, 0.0)
        can_dist_next[segment_idx_arr] = trip_static["can_distance_to_next_stop"].values
        result["can_distance_to_next_stop"] = can_dist_next

    # Route ID
    result["route_id"] = route_id

    # Time sin/cos
    if "time_seconds" in result:
        day_seconds = result["time_seconds"] % 86400
        result["time_sin"] = np.sin(2 * np.pi * day_seconds / 86400)
        result["time_cos"] = np.cos(2 * np.pi * day_seconds / 86400)

    # Traffic averages (using pre-built index)
    if traffic_avg_index and "h3_index" in result and "time_seconds" in result:
        hours = (result["time_seconds"] % 86400).astype(int) // 3600
        h3_indices = result["h3_index"]

        avg_speed_ratios = []
        avg_traffic_speeds = []

        for i in range(GRID_SIZE):
            key = (str(h3_indices[i]), day_type, hours[i])
            if key in traffic_avg_index:
                sr, ts = traffic_avg_index[key]
                avg_speed_ratios.append(sr)
                avg_traffic_speeds.append(ts)
            else:
                avg_speed_ratios.append(global_avg_speed_ratio)
                avg_traffic_speeds.append(global_avg_traffic_speed)

        result["can_avg_speed_ratio"] = avg_speed_ratios
        result["can_avg_traffic_speed"] = avg_traffic_speeds
    else:
        result["can_avg_speed_ratio"] = [0.0] * GRID_SIZE
        result["can_avg_traffic_speed"] = [0.0] * GRID_SIZE

    # Compute static features
    static_result = compute_static_features(trip_df)
    result.update(static_result)

    # Post-interpolation validation
    if "time_seconds" in result:
        if result["time_seconds"].min() < 0 or result["time_seconds"].max() > 259200:
            return None

    if "schedule_adherence" in result:
        if np.abs(result["schedule_adherence"]).max() > 3600:
            return None

    # Build DataFrame
    df_result = pd.DataFrame(result)
    df_result[trip_col] = trip_id

    return df_result


def compute_static_features(trip_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute static trip features."""
    first_row = trip_df.iloc[0]
    result = {}

    for feat in STATIC_TRIP_FEATURES:
        if feat in trip_df.columns:
            result[feat] = [first_row[feat]] * GRID_SIZE

    # Start time features
    start_time_cols = [
        "starting_time_sin",
        "starting_time_cos",
        "sch_starting_time_sin",
        "sch_starting_time_cos",
    ]

    has_start_times = all(col in trip_df.columns for col in start_time_cols)
    if has_start_times:
        valid_mask = (
            trip_df["starting_time_sin"].notna()
            & trip_df["starting_time_cos"].notna()
            & trip_df["sch_starting_time_sin"].notna()
            & trip_df["sch_starting_time_cos"].notna()
        )
        if valid_mask.any():
            row = trip_df[valid_mask].iloc[0]
            result["actual_start_time_sin"] = [row["starting_time_sin"]] * GRID_SIZE
            result["actual_start_time_cos"] = [row["starting_time_cos"]] * GRID_SIZE
            result["scheduled_start_time_sin"] = [
                row["sch_starting_time_sin"]
            ] * GRID_SIZE
            result["scheduled_start_time_cos"] = [
                row["sch_starting_time_cos"]
            ] * GRID_SIZE
            return result

    # Compute start times from distance_to_next_stop
    if (
        "distance_to_next_stop" in trip_df.columns
        and "time_seconds_unrolled" in trip_df.columns
    ):
        trip_sorted = trip_df.sort_values("ts")
        dist_to_next = trip_sorted["distance_to_next_stop"].values
        time_unrolled = trip_sorted["time_seconds_unrolled"].values

        # Find first point where distance decreases by > 100m
        start_idx = None
        for i in range(1, len(dist_to_next)):
            if not pd.isna(dist_to_next[i]) and not pd.isna(dist_to_next[i - 1]):
                if dist_to_next[i - 1] - dist_to_next[i] > 100:
                    start_idx = i
                    break

        if start_idx is not None:
            actual_start = time_unrolled[start_idx]
            if "schedule_adherence" in trip_sorted.columns:
                sched_adh = trip_sorted["schedule_adherence"].iloc[start_idx]
                scheduled_start = (
                    actual_start - sched_adh if not pd.isna(sched_adh) else actual_start
                )
            else:
                scheduled_start = actual_start
        else:
            actual_start = time_unrolled[0] if len(time_unrolled) > 0 else 0
            if "schedule_adherence" in trip_sorted.columns and len(trip_sorted) > 0:
                sched_adh = trip_sorted["schedule_adherence"].iloc[0]
                scheduled_start = (
                    actual_start - sched_adh if not pd.isna(sched_adh) else actual_start
                )
            else:
                scheduled_start = actual_start

        actual_sec = actual_start % 86400
        scheduled_sec = scheduled_start % 86400

        result["actual_start_time_sin"] = [
            np.sin(2 * np.pi * actual_sec / 86400)
        ] * GRID_SIZE
        result["actual_start_time_cos"] = [
            np.cos(2 * np.pi * actual_sec / 86400)
        ] * GRID_SIZE
        result["scheduled_start_time_sin"] = [
            np.sin(2 * np.pi * scheduled_sec / 86400)
        ] * GRID_SIZE
        result["scheduled_start_time_cos"] = [
            np.cos(2 * np.pi * scheduled_sec / 86400)
        ] * GRID_SIZE

    return result


def process_data(start_date: str = None):
    """
    Main entry point for vector processing.

    Args:
        start_date: Optional start date (YYYY-MM-DD). Only process data from this date onwards.
    """
    print("=" * 60)
    print("VECTOR PROCESSING (Optimized)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    static_map = load_static_map()
    traffic_avg = load_traffic_avg()
    df = load_dynamic_data(start_date=start_date)

    if df.empty:
        print("No dynamic data to process.")
        return

    # Drop deprecated column
    if "time_feat" in df.columns:
        print("Dropping deprecated 'time_feat' column.")
        df.drop(columns=["time_feat"], inplace=True)

    # Determine trip column - must exist
    if "trip_id_synthetic" in df.columns:
        trip_col = "trip_id_synthetic"
    elif "trip_id" in df.columns:
        trip_col = "trip_id"
    else:
        print("Error: No trip_id or trip_id_synthetic column found in data.")
        print(f"Available columns: {list(df.columns)}")
        return
    print(f"Using trip column: {trip_col}")

    # Pre-build indexes
    print("\nBuilding lookup indexes...")
    static_map_index = build_static_map_index(static_map)
    traffic_avg_index = build_traffic_avg_index(traffic_avg)
    h3_encoding = build_h3_encoding(static_map)

    # Pre-compute global traffic averages
    global_avg_speed_ratio = (
        traffic_avg["avg_speed_ratio"].mean() if not traffic_avg.empty else 0.0
    )
    global_avg_traffic_speed = (
        traffic_avg["avg_current_traffic_speed"].mean()
        if not traffic_avg.empty
        else 0.0
    )

    # Compute outlier stats
    print("\nComputing outlier statistics...")
    outlier_stats = compute_outlier_stats(df, trip_col)
    print(f"  Computed stats for {len(outlier_stats)} features")

    # Unroll time
    print("\nUnrolling time...")
    df = unroll_time(df, trip_col)

    # Fill distance_to_next_stop NaN
    if "distance_to_next_stop" in df.columns:
        df["distance_to_next_stop"] = df["distance_to_next_stop"].fillna(-1000)

    # Attach canonical length and compute progress
    print("\nAttaching canonical length...")
    lookup = static_map[
        ["route_id", "direction_id", "canonical_len_m"]
    ].drop_duplicates()
    df = df.merge(lookup, on=["route_id", "direction_id"], how="left")
    df = df.dropna(subset=["canonical_len_m"])
    df["progress"] = df["shape_dist_travelled"] / df["canonical_len_m"]
    df["progress"] = df["progress"].clip(0.0, 1.0)

    # Combined filtering
    print("\n" + "-" * 40)
    df = filter_trips_combined(df, trip_col, outlier_stats)

    # Interpolate schedule_adherence
    df = interpolate_schedule_adherence(df, trip_col)

    # Drop stop_sequence (will be replaced from static map)
    if "stop_sequence" in df.columns:
        df.drop(columns=["stop_sequence"], inplace=True)

    # Parallel processing
    print("\n" + "-" * 40)
    print("Processing trips in parallel...")

    trip_groups = list(df.groupby(trip_col))
    total_trips = len(trip_groups)
    print(f"Processing {total_trips} trips...")

    results = Parallel(
        n_jobs=-1,
        backend="loky",
        verbose=10,
        batch_size="auto",
    )(
        delayed(process_single_trip)(
            trip_id,
            trip_df,
            static_map_index,
            traffic_avg_index,
            h3_encoding,
            global_avg_speed_ratio,
            global_avg_traffic_speed,
            trip_col,
        )
        for trip_id, trip_df in trip_groups
    )

    # Filter None results
    print("\nCombining results...")
    final_dfs = [r for r in results if r is not None]

    if not final_dfs:
        print("No valid trips generated.")
        return

    combined_df = pd.concat(final_dfs, ignore_index=True)

    # Save encoding mappings
    print("\nSaving encoding mappings...")
    unique_routes = sorted(static_map["route_id"].unique())
    route_map = {str(route): idx for idx, route in enumerate(unique_routes)}

    with open(ENCODING_MAP_FILE, "w") as f:
        json.dump(route_map, f)

    h3_map_file = os.path.join(PARQUET_DIR, "h3_encoding.json")
    with open(h3_map_file, "w") as f:
        json.dump(h3_encoding, f)

    # Final filter: drop NaN h3_index
    combined_df = combined_df.dropna(subset=["h3_index"])

    # Validate 100 rows per trip
    trip_counts = combined_df[trip_col].value_counts()
    valid_trips = trip_counts[trip_counts == 100].index
    combined_df = combined_df[combined_df[trip_col].isin(valid_trips)]

    print(f"\nFinal dataset: {len(valid_trips)} trips, {len(combined_df)} rows.")

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    combined_df.to_parquet(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    process_data()
