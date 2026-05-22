#!/usr/bin/env python
"""Build the ML training dataset end-to-end.

This module owns the dataset preparation pipeline as one deep module with a
small public surface:

- build_dataset(...)
- build_canonical_shape_map(...)
- extract_historical_training_rows(...)
- preprocess_historical_rows(...)
- build_stop_level_dataset(...)
- scale_dataset(...)
- Filter
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h3
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import PchipInterpolator, interp1d
from sklearn.preprocessing import LabelEncoder

from application.runtime import ApplicationContext
from application.domain.artifacts import Artifact, DEFAULT_ARTIFACTS
from application.domain.static_data_fetcher import StaticDataFetcher
from persistence.gateway import create_persistence_gateway


PROJECT_ROOT = DEFAULT_ARTIFACTS.project_root
PARQUET_DIR = DEFAULT_ARTIFACTS.parquet_dir
CANONICAL_MAP_PATH = DEFAULT_ARTIFACTS.path(Artifact.CANONICAL_STOP_MAP)
STOP_ROUTE_CONFIG_PATH = DEFAULT_ARTIFACTS.path(Artifact.CANONICAL_STOP_CONFIG)
TRAFFIC_AVERAGES_PATH = DEFAULT_ARTIFACTS.path(Artifact.TRAFFIC_AVERAGES)
UNSCALED_DATASET_PATH = PARQUET_DIR / "dataset_lstm_unscaled.parquet"
FINAL_DATASET_PATH = PARQUET_DIR / "dataset_lstm_final.parquet"
ROUTE_ENCODER_PATH = PARQUET_DIR / "route_encoder.pkl"
ROUTE_ENCODING_PATH = PARQUET_DIR / "route_encoding.json"
H3_ENCODING_PATH = PARQUET_DIR / "h3_encoding.json"
GTFS_MD5_PATH = DEFAULT_ARTIFACTS.path(Artifact.GTFS_MD5)
CONFIG_CONTEXT = ApplicationContext()
PERSISTENCE = create_persistence_gateway(CONFIG_CONTEXT.config)

H3_RESOLUTION = 9
R_EARTH = 6371000

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


@dataclass
class Filter:
    """Configurable trip-level dataset filter."""

    outlier_sigma: float = 5.0
    max_trip_duration_sec: float = 7200.0
    min_schedule_coverage: float = 0.70
    min_progress_coverage: float = 0.20
    max_abs_schedule_adherence: float = 3600.0

    def filter(self, df: pd.DataFrame, trip_col: str = "trip_id") -> pd.DataFrame:
        """Return a filtered copy of ``df`` according to this filter state."""
        if df.empty:
            return df

        print("Filtering trips...")
        outlier_stats = self._compute_outlier_stats(df, trip_col)
        valid_trips = []
        total_trips = df[trip_col].nunique()
        dropped_outlier = 0
        dropped_duration = 0
        dropped_schedule = 0
        dropped_coverage = 0
        dropped_abs_schedule = 0

        for trip_id, group in df.groupby(trip_col):
            if self._has_outlier(group, outlier_stats):
                dropped_outlier += 1
                continue

            if self._exceeds_duration(group):
                dropped_duration += 1
                continue

            if self._has_low_schedule_coverage(group):
                dropped_schedule += 1
                continue

            if self._has_low_progress_coverage(group):
                dropped_coverage += 1
                continue

            if self._has_extreme_schedule_adherence(group):
                dropped_abs_schedule += 1
                continue

            valid_trips.append(trip_id)

        print(f"  Total trips: {total_trips}")
        print(f"  Dropped by outlier: {dropped_outlier}")
        print(f"  Dropped by duration: {dropped_duration}")
        print(f"  Dropped by schedule coverage: {dropped_schedule}")
        print(f"  Dropped by progress coverage: {dropped_coverage}")
        print(f"  Dropped by abs(schedule_adherence): {dropped_abs_schedule}")
        print(f"  Remaining: {len(valid_trips)}")

        return df[df[trip_col].isin(valid_trips)].copy()

    def _compute_outlier_stats(
        self,
        df: pd.DataFrame,
        trip_col: str,
    ) -> Dict[str, Tuple[float, float]]:
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
            outlier_stats[col] = (df[col].mean(), col_std)
        print(f"  Computed outlier stats for {len(outlier_stats)} features")
        return outlier_stats

    def _has_outlier(
        self,
        group: pd.DataFrame,
        outlier_stats: Dict[str, Tuple[float, float]],
    ) -> bool:
        for col, (mean, std) in outlier_stats.items():
            col_vals = group[col].dropna()
            if len(col_vals) and (np.abs(col_vals - mean) > self.outlier_sigma * std).any():
                return True
        return False

    def _exceeds_duration(self, group: pd.DataFrame) -> bool:
        if "stop_sequence" not in group.columns or "time_seconds_unrolled" not in group.columns:
            return False
        min_stop = group["stop_sequence"].min()
        max_stop = group["stop_sequence"].max()
        if pd.isna(min_stop) or pd.isna(max_stop):
            return False
        time_min = group.loc[group["stop_sequence"] == min_stop, "time_seconds_unrolled"]
        time_max = group.loc[group["stop_sequence"] == max_stop, "time_seconds_unrolled"]
        if len(time_min) == 0 or len(time_max) == 0:
            return False
        return (time_max.iloc[0] - time_min.iloc[0]) > self.max_trip_duration_sec

    def _has_low_schedule_coverage(self, group: pd.DataFrame) -> bool:
        if "schedule_adherence" not in group.columns:
            return False
        coverage = group["schedule_adherence"].notna().sum() / len(group)
        return coverage < self.min_schedule_coverage

    def _has_low_progress_coverage(self, group: pd.DataFrame) -> bool:
        if "progress" not in group.columns:
            return False
        coverage = group["progress"].max() - group["progress"].min()
        return coverage < self.min_progress_coverage

    def _has_extreme_schedule_adherence(self, group: pd.DataFrame) -> bool:
        if "schedule_adherence" not in group.columns:
            return False
        values = group["schedule_adherence"].dropna()
        if values.empty:
            return False
        return values.abs().max() > self.max_abs_schedule_adherence


@contextmanager
def working_directory(path: Path):
    """Temporarily switch cwd for GTFS download/extraction helpers."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def get_remote_gtfs_md5() -> str | None:
    """Fetch the remote GTFS MD5 hash."""
    import requests as rq

    headers = {
        "Referer": CONFIG_CONTEXT.config.urls.gtfs_referer,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    try:
        response = rq.get(CONFIG_CONTEXT.config.urls.gtfs_md5, headers=headers, timeout=15)
        return response.text.strip()
    except Exception as e:
        print(f"Warning: Could not fetch remote MD5: {e}")
        return None


def get_local_gtfs_md5() -> str | None:
    """Return the MD5 of the local GTFS zip file, if present."""
    gtfs_zip = PROJECT_ROOT / CONFIG_CONTEXT.config.paths.gtfs_static_zip
    if not gtfs_zip.exists():
        return None

    md5 = hashlib.md5()
    with gtfs_zip.open("rb") as f:
        while chunk := f.read(4096):
            md5.update(chunk)
    return md5.hexdigest()


def get_cached_md5() -> str | None:
    """Return the MD5 recorded during the last canonical-map build."""
    if not GTFS_MD5_PATH.exists():
        return None
    try:
        with GTFS_MD5_PATH.open("r", encoding="utf-8") as f:
            return json.load(f).get("md5")
    except Exception:
        return None


def save_cached_md5(md5: str):
    """Persist the GTFS MD5 used for the canonical map."""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    with GTFS_MD5_PATH.open("w", encoding="utf-8") as f:
        json.dump({"md5": md5}, f)


def needs_canonical_regeneration(force: bool = False) -> tuple[bool, str]:
    """Return whether the stop-route map must be regenerated, plus a reason."""
    if force:
        return True, "forced by user"
    if not CANONICAL_MAP_PATH.exists():
        return True, "canonical map does not exist"

    remote_md5 = get_remote_gtfs_md5()
    if remote_md5 is None:
        local_md5 = get_local_gtfs_md5()
        if local_md5 is None:
            return True, "no GTFS data available"
        return False, f"cannot check remote, using local GTFS (md5: {local_md5[:8]}...)"

    cached_md5 = get_cached_md5()
    if cached_md5 is None:
        return True, f"no cached MD5, remote has new version ({remote_md5[:8]}...)"
    if cached_md5 != remote_md5:
        return True, f"GTFS updated: cached {cached_md5[:8]}... vs remote {remote_md5[:8]}..."
    return False, f"canonical map is up to date (md5: {cached_md5[:8]}...)"


def get_h3_index(lat: float, lng: float, resolution: int = H3_RESOLUTION) -> str:
    """Return the H3 cell ID at ``resolution`` for a (lat, lng) pair."""
    return h3.latlng_to_cell(lat, lng, resolution)


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in meters between lat/lon arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * R_EARTH


def build_canonical_shape_map(
    trips_path: str | Path | None = None,
    stop_times_path: str | Path | None = None,
    stops_path: str | Path | None = None,
    output_path: str | Path = CANONICAL_MAP_PATH,
    config_output_path: str | Path = STOP_ROUTE_CONFIG_PATH,
    traffic_output_path: str | Path = TRAFFIC_AVERAGES_PATH,
    fetch_static: bool = True,
) -> pd.DataFrame:
    """Build the canonical stop-route map and traffic averages."""
    print("Starting Stop-Based Route Mapping...")

    trips_path = Path(trips_path or PROJECT_ROOT / "trips.txt")
    stop_times_path = Path(stop_times_path or PROJECT_ROOT / "stop_times.txt")
    stops_path = Path(stops_path or PROJECT_ROOT / "stops.txt")

    if fetch_static:
        try:
            with working_directory(PROJECT_ROOT):
                fetcher = StaticDataFetcher(
                    zip_path=str(PROJECT_ROOT / CONFIG_CONTEXT.config.paths.gtfs_static_zip),
                    static_url=CONFIG_CONTEXT.config.urls.gtfs_static,
                    md5_url=CONFIG_CONTEXT.config.urls.gtfs_md5,
                    referer=CONFIG_CONTEXT.config.urls.gtfs_referer,
                )
                fetcher.fetch()
        except Exception as e:
            print(f"Warning: StaticDataFetcher failed: {e}")
            print("Checking if files exist locally...")

    for path in [trips_path, stop_times_path, stops_path]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Ensure GTFS data is available.")

    stop_map = process_stop_route_map(
        trips_path=trips_path,
        stop_times_path=stop_times_path,
        stops_path=stops_path,
        output_path=output_path,
        config_output_path=config_output_path,
    )

    print("\n--- Computing Traffic Averages ---")
    compute_traffic_averages(output_path=traffic_output_path)

    local_md5 = get_local_gtfs_md5()
    if local_md5:
        save_cached_md5(local_md5)
        print(f"Saved GTFS MD5: {local_md5}")

    return stop_map


def process_stop_route_map(
    trips_path: str | Path,
    stop_times_path: str | Path,
    stops_path: str | Path,
    output_path: str | Path = CANONICAL_MAP_PATH,
    config_output_path: str | Path = STOP_ROUTE_CONFIG_PATH,
) -> pd.DataFrame:
    """Build a stop-based route map from GTFS trips, stop_times and stops."""
    trips_path = Path(trips_path)
    stop_times_path = Path(stop_times_path)
    stops_path = Path(stops_path)
    output_path = Path(output_path)
    config_output_path = Path(config_output_path)

    print(f"Loading GTFS data from {trips_path}, {stop_times_path}, {stops_path}...")
    trips = pd.read_csv(trips_path, dtype={"route_id": str}, low_memory=False)
    stop_times = pd.read_csv(stop_times_path, low_memory=False)
    stops = pd.read_csv(stops_path, low_memory=False)

    print("Phase A: Identifying canonical stop patterns...")
    trip_route = trips[["trip_id", "route_id", "direction_id"]].dropna(
        subset=["route_id", "direction_id"]
    )
    st = stop_times.merge(trip_route, on="trip_id", how="inner")

    stops_per_trip = (
        st.groupby(["route_id", "direction_id", "trip_id"])
        .size()
        .reset_index(name="n_stops")
    )
    most_common_count = (
        stops_per_trip.groupby(["route_id", "direction_id", "n_stops"])
        .size()
        .reset_index(name="freq")
        .sort_values(["route_id", "direction_id", "freq"], ascending=[True, True, False])
        .drop_duplicates(subset=["route_id", "direction_id"], keep="first")
    )
    canonical_trips = most_common_count.merge(
        stops_per_trip,
        on=["route_id", "direction_id", "n_stops"],
        how="inner",
    ).drop_duplicates(subset=["route_id", "direction_id"], keep="first")
    print(f"Identified {len(canonical_trips)} canonical route+direction pairs.")

    print("Phase B: Extracting stop sequences with coordinates...")
    canonical_trip_ids = set(canonical_trips["trip_id"].values)
    canonical_st = st[st["trip_id"].isin(canonical_trip_ids)].copy()
    canonical_st = canonical_st.merge(
        stops[["stop_id", "stop_lat", "stop_lon"]],
        on="stop_id",
        how="left",
    )

    before = len(canonical_st)
    canonical_st = canonical_st.dropna(subset=["stop_lat", "stop_lon"])
    if len(canonical_st) < before:
        print(f"  Dropped {before - len(canonical_st)} stops without coordinates.")

    canonical_st = canonical_st.sort_values(
        ["route_id", "direction_id", "trip_id", "stop_sequence"]
    )

    print("Phase C: Computing inter-stop distances and H3 indices...")
    results = []
    max_stops = 0
    for (route_id, direction_id, _trip_id), group in canonical_st.groupby(
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

        cum_dists = [0.0]
        for i in range(1, n_stops):
            distance = float(haversine_np(lats[i - 1], lons[i - 1], lats[i], lons[i]))
            cum_dists.append(cum_dists[-1] + distance)

        route_len = cum_dists[-1]
        if route_len == 0:
            continue

        max_stops = max(max_stops, n_stops)
        for i in range(n_stops):
            dist_to_next = cum_dists[i + 1] - cum_dists[i] if i < n_stops - 1 else 0.0
            results.append(
                {
                    "route_id": route_id,
                    "direction_id": int(direction_id),
                    "stop_idx": i,
                    "stop_id": str(stop_ids[i]),
                    "stop_sequence": int(stop_seqs[i]),
                    "h3_index": get_h3_index(lats[i], lons[i]),
                    "stop_lat": lats[i],
                    "stop_lon": lons[i],
                    "shape_dist_at_stop": cum_dists[i],
                    "distance_to_next_stop": dist_to_next,
                    "num_stops": n_stops,
                    "route_len_m": route_len,
                }
            )

    if not results:
        raise RuntimeError("No valid routes produced. Check GTFS data.")

    final_df = pd.DataFrame(results)
    n_routes = final_df.groupby(["route_id", "direction_id"]).ngroups
    stop_counts = final_df.drop_duplicates(subset=["route_id", "direction_id"])[
        "num_stops"
    ]
    print(f"\nBuilt stop map for {n_routes} route+direction pairs.")
    print(f"MAX_STOPS = {max_stops}")
    print("Stop count distribution:")
    print(
        f"  Min: {stop_counts.min()}, Median: {stop_counts.median():.0f}, "
        f"Max: {stop_counts.max()}, Mean: {stop_counts.mean():.1f}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(final_df)} rows to {output_path}...")
    final_df.to_parquet(output_path, index=False)

    with config_output_path.open("w", encoding="utf-8") as f:
        json.dump({"max_stops": max_stops}, f, indent=2)
    print(f"Saved stop route config to {config_output_path}")

    return final_df


def compute_traffic_averages(
    traffic_rows: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute per-H3/day/hour traffic averages from DB-returned rows."""
    if traffic_rows is None:
        print("Fetching traffic data from database...")
        traffic_rows = PERSISTENCE.read_historical_traffic_rows()

    if traffic_rows is None or traffic_rows.empty:
        print("No traffic data found.")
        return pd.DataFrame()

    df = traffic_rows.copy()
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
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        averages.to_parquet(output_path, index=False)
        print(f"Saved traffic averages to {output_path}")

    return averages


def get_processed_dates(parquet_dir: Path = PARQUET_DIR) -> set[str]:
    """Return dates already written as `dataset_YYYY-MM-DD.parquet` files."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    processed_dates = set()
    for file_path in parquet_dir.glob("dataset_*.parquet"):
        filename = file_path.name
        try:
            date_str = filename.replace("dataset_", "").replace(".parquet", "")
            datetime.strptime(date_str, "%Y-%m-%d")
            processed_dates.add(date_str)
        except ValueError:
            print(f"Warning: Skipping file with unexpected name format: {filename}")
    return processed_dates


def extract_historical_training_rows(start_date: str = None) -> pd.DataFrame:
    """Read historical measurement rows through the persistence facade."""
    print("Executing query and loading data (this may take a while)...")
    df = PERSISTENCE.read_historical_training_rows(start_date=start_date)
    if df is None:
        raise RuntimeError("Ledger database not configured. Check config.ini.")
    df = normalize_historical_training_rows(df)
    print(f"Loaded {len(df)} rows from database.")
    return df


def normalize_historical_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Format historical ledger rows for dataset preprocessing."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if "ts" not in df.columns and "measurement_time" in df.columns:
        df["ts"] = df["measurement_time"]
    df["ts"] = pd.to_datetime(df["ts"])
    seconds = (
        df["ts"].dt.hour * 3600
        + df["ts"].dt.minute * 60
        + df["ts"].dt.second
    )
    df["time_seconds"] = seconds.astype(int)
    df["day_type"] = df["ts"].dt.weekday.map(
        lambda weekday: 1 if weekday == 5 else 2 if weekday == 6 else 0
    )
    df["rush_hour_status"] = (
        ((df["time_seconds"] >= 25200) & (df["time_seconds"] <= 32400))
        | ((df["time_seconds"] >= 61200) & (df["time_seconds"] <= 70200))
    ).astype(int)
    if "distance_to_next_stop" in df.columns:
        df["far_status"] = (df["distance_to_next_stop"].fillna(0) > 250).astype(int)
    else:
        df["far_status"] = 0
    df["served_ratio"] = df.get("served_ratio", 1.0)

    for column in [
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
    ]:
        if column not in df.columns:
            df[column] = 0

    return df


def generate_synthetic_trip_id(df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic trip IDs from route/direction/time/stop reset breaks."""
    print("Generating Synthetic Trip IDs...")
    df = df.copy()
    df.sort_values(by=["route_id", "direction_id", "ts"], inplace=True)

    prev_route = df["route_id"].shift(1)
    prev_dir = df["direction_id"].shift(1)
    prev_seq = df["stop_sequence"].shift(1)
    prev_time = df["time_seconds"].shift(1)

    route_change = df["route_id"] != prev_route
    dir_change = df["direction_id"] != prev_dir
    seq_reset = df["stop_sequence"] < prev_seq
    large_time_gap = (df["time_seconds"] - prev_time) > 600
    time_backward = (df["time_seconds"] < prev_time) & (
        (prev_time - df["time_seconds"]) < 40000
    )

    df["trip_id"] = (
        route_change | dir_change | seq_reset | large_time_gap | time_backward
    ).cumsum()
    print(f"Generated {df['trip_id'].max()} unique trips.")
    return df


def preprocess_historical_rows(
    df: pd.DataFrame,
    start_date: str = None,
    write_daily_parquets: bool = True,
    parquet_dir: Path = PARQUET_DIR,
) -> pd.DataFrame:
    """Normalize DB rows into the dynamic dataframe consumed by stop-level processing."""
    if df is None or df.empty:
        print("No data found in database.")
        return pd.DataFrame()

    df = df.copy()
    if start_date and "ts" in df.columns:
        print(f"Filtering data from {start_date} onwards...")
        print(f"Data range: {df['ts'].min()} to {df['ts'].max()}")

    df["ts"] = pd.to_datetime(df["ts"])
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)

    if "trip_id" in df.columns:
        has_trip_id = df["trip_id"].notna()
        if has_trip_id.any():
            print(f"Found {has_trip_id.sum()} rows with existing trip_id.")
            df["trip_id"] = df["trip_id"].astype(object)
            df.loc[has_trip_id, "trip_id"] = df.loc[has_trip_id, "trip_id"].astype(str)

            if (~has_trip_id).any():
                print(f"Generating synthetic trip_id for {(~has_trip_id).sum()} rows without it...")
                missing_df = generate_synthetic_trip_id(df[~has_trip_id].copy())
                df.loc[~has_trip_id, "trip_id"] = missing_df["trip_id"].apply(
                    lambda value: f"synthetic_{value}"
                )
        else:
            df = generate_synthetic_trip_id(df)
    else:
        df = generate_synthetic_trip_id(df)

    if write_daily_parquets:
        write_daily_dynamic_parquets(df, parquet_dir=parquet_dir)
    return df

def write_daily_dynamic_parquets(
    df: pd.DataFrame,
    parquet_dir: Path = PARQUET_DIR,
):
    """Write normalized dynamic rows to daily cache parquet files."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    processed_dates = get_processed_dates(parquet_dir)
    print(f"Found {len(processed_dates)} already processed days: {sorted(processed_dates)}")

    df = df.copy()
    df["date_str"] = df["ts"].dt.date.astype(str)
    unique_days = df["date_str"].unique()
    print(f"Processing data for {len(unique_days)} unique days found in dataset...")

    for day in unique_days:
        if day in processed_dates:
            print(f"Skipping {day} (already processed).")
            continue
        print(f"Saving data for {day}...")
        day_df = df[df["date_str"] == day].copy()
        day_df.drop(columns=["date_str"], inplace=True)
        output_path = parquet_dir / f"dataset_{day}.parquet"
        day_df.to_parquet(output_path, engine="pyarrow", index=False)
        print(f"Saved {output_path}")


def load_dynamic_data(start_date: str = None, parquet_dir: Path = PARQUET_DIR) -> pd.DataFrame:
    """Load cached dynamic daily parquet files."""
    files = [
        f
        for f in glob.glob(str(parquet_dir / "dataset_*.parquet"))
        if "dataset_lstm_unscaled" not in f and "dataset_lstm_final" not in f
    ]

    if not files:
        print("No dynamic dataset files found in ./parquets/")
        return pd.DataFrame()

    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        filtered_files = []
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                date_str = filename.replace("dataset_", "").replace(".parquet", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if file_date >= start_dt:
                    filtered_files.append(file_path)
            except ValueError:
                continue
        files = filtered_files
        print(f"Filtering to {len(files)} files from {start_date} onwards")
    else:
        print(f"Loading {len(files)} dynamic dataset files...")

    dfs = [pd.read_parquet(file_path) for file_path in sorted(files)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_static_map() -> pd.DataFrame:
    """Load the canonical stop-route map."""
    if not CANONICAL_MAP_PATH.exists():
        raise FileNotFoundError(f"Static map file '{CANONICAL_MAP_PATH}' not found.")
    return pd.read_parquet(CANONICAL_MAP_PATH)


def load_traffic_avg() -> pd.DataFrame:
    """Load precomputed per-hex traffic averages, if present."""
    if not TRAFFIC_AVERAGES_PATH.exists():
        print(f"Warning: Traffic averages file '{TRAFFIC_AVERAGES_PATH}' not found.")
        return pd.DataFrame()
    return pd.read_parquet(TRAFFIC_AVERAGES_PATH)


def build_static_map_index(static_map: pd.DataFrame) -> Dict[Tuple[str, int], pd.DataFrame]:
    """Pre-index static map by (route_id, direction_id)."""
    print("Building static map index...")
    index = {}
    for (route_id, direction_id), group in static_map.groupby(["route_id", "direction_id"]):
        index[(str(route_id), int(direction_id))] = group.sort_values("stop_idx").reset_index(drop=True)
    print(f"  Indexed {len(index)} route+direction combinations")
    return index


def build_traffic_avg_index(
    traffic_avg: pd.DataFrame,
) -> Dict[Tuple[str, int, int], Tuple[float, float]]:
    """Pre-index traffic averages by (h3_index, day_type, hour)."""
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
    """Build h3_index -> integer encoding."""
    unique_h3 = sorted(static_map["h3_index"].dropna().unique())
    return {str(h3_index): i for i, h3_index in enumerate(unique_h3)}


def unroll_time(df: pd.DataFrame, trip_col: str = "trip_id") -> pd.DataFrame:
    """Unroll time to handle midnight crossings within a trip."""
    df = df.sort_values(by=[trip_col, "ts"]).copy()
    df["time_diff"] = df.groupby(trip_col)["time_seconds"].diff()
    df["day_cycle"] = df.groupby(trip_col)["time_diff"].transform(
        lambda x: (x < -40000).cumsum()
    )
    df["time_seconds_unrolled"] = df["time_seconds"] + (df["day_cycle"] * 86400)
    return df


def interpolate_schedule_adherence(df: pd.DataFrame, trip_col: str) -> pd.DataFrame:
    """Interpolate missing schedule_adherence values per trip."""
    if "schedule_adherence" not in df.columns:
        return df

    print("Interpolating missing schedule_adherence values...")
    df = df.sort_values([trip_col, "ts"])
    trip_col_values = df[trip_col].values.copy()

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
            group["schedule_adherence"] = group["schedule_adherence"].fillna(valid_vals[0])
        return group

    df = df.groupby(trip_col, group_keys=False).apply(interp_group)
    if trip_col not in df.columns:
        df[trip_col] = trip_col_values[: len(df)]
    return df


def prepare_dynamic_dataframe(
    df: pd.DataFrame,
    static_map: pd.DataFrame,
    dataset_filter: Filter,
) -> tuple[pd.DataFrame, str]:
    """Prepare dynamic rows for stop-level interpolation."""
    if df.empty:
        return df, "trip_id"

    df = df.copy()
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)[:10]}...")
    if "time_feat" in df.columns:
        print("Dropping deprecated 'time_feat' column.")
        df.drop(columns=["time_feat"], inplace=True)

    if "trip_id_synthetic" in df.columns:
        trip_col = "trip_id_synthetic"
    elif "trip_id" in df.columns:
        trip_col = "trip_id"
    else:
        raise ValueError(f"No trip_id or trip_id_synthetic column found. Columns: {list(df.columns)}")
    print(f"Using trip column: {trip_col}")

    print("\nUnrolling time...")
    df = unroll_time(df, trip_col)

    if "distance_to_next_stop" in df.columns:
        df["distance_to_next_stop"] = df["distance_to_next_stop"].fillna(-1000)

    print("\nAttaching route length...")
    lookup = static_map[["route_id", "direction_id", "route_len_m"]].drop_duplicates()
    df = df.merge(lookup, on=["route_id", "direction_id"], how="left")
    df = df.dropna(subset=["route_len_m"])
    df["progress"] = (df["shape_dist_travelled"] / df["route_len_m"]).clip(0.0, 1.0)

    print("\n" + "-" * 40)
    df = dataset_filter.filter(df, trip_col=trip_col)
    if df.empty:
        return df, trip_col

    df = interpolate_schedule_adherence(df, trip_col)
    if "stop_sequence" in df.columns:
        df.drop(columns=["stop_sequence"], inplace=True)
    return df, trip_col


def build_stop_level_dataset(
    dynamic_df: pd.DataFrame | None = None,
    start_date: str = None,
    dataset_filter: Filter | None = None,
    output_path: str | Path = UNSCALED_DATASET_PATH,
    save: bool = True,
) -> pd.DataFrame:
    """Transform dynamic rows into the padded stop-level unscaled dataset."""
    print("=" * 60)
    print("STOP-LEVEL DATASET BUILD")
    print("=" * 60)

    if not STOP_ROUTE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Stop route config not found at {STOP_ROUTE_CONFIG_PATH}")
    with STOP_ROUTE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        stop_config = json.load(f)
    max_stops = stop_config["max_stops"]
    print(f"MAX_STOPS = {max_stops}")

    print("\nLoading static inputs...")
    static_map = load_static_map()
    traffic_avg = load_traffic_avg()
    if dynamic_df is None:
        dynamic_df = load_dynamic_data(start_date=start_date)
    if dynamic_df.empty:
        print("No dynamic data to process.")
        return pd.DataFrame()

    print("\nBuilding lookup indexes...")
    static_map_index = build_static_map_index(static_map)
    traffic_avg_index = build_traffic_avg_index(traffic_avg)
    h3_encoding = build_h3_encoding(static_map)
    global_avg_speed_ratio = traffic_avg["avg_speed_ratio"].mean() if not traffic_avg.empty else 0.0
    global_avg_traffic_speed = traffic_avg["avg_current_traffic_speed"].mean() if not traffic_avg.empty else 0.0

    dataset_filter = dataset_filter or Filter()
    df, trip_col = prepare_dynamic_dataframe(dynamic_df, static_map, dataset_filter)
    if df.empty:
        print("No trips remaining after filtering.")
        return pd.DataFrame()

    print("\n" + "-" * 40)
    print("Processing trips...")
    trip_groups = list(df.groupby(trip_col))
    print(f"Processing {len(trip_groups)} trips...")

    try:
        results = Parallel(n_jobs=1, verbose=10)(
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
    except Exception as e:
        import traceback

        print(f"\nError during trip processing: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    print("\nCombining results...")
    final_dfs = [result for result in results if result is not None]
    if not final_dfs:
        print("No valid trips generated.")
        return pd.DataFrame()

    combined_df = pd.concat(final_dfs, ignore_index=True)
    combined_df = finalize_stop_level_dataset(
        combined_df,
        static_map=static_map,
        h3_encoding=h3_encoding,
        trip_col=trip_col,
        max_stops=max_stops,
    )

    print(
        f"\nFinal dataset: {combined_df[trip_col].nunique()} trips, "
        f"{len(combined_df)} rows (padded to {max_stops} stops each)."
    )

    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path}...")
        combined_df.to_parquet(output_path, index=False)
        print("Done.")

    return combined_df


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
    """Interpolate one raw trip onto its canonical stop sequence."""
    try:
        route_id = str(trip_df["route_id"].iloc[0])
        direction_id = int(trip_df["direction_id"].iloc[0])
        day_type = int(trip_df["day_type"].iloc[0]) if "day_type" in trip_df.columns else 0
    except KeyError as e:
        print(f"KeyError in trip {trip_id}: {e}")
        print(f"Available columns: {list(trip_df.columns)}")
        raise

    trip_static = static_map_index.get((route_id, direction_id))
    if trip_static is None or trip_static.empty:
        return None
    if "shape_dist_travelled" not in trip_df.columns:
        return None

    num_stops = int(trip_static["num_stops"].iloc[0])
    stop_dists = trip_static["shape_dist_at_stop"].values
    trip_df = trip_df.sort_values("shape_dist_travelled").drop_duplicates(
        subset=["shape_dist_travelled"],
        keep="last",
    )
    if len(trip_df) < 2:
        return None

    x = trip_df["shape_dist_travelled"].values
    x_new = stop_dists
    result = {"stop_idx": np.arange(num_stops)}

    stop_has_measurement = np.zeros(num_stops, dtype=bool)
    for shape_dist in x:
        idx = np.searchsorted(stop_dists, shape_dist)
        if idx == 0:
            stop_i = 0
        elif idx >= num_stops:
            stop_i = num_stops - 1
        else:
            dist_left = abs(shape_dist - stop_dists[idx - 1])
            dist_right = abs(shape_dist - stop_dists[idx])
            stop_i = idx - 1 if dist_left < dist_right else idx
        stop_has_measurement[stop_i] = True
    result["is_genuine"] = stop_has_measurement.astype(int)

    interpolate_dynamic_features(trip_df, x, x_new, result, num_stops)

    if "time_seconds" in result:
        day_seconds = result["time_seconds"] % 86400
        is_morning = (day_seconds >= 25200) & (day_seconds <= 32400)
        is_evening = (day_seconds >= 61200) & (day_seconds <= 70200)
        result["rush_hour_status"] = (is_morning | is_evening).astype(int)

    result["far_status"] = (
        (trip_static["distance_to_next_stop"].values > 250).astype(int)
        if "distance_to_next_stop" in trip_static.columns
        else np.ones(num_stops, dtype=int)
    )

    h3_values = trip_static["h3_index"].values
    result["h3_index"] = h3_values
    result["h3_index_encoded"] = np.array(
        [h3_encoding.get(str(h3_value), -1) for h3_value in h3_values],
        dtype=int,
    )
    result["stop_sequence"] = trip_static["stop_sequence"].values.astype(int)
    result["can_shape_dist_travelled"] = trip_static["shape_dist_at_stop"].values
    result["can_distance_to_next_stop"] = trip_static["distance_to_next_stop"].values
    result["route_id"] = route_id
    result["num_stops"] = num_stops

    if "time_seconds" in result:
        day_seconds = result["time_seconds"] % 86400
        result["time_sin"] = np.sin(2 * np.pi * day_seconds / 86400)
        result["time_cos"] = np.cos(2 * np.pi * day_seconds / 86400)

    attach_traffic_averages(
        result,
        h3_values=h3_values,
        day_type=day_type,
        num_stops=num_stops,
        traffic_avg_index=traffic_avg_index,
        global_avg_speed_ratio=global_avg_speed_ratio,
        global_avg_traffic_speed=global_avg_traffic_speed,
    )

    result.update(compute_static_features(trip_df, num_stops))

    if "time_seconds" in result:
        if result["time_seconds"].min() < 0 or result["time_seconds"].max() > 259200:
            return None
    if "schedule_adherence" in result:
        if np.abs(result["schedule_adherence"]).max() > 3600:
            return None

    df_result = pd.DataFrame(result)
    df_result[trip_col] = trip_id
    return df_result


def interpolate_dynamic_features(
    trip_df: pd.DataFrame,
    x: np.ndarray,
    x_new: np.ndarray,
    result: dict,
    num_stops: int,
):
    """Interpolate dynamic feature columns onto stop positions."""
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
                result[feat] = np.full(num_stops, y[0])
        elif feat == "occupancy_status":
            f_zoh = interp1d(
                x,
                y,
                kind="nearest",
                bounds_error=False,
                fill_value=(y[0], y[-1]),
            )
            y_new = np.clip(f_zoh(x_new), 0, 7)
            result[feat] = np.round(y_new).astype(int)
            result["occupancy_status_available"] = ((y_new >= 0) & (y_new <= 6)).astype(int)
        elif feat in ["far_status", "rush_hour_status"]:
            continue
        else:
            try:
                interp = PchipInterpolator(x, y, extrapolate=True)
                result[feat] = interp(x_new)
            except Exception:
                f = interp1d(x, y, kind="linear", fill_value="extrapolate")
                result[feat] = f(x_new)


def attach_traffic_averages(
    result: dict,
    h3_values: np.ndarray,
    day_type: int,
    num_stops: int,
    traffic_avg_index: Dict[Tuple[str, int, int], Tuple[float, float]],
    global_avg_speed_ratio: float,
    global_avg_traffic_speed: float,
):
    """Attach traffic averages to the per-stop result dict."""
    if not traffic_avg_index or "time_seconds" not in result:
        result["can_avg_speed_ratio"] = [0.0] * num_stops
        result["can_avg_traffic_speed"] = [0.0] * num_stops
        return

    hours = (result["time_seconds"] % 86400).astype(int) // 3600
    avg_speed_ratios = []
    avg_traffic_speeds = []
    for i in range(num_stops):
        key = (str(h3_values[i]), day_type, hours[i])
        if key in traffic_avg_index:
            speed_ratio, traffic_speed = traffic_avg_index[key]
            avg_speed_ratios.append(speed_ratio)
            avg_traffic_speeds.append(traffic_speed)
        else:
            avg_speed_ratios.append(global_avg_speed_ratio)
            avg_traffic_speeds.append(global_avg_traffic_speed)
    result["can_avg_speed_ratio"] = avg_speed_ratios
    result["can_avg_traffic_speed"] = avg_traffic_speeds


def compute_static_features(trip_df: pd.DataFrame, num_stops: int) -> Dict[str, Any]:
    """Compute trip-level static features and replicate them per stop."""
    first_row = trip_df.iloc[0]
    result = {}
    for feat in STATIC_TRIP_FEATURES:
        if feat in trip_df.columns:
            result[feat] = [first_row[feat]] * num_stops

    start_time_cols = [
        "starting_time_sin",
        "starting_time_cos",
        "sch_starting_time_sin",
        "sch_starting_time_cos",
    ]
    if all(col in trip_df.columns for col in start_time_cols):
        valid_mask = (
            trip_df["starting_time_sin"].notna()
            & trip_df["starting_time_cos"].notna()
            & trip_df["sch_starting_time_sin"].notna()
            & trip_df["sch_starting_time_cos"].notna()
        )
        if valid_mask.any():
            row = trip_df[valid_mask].iloc[0]
            result["actual_start_time_sin"] = [row["starting_time_sin"]] * num_stops
            result["actual_start_time_cos"] = [row["starting_time_cos"]] * num_stops
            result["scheduled_start_time_sin"] = [row["sch_starting_time_sin"]] * num_stops
            result["scheduled_start_time_cos"] = [row["sch_starting_time_cos"]] * num_stops
            return result

    if "distance_to_next_stop" in trip_df.columns and "time_seconds_unrolled" in trip_df.columns:
        trip_sorted = trip_df.sort_values("ts")
        dist_to_next = trip_sorted["distance_to_next_stop"].values
        time_unrolled = trip_sorted["time_seconds_unrolled"].values

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
                scheduled_start = actual_start - sched_adh if not pd.isna(sched_adh) else actual_start
            else:
                scheduled_start = actual_start
        else:
            actual_start = time_unrolled[0] if len(time_unrolled) > 0 else 0
            if "schedule_adherence" in trip_sorted.columns and len(trip_sorted) > 0:
                sched_adh = trip_sorted["schedule_adherence"].iloc[0]
                scheduled_start = actual_start - sched_adh if not pd.isna(sched_adh) else actual_start
            else:
                scheduled_start = actual_start

        actual_sec = actual_start % 86400
        scheduled_sec = scheduled_start % 86400
        result["actual_start_time_sin"] = [np.sin(2 * np.pi * actual_sec / 86400)] * num_stops
        result["actual_start_time_cos"] = [np.cos(2 * np.pi * actual_sec / 86400)] * num_stops
        result["scheduled_start_time_sin"] = [np.sin(2 * np.pi * scheduled_sec / 86400)] * num_stops
        result["scheduled_start_time_cos"] = [np.cos(2 * np.pi * scheduled_sec / 86400)] * num_stops

    return result


def finalize_stop_level_dataset(
    combined_df: pd.DataFrame,
    static_map: pd.DataFrame,
    h3_encoding: Dict[str, int],
    trip_col: str,
    max_stops: int,
) -> pd.DataFrame:
    """Save encodings, validate stop counts, pad trips and compute t_grid."""
    print("\nSaving encoding mappings...")
    unique_routes = sorted(static_map["route_id"].unique())
    route_map = {str(route): idx for idx, route in enumerate(unique_routes)}
    with ROUTE_ENCODING_PATH.open("w", encoding="utf-8") as f:
        json.dump(route_map, f)
    with H3_ENCODING_PATH.open("w", encoding="utf-8") as f:
        json.dump(h3_encoding, f)

    combined_df = combined_df.dropna(subset=["h3_index"])
    trip_counts = combined_df.groupby(trip_col).agg(
        actual_rows=("stop_idx", "count"),
        expected_rows=("num_stops", "first"),
    )
    valid_trips = trip_counts[trip_counts["actual_rows"] == trip_counts["expected_rows"]].index
    combined_df = combined_df[combined_df[trip_col].isin(valid_trips)]
    print(f"\nValid trips (correct stop count): {len(valid_trips)}")

    print(f"Padding trips to MAX_STOPS={max_stops}...")
    padded_dfs = []
    for trip_id, group in combined_df.groupby(trip_col):
        n_rows = len(group)
        if n_rows < max_stops:
            pad_rows = max_stops - n_rows
            pad_df = pd.DataFrame(0, index=range(pad_rows), columns=group.columns)
            pad_df[trip_col] = trip_id
            pad_df["stop_idx"] = range(n_rows, max_stops)
            pad_df["num_stops"] = group["num_stops"].iloc[0]
            pad_df["route_id"] = group["route_id"].iloc[0]
            pad_df["direction_id"] = group["direction_id"].iloc[0]
            pad_df["is_genuine"] = 0
            pad_df["h3_index"] = ""
            pad_df["h3_index_encoded"] = -1
            pad_df["stop_sequence"] = -1
            padded_dfs.append(pd.concat([group, pad_df], ignore_index=True))
        else:
            padded_dfs.append(group)

    if not padded_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(padded_dfs, ignore_index=True)
    if trip_col != "trip_id" and "trip_id" not in combined_df.columns:
        combined_df["trip_id"] = combined_df[trip_col]

    print("Computing t_grid...")
    t_grid_values = []
    for _, group in combined_df.groupby(trip_col):
        n_stops = int(group["num_stops"].iloc[0])
        dists = group["can_shape_dist_travelled"].values
        route_len = dists[:n_stops].max() if n_stops > 0 and dists[:n_stops].max() > 0 else 1.0
        t_grid = dists / route_len
        t_grid[n_stops:] = 0.0
        t_grid_values.extend(t_grid.tolist())
    combined_df["t_grid"] = t_grid_values
    return combined_df


def scale_dataset(
    df: pd.DataFrame | None = None,
    input_parquet: str | Path = UNSCALED_DATASET_PATH,
    output_parquet: str | Path = FINAL_DATASET_PATH,
    encoder_path: str | Path = ROUTE_ENCODER_PATH,
    save: bool = True,
) -> pd.DataFrame:
    """Apply physical scaling and fit the route LabelEncoder."""
    print("Reading Dataset...")
    if df is None:
        df = pd.read_parquet(input_parquet)
    else:
        df = df.copy()

    with STOP_ROUTE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        max_stops = json.load(f)["max_stops"]
    print(f"MAX_STOPS = {max_stops}")

    print("Individuazione anomalie nei ritardi...")
    anomalous_trips = df[
        (df["schedule_adherence"] > 3600) | (df["schedule_adherence"] < -3600)
    ]["trip_id"].unique()
    print(f"Rimozione di {len(anomalous_trips)} viaggi interi per guasti/anomalie...")
    df = df[~df["trip_id"].isin(anomalous_trips)].copy()
    df.reset_index(drop=True, inplace=True)

    print("Encoding categorical strings...")
    route_encoder = LabelEncoder()
    df["route_id"] = route_encoder.fit_transform(df["route_id"])

    print("Applicazione dello scaling logico...")
    df["door_number"] = df["door_number"].clip(1, 3) / 3.0
    df["stop_idx"] = df["stop_idx"] / max(max_stops - 1, 1)
    df["bus_type"] = df["bus_type"] / 9.0
    df["weather_code"] = df["weather_code"] / 33
    df["can_shape_dist_travelled"] = df["can_shape_dist_travelled"].clip(0, 15000) / 15000.0
    df["can_distance_to_next_stop"] = df["can_distance_to_next_stop"].clip(0, 1000) / 1000.0
    df["can_avg_traffic_speed"] = df["can_avg_traffic_speed"].clip(0, 65) / 65.0
    df["schedule_adherence"] = df["schedule_adherence"] / 600.0

    if save:
        print("Salvataggio dell'Encoder della Linea...")
        joblib.dump({"route": route_encoder}, encoder_path)
        print("Salvataggio dataset pulito e scalato...")
        df.to_parquet(output_parquet)
        print("Fatto.")

    return df


def descaling(value: float, scaler_path: str = None):
    """Decode scaled schedule adherence back to seconds."""
    return value * 3600.0


def build_dataset(
    skip_db: bool = False,
    force_canonical: bool = False,
    start_date: str = None,
    dataset_filter: Filter | None = None,
) -> pd.DataFrame:
    """Build the complete final training dataset."""
    print("=" * 60)
    print("DATASET PREPARATION PIPELINE")
    print("=" * 60)

    if start_date:
        print(f"\nStart date filter: {start_date}")
        existing_files = list(PARQUET_DIR.glob("dataset_*.parquet")) if PARQUET_DIR.exists() else []
        if existing_files and not skip_db:
            print(f"Warning: {len(existing_files)} existing parquet files found.")
            print("Only NEW dates (not already processed) will be written to daily cache.")

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    needs_regen, reason = needs_canonical_regeneration(force=force_canonical)
    print(f"\nCanonical map check: {reason}")
    if needs_regen:
        print("\n" + "=" * 60)
        print("STAGE 1: Canonical Shape Map")
        print("=" * 60)
        build_canonical_shape_map()
    else:
        print("\nSkipping Stage 1: Canonical map is up to date")

    if skip_db:
        print("\nSkipping DB extraction: using existing dataset_*.parquet files")
        dynamic_df = load_dynamic_data(start_date=start_date)
    else:
        print("\n" + "=" * 60)
        print("STAGE 2: Historical Measurement Extraction + Preprocessing")
        print("=" * 60)
        raw_rows = extract_historical_training_rows(start_date=start_date)
        dynamic_df = preprocess_historical_rows(raw_rows, start_date=start_date)

    print("\n" + "=" * 60)
    print("STAGE 3: Stop-Level Dataset")
    print("=" * 60)
    unscaled_df = build_stop_level_dataset(
        dynamic_df=dynamic_df,
        start_date=start_date,
        dataset_filter=dataset_filter or Filter(),
        save=True,
    )
    if unscaled_df.empty:
        raise RuntimeError("No unscaled dataset rows produced.")

    print("\n" + "=" * 60)
    print("STAGE 4: Scaling")
    print("=" * 60)
    final_df = scale_dataset(unscaled_df, save=True)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  - {CANONICAL_MAP_PATH}")
    print(f"  - {FINAL_DATASET_PATH}")
    print(f"  - {ROUTE_ENCODER_PATH}")
    print(f"  - {ROUTE_ENCODING_PATH}")
    print(f"  - {H3_ENCODING_PATH}")
    return final_df


def extract_historical(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Extract stop-level historical arrival data from the ledger database."""
    date_start = datetime.strptime(start_date, "%Y-%m-%d").timestamp() if start_date else None
    date_end = datetime.strptime(end_date, "%Y-%m-%d").timestamp() if end_date else None
    df = PERSISTENCE.read_historical_records(
        CONFIG_CONTEXT.config.ledger.db_connection,
        CONFIG_CONTEXT.config.ledger.historical_table,
        date_start=date_start,
        date_end=date_end,
    )
    print(f"Loaded {len(df)} historical arrival records from ledger.")
    return df


def extract_vehicle_trips(
    start_date: str = None,
    end_date: str = None,
    route_id: str = None,
) -> pd.DataFrame:
    """Extract vehicle-level trip performance data from the ledger database."""
    df = PERSISTENCE.read_vehicle_trip_records(
        CONFIG_CONTEXT.config.ledger.db_connection,
        CONFIG_CONTEXT.config.ledger.vehicle_table,
        route_id=route_id,
        date_start=start_date,
        date_end=end_date,
    )
    print(f"Loaded {len(df)} vehicle trip records from ledger.")
    return df


def main():
    """CLI entry point for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages:
  1. Build canonical stop-route map from GTFS
  2. Extract and preprocess historical ledger rows
  3. Build stop-level unscaled dataset
  4. Scale final training dataset
        """,
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip DB extraction (use existing dataset_*.parquet files)",
    )
    parser.add_argument(
        "--force-canonical",
        action="store_true",
        help="Force regeneration of canonical route map",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Only process data from this date onwards (YYYY-MM-DD format)",
    )
    args = parser.parse_args()
    build_dataset(
        skip_db=args.skip_db,
        force_canonical=args.force_canonical,
        start_date=args.start_date,
    )


if __name__ == "__main__":
    main()
