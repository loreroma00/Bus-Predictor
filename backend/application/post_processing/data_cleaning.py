"""
Data Cleaning Pipeline for post-processing measurements.

Provides functions to:
- Remove duplicate measurements
- Filter outliers (random bus appearances)
- Validate runs (terminus checks, measurement counts)
- Vectorize cleaned measurements
"""

import logging
import pandas as pd
import csv
import bisect
import os
from datetime import datetime
from typing import TYPE_CHECKING, Union, overload
from types import SimpleNamespace

from ..domain.live_data import GPSData
from ..domain.weather import Weather
from ..domain.time_utils import to_unix_time, get_timestamp_components
from application.domain.interfaces import Pipeline
from . import vectorization

if TYPE_CHECKING:
    from ..domain.observers import Measurement, Diary

logger = logging.getLogger(__name__)


class PredictionPipeline(Pipeline):
    """Pipeline for data cleaning and vectorization."""

    def __init__(
        self,
        diary: "Diary",
        topology=None,
        served_ratio: float = 1.0,
        config: dict = None,
    ):
        """
        Initialize pipeline with diary and static data.

        Args:
            diary: The Diary object containing measurements
            topology: TopologyLedger with routes, trips, shapes
            served_ratio: Ratio from scan_trip_adherence
            config: Configuration dictionary
        """
        self.diary = diary
        self.topology = topology
        self.served_ratio = served_ratio

        # Load cleaning parameters from config or defaults
        cleaning_cfg = config.get("data_cleaning", {}) if config else {}
        self.min_measurements = int(cleaning_cfg.get("min_measurements_per_run", 7))
        self.max_bus_speed = int(cleaning_cfg.get("max_bus_speed_kmh", 120))
        self.max_time_gap = int(cleaning_cfg.get("max_time_gap_seconds", 600))
        self.uptime_lookback_seconds = (
            int(cleaning_cfg.get("served_ratio_lookback_minutes", 60)) * 60
        )

        self.uptime_timestamps = _load_uptime_data("uptime.csv")

    def clean(
        self,
    ) -> list[
        tuple[vectorization.PredictionVector, vectorization.PredictionLabel, float]
    ]:
        """
        Clean diary measurements and vectorize them.

        Returns:
            List of (Vector, Label, measurement_time) tuples for DB insert.
        """
        if not self.diary or not self.diary.measurements:
            return []

        logger.info(
            f"🚀 Starting pipeline for diary {self.diary.trip_id} "
            f"with {len(self.diary.measurements)} measurements"
        )

        # Step -1: Validate Uptime Coverage for Served Ratio (Configurable lookback)
        # If served_ratio < 1.0, we must ensure the "missed" trips weren't due to downtime.
        if self.served_ratio < 1.0 and self.uptime_timestamps:
            first_ts = self.diary.measurements[0].measurement_time
            if not _is_uptime_sufficient(
                first_ts,
                self.uptime_timestamps,
                window_seconds=self.uptime_lookback_seconds,
            ):
                logger.warning(
                    f"⚠️ Discarding diary {self.diary.trip_id}: served_ratio {self.served_ratio:.2f} "
                    f"but insufficient uptime coverage during the lookback window ({self.uptime_lookback_seconds}s)."
                )
                return []

        # Build measurements dict from diary
        trip_key = str(self.diary.trip_id)
        measurements = {trip_key: self.diary.measurements}
        initial_count = len(self.diary.measurements)

        # Step 0: Data Integrity (Zeros & Uptime)
        cleaned = _check_data_integrity(
            measurements, self.uptime_timestamps, self.served_ratio
        )
        if not cleaned or len(cleaned.get(trip_key, [])) < initial_count:
            logger.info(
                f"🚫 Trip {trip_key} discarded: Integrity check failed (faulty measurements)"
            )
            return []

        # Step 1: Remove duplicates
        cleaned = _check_for_duplicates(cleaned)

        # Step 2: Remove outliers
        # Update count after duplicates
        post_dup_count = len(cleaned.get(trip_key, []))
        cleaned = _remove_outliers(cleaned, self.max_bus_speed, self.max_time_gap)
        if not cleaned or len(cleaned.get(trip_key, [])) < post_dup_count:
            logger.info(
                f"🚫 Trip {trip_key} discarded: Outliers detected"
            )
            return []

        # Step 3: Check for valid runs
        cleaned = _check_for_runs(cleaned, self.topology, self.min_measurements)
        if not cleaned:
            return []

        # Step 4: Vectorize with served_ratio
        vectors = _vectorize(cleaned, self.topology, self.served_ratio)

        # Step 5: Check Projection Consistency
        if _has_projection_errors(vectors):
            logger.info(f"🚫 Trip {trip_key} discarded: Projection errors detected")
            return []

        return vectors


class LenientPipeline(Pipeline):
    """
    A lenient pipeline that only drops data on specific validity failures.
    """

    def __init__(
        self,
        diary: "Diary",
        topology=None,
        served_ratio: float = 1.0,
        config: dict = None,
    ):
        self.diary = diary
        self.topology = topology
        self.served_ratio = served_ratio
        # Config is unused but kept for interface consistency

    def clean(
        self,
    ) -> list[
        tuple[vectorization.PredictionVector, vectorization.PredictionLabel, float]
    ]:
        if not self.diary or not self.diary.measurements:
            return []

        # Vectorize everything first
        vectors = _vectorize(self.diary, self.topology, self.served_ratio)

        # Apply lenient filtering rules
        for vec, label, timestamp in vectors:
            # 1. Drop if schedule_adherence is -1000.0 (indicates failed projection usually)
            if vec.schedule_adherence == -1000.0:
                return []

            # 2. Drop if scheduled start time encoding is missing (both 0)
            if vec.sch_starting_time_cos == 0.0 and vec.sch_starting_time_sin == 0.0:
                return []

            # 3. Drop if occupancy_status is 7
            if label.occupancy_status == 7:
                return []

        # 4. Check for massive projection errors
        if _has_projection_errors(vectors):
            return []

        return vectors


class TrafficPipeline(Pipeline):
    """Pipeline for traffic analysis vectorization."""

    def __init__(
        self,
        diary: "Diary",
        config: dict = None,
    ):
        self.diary = diary

        # Load cleaning parameters
        cleaning_cfg = config.get("data_cleaning", {}) if config else {}
        self.max_bus_speed = int(cleaning_cfg.get("max_bus_speed_kmh", 120))
        self.max_time_gap = int(cleaning_cfg.get("max_time_gap_seconds", 600))

    def clean(
        self,
    ) -> list[tuple[vectorization.TrafficVector, vectorization.TrafficLabel, float]]:
        """
        Clean diary measurements and vectorize them for traffic analysis.
        """
        if not self.diary or not self.diary.measurements:
            return []

        logger.info(f"🚗 Starting traffic pipeline for diary {self.diary.trip_id}")

        measurements = {str(self.diary.trip_id): self.diary.measurements}

        # Step 0: Data Integrity (Zeros checks only)
        # We pass empty uptime list to skip uptime checks, and irrelevant served_ratio
        cleaned = _check_data_integrity(
            measurements, uptime_timestamps=[], served_ratio=1.0
        )

        # Step 1: Remove duplicates (Time-based for traffic)
        cleaned = _check_for_duplicates(cleaned, keys=["measurement_time"])

        # Step 2: Remove outliers
        cleaned = _remove_outliers(cleaned, self.max_bus_speed, self.max_time_gap)

        # Step 3: Vectorize (Traffic only)
        vectors = _vectorize_traffic(cleaned)

        return vectors


class VehiclePipeline(Pipeline):
    """Pipeline for vehicle classification vectorization."""

    def __init__(
        self,
        diary: "Diary",
        topology=None,
        config: dict = None,
        vehicle_type_name: str = "Unknown",
    ):
        self.diary = diary
        self.topology = topology
        self.vehicle_type_name = vehicle_type_name

        # Load cleaning parameters
        cleaning_cfg = config.get("data_cleaning", {}) if config else {}
        self.min_measurements = int(cleaning_cfg.get("min_measurements_per_run", 7))
        self.max_bus_speed = int(cleaning_cfg.get("max_bus_speed_kmh", 120))
        self.max_time_gap = int(cleaning_cfg.get("max_time_gap_seconds", 600))
        
        self.uptime_timestamps = _load_uptime_data("uptime.csv")

    def clean(
        self,
    ) -> list[tuple[vectorization.VehicleVector, vectorization.VehicleLabel, float]]:
        """
        Clean diary measurements and vectorize the first one for vehicle classification.
        """
        if not self.diary or not self.diary.measurements:
            return []

        logger.info(f"🚌 Starting vehicle pipeline for diary {self.diary.trip_id}")

        measurements = {str(self.diary.trip_id): self.diary.measurements}

        # Step 0: Data Integrity
        cleaned = _check_data_integrity(
            measurements, self.uptime_timestamps, served_ratio=1.0
        )

        # Step 1: Remove duplicates
        cleaned = _check_for_duplicates(cleaned)

        # Step 2: Remove outliers
        cleaned = _remove_outliers(cleaned, self.max_bus_speed, self.max_time_gap)

        # Step 3: Check for valid runs
        cleaned = _check_for_runs(cleaned, self.topology, self.min_measurements)

        # Step 4: Vectorize (First measurement only)
        vectors = _vectorize_vehicle(cleaned, self.topology, self.vehicle_type_name)

        return vectors


def _vectorize_vehicle(
    data: Union["Diary", dict[str, list["Measurement"]]],
    topology=None,
    vehicle_type_name: str = "Unknown",
) -> list[tuple[vectorization.VehicleVector, vectorization.VehicleLabel, float]]:
    """
    Vectorize first measurement using VehicleVectorizer.
    """
    if vehicle_type_name == "Unknown":
        logger.warning("🚌 Skipping vehicle vectorization: Vehicle type is Unknown")
        return []

    filtered_measurements: dict[str, list["Measurement"]] = {}
    if hasattr(data, "trip_id") and hasattr(data, "measurements"):
        filtered_measurements = {str(data.trip_id): data.measurements}
    else:
        filtered_measurements = data

    results: list[
        tuple[vectorization.VehicleVector, vectorization.VehicleLabel, float]
    ] = []

    trips = topology.trips if topology and hasattr(topology, "trips") else {}

    for trip_id, trip_measurements in filtered_measurements.items():
        if not trip_measurements:
            continue
            
        trip = trips.get(trip_id)
        if not trip:
            logger.warning(f"⚠️ Trip {trip_id} skipped: Not found in topology")
            continue

        # Use the first measurement
        m = trip_measurements[0]

        try:
            vectorizer = vectorization.VehicleVectorizer(
                measurement=m,
                trip=trip,
                vehicle_type_name=vehicle_type_name
            )
            vector, label = vectorizer.vectorize()
            results.append((vector, label, m.measurement_time))
        except Exception:
            logger.exception(f"Failed to vectorize vehicle measurement {m.id}")
            continue

    return results



def _vectorize_traffic(
    data: Union["Diary", dict[str, list["Measurement"]]],
) -> list[tuple[vectorization.TrafficVector, vectorization.TrafficLabel, float]]:
    """
    Vectorize measurements using TrafficVectorizer.
    """
    filtered_measurements: dict[str, list["Measurement"]] = {}
    if hasattr(data, "trip_id") and hasattr(data, "measurements"):
        filtered_measurements = {str(data.trip_id): data.measurements}
    else:
        filtered_measurements = data

    results: list[
        tuple[vectorization.TrafficVector, vectorization.TrafficLabel, float]
    ] = []

    for _, trip_measurements in filtered_measurements.items():
        for m in trip_measurements:
            try:
                vectorizer = vectorization.TrafficVectorizer(measurement=m)
                vector, label = vectorizer.vectorize()
                results.append((vector, label, m.measurement_time))
            except Exception:
                logger.exception(f"Failed to vectorize traffic measurement {m.id}")
                continue

    return results


def _load_uptime_data(filepath: str) -> list[int]:
    """Load sorted list of uptime timestamps."""
    if not os.path.exists(filepath):
        logger.warning(f"Uptime file {filepath} not found. Skipping uptime checks.")
        return []

    timestamps = []
    try:
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row and row[0].isdigit():
                    timestamps.append(int(row[0]))
        timestamps.sort()
    except Exception as e:
        logger.error(f"Error loading uptime data: {e}")
    return timestamps


def _is_uptime_sufficient(
    target_ts: int, uptime_timestamps: list[int], window_seconds: int = 3600
) -> bool:
    """
    Check if the bot was ALIVE during the lookback window preceding target_ts.
    Returns False if there are significant gaps (> 3 mins) in uptime coverage.
    """
    start_ts = target_ts - window_seconds
    end_ts = target_ts

    try:
        # Find uptime slice for this window
        start_idx = bisect.bisect_left(uptime_timestamps, start_ts)
        end_idx = bisect.bisect_right(uptime_timestamps, end_ts)

        pings_in_window = uptime_timestamps[start_idx:end_idx]

        if not pings_in_window:
            logger.debug(
                f"Discarded due to insufficient uptime (no pings in lookback window of {window_seconds}s)"
            )
            return False

        # Check for gaps inside the window
        # 1. Gap from start of window to first ping
        if (pings_in_window[0] - start_ts) > 180:  # 3 mins tolerance
            logger.debug(
                "Discarded due to insufficient uptime (hasn't been active for at least a consecutive hour)"
            )
            return False

        # 2. Gap from last ping to end of window
        if (end_ts - pings_in_window[-1]) > 180:
            logger.debug(
                "Discarded due to insufficient uptime (inactive immediately prior to measurement)"
            )
            return False

        # 3. Gaps between pings
        for i in range(len(pings_in_window) - 1):
            if (pings_in_window[i + 1] - pings_in_window[i]) > 180:
                logger.debug(
                    f"Discarded due to insufficient uptime (gap of {pings_in_window[i + 1] - pings_in_window[i]}s during lookback window)"
                )
                return False

        return True

    except Exception as e:
        logger.error(f"Error checking uptime sufficiency: {e}")
        return True  # Fail safe (keep data)


@overload
def _check_data_integrity(
    data: "Diary", uptime_timestamps: list[int], served_ratio: float
) -> "Diary": ...
@overload
def _check_data_integrity(
    data: dict[str, list["Measurement"]],
    uptime_timestamps: list[int],
    served_ratio: float,
) -> dict[str, list["Measurement"]]: ...


def _check_data_integrity(
    data: Union["Diary", dict[str, list["Measurement"]]],
    uptime_timestamps: list[int],
    served_ratio: float,
) -> Union["Diary", dict[str, list["Measurement"]]]:
    """
    Discard measurements with invalid data (zeros) or when bot was offline.
    """

    def is_valid(m: "Measurement") -> bool:
        # 1. Zero check
        # Explicit check for 0.0 (float) or 0 (int)
        # Note: None is technically allowed here (filtered elsewhere maybe?)
        # but user said "if ... are 0"

        # Helper to safely check for 0
        def is_zero(val):
            return val is not None and (val == 0 or val == 0.0)

        # Note: current_speed might be from GPS or derived.
        # m.current_speed is usually the one from traffic/live data

        if is_zero(m.speed_ratio):
            return False
        if is_zero(m.current_speed):  # traffic_speed in some contexts
            return False
        # The user said "current_speed" - in Measurement object:
        # self.current_speed = current_speed (traffic speed)
        # self.gpsdata.speed = GPS speed
        # Assuming they meant the traffic-related fields on the measurement object

        # 2. Uptime check
        # Only relevant if we are claiming the trip was served (served_ratio > 0)
        # If we say it was served, we better be sure we were watching.
        # If uptime check fails, we can't be sure about served_ratio,
        # but here we are filtering individual measurements.

        # Actually, if the bot wasn't running, we shouldn't have recorded a measurement
        # UNLESS it's a reconstructed diary or delayed processing.
        # But if we are processing a diary, and the timestamp says 10:00,
        # and uptime says we were OFF at 10:00, then this data is suspect?
        # Or does "served_ratio computed on non-existing data" mean the Trip's
        # served_ratio is invalid if the bot wasn't up during the trip?

        # The prompt says: "if served_ratio is computed on non-existing data (bot wasn't running...), then the measurement is also not valid"
        # This implies we drop measurements if the bot wasn't alive at measurement_time.

        if uptime_timestamps:
            # Check if there is a ping within 90 seconds
            idx = bisect.bisect_left(uptime_timestamps, m.measurement_time)

            # Check left and right neighbors
            closest_dist = float("inf")

            if idx < len(uptime_timestamps):
                closest_dist = abs(uptime_timestamps[idx] - m.measurement_time)

            if idx > 0:
                dist_prev = abs(uptime_timestamps[idx - 1] - m.measurement_time)
                closest_dist = min(closest_dist, dist_prev)

            if closest_dist > 90:  # 90 seconds tolerance
                return False

        return True

    # Handle Diary
    if hasattr(data, "trip_id") and hasattr(data, "measurements"):
        diary: "Diary" = data
        initial = len(diary.measurements)
        diary.measurements = [m for m in diary.measurements if is_valid(m)]
        dropped = initial - len(diary.measurements)
        if dropped > 0:
            logger.debug(
                f"🛡️ Integrity Check: Dropped {dropped} measurements from diary {diary.trip_id}"
            )
        return diary

    # Handle Dict
    measurements_dict: dict[str, list["Measurement"]] = data
    cleaned_dict: dict[str, list["Measurement"]] = {}

    total_dropped = 0
    for trip_id, measurements in measurements_dict.items():
        valid_m = [m for m in measurements if is_valid(m)]
        if valid_m:
            cleaned_dict[trip_id] = valid_m
        total_dropped += len(measurements) - len(valid_m)

    if total_dropped > 0:
        logger.debug(f"🛡️ Integrity Check: Dropped {total_dropped} measurements")

    return cleaned_dict


def _process_parquet_file(file_path: str) -> list["Diary"]:
    """
    Reads a parquet file and reconstructs Diary objects.

    Args:
        file_path: Path to the .parquet file.

    Returns:
        List of Diary objects (one per trip_id found in the file).
    """
    from ..domain.observers import Diary, Measurement

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to read parquet file {file_path}: {e}")
        return []

    diaries: list["Diary"] = []

    # Group by trip_id
    if "trip_id" not in df.columns:
        logger.error("Parquet file missing 'trip_id' column")
        return []

    grouped = df.groupby("trip_id")

    for trip_id, group in grouped:
        # Create a new Diary for this trip
        # Observer is None as we are reconstructing from file
        diary = Diary(observer=None, trip_id=str(trip_id))

        for _, row in group.iterrows():
            # Reconstruct Weather
            weather = Weather(
                valid_time=row.get("measurement_time"),
                temperature=row.get("temperature"),
                apparent_temperature=row.get("apparent_temperature"),
                humidity=row.get("humidity"),
                precip_intensity=row.get("precip_intensity"),
                wind_speed=row.get("wind_speed"),
                weather_code=row.get("weather_code") or 0,
            )

            # Reconstruct GPSData
            # GPSData expects a trip object with an .id attribute
            dummy_trip = SimpleNamespace(id=str(trip_id))

            gps_data = GPSData(
                id=row.get("stop_id") or 0,  # Or some ID
                trip=dummy_trip,
                timestamp=row.get("gps_timestamp"),
                latitude=row.get("lat"),
                longitude=row.get("lon"),
                speed=row.get("speed"),
                heading=row.get("bearing"),
                next_stop_id=row.get("next_stop"),
                current_stop_sequence=0,  # Not typically in flattened parquet
                # unless saved
                current_status=None,
            )

            # Reconstruct Measurement
            m = Measurement(
                id=row.get("stop_id"),
                autobus_id=row.get("autobus_id", 0),
                next_stop=row.get("next_stop"),
                next_stop_distance=row.get("next_stop_distance"),
                gpsdata=gps_data,
                trip_id=str(trip_id),
                weather=weather,
                occupancy_status=row.get("occupancy"),
                speed_ratio=row.get("speed_ratio"),
                current_speed=row.get("traffic_speed"),
                derived_speed=row.get("derived_speed"),
                derived_bearing=row.get("derived_bearing"),
                is_in_preferential=row.get("is_in_preferential"),
                hexagon_id=row.get("hexagon_id"),
                traffic_data_pending=False,
                schedule_adherence=row.get("schedule_adherence", -1000.0),
                bus_type=row.get("bus_type", 0),
                door_number=row.get("door_number", 0),
                deposits=row.get("deposits", []),
                measurement_time=row.get("measurement_time"),
            )

            diary.add_measurement(m)

        diaries.append(diary)

    logger.info(f"Reconstructed {len(diaries)} diaries from {file_path}")
    return diaries


def _get_attr_path(obj, path):
    """Recursively get attribute value from dot-notation path."""
    try:
        for attr in path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return None


@overload
def _check_for_duplicates(data: "Diary", keys: list[str] = None) -> "Diary": ...
@overload
def _check_for_duplicates(
    data: dict[str, list["Measurement"]], keys: list[str] = None
) -> dict[str, list["Measurement"]]: ...


def _check_for_duplicates(
    data: Union["Diary", dict[str, list["Measurement"]]], keys: list[str] = None
) -> Union["Diary", dict[str, list["Measurement"]]]:
    """
    Remove duplicate measurements based on provided keys.
    Default keys: ["gpsdata.latitude", "gpsdata.longitude"]
    """
    if keys is None:
        keys = ["gpsdata.latitude", "gpsdata.longitude"]

    # Helper for list processing
    def process_list(measurements: list["Measurement"]) -> list["Measurement"]:
        seen = set()
        unique = []
        for m in measurements:
            # Build identity key tuple
            identity = tuple(_get_attr_path(m, k) for k in keys)

            if identity not in seen:
                seen.add(identity)
                unique.append(m)
        return unique

    # Handle Diary
    if hasattr(data, "trip_id") and hasattr(data, "measurements"):
        diary: "Diary" = data
        initial_count = len(diary.measurements)
        diary.measurements = process_list(diary.measurements)

        removed = initial_count - len(diary.measurements)
        if removed > 0:
            pct = (removed / initial_count) * 100 if initial_count > 0 else 0
            logger.debug(
                f"🗑️ Removed {removed} duplicate measurements ({pct:.1f}%) "
                f"from diary {diary.trip_id} using keys {keys}"
            )
        return diary

    # Handle Dict
    measurements_dict: dict[str, list["Measurement"]] = data
    cleaned_dict: dict[str, list["Measurement"]] = {}

    total_removed = 0
    total_initial = 0

    for trip_id, measurements in measurements_dict.items():
        total_initial += len(measurements)
        cleaned_list = process_list(measurements)
        cleaned_dict[trip_id] = cleaned_list
        total_removed += len(measurements) - len(cleaned_list)

    if total_removed > 0:
        pct = (total_removed / total_initial) * 100 if total_initial > 0 else 0
        logger.debug(
            f"🗑️ Removed {total_removed} duplicate measurements ({pct:.1f}%) using keys {keys}"
        )

    return cleaned_dict


@overload
def _remove_outliers(
    data: "Diary", max_bus_speed_kmh: int = 120, max_time_gap_seconds: int = 600
) -> "Diary": ...
@overload
def _remove_outliers(
    data: dict[str, list["Measurement"]],
    max_bus_speed_kmh: int = 120,
    max_time_gap_seconds: int = 600,
) -> dict[str, list["Measurement"]]: ...


def _remove_outliers(
    data: Union["Diary", dict[str, list["Measurement"]]],
    max_bus_speed_kmh: int = 120,
    max_time_gap_seconds: int = 600,
) -> Union["Diary", dict[str, list["Measurement"]]]:
    """
    Remove outlier measurements (random appearances/disappearances).
    """

    # Helper to process a list of measurements
    def process_list(measurements: list["Measurement"]) -> list["Measurement"]:
        if len(measurements) < 2:
            return []  # Filter out single measurements

        sorted_m = sorted(measurements, key=lambda m: m.measurement_time)
        valid_measurements = []
        for i, m in enumerate(sorted_m):
            is_valid = True
            # Speed check
            if hasattr(m, "derived_speed") and m.derived_speed:
                if m.derived_speed > max_bus_speed_kmh:
                    is_valid = False

            # Gap check
            if i > 0 and i < len(sorted_m) - 1:
                prev_gap = m.measurement_time - sorted_m[i - 1].measurement_time
                next_gap = sorted_m[i + 1].measurement_time - m.measurement_time
                if prev_gap > max_time_gap_seconds and next_gap > max_time_gap_seconds:
                    is_valid = False

            if is_valid:
                valid_measurements.append(m)
        return valid_measurements

    # Handle Diary input
    if hasattr(data, "trip_id") and hasattr(data, "measurements"):
        diary: "Diary" = data
        original_len = len(diary.measurements)
        diary.measurements = process_list(diary.measurements)
        if len(diary.measurements) < original_len:
            logger.debug(
                f"🧹 Removed {original_len - len(diary.measurements)} outliers from diary {diary.trip_id}"
            )
        return diary

    # Handle Dict input
    measurements_dict: dict[str, list["Measurement"]] = data
    filtered: dict[str, list["Measurement"]] = {}

    for trip_id, trip_measurements in measurements_dict.items():
        valid = process_list(trip_measurements)
        if valid:
            filtered[trip_id] = valid

    removed = sum(len(v) for v in measurements_dict.values()) - sum(
        len(v) for v in filtered.values()
    )
    if removed > 0:
        logger.debug(f"🧹 Removed {removed} outlier measurements")

    return filtered


@overload
def _check_for_runs(
    data: "Diary", topology=None, min_measurements_per_run: int = 7
) -> "Diary": ...
@overload
def _check_for_runs(
    data: dict[str, list["Measurement"]],
    topology=None,
    min_measurements_per_run: int = 7,
) -> dict[str, list["Measurement"]]: ...


def _check_for_runs(
    data: Union["Diary", dict[str, list["Measurement"]]],
    topology=None,
    min_measurements_per_run: int = 7,
) -> Union["Diary", dict[str, list["Measurement"]]]:
    """
    Validate complete runs and remove ghost runs.
    """

    def is_valid_run(trip_id: str, measurements: list["Measurement"]) -> bool:
        if len(measurements) < min_measurements_per_run:
            logger.info(f"🚌 Trip {trip_id} dropped: < min measurements")
            return False

        if topology and hasattr(topology, "trips"):
            trip = topology.trips.get(trip_id)
            if trip and hasattr(trip, "stop_times") and trip.stop_times:
                first_stop_id = str(trip.stop_times[0].get("stop_id"))
                last_stop_id = str(trip.stop_times[-1].get("stop_id"))

                sorted_m = sorted(measurements, key=lambda m: m.measurement_time)
                first_m = sorted_m[0]
                last_m = sorted_m[-1]

                starts_at_terminus = (
                    hasattr(first_m, "next_stop")
                    and str(first_m.next_stop) == first_stop_id
                )
                ends_at_terminus = (
                    hasattr(last_m, "next_stop")
                    and str(last_m.next_stop) == last_stop_id
                )

                if not (starts_at_terminus or ends_at_terminus):
                    logger.info(f"👻 Trip {trip_id} dropped: Ghost run")
                    return False
        return True

    # Handle Diary
    if hasattr(data, "trip_id") and hasattr(data, "measurements"):
        diary: "Diary" = data
        if not is_valid_run(str(diary.trip_id), diary.measurements):
            diary.measurements = []  # Clear invalid run? or return empty diary?
        return diary

    # Handle Dict
    filtered: dict[str, list["Measurement"]] = {}
    for trip_id, trip_measurements in data.items():
        if is_valid_run(trip_id, trip_measurements):
            filtered[trip_id] = trip_measurements

    removed_trips = len(data) - len(filtered)
    if removed_trips > 0:
        logger.debug(f"🚌 Removed {removed_trips} incomplete runs")

    return filtered


@overload
def _vectorize(
    data: "Diary", topology=None, served_ratio: float = 1.0
) -> list[
    tuple[vectorization.PredictionVector, vectorization.PredictionLabel, float]
]: ...
@overload
def _vectorize(
    data: dict[str, list["Measurement"]], topology=None, served_ratio: float = 1.0
) -> list[
    tuple[vectorization.PredictionVector, vectorization.PredictionLabel, float]
]: ...


def _vectorize(
    data: Union["Diary", dict[str, list["Measurement"]]],
    topology=None,
    served_ratio: float = 1.0,
) -> list[tuple[vectorization.PredictionVector, vectorization.PredictionLabel, float]]:
    """
    Vectorize all measurements using the Vectorizer.
    """
    # Normalize input to dict for unified processing
    filtered_measurements: dict[str, list["Measurement"]] = {}
    if hasattr(data, "trip_id") and hasattr(data, "measurements"):
        filtered_measurements = {str(data.trip_id): data.measurements}
    else:
        filtered_measurements = data

    results: list[
        tuple[vectorization.PredictionVector, vectorization.PredictionLabel, float]
    ] = []

    if not topology:
        logger.warning("⚠️ No topology provided, cannot vectorize")
        return results

    trips = topology.trips
    routes = topology.routes

    for trip_id, trip_measurements in filtered_measurements.items():
        trip = trips.get(trip_id)
        if not trip:
            logger.warning(f"⚠️ Trip {trip_id} skipped: Not found in topology")
            continue

        route = trip.route if hasattr(trip, "route") else None
        if not route:
            logger.warning(f"⚠️ Trip {trip_id} skipped: Route {trip.route_id} not found")
            continue

        shape = trip.shape if hasattr(trip, "shape") else None
        if not shape:
            logger.warning(f"⚠️ Trip {trip_id} skipped: Shape not found on trip object")
            continue

        for m in trip_measurements:
            # Skip if bus info is missing
            if getattr(m, 'bus_type', 0) == 0:
                # logger.debug(f"Skipping vectorization for measurement {m.id}: Unknown bus type")
                continue

            try:
                vectorizer = vectorization.PredictionVectorizer(
                    measurement=m,
                    route=route,
                    shape=shape,
                    trip=trip,
                    served_ratio=served_ratio,
                )
                vector, label = vectorizer.vectorize()
                results.append((vector, label, m.measurement_time))
            except Exception:
                logger.exception(f"Failed to vectorize measurement {m.id}")
                continue

    logger.info(f"✅ Vectorized {len(results)} measurements")
    return results


def _has_projection_errors(
    vectors: list[
        tuple[vectorization.PredictionVector, vectorization.PredictionLabel, float]
    ],
) -> bool:
    """
    Check for massive jumps in shape_dist_travelled indicating projection errors.
    """
    if not vectors:
        return False

    # Sort by time just in case
    sorted_v = sorted(vectors, key=lambda x: x[2])  # x[2] is measurement_time

    for i in range(1, len(sorted_v)):
        v_curr = sorted_v[i][0]
        v_prev = sorted_v[i - 1][0]

        # Check for backwards jumps (loop snapping error)
        # Allow small buffer (e.g. 50m) for GPS jitter
        delta_dist = v_curr.shape_dist_travelled - v_prev.shape_dist_travelled
        if delta_dist < -50.0:
            logger.debug(
                f"🚫 Projection Error: Backward jump of {delta_dist:.1f}m detected."
            )
            return True

        # Check for massive forward jumps (impossible speed)
        # e.g. > 160 km/h (45 m/s)
        delta_time = sorted_v[i][2] - sorted_v[i - 1][2]
        if delta_time > 0:
            speed = delta_dist / delta_time
            if speed > 45.0:
                logger.debug(
                    f"🚫 Projection Error: Impossible speed {speed*3.6:.1f} km/h detected."
                )
                return True

    return False
