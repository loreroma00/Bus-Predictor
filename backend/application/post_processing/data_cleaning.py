"""
Lightweight post-processing pipelines for measurements.

Provides functions to:
- Drop structurally invalid measurements
- Remove exact duplicate measurements
- Vectorize measurements
"""

import logging
import pandas as pd
import csv
import bisect
import os
import math
from typing import TYPE_CHECKING, Union, overload
from types import SimpleNamespace

from ..domain.live_data import GPSData
from ..domain.weather import Weather
from application.domain.interfaces import Pipeline
from . import vectorization

if TYPE_CHECKING:
    from ..domain.live_data import LiveTrip, Measurement

logger = logging.getLogger(__name__)


class PredictionPipeline(Pipeline):
    """Pipeline for fast integrity cleaning and prediction vectorization."""

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
            diary: LiveTrip-like object containing measurements
            topology: TopologyLedger with routes, trips, shapes
            served_ratio: Ratio from scan_trip_adherence
            config: Configuration dictionary
        """
        self.diary = diary
        self.topology = topology
        self.served_ratio = served_ratio

        cleaning_cfg = config.get("data_cleaning", {}) if config else {}
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
        Clean measurements with universal checks and vectorize them.

        Returns:
            List of (Vector, Label, measurement_time) tuples for DB insert.
        """
        if not self.diary or not self.diary.measurements:
            return []

        logger.info(
            f"🚀 Starting pipeline for diary {self.diary.trip_id} "
            f"with {len(self.diary.measurements)} measurements"
        )

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

        trip_key = str(self.diary.trip_id)
        measurements = {trip_key: self.diary.measurements}
        cleaned = _clean_measurements(
            measurements,
            uptime_timestamps=self.uptime_timestamps,
            served_ratio=self.served_ratio,
        )
        if not cleaned or not cleaned.get(trip_key):
            return []

        vectors = _vectorize(cleaned, self.topology, self.served_ratio)

        if _has_vector_integrity_errors(vectors):
            logger.info(f"🚫 Trip {trip_key} discarded: Invalid vector data detected")
            return []

        return vectors


class LenientPipeline(PredictionPipeline):
    """
    Backward-compatible alias for the now-lightweight prediction pipeline.
    """


class TrafficPipeline(Pipeline):
    """Pipeline for traffic analysis vectorization."""

    def __init__(
        self,
        diary: "Diary",
        config: dict = None,
    ):
        """Bind the diary. Config is accepted for interface compatibility."""
        self.diary = diary

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
        cleaned = _clean_measurements(
            measurements,
            uptime_timestamps=[],
            served_ratio=1.0,
        )
        if not cleaned:
            return []
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
        """Bind the diary, topology, vehicle-type label, and cleaning thresholds."""
        self.diary = diary
        self.topology = topology
        self.vehicle_type_name = vehicle_type_name

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
        cleaned = _clean_measurements(
            measurements,
            uptime_timestamps=self.uptime_timestamps,
            served_ratio=1.0,
        )
        if not cleaned:
            return []
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


def _clean_measurements(
    data: Union["Diary", dict[str, list["Measurement"]]],
    uptime_timestamps: list[int],
    served_ratio: float,
    duplicate_keys: list[str] = None,
) -> Union["Diary", dict[str, list["Measurement"]]]:
    """Apply the lightweight universal cleaning rules used by all pipelines."""
    cleaned = _check_data_integrity(data, uptime_timestamps, served_ratio)
    return _check_for_duplicates(cleaned, keys=duplicate_keys)


def _is_finite_number(value) -> bool:
    """Return True for finite int/float values, excluding booleans."""
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@overload
def _check_data_integrity(
    data: "Diary", uptime_timestamps: list[int], served_ratio: float
) -> "Diary":
    """Diary overload — see implementation below."""
    ...
@overload
def _check_data_integrity(
    data: dict[str, list["Measurement"]],
    uptime_timestamps: list[int],
    served_ratio: float,
) -> dict[str, list["Measurement"]]:
    """Dict-of-measurements overload — see implementation below."""
    ...


def _check_data_integrity(
    data: Union["Diary", dict[str, list["Measurement"]]],
    uptime_timestamps: list[int],
    served_ratio: float,
) -> Union["Diary", dict[str, list["Measurement"]]]:
    """
    Discard measurements with invalid structural data or offline timestamps.
    """

    def is_valid(m: "Measurement") -> bool:
        """Return True if a measurement has the fields required for vectorization."""
        measurement_time = getattr(m, "measurement_time", None)
        if not _is_finite_number(measurement_time):
            return False

        speed_ratio = getattr(m, "speed_ratio", None)
        if not _is_finite_number(speed_ratio) or float(speed_ratio) <= 0:
            return False

        current_speed = getattr(m, "current_speed", None)
        if not _is_finite_number(current_speed) or float(current_speed) < 0:
            return False

        gpsdata = getattr(m, "gpsdata", None)
        if gpsdata is None:
            return False
        if not _is_finite_number(getattr(gpsdata, "latitude", None)):
            return False
        if not _is_finite_number(getattr(gpsdata, "longitude", None)):
            return False

        if uptime_timestamps:
            idx = bisect.bisect_left(uptime_timestamps, measurement_time)
            closest_dist = float("inf")
            if idx < len(uptime_timestamps):
                closest_dist = abs(uptime_timestamps[idx] - measurement_time)
            if idx > 0:
                dist_prev = abs(uptime_timestamps[idx - 1] - measurement_time)
                closest_dist = min(closest_dist, dist_prev)
            if closest_dist > 90:
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


def _process_parquet_file(file_path: str) -> list["LiveTrip"]:
    """
    Reads a parquet file and reconstructs lightweight LiveTrip-like objects.

    Args:
        file_path: Path to the .parquet file.

    Returns:
        List of LiveTrip-like objects (one per trip_id found in the file).
    """
    from ..domain.live_data import Measurement

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to read parquet file {file_path}: {e}")
        return []

    live_trips: list["LiveTrip"] = []

    # Group by trip_id
    if "trip_id" not in df.columns:
        logger.error("Parquet file missing 'trip_id' column")
        return []

    grouped = df.groupby("trip_id")

    for trip_id, group in grouped:
        live_trip = SimpleNamespace(
            trip_id=str(trip_id),
            measurements=[],
            scheduled_start_time="",
            actual_start_time=0.0,
        )

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
                vehicle_id=row.get("vehicle_id", row.get("autobus_id", 0)),
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

            live_trip.measurements.append(m)

        live_trips.append(live_trip)

    logger.info(f"Reconstructed {len(live_trips)} live trips from {file_path}")
    return live_trips


def _get_attr_path(obj, path):
    """Recursively get attribute value from dot-notation path."""
    try:
        for attr in path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return None


@overload
def _check_for_duplicates(data: "Diary", keys: list[str] = None) -> "Diary":
    """Diary overload — see implementation below."""
    ...
@overload
def _check_for_duplicates(
    data: dict[str, list["Measurement"]], keys: list[str] = None
) -> dict[str, list["Measurement"]]:
    """Dict-of-measurements overload — see implementation below."""
    ...


def _check_for_duplicates(
    data: Union["Diary", dict[str, list["Measurement"]]], keys: list[str] = None
) -> Union["Diary", dict[str, list["Measurement"]]]:
    """
    Remove duplicate measurements based on provided keys.
    Default keys describe a repeated ping, not a vehicle legitimately standing still.
    """
    if keys is None:
        keys = ["measurement_time", "gpsdata.latitude", "gpsdata.longitude"]

    # Helper for list processing
    def process_list(measurements: list["Measurement"]) -> list["Measurement"]:
        """Return ``measurements`` with duplicates (same tuple of ``keys``) removed, preserving order."""
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
def _vectorize(
    data: "Diary", topology=None, served_ratio: float = 1.0
) -> list[
    tuple[vectorization.PredictionVector, vectorization.PredictionLabel, float]
]:
    """Diary overload — see implementation below."""
    ...
@overload
def _vectorize(
    data: dict[str, list["Measurement"]], topology=None, served_ratio: float = 1.0
) -> list[
    tuple[vectorization.PredictionVector, vectorization.PredictionLabel, float]
]:
    """Dict-of-measurements overload — see implementation below."""
    ...


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
            if getattr(m, "bus_type", 0) == 0:
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


def _has_vector_integrity_errors(
    vectors: list[
        tuple[vectorization.PredictionVector, vectorization.PredictionLabel, float]
    ],
) -> bool:
    """Return True when vectorization produced known invalid sentinel values."""
    for vector, label, _ in vectors:
        if getattr(vector, "schedule_adherence", 0.0) == -1000.0:
            return True
        if getattr(label, "occupancy_status", 0) == 7:
            return True
    return False
