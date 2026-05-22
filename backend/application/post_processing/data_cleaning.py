"""
Lightweight post-processing pipelines for measurements.

Provides functions to:
- Drop structurally invalid measurements
- Remove exact duplicate measurements
"""

import logging
import pandas as pd
import csv
import bisect
import os
import math
from typing import TYPE_CHECKING
from types import SimpleNamespace

from config import Config
from ..domain.live_data import GPSData
from ..domain.weather import Weather
from application.domain.interfaces import Pipeline

if TYPE_CHECKING:
    from ..domain.live_data import LiveTrip, Measurement

logger = logging.getLogger(__name__)


class PredictionPipeline(Pipeline):
    """Pipeline for fast integrity cleaning of prediction measurements."""

    def __init__(
        self,
        live_trip: "LiveTrip",
        topology=None,
        served_ratio: float = 1.0,
        config: Config | dict = None,
    ):
        """
        Initialize pipeline with one completed live trip and static data.

        Args:
            live_trip: LiveTrip object containing measurements
            topology: TopologyLedger with routes, trips, shapes
            served_ratio: Ratio from scan_trip_adherence
            config: Configuration dictionary
        """
        self.live_trip = live_trip
        self.topology = topology
        self.served_ratio = served_ratio

        config = Config.coerce(config)
        self.uptime_lookback_seconds = (
            config.data_cleaning.served_ratio_lookback_minutes * 60
        )

        self.uptime_timestamps = _load_uptime_data("uptime.csv")

    def clean(self) -> list["Measurement"]:
        """
        Clean measurements with universal integrity checks.

        Returns:
            Cleaned measurements, still expressed as domain objects.
        """
        if not self.live_trip or not self.live_trip.measurements:
            return []

        logger.info(
            f"🚀 Starting pipeline for live trip {self.live_trip.trip_id} "
            f"with {len(self.live_trip.measurements)} measurements"
        )

        if self.served_ratio < 1.0 and self.uptime_timestamps:
            first_ts = self.live_trip.measurements[0].measurement_time
            if not _is_uptime_sufficient(
                first_ts,
                self.uptime_timestamps,
                window_seconds=self.uptime_lookback_seconds,
            ):
                logger.warning(
                    f"⚠️ Discarding live trip {self.live_trip.trip_id}: served_ratio {self.served_ratio:.2f} "
                    f"but insufficient uptime coverage during the lookback window ({self.uptime_lookback_seconds}s)."
                )
                return []

        trip_key = str(self.live_trip.trip_id)
        measurements = {trip_key: self.live_trip.measurements}
        cleaned = _clean_measurements(
            measurements,
            uptime_timestamps=self.uptime_timestamps,
            served_ratio=self.served_ratio,
        )
        if not cleaned or not cleaned.get(trip_key):
            return []

        return cleaned.get(trip_key, [])


class TrafficPipeline(Pipeline):
    """Pipeline for traffic measurement integrity cleaning."""

    def __init__(
        self,
        live_trip: "LiveTrip",
    ):
        """Bind the completed live trip."""
        self.live_trip = live_trip

    def clean(self) -> list["Measurement"]:
        """Clean live-trip measurements for traffic analysis."""
        if not self.live_trip or not self.live_trip.measurements:
            return []

        logger.info(f"🚗 Starting traffic pipeline for live trip {self.live_trip.trip_id}")

        measurements = {str(self.live_trip.trip_id): self.live_trip.measurements}
        cleaned = _clean_measurements(
            measurements,
            uptime_timestamps=[],
            served_ratio=1.0,
        )
        return cleaned.get(str(self.live_trip.trip_id), [])


class VehiclePipeline(Pipeline):
    """Pipeline for vehicle measurement integrity cleaning."""

    def __init__(
        self,
        live_trip: "LiveTrip",
        topology=None,
        vehicle_type_name: str = "Unknown",
    ):
        """Bind the live trip, topology, vehicle-type label, and cleaning thresholds."""
        self.live_trip = live_trip
        self.topology = topology
        self.vehicle_type_name = vehicle_type_name

        self.uptime_timestamps = _load_uptime_data("uptime.csv")

    def clean(self) -> list["Measurement"]:
        """Clean live-trip measurements for vehicle history."""
        if not self.live_trip or not self.live_trip.measurements:
            return []

        logger.info(f"🚌 Starting vehicle pipeline for live trip {self.live_trip.trip_id}")

        measurements = {str(self.live_trip.trip_id): self.live_trip.measurements}
        cleaned = _clean_measurements(
            measurements,
            uptime_timestamps=self.uptime_timestamps,
            served_ratio=1.0,
        )
        return cleaned.get(str(self.live_trip.trip_id), [])


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
    data: dict[str, list["Measurement"]],
    uptime_timestamps: list[int],
    served_ratio: float,
    duplicate_keys: list[str] = None,
) -> dict[str, list["Measurement"]]:
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


def _check_data_integrity(
    data: dict[str, list["Measurement"]],
    uptime_timestamps: list[int],
    served_ratio: float,
) -> dict[str, list["Measurement"]]:
    """
    Discard measurements with invalid structural data or offline timestamps.
    """

    def is_valid(m: "Measurement") -> bool:
        """Return True if a measurement has the fields required for persistence."""
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

    cleaned_dict: dict[str, list["Measurement"]] = {}

    total_dropped = 0
    for trip_id, measurements in data.items():
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
                vehicle_id=row.get("vehicle_id", 0),
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


def _check_for_duplicates(
    data: dict[str, list["Measurement"]], keys: list[str] = None
) -> dict[str, list["Measurement"]]:
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

    cleaned_dict: dict[str, list["Measurement"]] = {}

    total_removed = 0
    total_initial = 0

    for trip_id, measurements in data.items():
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
