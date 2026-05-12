"""GTFS static-data domain objects: fleet enums, VehicleType, Route/Trip/Stop/Shape."""

import numpy as np
import scipy as sp
from scipy.spatial import KDTree
import statistics
import time as _time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from .time_utils import to_unix_time

class FuelType(Enum):
    """Enum of propulsion types used by ATAC fleet."""
    DIESEL = 0
    ELECTRIC_NMC = 1
    CNG = 2
    HYBRID_DIESEL_ELECTRIC = 3
    HYBRID_LTO = 4
    ELECTRIC_NINACL = 5
    ELECTRIC_CATL_LFP = 6
    ELECTRIC_LFP = 7
    ELECTRIC = 8
    DUAL_DIESEL_ELECTRIC = 9

class EuroType(Enum):
    """Euro emission class + ZEV marker."""
    EURO_0 = 0
    EURO_1 = 1
    EURO_2 = 2
    EURO_3 = 3
    EURO_4 = 4
    EURO_5 = 5
    EURO_6 = 6
    EURO_7 = 7
    EURO_8 = 8
    EURO_9 = 9
    ZEV = 10

class Engine:
    """Descriptor for an engine variant (name + fuel + emission class)."""
    def __init__(self,
                 name: str,
                 fuel: FuelType,
                 euro: EuroType):
        """Store engine name, fuel type, and Euro classification."""
        self.name = name
        self.fuel = fuel
        self.euro = euro
    
class VehicleType:
    """Descriptor for a fleet vehicle model (ID ranges, engine, physical dims, capacity)."""
    def __init__(self,
                 name: str,
                 ids: list[list[int]], # Each list is an interval
                 amount: int,
                 active: int, # amount of active vehicles of this model
                 agency: str,
                 deposits: list[str],
                 doors: int, # amount of doors
                 engine: Engine,
                 length: float = None,
                 width: float = None,
                 height: float = None,
                 weight: float = None,
                 capacity_sitting: int = 0,
                 capacity_standing: int = 0,
                 capacity_total: int = 0,
                 construction_year: int = 0,
                 constructors: list[str] = None):
        """Store the full vehicle-type profile and derive total capacity when not supplied."""
        self.name = name
        self.ids = ids
        self.amount = amount
        self.active = active
        self.deposits = deposits
        self.doors = doors
        self.engine = engine
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.capacity_sitting = capacity_sitting
        self.capacity_standing = capacity_standing
        self.capacity_total = (self.capacity_sitting + self.capacity_standing) if capacity_total == 0 else capacity_total
        self.construction_year = construction_year
        self.constructors = constructors        


class Vehicle:
    """Static identity of a physical fleet vehicle.

    Runtime state belongs to ``LiveTrip``. This object only describes the
    physical vehicle and can lazily ask an injected history loader for past
    served trips.
    """

    def __init__(
        self,
        id: str,
        label: str = None,
        vehicle_type: VehicleType = None,
        history_loader=None,
        history_ledger=None,
        history_ttl_seconds: int = 300,
    ):
        """Store stable vehicle identity and optional lazy history access."""
        self.id = str(id)
        self.label = str(label) if label else str(id)
        self.vehicle_type = vehicle_type
        self._history_loader = history_loader
        self._history_ledger = history_ledger
        self._history_ttl_seconds = history_ttl_seconds
        self._history_cache = None
        self._history_loaded_at = 0.0

    def get_label(self) -> str:
        """Return the public fleet label."""
        return self.label

    def set_vehicle_type(self, vehicle_type: VehicleType):
        """Set the static vehicle type descriptor."""
        self.vehicle_type = vehicle_type

    def get_vehicle_type(self) -> VehicleType | None:
        """Return the static vehicle type descriptor."""
        return self.vehicle_type

    def bind_history_loader(self, history_loader):
        """Bind a lazy history provider without coupling Vehicle to persistence."""
        self._history_loader = history_loader

    def _get_history_ledger(self):
        """Return this vehicle's lazy mini-ledger."""
        if self._history_ledger is None:
            self._history_ledger = VehicleHistoryLedger()
        return self._history_ledger

    def get_history(self, force_refresh: bool = False):
        """Return cached served-trip history, refreshing lazily when needed."""
        import time

        now = time.time()
        is_stale = now - self._history_loaded_at > self._history_ttl_seconds
        if force_refresh or self._history_cache is None or is_stale:
            if self._history_loader is not None:
                self._history_cache = self._history_loader(self.label)
            else:
                self._history_cache = self._get_history_ledger().get_history(
                    self.label,
                    force_refresh=force_refresh,
                )
            self._history_loaded_at = now
        return self._history_cache

    def record_trip(self, record):
        """Append a served-trip summary to this vehicle's mini-ledger."""
        ledger = self._get_history_ledger()
        ledger.record_trip(record)
        ledger.push_to_db()
        self._history_cache = None
        self._history_loaded_at = 0.0

    def get_today_vehicle_trips(self) -> list[dict]:
        """Return records appended during this process for this vehicle."""
        if self._history_ledger is None:
            return []
        return self._history_ledger.get_today_vehicle_trips()


@dataclass
class VehicleTripRecord:
    """One completed trip as observed from a specific vehicle."""

    vehicle_id: str
    trip_id: str
    route_id: str
    direction_id: int
    vehicle_type_name: str
    fuel_type: int
    euro_class: int
    capacity_total: int
    trip_date: str
    scheduled_start: str
    actual_start_time: float
    trip_end_time: float
    trip_duration_sec: float
    mean_delay_sec: float
    median_delay_sec: float
    max_delay_sec: float
    min_delay_sec: float
    std_delay_sec: float
    mean_occupancy: float
    max_occupancy: int
    measurement_count: int
    preferential_ratio: float
    recorded_at: float


class VehicleHistoryLedger:
    """Mini-ledger for per-vehicle served-trip history."""

    _FIELDS = [
        "vehicle_id", "trip_id", "route_id", "direction_id",
        "vehicle_type_name", "fuel_type", "euro_class", "capacity_total",
        "trip_date", "scheduled_start", "actual_start_time",
        "trip_end_time", "trip_duration_sec",
        "mean_delay_sec", "median_delay_sec", "max_delay_sec",
        "min_delay_sec", "std_delay_sec",
        "mean_occupancy", "max_occupancy",
        "measurement_count", "preferential_ratio", "recorded_at",
    ]

    def __init__(self, connection_string: str = None, table_name: str = None):
        """Bind the mini-ledger to its DB destination configuration."""
        from config import Ledger

        self._conn_str = connection_string or Ledger.DB_CONNECTION
        self._table = table_name or Ledger.VEHICLE_TABLE
        self._today_records: list[dict] = []
        self._history_cache: dict[str, tuple[float, list[dict]]] = {}
        self._history_ttl_seconds = 300
        self._pending_db_rows: list[dict] = []

    def record_trip(self, record: VehicleTripRecord):
        """Append a single vehicle trip record."""
        return self.record_trips([record])

    def record_trips(self, records: list[VehicleTripRecord]):
        """Append vehicle trip records to the in-memory buffer."""
        if not records:
            return []
        rows = [{field: getattr(record, field) for field in self._FIELDS} for record in records]
        self._today_records.extend(rows)
        self._pending_db_rows.extend(rows)
        return rows

    def push_to_db(self):
        """Push pending vehicle-trip rows to the database layer."""
        if not self._pending_db_rows:
            return
        from persistence.database import write_vehicle_trips

        write_vehicle_trips(self._conn_str, self._table, self._pending_db_rows)
        self._pending_db_rows = []

    def get_today_vehicle_trips(self) -> list[dict]:
        """Return records appended during this process."""
        return self._today_records

    def get_history(self, vehicle_id: str, force_refresh: bool = False) -> list[dict]:
        """Return cached served-trip history for one vehicle, refreshing lazily."""
        now = _time.time()
        cached = self._history_cache.get(str(vehicle_id))
        if cached and not force_refresh and now - cached[0] <= self._history_ttl_seconds:
            return cached[1]

        df = self.query(vehicle_id=str(vehicle_id))
        records = df.to_dict("records") if df is not None and not df.empty else []
        self._history_cache[str(vehicle_id)] = (now, records)
        return records

    def query(
        self,
        vehicle_id: str = None,
        route_id: str = None,
        fuel_type: int = None,
        date_start: str = None,
        date_end: str = None,
    ):
        """Query vehicle trip records through the DB layer."""
        from persistence.database import read_vehicle_trips

        return read_vehicle_trips(
            self._conn_str,
            self._table,
            vehicle_id=vehicle_id,
            route_id=route_id,
            fuel_type=fuel_type,
            date_start=date_start,
            date_end=date_end,
        )


def summarize_live_trip_for_vehicle(
    live_trip,
    route_id: str = None,
    direction_id: int = None,
    vehicle_type_name: str = "Unknown",
) -> Optional[VehicleTripRecord]:
    """Summarize a completed live trip into a vehicle trip performance record."""
    if not live_trip or not live_trip.measurements:
        return None

    vehicle = live_trip.vehicle
    vehicle_id = str(getattr(vehicle, "label", "?") or "?")
    vehicle_type = getattr(vehicle, "vehicle_type", None)

    fuel_type = 0
    euro_class = 0
    capacity_total = 0
    if vehicle_type:
        engine = getattr(vehicle_type, "engine", None)
        if engine:
            fuel_type = getattr(getattr(engine, "fuel", None), "value", 0) or 0
            euro_class = getattr(getattr(engine, "euro", None), "value", 0) or 0
        capacity_total = getattr(vehicle_type, "capacity_total", 0) or 0

    first_measurement = live_trip.measurements[0]
    last_measurement = live_trip.measurements[-1]
    actual_start = (
        live_trip.actual_start_time
        if live_trip.actual_start_time
        else first_measurement.measurement_time
    )
    trip_end = last_measurement.measurement_time
    trip_duration = trip_end - actual_start if actual_start else 0.0
    trip_date = datetime.fromtimestamp(actual_start).strftime("%Y-%m-%d") if actual_start else ""
    scheduled_start = live_trip.scheduled_start_time or ""

    valid_delays = [
        measurement.schedule_adherence
        for measurement in live_trip.measurements
        if measurement.schedule_adherence is not None
        and measurement.schedule_adherence != -1000.0
        and abs(measurement.schedule_adherence) <= 7200
    ]
    if valid_delays:
        mean_delay = statistics.mean(valid_delays)
        median_delay = statistics.median(valid_delays)
        max_delay = max(valid_delays)
        min_delay = min(valid_delays)
        std_delay = statistics.stdev(valid_delays) if len(valid_delays) > 1 else 0.0
    else:
        mean_delay = median_delay = max_delay = min_delay = std_delay = 0.0

    occupancy_values = [
        measurement.occupancy_status
        for measurement in live_trip.measurements
        if measurement.occupancy_status is not None
    ]
    mean_occupancy = statistics.mean(occupancy_values) if occupancy_values else 0.0
    max_occupancy = max(occupancy_values) if occupancy_values else 0

    preferential_count = sum(
        1
        for measurement in live_trip.measurements
        if getattr(measurement, "is_in_preferential", False)
    )
    preferential_ratio = preferential_count / len(live_trip.measurements)

    trip = live_trip.trip
    route_id = route_id or (trip.route.id if trip and trip.route else "")
    direction_id = (
        direction_id
        if direction_id is not None
        else (trip.direction_id if trip else 0)
    )
    if vehicle_type_name == "Unknown" and vehicle_type:
        vehicle_type_name = vehicle_type.name

    return VehicleTripRecord(
        vehicle_id=vehicle_id,
        trip_id=live_trip.trip_id,
        route_id=route_id,
        direction_id=direction_id,
        vehicle_type_name=vehicle_type_name,
        fuel_type=fuel_type,
        euro_class=euro_class,
        capacity_total=capacity_total,
        trip_date=trip_date,
        scheduled_start=scheduled_start,
        actual_start_time=actual_start or 0.0,
        trip_end_time=trip_end,
        trip_duration_sec=trip_duration,
        mean_delay_sec=mean_delay,
        median_delay_sec=median_delay,
        max_delay_sec=max_delay,
        min_delay_sec=min_delay,
        std_delay_sec=std_delay,
        mean_occupancy=mean_occupancy,
        max_occupancy=max_occupancy,
        measurement_count=len(live_trip.measurements),
        preferential_ratio=preferential_ratio,
        recorded_at=_time.time(),
    )


class Route:
    """GTFS route: id, agency, direction, optional shape and trip list."""
    def __init__(self, id, agency=None, direction=None, shape=None, trips=None):
        """Store the route identity and its optional shape/trips."""
        self.id = id
        self.agency = agency
        self.direction = direction
        self.shape = shape
        self.trips = trips

    @property
    def get_trips(self):  # Returns the list of trips of the route
        """Return the list of trips attached to this route."""
        return self.trips


class Trip:
    """GTFS trip instance: stop times + interpolated time/spatial laws (lazy-built)."""
    def __init__(
        self,
        id,
        route,
        dates,
        direction_id=None,
        shape=None,
        stop_times=None,
        trip_headsign=None,
    ):
        """Store trip metadata; time/spatial laws are built lazily on first access."""
        self.id = id
        self.route = route
        self.direction_id = direction_id
        self.shape = shape  # The argument shape_id actually receives a Shape object now
        self.stop_times = stop_times
        self.direction_name = trip_headsign
        self.dates = dates
        self.time_law = None
        self.spatial_law = None

    def set_stop_times(self, stop_times):
        """Replace the list of stop times for this trip."""
        self.stop_times = stop_times

    def get_stop_times(self):
        """Return the list of stop-time records."""
        return self.stop_times

    def _build_laws(self):
        """Build the time↔distance interpolators (``time_law``/``spatial_law``), monotonising midnight-crossing trips."""
        stop_times = []
        distances = []
        for row in self.stop_times:
            # Parse time string directly to preserve >24h values for night buses.
            # Using get_seconds_since_midnight wraps to 0-86399, breaking
            # interpolation for midnight-crossing trips.
            h, m, s = map(int, row["arrival_time"].split(":"))
            seconds = h * 3600 + m * 60 + s
            stop_times.append(seconds)
            distances.append(float(row["shape_dist_traveled"]))

        # Ensure monotonicity for midnight-crossing trips where times
        # wrap around (e.g., "23:30:00" → "00:05:00" should become 86700).
        for i in range(1, len(stop_times)):
            while stop_times[i] < stop_times[i - 1] - 43200:
                stop_times[i] += 86400

        self.time_law = sp.interpolate.interp1d(stop_times, distances, kind="linear")
        self.spatial_law = sp.interpolate.interp1d(distances, stop_times, kind="linear")

    def get_time_law(self):
        """Return the time→distance interpolator, building it lazily on first call."""
        if self.time_law is None:
            self._build_laws()
        return self.time_law

    def get_spatial_law(self):
        """Return the distance→time interpolator, building it lazily on first call."""
        if self.spatial_law is None:
            self._build_laws()
        return self.spatial_law

    def get_shape(self):
        """Return the Shape object associated with this trip."""
        return self.shape


class Stop:
    """GTFS stop: identifier, human name, and geographic coordinates."""
    def __init__(self, id: int, name: str, latitude: float, longitude: float):
        """Store the stop identity and its lat/lon coordinates."""
        self.id = id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude


class Shape:
    """GTFS shape with a KDTree index and flat-earth projection optimised for Rome."""
    def __init__(self, id: int, points: list[dict[str, float]]):
        """Store polyline points, build a KDTree, and precompute the Rome longitude-scale factor."""
        self.id = id
        # points is a list of {'lat', 'lon', 'dist'}
        # Extract coordinates for KDTree (Lat, Lon)
        self.coords = np.array([[p["lat"], p["lon"]] for p in points])
        self.distances = np.array([p["dist"] for p in points])

        # Build optimized spatial index
        self.tree = KDTree(self.coords)

        # Rome Latitude for Flat Earth Projection (approx 41.9)
        # We pre-calculate scaling factor
        mean_lat = np.radians(41.9)
        self.lon_scale = np.cos(mean_lat)

    def project(self, lat: float, lon: float) -> float:
        """
        Projects a (lat, lon) point onto the shape and returns the distance traveled along the shape (meters).
        Uses KDTree for O(logN) lookup + Vector Projection on nearest segments.
        """
        # 1. Find nearest vertex (Coarse Search)
        _, idx = self.tree.query([lat, lon])  # Returns distance, index

        # 2. Identify Candidate Segments (Previous and Next)
        # We need to check segment (idx-1 -> idx) and (idx -> idx+1)
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, idx))
        if idx < len(self.coords) - 1:
            candidates.append((idx, idx + 1))

        # If no candidates (single point shape?), return point distance
        if not candidates:
            return self.distances[idx]

        best_dist = self.distances[idx]
        min_perp_dist = float("inf")

        # 3. Project onto segments
        P = np.array([lon * self.lon_scale, lat])  # Note: XY order -> Lon, Lat

        for i_start, i_end in candidates:
            # Get Segment Endpoints in Flat Space
            A_lat, A_lon = self.coords[i_start]
            B_lat, B_lon = self.coords[i_end]

            A = np.array([A_lon * self.lon_scale, A_lat])
            B = np.array([B_lon * self.lon_scale, B_lat])

            # Vector A->B and A->P
            AB = B - A
            AP = P - A

            # Length squared of segment
            len_sq = np.dot(AB, AB)

            if len_sq == 0:
                continue  # Zero length segment

            # Project AP onto AB (Scalar projection factor t)
            t = np.dot(AP, AB) / len_sq

            # Clamp t to segment [0, 1]
            t_clamped = max(0, min(1, t))

            # Compute 'Shadow' point on line
            Shadow = A + t_clamped * AB

            # Distance from Point to Shadow (Perpendicular distance)
            perp_vec = P - Shadow
            perp_dist = np.dot(perp_vec, perp_vec)

            # If this is the closest segment so far, update result
            if perp_dist < min_perp_dist:
                min_perp_dist = perp_dist

                # Interpolate shape distance
                d_start = self.distances[i_start]
                d_end = self.distances[i_end]
                best_dist = d_start + t_clamped * (d_end - d_start)

        return float(best_dist)
