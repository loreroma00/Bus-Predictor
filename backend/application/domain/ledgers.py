"""
Ledger Types — Split the monolithic ledger into purpose-specific ledgers.

TopologyLedger:  Static physical network (routes, stops, shapes, trips).
ScheduleLedger:  Timetable index (route → direction → date → start times).
HistoricalLedger: Per-measurement observations, backed by database (append-only).
PredictedLedger:  Model predictions, backed by database (append-only).
VehicleHistoryLedger: Lazy per-vehicle served-trip history, backed by database.
"""

import json
import logging
import statistics
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from .static_data import Route, Shape, Trip

if TYPE_CHECKING:
    from .live_data import LiveTrip, Measurement, Schedule

logger = logging.getLogger(__name__)


# ============================================================
#  Topology Ledger
# ============================================================

@dataclass
class TopologyLedger:
    """Static physical network: routes, stops, shapes, and trips.

    Rebuilt only on GTFS update.  Pickle-cached.
    """

    routes: Dict[str, Route] = field(default_factory=dict)
    stops: Dict[str, dict] = field(default_factory=dict)
    shapes: Dict[str, Shape] = field(default_factory=dict)
    trips: Dict[str, Trip] = field(default_factory=dict)

    source_md5: Optional[str] = None

    # ---- convenience accessors ----

    def get_trip(self, trip_id: str) -> Optional[Trip]:
        """Return the Trip for ``trip_id`` or None."""
        return self.trips.get(trip_id)

    def get_stop(self, stop_id: str) -> Optional[dict]:
        """Return the stop metadata dict for ``stop_id`` or None."""
        return self.stops.get(stop_id)

    def get_route(self, route_id: str) -> Optional[Route]:
        """Return the Route for ``route_id`` or None."""
        return self.routes.get(route_id)

    def get_shape_for_trip(self, trip_id: str) -> Optional[Shape]:
        """Return the Shape attached to ``trip_id``, or None if the trip or its shape is missing."""
        trip = self.trips.get(trip_id)
        return trip.get_shape() if trip else None

    def build_stops_map(
        self, route_id: str, direction_id
    ) -> Dict[int, dict]:
        """Build stop_sequence → {stop_id, stop_name, stop_lat, stop_lon, shape_dist_travelled}.

        Used by validator, live_validator, and API endpoints.
        """
        stops_map: dict[int, dict] = {}
        for trip in list(self.trips.values()):
            if trip.route.id == route_id and trip.direction_id == direction_id:
                for st in trip.get_stop_times() or []:
                    seq = int(st.get("stop_sequence", 0) or 0)
                    if seq not in stops_map:
                        stop_id = st.get("stop_id")
                        stop_info = self.stops.get(stop_id, {})
                        shape_dist = st.get("shape_dist_traveled") or st.get(
                            "shape_dist_travelled"
                        )
                        stops_map[seq] = {
                            "stop_id": stop_id or "",
                            "stop_name": stop_info.get("stop_name", ""),
                            "stop_lat": float(stop_info.get("stop_lat", 0) or 0),
                            "stop_lon": float(stop_info.get("stop_lon", 0) or 0),
                            "shape_dist_travelled": float(shape_dist)
                            if shape_dist
                            else None,
                        }
                break  # first matching trip is sufficient
        return stops_map


# ============================================================
#  Schedule Ledger
# ============================================================

@dataclass
class ScheduleLedger:
    """Timetable index: route → direction → date → [start times].

    Rebuilt only on GTFS update.  Pickle-cached.
    """

    schedule: "Schedule" = None
    source_md5: Optional[str] = None

    def get_times(self, route_id: str, direction_id, date_str: str) -> list:
        """Sorted list of scheduled start times (Unix timestamps)."""
        if self.schedule is None:
            return []
        return self.schedule.get(route_id, direction_id, date_str)


# ============================================================
#  Historical Ledger  (measurement-level, database-backed)
# ============================================================

@dataclass
class MeasurementRecord:
    """A single observed measurement (one GPS ping).

    Stores raw data from each ping.  Derived features (sin/cos encodings,
    deposit flags, day_type, rush_hour, far_status) are computed during
    preprocessing — only raw values are persisted here.
    """

    # Identity
    trip_id: str
    route_id: str
    direction_id: int
    vehicle_id: str

    # Position
    latitude: float
    longitude: float
    hexagon_id: str
    stop_sequence: int
    shape_dist_travelled: float     # projected distance along shape (metres)
    distance_to_next_stop: float
    is_in_preferential: bool

    # Time
    measurement_time: float         # Unix timestamp
    actual_start_time: float        # Unix ts of first non-terminus ping

    # Schedule
    schedule_adherence: float       # delay in seconds
    scheduled_start_time: str       # HH:MM:SS
    delay_genuine: int

    # Speed / Traffic
    current_speed: float            # GPS speed or derived (km/h)
    speed_ratio: float              # current / free-flow
    current_traffic_speed: float    # free-flow speed (km/h)

    # Weather (full)
    temperature: float              # °C
    apparent_temperature: float     # °C (feels-like)
    humidity: float                 # %
    precipitation: float            # mm/h
    wind_speed: float               # m/s
    weather_code: int               # WMO code

    # Vehicle / occupancy
    bus_type: int
    door_number: int
    occupancy_status: int
    deposits: str                   # JSON list of depot names


# Keep backward-compat alias so existing imports don't break
StopArrival = MeasurementRecord


class HistoricalLedger:
    """Append-only record of per-measurement observations, backed by database.

    Maintains an in-memory buffer keyed by trip_id for fast GUI access.
    DB is used as persistent storage and as a bootstrap source on restart.
    """

    def __init__(self, connection_string: str = None, table_name: str = None):
        """Bind the ledger to a DB connection and target table; defaults come from config."""
        from config import Ledger
        self._conn_str = connection_string or Ledger.DB_CONNECTION
        self._table = table_name or Ledger.HISTORICAL_TABLE
        self._today_by_trip: Dict[str, list[dict]] = {}
        self._pending_db_rows: list[dict] = []

    def record_measurements(self, records: list[MeasurementRecord]):
        """Append a batch of measurement records to the in-memory buffer."""
        if not records:
            return []
        rows = [
            {
                "trip_id": r.trip_id,
                "route_id": r.route_id,
                "direction_id": r.direction_id,
                "vehicle_id": r.vehicle_id,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "hexagon_id": r.hexagon_id,
                "stop_sequence": r.stop_sequence,
                "shape_dist_travelled": r.shape_dist_travelled,
                "distance_to_next_stop": r.distance_to_next_stop,
                "is_in_preferential": r.is_in_preferential,
                "measurement_time": r.measurement_time,
                "actual_start_time": r.actual_start_time,
                "schedule_adherence": r.schedule_adherence,
                "scheduled_start_time": r.scheduled_start_time,
                "delay_genuine": r.delay_genuine,
                "current_speed": r.current_speed,
                "speed_ratio": r.speed_ratio,
                "current_traffic_speed": r.current_traffic_speed,
                "temperature": r.temperature,
                "apparent_temperature": r.apparent_temperature,
                "humidity": r.humidity,
                "precipitation": r.precipitation,
                "wind_speed": r.wind_speed,
                "weather_code": r.weather_code,
                "bus_type": r.bus_type,
                "door_number": r.door_number,
                "occupancy_status": r.occupancy_status,
                "deposits": r.deposits,
            }
            for r in records
        ]

        # In-memory buffer
        for row in rows:
            tid = row["trip_id"]
            if tid not in self._today_by_trip:
                self._today_by_trip[tid] = []
            self._today_by_trip[tid].append(row)

        self._pending_db_rows.extend(rows)
        return rows

    def push_to_db(self):
        """Push pending measurement rows to the database layer."""
        if not self._pending_db_rows:
            return
        from persistence.database import write_historical
        write_historical(self._conn_str, self._table, self._pending_db_rows)
        self._pending_db_rows = []

    # Keep old method as alias for backward compat
    record_arrivals = record_measurements

    def get_trip_measurements(self, trip_id: str) -> list[dict]:
        """Return in-memory measurements for a trip (for GUI detail popup)."""
        return self._today_by_trip.get(trip_id, [])

    def get_today_trip_count(self) -> int:
        """Return count of trips recorded today (in memory)."""
        return len(self._today_by_trip)

    def query(
        self,
        trip_id: str = None,
        route_id: str = None,
        date_start: float = None,
        date_end: float = None,
    ) -> pd.DataFrame:
        """Query historical measurements from DB.  Returns empty DataFrame if no data."""
        from persistence.database import read_historical
        return read_historical(
            self._conn_str, self._table,
            trip_id=trip_id, route_id=route_id,
            date_start=date_start, date_end=date_end,
        )


# ============================================================
#  Predicted Ledger  (database-backed, append-only)
# ============================================================

def _dd_mm_yyyy_to_iso(date_str: str) -> str:
    """Convert 'DD-MM-YYYY' → 'YYYY-MM-DD' for SQL DATE comparisons."""
    from datetime import datetime
    return datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")

@dataclass
class StopPredictionRecord:
    """A single predicted arrival at a stop."""

    route_id: str
    direction_id: int
    trip_date: str              # YYYY-MM-DD
    scheduled_start: str        # HH:MM or HH:MM:SS
    stop_id: str
    stop_sequence: int
    predicted_arrival: str      # HH:MM:SS (from model)
    predicted_delay_sec: float
    predicted_crowd_level: int
    prediction_timestamp: float # when the prediction was made (Unix)


class PredictedLedger:
    """Append-only record of model predictions, backed by database.

    Maintains in-memory buffers for fast GUI access:
    - _today_trips: one summary dict per trip (for table view)
    - _today_stops: per-stop records keyed by trip key (for detail popup)
    DB is used as persistent storage and as a bootstrap source on restart.
    """

    def __init__(self, connection_string: str = None, table_name: str = None):
        """Bind the ledger to a DB connection and table; defaults come from config."""
        from config import Ledger
        self._conn_str = connection_string or Ledger.DB_CONNECTION
        self._table = table_name or Ledger.PREDICTED_TABLE
        # In-memory buffers
        self._today_trips: list[dict] = []
        self._today_stops: Dict[str, list[dict]] = {}  # key: "route_dir_start"
        self._today_trip_keys: set = set()  # for dedup
        self._pending_db_rows: list[dict] = []

    def _trip_key(self, route_id, direction_id, scheduled_start):
        """Compose the composite in-memory key used to deduplicate trip summaries."""
        return f"{route_id}_{direction_id}_{scheduled_start}"

    def record_predictions(self, predictions: list[StopPredictionRecord]):
        """Append prediction records to the in-memory buffers."""
        if not predictions:
            return []
        records = [
            {
                "route_id": p.route_id,
                "direction_id": p.direction_id,
                "trip_date": p.trip_date,
                "scheduled_start": p.scheduled_start,
                "stop_id": p.stop_id,
                "stop_sequence": p.stop_sequence,
                "predicted_arrival": p.predicted_arrival,
                "predicted_delay_sec": p.predicted_delay_sec,
                "predicted_crowd_level": p.predicted_crowd_level,
                "prediction_timestamp": p.prediction_timestamp,
            }
            for p in predictions
        ]

        # Populate in-memory buffers
        # Group by trip key to build summary + stops
        from collections import defaultdict
        grouped = defaultdict(list)
        for rec in records:
            key = self._trip_key(rec["route_id"], rec["direction_id"], rec["scheduled_start"])
            grouped[key].append(rec)

        for key, stop_records in grouped.items():
            # Store per-stop data (overwrite if same trip predicted again)
            self._today_stops[key] = sorted(stop_records, key=lambda r: r["stop_sequence"])

            # Build/update trip summary
            delays = [r["predicted_delay_sec"] for r in stop_records]
            first = stop_records[0]
            summary = {
                "route_id": first["route_id"],
                "direction_id": first["direction_id"],
                "trip_date": first["trip_date"],
                "scheduled_start": first["scheduled_start"],
                "stop_count": len(stop_records),
                "avg_delay": round(sum(delays) / len(delays), 1) if delays else 0,
                "max_delay": round(max(delays), 1) if delays else 0,
                "prediction_timestamp": first["prediction_timestamp"],
            }

            if key in self._today_trip_keys:
                # Update existing summary
                for i, t in enumerate(self._today_trips):
                    if self._trip_key(t["route_id"], t["direction_id"], t["scheduled_start"]) == key:
                        self._today_trips[i] = summary
                        break
            else:
                self._today_trips.append(summary)
                self._today_trip_keys.add(key)

        self._pending_db_rows.extend(records)
        return records

    def push_to_db(self):
        """Push pending prediction rows to the database layer."""
        if not self._pending_db_rows:
            return
        from persistence.database import write_predicted
        write_predicted(self._conn_str, self._table, self._pending_db_rows)
        self._pending_db_rows = []

    def get_today_predictions(self) -> list[dict]:
        """Return trip summaries from in-memory buffer (for GUI table)."""
        return self._today_trips

    def get_trip_stops(self, route_id: str, direction_id: int, scheduled_start: str) -> list[dict]:
        """Return per-stop prediction records for a specific trip (for detail popup)."""
        key = self._trip_key(route_id, direction_id, scheduled_start)
        return self._today_stops.get(key, [])

    def query(
        self,
        route_id: str = None,
        trip_date: str = None,       # "DD-MM-YYYY" — converted to ISO for SQL
    ) -> pd.DataFrame:
        """Query predicted arrivals for a route/date from DB."""
        from persistence.database import read_predicted
        trip_date_iso = _dd_mm_yyyy_to_iso(trip_date) if trip_date else None
        return read_predicted(
            self._conn_str, self._table,
            route_id=route_id, trip_date=trip_date_iso,
        )

    def query_trip(
        self,
        route_id: str,
        direction_id: int,
        trip_date: str,       # "DD-MM-YYYY"
        scheduled_start: str, # "HH:MM"
    ) -> pd.DataFrame:
        """Return all stop predictions for one specific trip, or empty DataFrame."""
        from persistence.database import read_predicted
        trip_date_iso = _dd_mm_yyyy_to_iso(trip_date)
        return read_predicted(
            self._conn_str, self._table,
            route_id=route_id,
            direction_id=direction_id,
            trip_date=trip_date_iso,
            scheduled_start=scheduled_start,
        )


# ============================================================
#  Extraction Utility  (LiveTrip -> measurement records)
# ============================================================

def extract_measurements_from_live_trip(
    live_trip: "LiveTrip",
    route_id: str = None,
) -> list[MeasurementRecord]:
    """Convert live-trip measurements into MeasurementRecords for the historical ledger.

    Each measurement is projected onto the trip's shape to compute
    ``shape_dist_travelled``.  All raw fields are preserved so that
    the preprocessing pipeline can derive the training features later.

    Returns one MeasurementRecord per ping, or empty list if data is
    insufficient.
    """
    trip = live_trip.trip if live_trip else None
    if not live_trip or not live_trip.measurements or not trip:
        return []

    shape = trip.shape
    direction_id = trip.direction_id or 0
    route_id = route_id or (trip.route.id if trip.route else "")
    vehicle_id = str(live_trip.vehicle.label or live_trip.vehicle.id or "")
    scheduled_start = live_trip.scheduled_start_time or ""

    # Derive actual_start_time: first measurement NOT at stop_sequence 1
    # (i.e. the bus has left the terminus).
    actual_start_time = live_trip.actual_start_time or 0.0
    if not actual_start_time:
        for m in live_trip.measurements:
            seq = (
                m.gpsdata.current_stop_sequence
                if hasattr(m.gpsdata, "current_stop_sequence")
                else None
            )
            if seq is not None and seq > 1:
                actual_start_time = m.measurement_time
                break
        if not actual_start_time and live_trip.measurements:
            actual_start_time = live_trip.measurements[0].measurement_time

    records: list[MeasurementRecord] = []
    for m in live_trip.measurements:
        # Project GPS position onto shape (if available)
        if shape:
            try:
                shape_dist = shape.project(m.gpsdata.latitude, m.gpsdata.longitude)
            except Exception:
                shape_dist = 0.0
        else:
            shape_dist = 0.0

        # Current speed: prefer GPS, fall back to derived
        gps_speed = m.gpsdata.speed if m.gpsdata else 0.0
        current_speed = gps_speed if gps_speed else (m.derived_speed or 0.0)

        # Weather
        w = m.weather
        temperature = w.temperature if w else 0.0
        apparent_temperature = w.apparent_temperature if w else 0.0
        humidity = w.humidity if w else 0.0
        precipitation = w.precip_intensity if w else 0.0
        wind_speed = w.wind_speed if w else 0.0
        weather_code = w.weather_code if w else 0

        records.append(
            MeasurementRecord(
                trip_id=live_trip.trip_id,
                route_id=route_id,
                direction_id=direction_id,
                vehicle_id=vehicle_id,
                latitude=m.gpsdata.latitude,
                longitude=m.gpsdata.longitude,
                hexagon_id=m.hexagon_id or "",
                stop_sequence=m.gpsdata.current_stop_sequence or 0
                if hasattr(m.gpsdata, "current_stop_sequence")
                else 0,
                shape_dist_travelled=shape_dist,
                distance_to_next_stop=m.next_stop_distance or 0.0,
                is_in_preferential=bool(getattr(m, "is_in_preferential", False)),
                measurement_time=m.measurement_time,
                actual_start_time=actual_start_time,
                schedule_adherence=m.schedule_adherence or 0.0,
                scheduled_start_time=scheduled_start,
                delay_genuine=getattr(m, "delay_genuine", 0),
                current_speed=current_speed,
                speed_ratio=m.speed_ratio or 1.0,
                current_traffic_speed=m.current_speed or 0.0,
                temperature=temperature,
                apparent_temperature=apparent_temperature,
                humidity=humidity,
                precipitation=precipitation,
                wind_speed=wind_speed,
                weather_code=weather_code,
                bus_type=getattr(m, "bus_type", 0),
                door_number=getattr(m, "door_number", 0),
                occupancy_status=m.occupancy_status or 0,
                deposits=json.dumps(getattr(m, "deposits", []) or []),
            )
        )

    return records


def project_live_trip_to_measurements(live_trip: "LiveTrip") -> list[MeasurementRecord]:
    """Return measurement records for a completed live trip."""
    return extract_measurements_from_live_trip(live_trip)


# ============================================================
#  Vehicle History Ledger  (database-backed, append-only + lazy cache)
# ============================================================

@dataclass
class VehicleTripRecord:
    """One completed trip as observed from a specific vehicle."""

    # Identity
    vehicle_id: str
    trip_id: str
    route_id: str
    direction_id: int

    # Vehicle characteristics (denormalized for analytics)
    vehicle_type_name: str
    fuel_type: int          # FuelType enum value (0=Diesel, 1=Electric, …)
    euro_class: int         # EuroType enum value
    capacity_total: int

    # Timing
    trip_date: str          # YYYY-MM-DD
    scheduled_start: str    # HH:MM:SS
    actual_start_time: float  # Unix timestamp
    trip_end_time: float      # Unix timestamp
    trip_duration_sec: float

    # Delay summary (computed from valid measurements only)
    mean_delay_sec: float
    median_delay_sec: float
    max_delay_sec: float
    min_delay_sec: float
    std_delay_sec: float

    # Occupancy summary
    mean_occupancy: float
    max_occupancy: int

    # Trip quality
    measurement_count: int
    preferential_ratio: float  # fraction in bus lanes (0.0–1.0)

    # Metadata
    recorded_at: float      # Unix timestamp


class VehicleHistoryLedger:
    """Mini-ledger for per-vehicle served-trip history.

    New completed trips are appended to the DB. Per-vehicle history is loaded
    lazily and cached to avoid repeated reads for static Vehicle objects.
    """

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
        """Bind the ledger to a DB connection and table; defaults come from config."""
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
        rows = [{f: getattr(r, f) for f in self._FIELDS} for r in records]
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
        """Return all vehicle trip records from in-memory buffer (for GUI table)."""
        return self._today_records

    def get_history(self, vehicle_id: str, force_refresh: bool = False) -> list[dict]:
        """Return cached served-trip history for one vehicle, refreshing lazily."""
        now = _time.time()
        cached = self._history_cache.get(str(vehicle_id))
        if (
            cached
            and not force_refresh
            and now - cached[0] <= self._history_ttl_seconds
        ):
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
    ) -> pd.DataFrame:
        """Query vehicle trip records with optional filters."""
        from persistence.database import read_vehicle_trips
        return read_vehicle_trips(
            self._conn_str, self._table,
            vehicle_id=vehicle_id, route_id=route_id,
            fuel_type=fuel_type, date_start=date_start, date_end=date_end,
        )


# ============================================================
#  Projection Utility  (LiveTrip -> vehicle history ledger)
# ============================================================

def summarize_live_trip_for_vehicle(
    live_trip: "LiveTrip",
    route_id: str = None,
    direction_id: int = None,
    vehicle_type_name: str = "Unknown",
) -> Optional[VehicleTripRecord]:
    """Summarize a completed live trip into a vehicle trip performance record.

    Returns None if the live trip has no measurements.
    """
    if not live_trip or not live_trip.measurements:
        return None

    # ---- Vehicle identity ----
    vehicle = live_trip.vehicle
    vehicle_id = str(getattr(vehicle, "label", "?") or "?")

    vt = getattr(vehicle, "vehicle_type", None)
    fuel_type = 0
    euro_class = 0
    capacity_total = 0
    if vt:
        engine = getattr(vt, "engine", None)
        if engine:
            fuel_type = getattr(getattr(engine, "fuel", None), "value", 0) or 0
            euro_class = getattr(getattr(engine, "euro", None), "value", 0) or 0
        capacity_total = getattr(vt, "capacity_total", 0) or 0

    # ---- Timing ----
    first_m = live_trip.measurements[0]
    last_m = live_trip.measurements[-1]
    actual_start = live_trip.actual_start_time if live_trip.actual_start_time else first_m.measurement_time
    trip_end = last_m.measurement_time
    trip_duration = trip_end - actual_start if actual_start else 0.0
    trip_date = datetime.fromtimestamp(actual_start).strftime("%Y-%m-%d") if actual_start else ""
    scheduled_start = live_trip.scheduled_start_time or ""

    # ---- Delay statistics (filter invalid) ----
    valid_delays = [
        m.schedule_adherence for m in live_trip.measurements
        if m.schedule_adherence is not None
        and m.schedule_adherence != -1000.0
        and abs(m.schedule_adherence) <= 7200
    ]
    if valid_delays:
        mean_delay = statistics.mean(valid_delays)
        median_delay = statistics.median(valid_delays)
        max_delay = max(valid_delays)
        min_delay = min(valid_delays)
        std_delay = statistics.stdev(valid_delays) if len(valid_delays) > 1 else 0.0
    else:
        mean_delay = median_delay = max_delay = min_delay = std_delay = 0.0

    # ---- Occupancy ----
    occ_values = [
        m.occupancy_status for m in live_trip.measurements
        if m.occupancy_status is not None
    ]
    mean_occupancy = statistics.mean(occ_values) if occ_values else 0.0
    max_occupancy = max(occ_values) if occ_values else 0

    # ---- Preferential ratio ----
    pref_count = sum(1 for m in live_trip.measurements if getattr(m, "is_in_preferential", False))
    preferential_ratio = pref_count / len(live_trip.measurements)

    trip = live_trip.trip
    route_id = route_id or (trip.route.id if trip and trip.route else "")
    direction_id = direction_id if direction_id is not None else (trip.direction_id if trip else 0)
    if vehicle_type_name == "Unknown" and vt:
        vehicle_type_name = vt.name

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
