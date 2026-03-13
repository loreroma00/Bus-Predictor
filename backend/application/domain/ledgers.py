"""
Ledger Types — Split the monolithic ledger into purpose-specific ledgers.

TopologyLedger:  Static physical network (routes, stops, shapes, trips).
ScheduleLedger:  Timetable index (route → direction → date → start times).
HistoricalLedger: Observed stop arrivals, backed by database (append-only).
PredictedLedger:  Model predictions, backed by database (append-only).
VehicleLedger:   Per-vehicle trip performance, backed by database (append-only).
"""

import logging
import statistics
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, TYPE_CHECKING

import pandas as pd

from .static_data import Route, Shape, Trip

if TYPE_CHECKING:
    from .live_data import Schedule
    from .observers import Diary

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
        return self.trips.get(trip_id)

    def get_stop(self, stop_id: str) -> Optional[dict]:
        return self.stops.get(stop_id)

    def get_route(self, route_id: str) -> Optional[Route]:
        return self.routes.get(route_id)

    def get_shape_for_trip(self, trip_id: str) -> Optional[Shape]:
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
#  Historical Ledger  (parquet-backed, append-only)
# ============================================================

@dataclass
class StopArrival:
    """A single observed arrival at a stop."""

    trip_id: str
    stop_id: str
    stop_sequence: int
    actual_arrival_time: float   # Unix timestamp
    schedule_adherence: float    # delay in seconds
    occupancy_status: int
    vehicle_id: str = ""         # fleet number (default for backward compat)


class HistoricalLedger:
    """Append-only record of observed stop arrivals, backed by database."""

    def __init__(self, connection_string: str = None, table_name: str = None):
        from config import Ledger
        self._conn_str = connection_string or Ledger.DB_CONNECTION
        self._table = table_name or Ledger.HISTORICAL_TABLE

    def record_arrivals(self, arrivals: list[StopArrival]):
        """Append a batch of stop arrivals to the database."""
        if not arrivals:
            return
        from persistence.ledger_db import write_historical
        records = [
            {
                "trip_id": a.trip_id,
                "stop_id": a.stop_id,
                "stop_sequence": a.stop_sequence,
                "actual_arrival_time": a.actual_arrival_time,
                "schedule_adherence": a.schedule_adherence,
                "occupancy_status": a.occupancy_status,
                "vehicle_id": a.vehicle_id,
            }
            for a in arrivals
        ]
        write_historical(self._conn_str, self._table, records)

    def query(
        self,
        trip_id: str = None,
        date_start: float = None,
        date_end: float = None,
    ) -> pd.DataFrame:
        """Query historical arrivals.  Returns empty DataFrame if no data."""
        from persistence.ledger_db import read_historical
        return read_historical(
            self._conn_str, self._table,
            trip_id=trip_id, date_start=date_start, date_end=date_end,
        )


# ============================================================
#  Predicted Ledger  (parquet-backed, append-only)
# ============================================================

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
    """Append-only record of model predictions, backed by database."""

    def __init__(self, connection_string: str = None, table_name: str = None):
        from config import Ledger
        self._conn_str = connection_string or Ledger.DB_CONNECTION
        self._table = table_name or Ledger.PREDICTED_TABLE

    def record_predictions(self, predictions: list[StopPredictionRecord]):
        """Append prediction records to the database."""
        if not predictions:
            return
        from persistence.ledger_db import write_predicted
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
        write_predicted(self._conn_str, self._table, records)

    def query(
        self,
        route_id: str = None,
        trip_date: str = None,
    ) -> pd.DataFrame:
        """Query predicted arrivals."""
        from persistence.ledger_db import read_predicted
        return read_predicted(
            self._conn_str, self._table,
            route_id=route_id, trip_date=trip_date,
        )


# ============================================================
#  Projection Utility  (diary → historical ledger)
# ============================================================

def project_diary_to_stops(diary: "Diary", trip: Trip) -> list[StopArrival]:
    """Project diary measurements to the nearest stops.

    For each stop in the trip, finds the measurement whose GPS position
    projects closest to that stop on the shape, and records its
    measurement_time as the actual arrival time.

    Returns one StopArrival per stop (at most).
    """
    if (
        not diary.measurements
        or not trip
        or not trip.stop_times
        or not trip.shape
    ):
        return []

    shape = trip.shape

    # Extract vehicle ID from observer
    vehicle_id = ""
    if hasattr(diary, "observer") and diary.observer:
        vehicle = getattr(diary.observer, "assignedVehicle", None)
        if vehicle:
            vehicle_id = str(getattr(vehicle, "label", "") or "")

    # Pre-project all measurements onto the shape
    projected = []
    for m in diary.measurements:
        dist = shape.project(m.gpsdata.latitude, m.gpsdata.longitude)
        projected.append((dist, m))

    arrivals: list[StopArrival] = []
    for st in trip.stop_times:
        stop_dist = float(st.get("shape_dist_traveled", 0) or 0)
        stop_id = st.get("stop_id")
        stop_seq = int(st.get("stop_sequence", 0) or 0)

        # Find measurement with minimum |projected_dist – stop_dist|
        best_m = None
        best_delta = float("inf")
        for dist, m in projected:
            delta = abs(dist - stop_dist)
            if delta < best_delta:
                best_delta = delta
                best_m = m

        if best_m is not None:
            arrivals.append(
                StopArrival(
                    trip_id=diary.trip_id,
                    stop_id=stop_id,
                    stop_sequence=stop_seq,
                    actual_arrival_time=best_m.measurement_time,
                    schedule_adherence=best_m.schedule_adherence,
                    occupancy_status=best_m.occupancy_status
                    if best_m.occupancy_status is not None
                    else 0,
                    vehicle_id=vehicle_id,
                )
            )

    return arrivals


# ============================================================
#  Vehicle Ledger  (parquet-backed, append-only)
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


class VehicleLedger:
    """Append-only record of per-vehicle trip performance, backed by database."""

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
        from config import Ledger
        self._conn_str = connection_string or Ledger.DB_CONNECTION
        self._table = table_name or Ledger.VEHICLE_TABLE

    def record_trip(self, record: VehicleTripRecord):
        """Append a single vehicle trip record."""
        self.record_trips([record])

    def record_trips(self, records: list[VehicleTripRecord]):
        """Append vehicle trip records to the database."""
        if not records:
            return
        from persistence.ledger_db import write_vehicle_trips
        rows = [{f: getattr(r, f) for f in self._FIELDS} for r in records]
        write_vehicle_trips(self._conn_str, self._table, rows)

    def query(
        self,
        vehicle_id: str = None,
        route_id: str = None,
        fuel_type: int = None,
        date_start: str = None,
        date_end: str = None,
    ) -> pd.DataFrame:
        """Query vehicle trip records with optional filters."""
        from persistence.ledger_db import read_vehicle_trips
        return read_vehicle_trips(
            self._conn_str, self._table,
            vehicle_id=vehicle_id, route_id=route_id,
            fuel_type=fuel_type, date_start=date_start, date_end=date_end,
        )


# ============================================================
#  Projection Utility  (diary → vehicle ledger)
# ============================================================

def summarize_diary_for_vehicle(
    diary: "Diary",
    route_id: str,
    direction_id: int,
    vehicle_type_name: str = "Unknown",
) -> Optional[VehicleTripRecord]:
    """Summarize a completed diary into a vehicle trip performance record.

    Returns None if the diary has no measurements.
    """
    if not diary.measurements:
        return None

    # ---- Vehicle identity ----
    vehicle = getattr(diary.observer, "assignedVehicle", None) if diary.observer else None
    vehicle_id = str(getattr(vehicle, "label", "?") or "?")

    vt = getattr(vehicle, "vehicle_type", None) if vehicle else None
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
    first_m = diary.measurements[0]
    last_m = diary.measurements[-1]
    actual_start = diary.actual_start_time if diary.actual_start_time else first_m.measurement_time
    trip_end = last_m.measurement_time
    trip_duration = trip_end - actual_start if actual_start else 0.0
    trip_date = datetime.fromtimestamp(actual_start).strftime("%Y-%m-%d") if actual_start else ""
    scheduled_start = diary.scheduled_start_time or ""

    # ---- Delay statistics (filter invalid) ----
    valid_delays = [
        m.schedule_adherence for m in diary.measurements
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
        m.occupancy_status for m in diary.measurements
        if m.occupancy_status is not None
    ]
    mean_occupancy = statistics.mean(occ_values) if occ_values else 0.0
    max_occupancy = max(occ_values) if occ_values else 0

    # ---- Preferential ratio ----
    pref_count = sum(1 for m in diary.measurements if getattr(m, "is_in_preferential", False))
    preferential_ratio = pref_count / len(diary.measurements)

    return VehicleTripRecord(
        vehicle_id=vehicle_id,
        trip_id=diary.trip_id,
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
        measurement_count=len(diary.measurements),
        preferential_ratio=preferential_ratio,
        recorded_at=_time.time(),
    )
