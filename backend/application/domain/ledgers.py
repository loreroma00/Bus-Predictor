"""
Ledger Types — Split the monolithic ledger into four purpose-specific ledgers.

TopologyLedger:  Static physical network (routes, stops, shapes, trips).
ScheduleLedger:  Timetable index (route → direction → date → start times).
HistoricalLedger: Observed stop arrivals, backed by parquet (append-only).
PredictedLedger:  Model predictions, backed by parquet (append-only).
"""

import logging
import os
from dataclasses import dataclass, field
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
        for trip in self.trips.values():
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


class HistoricalLedger:
    """Append-only record of observed stop arrivals, backed by parquet."""

    def __init__(self, storage_dir: str = "ledgers/historical"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._file_path = os.path.join(storage_dir, "historical_arrivals.parquet")

    def record_arrivals(self, arrivals: list[StopArrival]):
        """Append a batch of stop arrivals to the parquet store."""
        if not arrivals:
            return
        records = [
            {
                "trip_id": a.trip_id,
                "stop_id": a.stop_id,
                "stop_sequence": a.stop_sequence,
                "actual_arrival_time": a.actual_arrival_time,
                "schedule_adherence": a.schedule_adherence,
                "occupancy_status": a.occupancy_status,
            }
            for a in arrivals
        ]
        df = pd.DataFrame(records)
        if os.path.exists(self._file_path) and os.path.getsize(self._file_path) > 0:
            existing = pd.read_parquet(self._file_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["trip_id", "stop_id", "actual_arrival_time"], keep="last"
            )
            combined.to_parquet(self._file_path, engine="pyarrow")
        else:
            df.to_parquet(self._file_path, engine="pyarrow")

    def query(
        self,
        trip_id: str = None,
        date_start: float = None,
        date_end: float = None,
    ) -> pd.DataFrame:
        """Query historical arrivals.  Returns empty DataFrame if no data."""
        if not os.path.exists(self._file_path):
            return pd.DataFrame()
        df = pd.read_parquet(self._file_path)
        if trip_id:
            df = df[df["trip_id"] == trip_id]
        if date_start is not None:
            df = df[df["actual_arrival_time"] >= date_start]
        if date_end is not None:
            df = df[df["actual_arrival_time"] < date_end]
        return df


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
    """Append-only record of model predictions, backed by parquet."""

    def __init__(self, storage_dir: str = "ledgers/predicted"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._file_path = os.path.join(storage_dir, "predicted_arrivals.parquet")

    def record_predictions(self, predictions: list[StopPredictionRecord]):
        """Append prediction records to the parquet store."""
        if not predictions:
            return
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
        df = pd.DataFrame(records)
        if os.path.exists(self._file_path) and os.path.getsize(self._file_path) > 0:
            existing = pd.read_parquet(self._file_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_parquet(self._file_path, engine="pyarrow")
        else:
            df.to_parquet(self._file_path, engine="pyarrow")

    def query(
        self,
        route_id: str = None,
        trip_date: str = None,
    ) -> pd.DataFrame:
        """Query predicted arrivals."""
        if not os.path.exists(self._file_path):
            return pd.DataFrame()
        df = pd.read_parquet(self._file_path)
        if route_id:
            df = df[df["route_id"] == route_id]
        if trip_date:
            df = df[df["trip_date"] == trip_date]
        return df


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
                )
            )

    return arrivals
