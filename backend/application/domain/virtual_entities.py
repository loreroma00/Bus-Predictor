"""
Observatory - Facade for all data processing operations.
Delegates to internal components while presenting a unified interface.
Uses Dependency Injection for external dependencies (e.g., caching).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

import pandas as pd

from .interfaces import CacheStrategy, GeocodingStrategy
from .static_data import Route, Shape, Trip, Vehicle, VehicleType
from .static_data_fetcher import StaticDataFetcher
from .cities import City
from .live_data import GPSData, LiveFeedRecord, LiveTrip, Update
from .fleet_loader import load_fleet
from application.services.persistence_gateway import get_persistence_gateway

if TYPE_CHECKING:
    from .live_data import Schedule


@dataclass
class TopologyLedger:
    """Static physical network: routes, stops, shapes, and trips."""

    routes: Dict[str, Route] = field(default_factory=dict)
    stops: Dict[str, dict] = field(default_factory=dict)
    shapes: Dict[str, Shape] = field(default_factory=dict)
    trips: Dict[str, Trip] = field(default_factory=dict)
    source_md5: Optional[str] = None

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
        """Return the Shape attached to ``trip_id``."""
        trip = self.trips.get(trip_id)
        return trip.get_shape() if trip else None

    def build_stops_map(self, route_id: str, direction_id) -> Dict[int, dict]:
        """Build stop_sequence -> stop metadata for one route/direction."""
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
                break
        return stops_map


@dataclass
class ScheduleLedger:
    """Timetable index: route -> direction -> date -> start times."""

    schedule: "Schedule" = None
    source_md5: Optional[str] = None

    def get_times(self, route_id: str, direction_id, date_str: str) -> list:
        """Sorted list of scheduled start times."""
        if self.schedule is None:
            return []
        return self.schedule.get(route_id, direction_id, date_str)


@dataclass
class MeasurementRecord:
    """A single observed measurement projected for historical persistence."""

    trip_id: str
    route_id: str
    direction_id: int
    vehicle_id: str
    latitude: float
    longitude: float
    hexagon_id: str
    stop_sequence: int
    shape_dist_travelled: float
    distance_to_next_stop: float
    is_in_preferential: bool
    measurement_time: float
    actual_start_time: float
    schedule_adherence: float
    scheduled_start_time: str
    delay_genuine: int
    current_speed: float
    speed_ratio: float
    current_traffic_speed: float
    temperature: float
    apparent_temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    weather_code: int
    bus_type: int
    door_number: int
    occupancy_status: int
    deposits: str


StopArrival = MeasurementRecord


class HistoricalLedger:
    """Append-only record of per-measurement observations owned by Observatory."""

    def __init__(
        self,
        connection_string: str = None,
        table_name: str = None,
        persistence_gateway=None,
    ):
        """Bind ledger state to its DB destination configuration."""
        from config import Ledger

        self._conn_str = connection_string or Ledger.DB_CONNECTION
        self._table = table_name or Ledger.HISTORICAL_TABLE
        self._persistence = persistence_gateway or get_persistence_gateway()
        self._today_by_trip: Dict[str, list[dict]] = {}
        self._pending_db_rows: list[dict] = []

    def record_measurements(self, records: list[MeasurementRecord]):
        """Append measurement records to the in-memory buffer."""
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

        for row in rows:
            tid = row["trip_id"]
            self._today_by_trip.setdefault(tid, []).append(row)

        self._pending_db_rows.extend(rows)
        return rows

    record_arrivals = record_measurements

    def push_to_db(self):
        """Push pending measurement rows to the database layer."""
        if not self._pending_db_rows:
            return
        self._persistence.write_historical_records(
            self._conn_str,
            self._table,
            self._pending_db_rows,
        )
        self._pending_db_rows = []

    def get_trip_measurements(self, trip_id: str) -> list[dict]:
        """Return in-memory measurements for a trip."""
        return self._today_by_trip.get(trip_id, [])

    def get_today_trip_count(self) -> int:
        """Return count of trips recorded today in memory."""
        return len(self._today_by_trip)

    def query(
        self,
        trip_id: str = None,
        route_id: str = None,
        date_start: float = None,
        date_end: float = None,
    ) -> pd.DataFrame:
        """Query historical measurements through the DB layer."""
        return self._persistence.read_historical_records(
            self._conn_str,
            self._table,
            trip_id=trip_id,
            route_id=route_id,
            date_start=date_start,
            date_end=date_end,
        )


def extract_measurements_from_live_trip(
    live_trip: "LiveTrip",
    route_id: str = None,
) -> list[MeasurementRecord]:
    """Convert live-trip measurements into historical ledger records."""
    trip = live_trip.trip if live_trip else None
    if not live_trip or not live_trip.measurements or not trip:
        return []

    shape = trip.shape
    direction_id = trip.direction_id or 0
    route_id = route_id or (trip.route.id if trip.route else "")
    vehicle_id = str(live_trip.vehicle.label or live_trip.vehicle.id or "")
    scheduled_start = live_trip.scheduled_start_time or ""

    actual_start_time = live_trip.actual_start_time or 0.0
    if not actual_start_time:
        for measurement in live_trip.measurements:
            seq = getattr(measurement.gpsdata, "current_stop_sequence", None)
            if seq is not None and seq > 1:
                actual_start_time = measurement.measurement_time
                break
        if not actual_start_time and live_trip.measurements:
            actual_start_time = live_trip.measurements[0].measurement_time

    records: list[MeasurementRecord] = []
    for measurement in live_trip.measurements:
        if shape:
            try:
                shape_dist = shape.project(
                    measurement.gpsdata.latitude,
                    measurement.gpsdata.longitude,
                )
            except Exception:
                shape_dist = 0.0
        else:
            shape_dist = 0.0

        gps_speed = measurement.gpsdata.speed if measurement.gpsdata else 0.0
        current_speed = gps_speed if gps_speed else (measurement.derived_speed or 0.0)
        weather = measurement.weather

        records.append(
            MeasurementRecord(
                trip_id=live_trip.trip_id,
                route_id=route_id,
                direction_id=direction_id,
                vehicle_id=vehicle_id,
                latitude=measurement.gpsdata.latitude,
                longitude=measurement.gpsdata.longitude,
                hexagon_id=measurement.hexagon_id or "",
                stop_sequence=getattr(
                    measurement.gpsdata,
                    "current_stop_sequence",
                    0,
                )
                or 0,
                shape_dist_travelled=shape_dist,
                distance_to_next_stop=measurement.next_stop_distance or 0.0,
                is_in_preferential=bool(
                    getattr(measurement, "is_in_preferential", False)
                ),
                measurement_time=measurement.measurement_time,
                actual_start_time=actual_start_time,
                schedule_adherence=measurement.schedule_adherence or 0.0,
                scheduled_start_time=scheduled_start,
                delay_genuine=getattr(measurement, "delay_genuine", 0),
                current_speed=current_speed,
                speed_ratio=measurement.speed_ratio or 1.0,
                current_traffic_speed=measurement.current_speed or 0.0,
                temperature=weather.temperature if weather else 0.0,
                apparent_temperature=weather.apparent_temperature if weather else 0.0,
                humidity=weather.humidity if weather else 0.0,
                precipitation=weather.precip_intensity if weather else 0.0,
                wind_speed=weather.wind_speed if weather else 0.0,
                weather_code=weather.weather_code if weather else 0,
                bus_type=getattr(measurement, "bus_type", 0),
                door_number=getattr(measurement, "door_number", 0),
                occupancy_status=measurement.occupancy_status or 0,
                deposits=json.dumps(getattr(measurement, "deposits", []) or []),
            )
        )

    return records


def project_live_trip_to_measurements(live_trip: "LiveTrip") -> list[MeasurementRecord]:
    """Return measurement records for a completed live trip."""
    return extract_measurements_from_live_trip(live_trip)


class Observatory:
    """
    Main entry point for data processing.
    Acts as a facade, delegating to specialized internal components.

    External dependencies (like cache) are injected, not imported directly.
    """

    def __init__(
        self,
        cache_strategy: "CacheStrategy" = None,
        geocoding_strategy: "GeocodingStrategy" = None,
        config: dict = None,
        persistence_gateway=None,
    ):
        """
        Initialize Observatory with optional injected dependencies.

        Args:
            cache_strategy: Implementation of CacheStrategy for ledger persistence.
                           If None, caching is disabled.
            geocoding_strategy: Implementation of GeocodingStrategy for street name
                               resolution. If None, geocoding is disabled.
            config: Configuration dictionary.
        """
        # Injected dependencies
        self._cache = cache_strategy
        self._geocoding = geocoding_strategy
        self._persistence = persistence_gateway or get_persistence_gateway()
        self.config = config or {}

        # Internal components (no external dependencies)
        from .ledger_builder import LedgerBuilder

        self._fetcher = StaticDataFetcher()
        self._builder = LedgerBuilder()
        # ---- Ledgers ----
        self.topology: TopologyLedger = None
        self.schedule_ledger: ScheduleLedger = None
        self.historical: HistoricalLedger = HistoricalLedger(
            persistence_gateway=self._persistence,
        )

        # Cache metadata
        self.current_md5: str = None
        self.last_update_check: float = 0

        # Cities & Fleet
        self.observed_cities: dict[str, City] = {}
        self.fleet = load_fleet("vehicles.csv")
        self.vehicles: dict[str, Vehicle] = {}
        self.live_trips_by_vehicle_id: dict[str, LiveTrip] = {}
        self.live_trips_by_trip_id: dict[str, LiveTrip] = {}
        self.completed_live_trips: list[LiveTrip] = []

    def add_city(self, city_name: str, static_bus_lanes: dict = None):
        """Add a city to the observed cities."""
        self.observed_cities[city_name] = City(city_name, static_bus_lanes)

    def get_city(self, city_name: str) -> City | None:
        """Get a city instance by name."""
        return self.observed_cities.get(city_name)

    def add_live_trip_to_city(
        self,
        city_name: str,
        live_trip: LiveTrip,
        latitude: float = None,
        longitude: float = None,
        hex_id: str = None,
    ):
        """Add a live trip to the city."""
        if hex_id is not None:
            self.observed_cities[city_name].add_live_trip_to_city(live_trip, hex_id=hex_id)
            live_trip.set_hexagon_id(hex_id)
        else:
            self.observed_cities[city_name].add_live_trip_to_city(
                live_trip, lat=latitude, lng=longitude
            )
            new_hex_id = self.observed_cities[city_name].get_hex_id(latitude, longitude)
            live_trip.set_hexagon_id(new_hex_id)

            # Trigger geocoding on creation if enabled
            if self._geocoding and latitude is not None:
                self._geocoding.enqueue(latitude, longitude, new_hex_id)
                street = self._geocoding.get_street(latitude, longitude)
                if street:
                    live_trip.set_location_name(street)

    def move_live_trip(
        self,
        city_name: str,
        live_trip: LiveTrip,
        new_hex_id: str = None,
        latitude: float = None,
        longitude: float = None,
    ):
        """Move a live trip in the city."""
        if new_hex_id is not None:
            self.observed_cities[city_name].move_live_trip(live_trip.id, new_hex_id)
            live_trip.set_hexagon_id(new_hex_id)
        else:
            self.observed_cities[city_name].move_live_trip(
                live_trip.id, latitude=latitude, longitude=longitude
            )
            new_hex_id = self.observed_cities[city_name].get_hex_id(latitude, longitude)
            live_trip.set_hexagon_id(new_hex_id)

            # Trigger geocoding if enabled
            if self._geocoding and latitude is not None:
                self._geocoding.enqueue(latitude, longitude, new_hex_id)
                street = self._geocoding.get_street(latitude, longitude)
                if street:
                    live_trip.set_location_name(street)

    def ingest_live_feed(
        self,
        records: list[LiveFeedRecord],
        city_name: str = "Rome",
    ) -> list[Update]:
        """Apply normalized GTFS-RT records to active LiveTrip aggregates."""
        topology = self.get_topology()
        updates: list[Update] = []
        processed_vehicles: set[str] = set()

        for record in records:
            try:
                vehicle_id = str(record.vehicle_id)
                if vehicle_id in processed_vehicles:
                    continue
                processed_vehicles.add(vehicle_id)

                trip = topology.get_trip(record.trip_id)
                if not trip and self.check_and_reload_ledger():
                    topology = self.get_topology()
                    trip = topology.get_trip(record.trip_id)

                if not trip:
                    logging.warning(
                        "Warning: Trip %s not found. Skipping.",
                        record.trip_id,
                    )
                    continue

                live_trip, status = self.get_or_create_live_trip(
                    vehicle_id=vehicle_id,
                    trip=trip,
                    label=record.vehicle_label,
                    scheduled_start_time=record.scheduled_start_time,
                )
                self._log_live_feed_status(record, live_trip, status)

                gps_data = self._create_gps_data(record, trip)
                live_trip.set_gps_data(gps_data)
                live_trip.set_occupancy_status(record.occupancy_status)
                live_trip.derive_speed()
                live_trip.derive_bearing()

                self._place_live_trip(city_name, live_trip, gps_data, status)

                update = Update(
                    live_trip=live_trip,
                    next_stops=record.stop_updates,
                )
                self._apply_live_feed_context(city_name, live_trip, update)
                updates.append(update)

            except Exception as e:
                label = record.vehicle_label or record.vehicle_id
                logging.error("Error processing update for %s: %s", label, e)
                continue

        return updates

    def _log_live_feed_status(
        self,
        record: LiveFeedRecord,
        live_trip: LiveTrip,
        status: int,
    ):
        """Log whether a GTFS-RT record created, changed, or updated a live trip."""
        display_id = record.vehicle_label or record.vehicle_id
        trip = live_trip.trip
        if status == 0:
            logging.info(
                "Found NEW LiveTrip: %s - %s %s",
                display_id,
                trip.route.id,
                trip.direction_name,
            )
        elif status == 2:
            logging.info("Detected trip change for %s: now %s", display_id, trip.id)
        else:
            logging.debug(
                "Update for LiveTrip: %s - %s %s",
                display_id,
                trip.route.id,
                trip.direction_name,
            )

    def _create_gps_data(self, record: LiveFeedRecord, trip) -> GPSData:
        """Create GPSData from a normalized feed record."""
        gps_id = f"{record.vehicle_id}_{record.timestamp}"
        return GPSData(
            id=gps_id,
            trip=trip,
            timestamp=record.timestamp,
            latitude=record.latitude,
            longitude=record.longitude,
            speed=record.speed,
            heading=record.bearing,
            next_stop_id=record.stop_id,
            current_stop_sequence=record.current_stop_sequence,
            current_status=record.current_status,
        )

    def _place_live_trip(
        self,
        city_name: str,
        live_trip: LiveTrip,
        gps_data: GPSData,
        status: int,
    ):
        """Create or move the city placement for a live trip."""
        if status in (0, 2):
            self.add_live_trip_to_city(
                city_name,
                live_trip,
                latitude=gps_data.latitude,
                longitude=gps_data.longitude,
            )
        else:
            self.move_live_trip(
                city_name,
                live_trip,
                latitude=gps_data.latitude,
                longitude=gps_data.longitude,
            )

    def _apply_live_feed_context(
        self,
        city_name: str,
        live_trip: LiveTrip,
        update: Update,
    ):
        """Compute city context and let LiveTrip append its measurement."""
        from application.domain.spatial_utils import get_cardinal_direction

        city = self.get_city(city_name)
        hexagon = city.get_hexagon(live_trip.get_hexagon_id()) if city else None
        if not hexagon:
            return

        bearing = live_trip.derived_bearing
        direction = get_cardinal_direction(bearing) if bearing else None
        weather = None
        try:
            weather = city.get_weather(live_trip.get_hexagon_id())
        except Exception:
            pass

        live_trip.apply_update(
            update,
            next_stop_distance=self.get_stop_distance(update),
            speed_ratio=hexagon.get_speed_ratio(direction),
            current_speed=hexagon.get_current_speed(direction),
            traffic_data_pending=hexagon.is_traffic_expired(),
            weather=weather,
        )

    # ==================== LEDGER OPERATIONS ====================

    def get_topology(self) -> TopologyLedger:
        """Public accessor for the topology ledger, building if necessary."""
        if self.topology is None:
            self._build_ledgers()
        return self.topology

    def get_schedule_ledger(self) -> ScheduleLedger:
        """Public accessor for the schedule ledger, building if necessary."""
        if self.schedule_ledger is None:
            self._build_ledgers()
        return self.schedule_ledger

    def _build_ledgers(self):
        """Orchestrates building both topology and schedule ledgers."""
        # 1. Fetch static data (download if needed)
        latest_md5 = self._fetcher.fetch()

        # 2. Try loading from cache (if cache strategy was injected)
        if self._cache:
            topology = self._cache.load_topology(expected_md5=latest_md5)
            schedule = self._cache.load_schedule(expected_md5=latest_md5)

            if topology is not None and schedule is not None:
                self.topology = topology
                self.schedule_ledger = schedule
                self.current_md5 = latest_md5
                return

        # 3. Build from CSVs
        self._builder.read_csvs()
        self.topology = self._builder.build_topology()
        self.schedule_ledger = self._builder.build_schedule(self.topology)

        # 4. Save to cache (if cache strategy was injected)
        if self._cache:
            self._cache.save_topology(self.topology, source_md5=latest_md5)
            self._cache.save_schedule(self.schedule_ledger, source_md5=latest_md5)

        self.current_md5 = latest_md5

    def _ensure_static_data_loaded(self):
        """Ensures topology + schedule are available."""
        if self.topology is None:
            self._build_ledgers()

    def check_and_reload_ledger(self) -> bool:
        """Hot-swap ledgers if a new version is available."""
        COOLDOWN = 300  # 5 minutes

        if time.time() - self.last_update_check < COOLDOWN:
            return False

        logging.info("Checking for static data updates...")
        self.last_update_check = time.time()

        try:
            latest_md5 = self._fetcher.fetch()

            if latest_md5 != self.current_md5:
                logging.info("Update Found! Hot swapping...")
                self.topology = None
                self.schedule_ledger = None
                self.get_topology()
                logging.info("Hot swap complete.")
                return True
            else:
                logging.info("Static data is up to date.")
                return False

        except Exception as e:
            logging.error(f"Error checking for updates: {e}")
            return False

    def search_trip(self, trip_id: str):
        """Search for a trip by ID."""
        if self.topology:
            return self.topology.get_trip(trip_id)
        return None

    # ==================== LIVE TRIP OPERATIONS ====================

    def get_or_create_vehicle(self, vehicle_id: str, label: str = None) -> Vehicle:
        """Return static Vehicle identity, creating it if needed."""
        vehicle_id = str(vehicle_id)
        display_id = str(label) if label else vehicle_id
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle:
            return vehicle

        vehicle_type = self.get_vehicle_type(display_id)
        if vehicle_type is None:
            logging.warning(
                "No VehicleType corresponding to ID %s; N/A values enabled.",
                display_id,
            )

        vehicle = Vehicle(
            id=vehicle_id,
            label=display_id,
            vehicle_type=vehicle_type,
            persistence_gateway=self._persistence,
        )
        self.vehicles[vehicle_id] = vehicle
        return vehicle

    def get_or_create_live_trip(
        self,
        vehicle_id: str,
        trip,
        label: str = None,
        scheduled_start_time: str = None,
    ) -> tuple[LiveTrip, int]:
        """Return active LiveTrip for a vehicle, finishing old trip on trip change.

        Status: 0 = newly created, 1 = existing same trip, 2 = trip changed.
        """
        vehicle_id = str(vehicle_id)
        current = self.live_trips_by_vehicle_id.get(vehicle_id)
        if current and current.trip.id == trip.id:
            return current, 1

        status = 0
        if current:
            status = 2
            self.finish_live_trip(current)

        vehicle = self.get_or_create_vehicle(vehicle_id, label=label)
        live_trip = LiveTrip(
            trip=trip,
            vehicle=vehicle,
            scheduled_start_time=scheduled_start_time,
        )
        self.live_trips_by_vehicle_id[vehicle_id] = live_trip
        self.live_trips_by_trip_id[trip.id] = live_trip
        self.update_fleet_count(vehicle.label)
        return live_trip, status

    def finish_live_trip(self, live_trip: LiveTrip):
        """Finish, archive, and unregister a live trip."""
        if not live_trip:
            return
        if not live_trip.is_finished:
            live_trip.finish()
        if live_trip not in self.completed_live_trips:
            self.completed_live_trips.append(live_trip)
        for city in self.observed_cities.values():
            try:
                city.remove_live_trip(live_trip)
            except Exception:
                pass
        self.live_trips_by_vehicle_id.pop(live_trip.vehicle_id, None)
        if self.live_trips_by_trip_id.get(live_trip.trip_id) is live_trip:
            self.live_trips_by_trip_id.pop(live_trip.trip_id, None)

    def get_active_live_trips(self) -> dict[str, LiveTrip]:
        """Return active live trips keyed by vehicle id."""
        return self.live_trips_by_vehicle_id

    def search_live_trip(self, trip_id: str) -> LiveTrip | None:
        """Search an active live trip by static trip id."""
        return self.live_trips_by_trip_id.get(trip_id)

    def search_completed_live_trip(self, trip_id: str) -> LiveTrip | None:
        """Search completed live trips by static trip id."""
        for live_trip in self.completed_live_trips:
            if live_trip.trip_id == trip_id:
                return live_trip
        return None

    def get_completed_measurements(self) -> list:
        """Return completed live-trip measurement dictionaries for persistence."""
        completed = []
        for live_trip in self.completed_live_trips:
            completed.extend(live_trip.to_dict_list())
        return completed

    def get_all_current_measurements(self) -> tuple:
        """Return active live-trip measurement dicts, count, active trip count."""
        all_measurements = []
        for live_trip in self.live_trips_by_vehicle_id.values():
            if live_trip.measurements:
                all_measurements.extend(live_trip.to_dict_list())
        return all_measurements, len(all_measurements), len(self.live_trips_by_vehicle_id)

    def get_vehicle_live_trips(self, vehicle_id: str) -> list[LiveTrip]:
        """Return completed and active live trips served by one vehicle."""
        vehicle_id = str(vehicle_id)
        live_trips = [
            lt for lt in self.completed_live_trips
            if lt.vehicle_id == vehicle_id or lt.vehicle.label == vehicle_id
        ]
        current = self.live_trips_by_vehicle_id.get(vehicle_id)
        if current:
            live_trips.append(current)
        return live_trips

    def get_id_by_label(self, label: str) -> str | None:
        """Find vehicle id by public label."""
        for vehicle in self.vehicles.values():
            if vehicle.label == label:
                return vehicle.id
        return None

    # ==================== UTILITY OPERATIONS ====================

    def get_stop_distance(self, update) -> float:
        """Calculate distance to next stop based on shape projection."""
        live_trip = update.get_live_trip()
        trip = self.search_trip(live_trip.trip.id)
        if not trip:
            return None

        shape = trip.get_shape()
        if not shape:
            return None

        gps = live_trip.get_gps_data()
        dist_travelled = shape.project(gps.get_latitude(), gps.get_longitude())

        next_stop = update.get_next_stop()
        if next_stop:
            for stop in trip.get_stop_times():
                if stop["stop_id"] == next_stop["stop_id"]:
                    return float(stop["shape_dist_traveled"]) - dist_travelled

        return None

    # ==================== TRIP ADHERENCE ANALYSIS ====================

    def scan_trip_adherence(
        self,
        route_id: str,
        time_window_start: float,
        time_window_end: float,
        direction_id: str = None,
        verification_strategy=None,
    ) -> float:
        """
        Calculate the ratio of trips served vs trips scheduled for a route.

        Args:
            route_id: The route to analyze
            time_window_start: Unix timestamp for window start
            time_window_end: Unix timestamp for window end
            direction_id: Optional direction filter
            verification_strategy: Strategy to verify trip validity
                                  (default: BasicTripVerification)

        Returns:
            Float ratio in range [0.0, 1.0] representing served/expected trips.
            Returns 1.0 if no trips were expected (prevents division by zero).
        """
        from .verification_strategies import BasicTripVerification

        if verification_strategy is None:
            verification_strategy = BasicTripVerification()

        self._ensure_static_data_loaded()

        # 1. Find expected trips for this route in the time window
        expected_trip_ids = self._get_expected_trips(
            route_id, time_window_start, time_window_end, direction_id
        )

        if not expected_trip_ids:
            return 1.0  # No trips expected = 100% adherence by definition

        # 2. Find served trips from diaries
        served_trip_ids = self._get_served_trip_ids(
            expected_trip_ids, verification_strategy
        )

        # 3. Calculate ratio
        served_count = len(served_trip_ids)
        expected_count = len(expected_trip_ids)

        return served_count / expected_count

    def get_vehicle_type(self, v_id: str) -> VehicleType | None:
        """Look up a VehicleType by ID, retrying with leading zeros stripped."""
        # 1. Strict lookup
        v_id_str = str(v_id)
        vt = self.fleet.get(v_id_str)
        if vt:
            return vt

        # 2. Fallback: Strip leading zeros (e.g. Feed "0839" -> Fleet "839")
        stripped = v_id_str.lstrip("0")
        if (
            stripped
        ):  # Ensure we don't look up an empty string if ID was just "0" or "00"
            vt = self.fleet.get(stripped)
            if vt:
                return vt

        return None

    def update_fleet_count(self, v_id: str):
        """Placeholder for dynamic fleet counting; currently unused."""
        # TODO: Implement dynamic fleet counting if needed
        pass

    def _get_expected_trips(
        self,
        route_id: str,
        time_window_start: float,
        time_window_end: float,
        direction_id: str = None,
    ) -> set:
        """
        Get set of trip IDs expected for a route in the time window.

        Uses first stop arrival time to determine if trip falls within window.
        """
        from .time_utils import to_unix_time
        from datetime import datetime

        expected = set()
        trips = self.topology.trips if self.topology else {}

        # Convert window to date string for efficient filtering
        local_tz = datetime.now().astimezone().tzinfo
        start_dt = datetime.fromtimestamp(time_window_start, tz=local_tz)
        end_dt = datetime.fromtimestamp(time_window_end, tz=local_tz)

        start_date_str = start_dt.strftime("%Y%m%d")
        end_date_str = end_dt.strftime("%Y%m%d")

        for trip_id, trip in trips.items():
            # Route filter
            if trip.route.id != route_id:
                continue

            # Direction filter (optional)
            if direction_id is not None and trip.direction_id != direction_id:
                continue

            # Date filter - trip must be active on a date within window
            if not trip.dates:
                continue

            trip_in_window = False
            for date_str in trip.dates:
                if start_date_str <= date_str <= end_date_str:
                    trip_in_window = True
                    break

            if not trip_in_window:
                continue

            # Time filter using first stop arrival
            if trip.stop_times:
                first_arrival = trip.stop_times[0].get("arrival_time")
                if first_arrival:
                    # Convert to unix time for the relevant date
                    for date_str in trip.dates:
                        if start_date_str <= date_str <= end_date_str:
                            try:
                                date_ref = datetime.strptime(date_str, "%Y%m%d")
                                trip_time = to_unix_time(first_arrival, date_ref)
                                if (
                                    trip_time
                                    and time_window_start
                                    <= trip_time
                                    <= time_window_end
                                ):
                                    expected.add(trip_id)
                                    break
                            except Exception:
                                continue

        return expected

    def _get_served_trip_ids(
        self,
        expected_trip_ids: set,
        verification_strategy,
    ) -> set:
        """
        Get set of trip IDs that were actually served (have valid diaries).

        Only checks trips that are in the expected set for efficiency.
        """
        served = set()

        for live_trip in list(self.live_trips_by_vehicle_id.values()) + self.completed_live_trips:
            trip_id = live_trip.trip_id
            if trip_id in expected_trip_ids:
                shape = self.topology.get_shape_for_trip(trip_id) if self.topology else None
                if verification_strategy.is_trip_valid(live_trip, shape):
                    served.add(trip_id)

        return served
