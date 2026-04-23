"""
Observatory - Facade for all data processing operations.
Delegates to internal components while presenting a unified interface.
Uses Dependency Injection for external dependencies (e.g., caching).
"""

import logging
import time

from .interfaces import CacheStrategy, GeocodingStrategy
from .static_data import VehicleType
from .static_data_fetcher import StaticDataFetcher
from .ledger_builder import LedgerBuilder
from .ledgers import TopologyLedger, ScheduleLedger, HistoricalLedger, PredictedLedger, VehicleLedger
from .observer_manager import ObserverManager
from .cities import City
from .live_data import Autobus
from typing import overload
from application.post_processing.data_cleaning import (
    PredictionPipeline,
    TrafficPipeline,
    LenientPipeline,
)
from .fleet_loader import load_fleet


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
        self.config = config or {}

        # Internal components (no external dependencies)
        self._fetcher = StaticDataFetcher()
        self._builder = LedgerBuilder()
        self._observer_manager = ObserverManager()

        # ---- Ledgers ----
        self.topology: TopologyLedger = None
        self.schedule_ledger: ScheduleLedger = None
        self.historical: HistoricalLedger = HistoricalLedger()
        self.predicted: PredictedLedger = PredictedLedger()
        self.vehicle_ledger: VehicleLedger = VehicleLedger()

        # Cache metadata
        self.current_md5: str = None
        self.last_update_check: float = 0

        # Cities & Fleet
        self.observed_cities: dict[str, City] = {}
        self.fleet = load_fleet("vehicles.csv")

    def add_city(self, city_name: str, static_bus_lanes: dict = None):
        """Add a city to the observed cities."""
        self.observed_cities[city_name] = City(city_name, static_bus_lanes)

    def get_city(self, city_name: str) -> City | None:
        """Get a city instance by name."""
        return self.observed_cities.get(city_name)

    @overload
    def add_bus_to_city(self, city_name: str, bus: Autobus, hex_id: str):
        """Overload: place the bus at a pre-resolved H3 hex."""
        ...

    @overload
    def add_bus_to_city(
        self, city_name: str, bus: Autobus, latitude: float, longitude: float
    ):
        """Overload: place the bus via latitude/longitude (hex is computed)."""
        ...

    def add_bus_to_city(
        self,
        city_name: str,
        bus: Autobus,
        latitude: float = None,
        longitude: float = None,
        hex_id: str = None,
    ):
        """Add a bus to the city."""
        # Assign Vehicle Type if not set
        if bus.vehicle_type is None:
            # Use the fuzzy lookup method with the label
            v_type = self.get_vehicle_type(bus.label)
            if v_type:
                bus.set_vehicle_type(v_type)

        if hex_id is not None:
            self.observed_cities[city_name].add_bus_to_city(bus, hex_id=hex_id)
            bus.set_hexagon_id(hex_id)
        else:
            self.observed_cities[city_name].add_bus_to_city(
                bus, lat=latitude, lng=longitude
            )
            new_hex_id = self.observed_cities[city_name].get_hex_id(latitude, longitude)
            bus.set_hexagon_id(new_hex_id)

            # Trigger geocoding on creation if enabled
            if self._geocoding and latitude is not None:
                self._geocoding.enqueue(latitude, longitude, new_hex_id)
                street = self._geocoding.get_street(latitude, longitude)
                if street:
                    bus.set_location_name(street)

    def remove_bus_from_city(self, city_name: str, bus: Autobus):
        """Remove a bus from the city."""
        self.observed_cities[city_name].remove_bus(bus)

    def prune_stale_buses(self, city_name: str, ttl: int = 300):
        """Removes buses that haven't updated in 'ttl' seconds."""
        if city_name not in self.observed_cities:
            return

        city = self.observed_cities[city_name]
        # Iterate over a copy of keys since we might mutate the dict
        for bus_id in list(city.bus_index.keys()):
            # We need to retrieve the bus to check its timestamp.
            # Since bus_index stores HexID, we use our new logic.
            hex_id = city.bus_index[bus_id]
            if hex_id in city.hexagons:
                bus = city.hexagons[hex_id].get_bus(bus_id)
                if bus:
                    # Use GPS timestamp if available (data freshness), else fallback to last_seen (system freshness)
                    # We want to prune if the DATA is old, even if we just saw it in the feed.
                    last_active = bus.last_seen_timestamp
                    if bus.GPSData and bus.GPSData.timestamp:
                        # timestamp might be int or float
                        last_active = float(bus.GPSData.timestamp)

                    if time.time() - last_active > ttl:
                        logging.info(
                            f"Pruning stale bus {bus_id} (inactive > {ttl}s)"
                        )
                        observer = bus.get_observer()
                        if observer and observer.current_diary and observer.current_diary.measurements:
                            observer.archive_current_diary()
                        city.bus_to_deposit(bus_id)

    @overload
    def move_bus(
        self, city_name: str, bus: Autobus, latitude: float, longitude: float
    ):
        """Overload: move the bus using new latitude/longitude (hex recomputed)."""
        ...

    @overload
    def move_bus(self, city_name: str, bus: Autobus, new_hex_id: str):
        """Overload: move the bus to a pre-resolved H3 hex."""
        ...

    def move_bus(
        self,
        city_name: str,
        bus: Autobus,
        new_hex_id: str = None,
        latitude: float = None,
        longitude: float = None,
    ):
        """Move a bus in the city."""
        if new_hex_id is not None:
            self.observed_cities[city_name].move_bus(bus.id, new_hex_id)
            bus.set_hexagon_id(new_hex_id)
        else:
            self.observed_cities[city_name].move_bus(
                bus.id, latitude=latitude, longitude=longitude
            )
            new_hex_id = self.observed_cities[city_name].get_hex_id(latitude, longitude)
            bus.set_hexagon_id(new_hex_id)

            # Trigger geocoding if enabled
            if self._geocoding and latitude is not None:
                self._geocoding.enqueue(latitude, longitude, new_hex_id)
                street = self._geocoding.get_street(latitude, longitude)
                if street:
                    bus.set_location_name(street)

    def get_bus(self, city_name: str, bus_id: str) -> Autobus | None:
        """Get a bus from a city."""
        if city_name in self.observed_cities:
            return self.observed_cities[city_name].get_bus(bus_id)
        return None

    def is_bus_in_deposit(self, city_name: str, bus_id: str) -> bool:
        """Check if a bus is in the deposit."""
        if city_name in self.observed_cities:
            return self.observed_cities[city_name].is_bus_in_deposit(bus_id)
        return False

    def update_weather(self, city_name: str):
        """Update the weather for a city."""
        if city_name in self.observed_cities:
            self.observed_cities[city_name].update_weather()

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

    # ==================== OBSERVER OPERATIONS (delegated) ====================

    def create_observer(
        self, vehicle, observer_type="strict", scheduled_start_time: str = None
    ):
        """Create an observer for a vehicle."""
        return self._observer_manager.create_observer(
            vehicle, self, scheduled_start_time
        )

    def get_observers(self) -> dict:
        """Get all observers."""
        return self._observer_manager.get_all_observers()

    def search_diary(self, trip_id: str):
        """Search for an active diary by trip ID."""
        return self._observer_manager.search_diary(trip_id)

    def search_history(self, trip_id: str):
        """Search for a diary in history by trip ID."""
        return self._observer_manager.search_history(trip_id)

    def get_completed_diaries(self) -> list:
        """Get all completed diary data."""
        return self._observer_manager.get_completed_diaries()

    def get_all_current_diaries(self) -> tuple:
        """Get all current diary data with counts."""
        return self._observer_manager.get_all_current_diaries()

    def get_vehicle_diaries(self, vehicle_id: str):
        """Return every diary (historical + current) tied to a specific vehicle."""
        return self._observer_manager.get_vehicle_diaries(vehicle_id)

    def get_id_by_label(self, label: str) -> str | None:
        """Find vehicle ID by label."""
        return self._observer_manager.get_id_by_label(label)

    # ==================== UTILITY OPERATIONS ====================

    def get_stop_distance(self, update) -> float:
        """Calculate distance to next stop based on shape projection."""
        trip = self.search_trip(update.get_autobus().get_trip().id)
        if not trip:
            return None

        shape = trip.get_shape()
        if not shape:
            return None

        gps = update.get_autobus().get_gpsData()
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

        # Iterate all observers
        for observer in self._observer_manager.observers.values():
            # Check current diary
            if observer.current_diary:
                trip_id = observer.current_diary.trip_id
                if trip_id in expected_trip_ids:
                    shape = self.topology.get_shape_for_trip(trip_id) if self.topology else None
                    if verification_strategy.is_trip_valid(
                        observer.current_diary, shape
                    ):
                        served.add(trip_id)

            # Check diary history
            for diary in observer.diary_history:
                trip_id = diary.trip_id
                if trip_id in expected_trip_ids:
                    shape = self.topology.get_shape_for_trip(trip_id) if self.topology else None
                    if verification_strategy.is_trip_valid(diary, shape):
                        served.add(trip_id)

        return served

    # ==================== VECTORIZATION LOOP ====================

    def process_completed_diary(
        self,
        diary,
        route_id: str,
        time_window_minutes: int = 60,
    ) -> list:
        """
        Process a completed diary: compute served_ratio, clean, vectorize.

        Args:
            diary: The completed Diary object
            route_id: Route ID for served_ratio calculation
            time_window_minutes: Time window for served_ratio (default: 60 min)

        Returns:
            List of vectors, or empty list if failed/filtered
        """

        # 1. Compute served_ratio using scan_trip_adherence
        if diary.measurements:
            times = [m.measurement_time for m in diary.measurements]
            time_window_start = min(times) - (time_window_minutes * 30)
            time_window_end = max(times) + (time_window_minutes * 30)
        else:
            now = time.time()
            time_window_start = now - (time_window_minutes * 60)
            time_window_end = now

        served_ratio = self.scan_trip_adherence(
            route_id=route_id,
            time_window_start=time_window_start,
            time_window_end=time_window_end,
        )

        # 2. Create pipeline and clean/vectorize
        PipelineClass = (
            LenientPipeline
            if self.config.get("lenient_pipeline")
            else PredictionPipeline
        )
        pipeline = PipelineClass(
            diary=diary,
            topology=self.topology,
            served_ratio=served_ratio,
            config=self.config,
        )
        return pipeline.clean()

    def process_traffic_diary(self, diary) -> list:
        """
        Process a completed diary for traffic analysis.

        Returns:
            List of traffic vectors.
        """
        pipeline = TrafficPipeline(diary=diary, config=self.config)
        return pipeline.clean()
