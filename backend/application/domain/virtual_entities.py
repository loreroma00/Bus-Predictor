"""
Observatory - Facade for all data processing operations.
Delegates to internal components while presenting a unified interface.
Uses Dependency Injection for external dependencies (e.g., caching).
"""

import logging
import time

from .interfaces import CacheStrategy, GeocodingStrategy
from .static_data import Vehicle, VehicleType
from .static_data_fetcher import StaticDataFetcher
from .ledger_builder import LedgerBuilder
from .ledgers import HistoricalLedger, ScheduleLedger, TopologyLedger
from .cities import City
from .live_data import LiveTrip
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
        # ---- Ledgers ----
        self.topology: TopologyLedger = None
        self.schedule_ledger: ScheduleLedger = None
        self.historical: HistoricalLedger = HistoricalLedger()

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
