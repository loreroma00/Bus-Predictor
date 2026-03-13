"""
Data Orchestrator - Manages real-time data collection and update processing.
Uses LiveFeedFetcher for data retrieval.

Observatory is injected at startup via initialize() - no persistence imports here.
"""

import logging
import time
import threading

from .. import domain as t
from .feed_fetcher import LiveFeedFetcher

# Threading events
SHUTDOWN_EVENT = threading.Event()
STOP_COLLECTION_EVENT = threading.Event()

# Global state
# CITIES: dict[str, t.City] = {}  <-- REMOVED (Split brain fix)
OBSERVATORY: t.Observatory = None  # Injected at startup via initialize()
CACHE_STRATEGY = None  # Injected at startup for saving loop
TRAFFIC_SERVICE = None  # Injected at startup for traffic updates
CONFIG: dict = None  # Injected at startup

# Feed fetcher instance
_feed_fetcher = None


class VehicleTypeNotFoundException(Exception):
    pass


def initialize(observatory: t.Observatory, cache_strategy=None, config: dict = None):
    """
    Initialize the data module with an Observatory instance.
    Called once at application startup from main.py.
    """
    global OBSERVATORY, CACHE_STRATEGY, CONFIG, _feed_fetcher
    OBSERVATORY = observatory
    CACHE_STRATEGY = cache_strategy
    CONFIG = config or {}

    # Initialize Feed Fetcher with config
    urls = CONFIG.get("urls", {})
    _feed_fetcher = LiveFeedFetcher(
        vehicles_url=urls.get("rtgtfs_vehicles"),
        trips_url=urls.get("rtgtfs_trip_updates"),
    )

    logging.info(f"Data module initialized with Observatory: {observatory}")


def wire_traffic_callback():
    """
    Wire the traffic update callback to the city.
    Must be called after TRAFFIC_SERVICE is initialized.
    """
    if OBSERVATORY and TRAFFIC_SERVICE:
        city = OBSERVATORY.get_city("Rome")
        if city:
            city.set_on_bus_entered_expired_hex(_on_bus_entered_expired_hex)
            logging.debug("  Traffic callback wired to city.")


def get_realtime_updates():
    """
    Fetches real-time data and returns a list of Update objects.
    Manages buses and observers for each active vehicle.
    """
    global OBSERVATORY
    # Ensure topology is available
    topology = OBSERVATORY.get_topology()

    # Fetch live data
    live_data = _feed_fetcher.fetch()

    # Debug: Feed freshness
    if not live_data.empty and "timestamp" in live_data.columns:
        newest_ts = live_data["timestamp"].max()
        logging.debug(
            f"  > Feed Timestamp: {newest_ts} ({t.to_readable_time(newest_ts)})"
        )

    updates = []
    processed_vehicles = set()

    for _, row in live_data.iterrows():
        try:
            v_id = row.get("vehicleId")
            label = row.get("vehicleLabel")
            display_id = label if label else v_id

            # Skip duplicates
            if v_id in processed_vehicles:
                continue
            processed_vehicles.add(v_id)

            t_id = row.get("tripId")

            # Get trip from topology
            trip = topology.get_trip(t_id)

            if not trip:
                # Hot swap check
                if OBSERVATORY.check_and_reload_ledger():
                    topology = OBSERVATORY.get_topology()
                    trip = topology.get_trip(t_id)

            if not trip:
                logging.warning(f"Warning: Trip {t_id} not found. Skipping.")
                continue

            # Get or create bus
            bus: t.Autobus = None
            status: int = 0
            try:
                scheduled_start_time = row.get("startTime")
                bus, status = _get_or_create_bus(
                    v_id, trip, label=label, scheduled_start_time=scheduled_start_time
                )
            except VehicleTypeNotFoundException as e:
                print(f"{e}")
                logging.error(e)
                continue

            if status == 0:
                logging.info(
                    f"Found NEW Bus: {display_id} - {trip.route.id} {trip.direction_name}"
                )
            else:
                logging.debug(
                    f"Update for known Bus: {display_id} - {trip.route.id} {trip.direction_name}"
                )

            # Create GPS data
            gps_data = _create_gps_data(v_id, trip, row)
            bus.set_gpsData(gps_data)
            bus.set_occupancy_status(row.get("occupancyStatus"))
            bus.derive_speed()
            bus.derive_bearing()

            # Add bus to city
            if status == 0:
                OBSERVATORY.add_bus_to_city(
                    "Rome",
                    bus,
                    latitude=gps_data.latitude,
                    longitude=gps_data.longitude,
                )
            else:
                OBSERVATORY.move_bus(
                    "Rome",
                    bus,
                    latitude=gps_data.latitude,
                    longitude=gps_data.longitude,
                )

            # Create update object
            update_obj = _create_update_object(bus, row)
            updates.append(update_obj)
            # bus.set_latest_update(update_obj) Do we need this?

        except Exception as e:
            logging.error(
                f"Error processing update for {row.get('vehicleLabel', row.get('vehicleId'))}: {e}"
            )
            continue

    return updates


def _get_or_create_bus(v_id, trip, label=None, scheduled_start_time: str = None):
    """Get existing bus or create new one."""
    bus = OBSERVATORY.get_bus("Rome", v_id)
    status = 1  # Existing

    # Use label for display and logic if available, else ID
    display_id = label if label else v_id

    if not bus:
        # Debug: Why not found?
        # print(f"DEBUG: Bus {v_id} NOT found in city. Creating NEW.")
        status = 0  # New

        # Vehicle Type Lookup uses LABEL (Fleet is keyed by Label/Visual ID)
        vehicle_type: t.static_data.VehicleType = OBSERVATORY.get_vehicle_type(
            display_id
        )

        if vehicle_type is None:
            logging.warning(
                f"No Vehicle corresponding to this ID ({display_id})! usage of N/A values enabled."
            )

        bus = t.Autobus(id=v_id, vehicle_type=vehicle_type, trip=trip, label=label)
        OBSERVATORY.create_observer(bus, scheduled_start_time=scheduled_start_time)
        OBSERVATORY.update_fleet_count(display_id)
    else:
        # Debug: Found
        # print(f"DEBUG: Bus {v_id} FOUND. Trip: {bus.trip.id} vs New: {trip.id}")

        # Check for trip change
        if bus.trip.id != trip.id:
            logging.info(
                f"DETECTED TRIP CHANGE: {display_id}: {bus.trip.id} -> {trip.id}"
            )
            observer = bus.get_observer()
            if observer:
                observer.start_new_trip(trip, scheduled_start_time=scheduled_start_time)
            bus.set_trip(trip)

    return bus, status


def _create_gps_data(v_id, trip, row):
    """Create GPSData object from row data."""
    gps_id = f"{v_id}_{row.get('timestamp')}"

    gps_data = t.GPSData(
        id=gps_id,
        trip=trip,
        timestamp=row.get("timestamp"),
        latitude=row.get("latitude"),
        longitude=row.get("longitude"),
        speed=row.get("speed", 0.0),
        heading=row.get("bearing", 0.0),
        next_stop_id=row.get("stopId"),
        current_stop_sequence=row.get("currentStopSequence"),
        current_status=row.get("currentStatus"),
    )
    return gps_data


def _create_update_object(bus, row):
    """Create Update object."""
    val = row.get("stopUpdates")
    next_stops = val if isinstance(val, list) else []
    return t.Update(autobus=bus, next_stops=next_stops)


def _feed_observers(updates: list[t.Update]):
    """Feed updates to observers for diary recording."""
    from application.domain.spatial_utils import get_cardinal_direction

    for update in updates:
        observer: t.Observer = update.autobus.get_observer()
        city: t.City = OBSERVATORY.get_city("Rome")
        hexagon = city.get_hexagon(update.autobus.get_hexagon_id())

        # Get traffic in the direction the bus is traveling
        bearing = update.autobus.derived_bearing
        direction = get_cardinal_direction(bearing) if bearing else None

        speed_ratio: float = hexagon.get_speed_ratio(direction)
        current_speed: float = hexagon.get_current_speed(direction)

        # Check if traffic data is expired - measurement will need correction
        traffic_pending = hexagon.is_traffic_expired()

        if observer:
            distance = OBSERVATORY.get_stop_distance(update)
            observer.updateDiary(
                update, distance, speed_ratio, current_speed, traffic_pending
            )


def print_tracking_summary():
    """Print table of currently tracked buses."""
    observers = OBSERVATORY.get_observers()
    if not observers:
        return

    print(
        f"\n[{t.to_readable_time(time.time())}] --- TRACKING STATUS ({len(observers)} Buses) ---"
    )
    print(
        f"{'Bus ID':<10} {'Type':<15} {'Trip ID':<15} {'Route':<10} {'Headsign':<20} {'Location':<25} {'Speed':<8} {'Status':<10} {'Last Seen':<12} {'Weather':<20} {'Samples'}"
    )
    print("-" * 180)

    sorted_obs = sorted(
        observers.values(), key=lambda x: x.assignedVehicle.trip.route.id
    )

    for obs in sorted_obs:
        trip = obs.assignedVehicle.trip
        diary = obs.current_diary
        rec_count = len(diary.measurements) if diary else 0
        headsign = (
            (trip.direction_name[:18] + "..")
            if trip.direction_name and len(trip.direction_name) > 20
            else trip.direction_name
        )

        last_seen = obs.assignedVehicle.last_seen_timestamp
        delta_min = int((time.time() - last_seen) / 60)
        last_seen_str = f"{delta_min}m ago" if delta_min > 0 else "Just now"

        # Determine Status
        status_str = "ACTIVE"
        if OBSERVATORY.is_bus_in_deposit("Rome", obs.assignedVehicle.id):
            status_str = "DEPOSIT"

        # Get Vehicle Type
        vehicle_type = "Unknown"
        if obs.assignedVehicle.vehicle_type:
            vehicle_type = obs.assignedVehicle.vehicle_type.name
            if len(vehicle_type) > 13:
                vehicle_type = vehicle_type[:11] + ".."

        # Get Location
        location_name = obs.assignedVehicle.get_location_name() or "Resolving..."
        if len(location_name) > 23:
            location_name = location_name[:21] + ".."

        # Get Speed
        bus = obs.assignedVehicle
        speed = (
            bus.GPSData.speed
            if bus.GPSData and bus.GPSData.speed
            else bus.derived_speed
        )
        speed_str = f"{speed:.1f}km/h"

        # Get Weather Info
        weather_str = "N/A"
        try:
            if diary:
                last_meas = diary.get_last_measurement()
                if last_meas and last_meas.weather:
                    w = last_meas.weather
                    weather_str = f"{w.description} ({w.precip_intensity}mm/h)"
        except Exception:
            pass

        print(
            f"{obs.assignedVehicle.label:<10} {vehicle_type:<15} {trip.id:<15} {trip.route.id:<10} {str(headsign):<20} {location_name:<25} {speed_str:<8} {status_str:<10} {last_seen_str:<12} {weather_str:<20} {rec_count}"
        )
    print("-" * 180)

    # Geocoding stats
    if OBSERVATORY._geocoding:
        resolved = OBSERVATORY._geocoding.get_and_reset_resolved_count()
        pending = OBSERVATORY._geocoding.get_queue_size()
        if resolved > 0 or pending > 0:
            logging.info(
                f"📍 Geocoding: {resolved} resolved this cycle, {pending} pending"
            )


def run_collection_loop():
    """Background thread function for data collection."""
    logging.info("Starting Real-time Observer... Press Ctrl+C or type 'quit' to stop.")

    try:
        while not SHUTDOWN_EVENT.is_set() and not STOP_COLLECTION_EVENT.is_set():
            logging.info(
                f"[{t.to_readable_time(time.time())}] Fetching and processing live data..."
            )

            updates = get_realtime_updates()
            _feed_observers(updates)

            # Helper: Prune stale buses
            if OBSERVATORY:
                OBSERVATORY.prune_stale_buses("Rome", ttl=600)  # 10 minutes TTL

            logging.info(f"[{t.to_readable_time(time.time())}] Cycle complete.")
            print_tracking_summary()
            print("\nCommand> ", end="", flush=True)

            SHUTDOWN_EVENT.wait(60)

    except Exception as e:
        logging.error(f"Error in collection loop: {e}")


def run_weather_loop(update_time: int = 900):
    """Background thread function for weather updates."""
    try:
        while not SHUTDOWN_EVENT.is_set() and not STOP_COLLECTION_EVENT.is_set():
            logging.info(
                f"[{t.to_readable_time(time.time())}] Fetching weather updates..."
            )
            OBSERVATORY.update_weather("Rome")
            logging.info(
                f"[{t.to_readable_time(time.time())}] Weather update complete."
            )
            SHUTDOWN_EVENT.wait(update_time)  # Update every fifteen minutes
    except Exception as e:
        logging.error(f"Error in weather loop: {e}")


def run_geocoding_loop():
    """
    Background thread function for geocoding queue processing.

    Processes one geocoding request per second to respect API rate limits.
    The actual rate limiting is handled by the RateLimiter in map_info.py,
    but we add a small sleep to prevent busy-waiting.
    """
    try:
        while not SHUTDOWN_EVENT.is_set() and not STOP_COLLECTION_EVENT.is_set():
            if OBSERVATORY and OBSERVATORY._geocoding:
                processed = OBSERVATORY._geocoding.process_one()
                if processed:
                    # Small delay after processing (rate limiter handles the real 1s delay)
                    time.sleep(0.1)
                else:
                    # Queue is empty, wait a bit before checking again
                    time.sleep(1.0)
            else:
                # Geocoding not enabled, sleep longer
                time.sleep(5.0)
    except Exception as e:
        logging.error(f"Error in geocoding loop: {e}")


def _on_bus_entered_expired_hex(bus_id: str, hex_id: str):
    """
    Callback when a bus enters a hexagon with expired traffic data.
    Immediately fetches fresh traffic and corrects pending measurements.
    """
    if not TRAFFIC_SERVICE:
        return

    label = bus_id
    if OBSERVATORY:
        b = OBSERVATORY.get_bus("Rome", bus_id)
        if b:
            label = b.label

    logging.debug(f"  Traffic: Bus {label} entered expired hex {hex_id[:12]}...")

    # 1. Fetch fresh traffic immediately (updates all hexagons in tile)
    updated_hex_ids = TRAFFIC_SERVICE.update_traffic_for_hexagon(hex_id)

    # 2. Correct pending measurements for this bus
    if OBSERVATORY and updated_hex_ids:
        _correct_pending_measurements(bus_id, updated_hex_ids)


def _correct_pending_measurements(bus_id: str, updated_hex_ids: list[str]):
    """Correct measurements that were recorded with outdated traffic data."""
    from application.domain.spatial_utils import get_cardinal_direction

    # Find the observer for this bus
    bus = OBSERVATORY.get_bus("Rome", bus_id)
    if not bus:
        return

    observer = bus.get_observer()
    if not observer or not observer.current_diary:
        return

    city = OBSERVATORY.get_city("Rome")
    pending = observer.current_diary.get_pending_traffic_measurements()

    for measurement in pending:
        # Get the hexagon for this measurement's location
        hex_id = measurement.hexagon_id if measurement.hexagon_id else None
        if not hex_id:
            # Derive from coordinates
            from application.domain import h3_utils

            hex_id = h3_utils.get_h3_index(
                measurement.gpsdata.latitude, measurement.gpsdata.longitude
            )

        if hex_id in updated_hex_ids:
            hexagon = city.get_hexagon(hex_id)
            if hexagon:
                # Get direction from measurement's bearing
                bearing = measurement.derived_bearing
                direction = get_cardinal_direction(bearing) if bearing else None

                # Update ONLY traffic fields
                measurement.update_traffic_data(
                    hexagon.get_speed_ratio(direction),
                    hexagon.get_current_speed(direction),
                )
                logging.debug(
                    f"  Traffic: Corrected measurement {measurement.id} for bus {bus.label}"
                )


def run_traffic_loop(update_interval: int = 300):
    """
    Background thread function for traffic updates.

    Fetches TomTom traffic data at regular intervals and updates hexagons.
    Rate limited to 10 QPS per TomTom API requirements internally.

    Args:
        update_interval: Seconds between update cycles (default 5 minutes)
    """
    try:
        while not SHUTDOWN_EVENT.is_set() and not STOP_COLLECTION_EVENT.is_set():
            if TRAFFIC_SERVICE:
                logging.debug(
                    f"[{t.to_readable_time(time.time())}] Fetching traffic updates..."
                )
                try:
                    updated_count = TRAFFIC_SERVICE.update_traffic()
                    logging.debug(
                        f"[{t.to_readable_time(time.time())}] Traffic update complete: "
                        f"{updated_count} hexagons updated."
                    )
                except Exception as e:
                    logging.error(f"Error in traffic update: {e}")
            else:
                # Traffic service not configured, wait longer
                pass
            SHUTDOWN_EVENT.wait(update_interval)
    except Exception as e:
        logging.error(f"Error in traffic loop: {e}")
