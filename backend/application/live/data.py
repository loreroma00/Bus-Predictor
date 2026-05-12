"""
Data orchestrator for real-time GTFS-RT ingestion.

The live feed is unpacked into ``Update`` objects, then applied to the owning
``LiveTrip`` aggregate. There are no observers: a live trip owns its current
state and its measurement list.
"""

from __future__ import annotations

import logging
import threading
import time

from .. import domain as t
from .feed_fetcher import LiveFeedFetcher

SHUTDOWN_EVENT = threading.Event()
STOP_COLLECTION_EVENT = threading.Event()

OBSERVATORY: t.Observatory = None
CACHE_STRATEGY = None
TRAFFIC_SERVICE = None
CONFIG: dict = None

_feed_fetcher = None


def initialize(observatory: t.Observatory, cache_strategy=None, config: dict = None):
    """Initialize this module with its runtime dependencies."""
    global OBSERVATORY, CACHE_STRATEGY, CONFIG, _feed_fetcher
    OBSERVATORY = observatory
    CACHE_STRATEGY = cache_strategy
    CONFIG = config or {}

    urls = CONFIG.get("urls", {})
    _feed_fetcher = LiveFeedFetcher(
        vehicles_url=urls.get("rtgtfs_vehicles"),
        trips_url=urls.get("rtgtfs_trip_updates"),
    )

    logging.info(f"Data module initialized with Observatory: {observatory}")


def wire_traffic_callback():
    """Wire the expired-traffic callback into the city object."""
    if OBSERVATORY and TRAFFIC_SERVICE:
        city = OBSERVATORY.get_city("Rome")
        if city:
            city.set_on_live_trip_entered_expired_hex(
                _on_live_trip_entered_expired_hex
            )
            logging.debug("  Traffic callback wired to city.")


def get_realtime_updates():
    """Fetch GTFS-RT rows and apply them to active LiveTrip objects."""
    global OBSERVATORY
    topology = OBSERVATORY.get_topology()
    live_data = _feed_fetcher.fetch()

    if not live_data.empty and "timestamp" in live_data.columns:
        newest_ts = live_data["timestamp"].max()
        logging.debug(
            f"  > Feed Timestamp: {newest_ts} ({t.to_readable_time(newest_ts)})"
        )

    updates = []
    processed_vehicles = set()

    for _, row in live_data.iterrows():
        try:
            vehicle_id = row.get("vehicleId")
            label = row.get("vehicleLabel")
            display_id = label if label else vehicle_id

            if vehicle_id in processed_vehicles:
                continue
            processed_vehicles.add(vehicle_id)

            trip_id = row.get("tripId")
            trip = topology.get_trip(trip_id)
            if not trip and OBSERVATORY.check_and_reload_ledger():
                topology = OBSERVATORY.get_topology()
                trip = topology.get_trip(trip_id)

            if not trip:
                logging.warning(f"Warning: Trip {trip_id} not found. Skipping.")
                continue

            scheduled_start_time = row.get("startTime")
            live_trip, status = OBSERVATORY.get_or_create_live_trip(
                vehicle_id=vehicle_id,
                trip=trip,
                label=label,
                scheduled_start_time=scheduled_start_time,
            )

            if status == 0:
                logging.info(
                    f"Found NEW LiveTrip: {display_id} - {trip.route.id} {trip.direction_name}"
                )
            elif status == 2:
                logging.info(
                    f"Detected trip change for {display_id}: now {trip.id}"
                )
            else:
                logging.debug(
                    f"Update for LiveTrip: {display_id} - {trip.route.id} {trip.direction_name}"
                )

            gps_data = _create_gps_data(vehicle_id, trip, row)
            live_trip.set_gps_data(gps_data)
            live_trip.set_occupancy_status(row.get("occupancyStatus"))
            live_trip.derive_speed()
            live_trip.derive_bearing()

            if status in (0, 2):
                OBSERVATORY.add_live_trip_to_city(
                    "Rome",
                    live_trip,
                    latitude=gps_data.latitude,
                    longitude=gps_data.longitude,
                )
            else:
                OBSERVATORY.move_live_trip(
                    "Rome",
                    live_trip,
                    latitude=gps_data.latitude,
                    longitude=gps_data.longitude,
                )

            update_obj = _create_update_object(live_trip, row)
            _apply_live_trip_update(live_trip, update_obj)
            updates.append(update_obj)

        except Exception as e:
            logging.error(
                f"Error processing update for {row.get('vehicleLabel', row.get('vehicleId'))}: {e}"
            )
            continue

    return updates


def _create_gps_data(vehicle_id, trip, row):
    """Create GPSData object from a feed row."""
    gps_id = f"{vehicle_id}_{row.get('timestamp')}"
    return t.GPSData(
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


def _create_update_object(live_trip: t.LiveTrip, row):
    """Create a combined Update object."""
    val = row.get("stopUpdates")
    next_stops = val if isinstance(val, list) else []
    return t.Update(live_trip=live_trip, next_stops=next_stops)


def _apply_live_trip_update(live_trip: t.LiveTrip, update: t.Update):
    """Compute context for one update and let LiveTrip append its measurement."""
    from application.domain.spatial_utils import get_cardinal_direction

    city: t.City = OBSERVATORY.get_city("Rome")
    hexagon = city.get_hexagon(live_trip.get_hexagon_id()) if city else None
    if not hexagon:
        return

    bearing = live_trip.derived_bearing
    direction = get_cardinal_direction(bearing) if bearing else None
    speed_ratio = hexagon.get_speed_ratio(direction)
    current_speed = hexagon.get_current_speed(direction)
    traffic_pending = hexagon.is_traffic_expired()
    distance = OBSERVATORY.get_stop_distance(update)

    weather = None
    try:
        weather = city.get_weather(live_trip.get_hexagon_id())
    except Exception:
        pass

    live_trip.apply_update(
        update,
        next_stop_distance=distance,
        speed_ratio=speed_ratio,
        current_speed=current_speed,
        traffic_data_pending=traffic_pending,
        weather=weather,
    )


def print_tracking_summary():
    """Print table of currently tracked live trips."""
    live_trips = OBSERVATORY.get_active_live_trips()
    if not live_trips:
        return

    print(
        f"\n[{t.to_readable_time(time.time())}] --- TRACKING STATUS ({len(live_trips)} LiveTrips) ---"
    )
    print(
        f"{'Vehicle':<10} {'Type':<15} {'Trip ID':<15} {'Route':<10} {'Headsign':<20} {'Location':<25} {'Speed':<8} {'Status':<10} {'Last Seen':<12} {'Weather':<20} {'Samples'}"
    )
    print("-" * 180)

    sorted_live_trips = sorted(
        live_trips.values(), key=lambda x: x.trip.route.id
    )

    for live_trip in sorted_live_trips:
        trip = live_trip.trip
        rec_count = len(live_trip.measurements)
        headsign = (
            (trip.direction_name[:18] + "..")
            if trip.direction_name and len(trip.direction_name) > 20
            else trip.direction_name
        )

        last_seen = live_trip.last_seen_timestamp
        delta_min = int((time.time() - last_seen) / 60)
        last_seen_str = f"{delta_min}m ago" if delta_min > 0 else "Just now"

        status_str = "ACTIVE"
        if OBSERVATORY.is_live_trip_in_deposit("Rome", live_trip.id):
            status_str = "DEPOSIT"

        vehicle_type = "Unknown"
        if live_trip.vehicle_type:
            vehicle_type = live_trip.vehicle_type.name
            if len(vehicle_type) > 13:
                vehicle_type = vehicle_type[:11] + ".."

        location_name = live_trip.get_location_name() or "Resolving..."
        if len(location_name) > 23:
            location_name = location_name[:21] + ".."

        gps = live_trip.get_gps_data()
        speed = gps.speed if gps and gps.speed else live_trip.derived_speed
        speed_str = f"{speed:.1f}km/h"

        weather_str = "N/A"
        try:
            last_meas = live_trip.get_last_measurement()
            if last_meas and last_meas.weather:
                w = last_meas.weather
                weather_str = f"{w.description} ({w.precip_intensity}mm/h)"
        except Exception:
            pass

        print(
            f"{live_trip.vehicle.label:<10} {vehicle_type:<15} {trip.id:<15} {trip.route.id:<10} {str(headsign):<20} {location_name:<25} {speed_str:<8} {status_str:<10} {last_seen_str:<12} {weather_str:<20} {rec_count}"
        )
    print("-" * 180)

    if OBSERVATORY._geocoding:
        resolved = OBSERVATORY._geocoding.get_and_reset_resolved_count()
        pending = OBSERVATORY._geocoding.get_queue_size()
        if resolved > 0 or pending > 0:
            logging.info(
                f"Geocoding: {resolved} resolved this cycle, {pending} pending"
            )


def run_collection_loop():
    """Background thread function for data collection."""
    logging.info("Starting real-time ingestion. Press Ctrl+C or type 'quit' to stop.")

    try:
        while not SHUTDOWN_EVENT.is_set() and not STOP_COLLECTION_EVENT.is_set():
            logging.info(
                f"[{t.to_readable_time(time.time())}] Fetching and processing live data..."
            )

            get_realtime_updates()

            if OBSERVATORY:
                OBSERVATORY.prune_stale_live_trips("Rome", ttl=600)

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
            SHUTDOWN_EVENT.wait(update_time)
    except Exception as e:
        logging.error(f"Error in weather loop: {e}")


def run_geocoding_loop():
    """Process async geocoding requests in the background."""
    try:
        while not SHUTDOWN_EVENT.is_set() and not STOP_COLLECTION_EVENT.is_set():
            if OBSERVATORY and OBSERVATORY._geocoding:
                processed = OBSERVATORY._geocoding.process_one()
                time.sleep(0.1 if processed else 1.0)
            else:
                time.sleep(5.0)
    except Exception as e:
        logging.error(f"Error in geocoding loop: {e}")


def _on_live_trip_entered_expired_hex(live_trip_id: str, hex_id: str):
    """Fetch fresh traffic when a live trip enters an expired hex."""
    if not TRAFFIC_SERVICE:
        return

    label = live_trip_id
    if OBSERVATORY:
        live_trip = OBSERVATORY.get_live_trip("Rome", live_trip_id)
        if live_trip:
            label = live_trip.vehicle.label

    logging.debug(f"  Traffic: LiveTrip {label} entered expired hex {hex_id[:12]}...")
    updated_hex_ids = TRAFFIC_SERVICE.update_traffic_for_hexagon(hex_id)

    if OBSERVATORY and updated_hex_ids:
        _correct_pending_measurements(live_trip_id, updated_hex_ids)


def _correct_pending_measurements(live_trip_id: str, updated_hex_ids: list[str]):
    """Correct measurements recorded with outdated traffic data."""
    from application.domain.spatial_utils import get_cardinal_direction

    live_trip = OBSERVATORY.get_live_trip("Rome", live_trip_id)
    if not live_trip:
        return

    city = OBSERVATORY.get_city("Rome")
    pending = live_trip.get_pending_traffic_measurements()

    for measurement in pending:
        hex_id = measurement.hexagon_id
        if not hex_id:
            from application.domain import h3_utils

            hex_id = h3_utils.get_h3_index(
                measurement.gpsdata.latitude,
                measurement.gpsdata.longitude,
            )

        if hex_id in updated_hex_ids:
            hexagon = city.get_hexagon(hex_id)
            if hexagon:
                bearing = measurement.derived_bearing
                direction = get_cardinal_direction(bearing) if bearing else None
                measurement.update_traffic_data(
                    hexagon.get_speed_ratio(direction),
                    hexagon.get_current_speed(direction),
                )
                logging.debug(
                    f"  Traffic: Corrected measurement {measurement.id} for {live_trip.vehicle.label}"
                )


def run_traffic_loop(update_interval: int = 300):
    """Background thread function for traffic updates."""
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
            SHUTDOWN_EVENT.wait(update_interval)
    except Exception as e:
        logging.error(f"Error in traffic loop: {e}")
