"""Background thread composition for the backend runtime."""

from __future__ import annotations

import logging
import threading
import time

from application.domain.time_utils import to_readable_time
from application.runtime import ApplicationContext
from persistence import log_uptime, saving_loop, saving_parquet


class ThreadLoader:
    """Create and manage the background threads needed by a runtime context."""

    def __init__(self, context: ApplicationContext):
        """Bind the loader to an already-built runtime context."""
        self.context = context
        if self.context.stop_event is None:
            self.context.stop_event = threading.Event()
        if self.context.shutdown_event is None:
            self.context.shutdown_event = threading.Event()
        if self.context.feed_fetcher is None:
            from application.live.feed_fetcher import LiveFeedFetcher

            urls = self.context.config.get("urls", {})
            self.context.feed_fetcher = LiveFeedFetcher(
                vehicles_url=urls.get("rtgtfs_vehicles"),
                trips_url=urls.get("rtgtfs_trip_updates"),
            )

    def start(self):
        """Start every configured background thread and store it on the context."""
        self.context.stop_event.clear()

        self._start_thread("collection", self._run_collection_loop)
        self._start_thread(
            "saving",
            saving_loop,
            args=(
                self.context.observatory,
                saving_parquet(),
                self.context.stop_event,
                self.context.cache_strategy,
            ),
        )
        self._start_thread("uptime", self._run_uptime_loop)
        self._start_thread("weather", self._run_weather_loop)

        if self._geocoding_is_async():
            self._start_thread("geocoding", self._run_geocoding_loop)
        else:
            logging.info(" > Geocoding Thread NOT needed (sync mode).")

        if self.context.traffic_service and self.context.city:
            self._start_thread("traffic", self._run_traffic_loop)
        else:
            logging.info(" > Traffic Thread NOT started (no API key).")

        if self.context.state_interface is not None:
            self._start_gui_thread()

    def stop(self):
        """Signal runtime service threads to stop and close owned UI services."""
        self.context.stop_event.set()
        try:
            from interaction import debug_gui

            debug_gui.stop_gui()
        except Exception:
            pass

    def join_core(self, timeout: float = 5.0):
        """Wait briefly for the core worker threads to exit."""
        for name in ("collection", "saving"):
            thread = self.context.threads.get(name)
            if thread is not None:
                thread.join(timeout=timeout)

    def _start_thread(self, name: str, target, args: tuple = ()):
        """Start one daemon thread unless a live one already exists."""
        current = self.context.threads.get(name)
        if current is not None and current.is_alive():
            logging.info(" > %s Thread already running.", name.title())
            return current

        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        self.context.threads[name] = thread
        logging.info(" > %s Thread Started.", name.title())
        return thread

    def _start_gui_thread(self):
        """Start the debug GUI thread if available."""
        current = self.context.threads.get("gui")
        if current is not None and current.is_alive():
            logging.info(" > Dashboard GUI already running.")
            return current

        from interaction import debug_gui

        services_cfg = self.context.config.get("services", {})
        gui_port = int(services_cfg.get("debug_gui_port", 8050))
        thread = debug_gui.start_gui(self.context.state_interface, port=gui_port)
        self.context.threads["gui"] = thread
        return thread

    def _run_collection_loop(self):
        """Poll GTFS-RT, let Observatory ingest records, and prune stale trips."""
        logging.info(
            "Starting real-time ingestion. Press Ctrl+C or type 'quit' to stop."
        )

        try:
            while not self._should_stop():
                logging.info(
                    "[%s] Fetching and processing live data...",
                    to_readable_time(time.time()),
                )

                records = self.context.feed_fetcher.fetch()
                self.context.observatory.ingest_live_feed(records, city_name="Rome")
                self._prune_stale_live_trips()

                logging.info("[%s] Cycle complete.", to_readable_time(time.time()))
                self._print_tracking_summary()
                print("\nCommand> ", end="", flush=True)

                self.context.stop_event.wait(60)

        except Exception as e:
            logging.error("Error in collection loop: %s", e)

    def _run_weather_loop(self):
        """Refresh city weather on the configured interval."""
        update_time = self._timing("update_interval", default=900)
        try:
            while not self._should_stop():
                logging.info(
                    "[%s] Fetching weather updates...",
                    to_readable_time(time.time()),
                )
                if self.context.city:
                    self.context.city.update_weather()
                logging.info(
                    "[%s] Weather update complete.",
                    to_readable_time(time.time()),
                )
                self.context.stop_event.wait(update_time)
        except Exception as e:
            logging.error("Error in weather loop: %s", e)

    def _run_traffic_loop(self):
        """Refresh traffic through the city-owned traffic service."""
        update_time = self._timing("traffic_update_interval", default=900)
        try:
            while not self._should_stop():
                logging.debug(
                    "[%s] Fetching traffic updates...",
                    to_readable_time(time.time()),
                )
                if not self.context.city:
                    self.context.stop_event.wait(update_time)
                    continue
                try:
                    updated_count = self.context.city.refresh_traffic()
                    logging.debug(
                        "[%s] Traffic update complete: %s hexagons updated.",
                        to_readable_time(time.time()),
                        updated_count,
                    )
                except Exception as e:
                    logging.error("Error in traffic update: %s", e)
                self.context.stop_event.wait(update_time)
        except Exception as e:
            logging.error("Error in traffic loop: %s", e)

    def _run_geocoding_loop(self):
        """Process async geocoding requests in the background."""
        service = self.context.geocoding_service
        try:
            while not self._should_stop():
                if service:
                    processed = service.process_one()
                    time.sleep(0.1 if processed else 1.0)
                else:
                    time.sleep(5.0)
        except Exception as e:
            logging.error("Error in geocoding loop: %s", e)

    def _run_uptime_loop(self):
        """Log a heartbeat once per minute while collection services run."""
        logging.info(" > Uptime Logger Started.")
        while not self._should_stop():
            log_uptime()
            for _ in range(60):
                if self._should_stop():
                    break
                time.sleep(1)
        logging.info(" > Uptime Logger Stopped.")

    def _prune_stale_live_trips(self):
        """Finish live trips that the city reports as stale."""
        if not self.context.city:
            return
        for live_trip in self.context.city.prune_stale_live_trips(ttl=600):
            logging.info(
                "Finishing stale live trip %s for vehicle %s (inactive > 600s)",
                live_trip.trip_id,
                live_trip.vehicle.label,
            )
            self.context.observatory.finish_live_trip(live_trip)

    def _print_tracking_summary(self):
        """Print a compact table of currently tracked live trips."""
        observatory = self.context.observatory
        live_trips = observatory.get_active_live_trips()
        if not live_trips:
            return

        print(
            f"\n[{to_readable_time(time.time())}] "
            f"--- TRACKING STATUS ({len(live_trips)} LiveTrips) ---"
        )
        print(
            f"{'Vehicle':<10} {'Type':<15} {'Trip ID':<15} {'Route':<10} "
            f"{'Headsign':<20} {'Location':<25} {'Speed':<8} {'Status':<10} "
            f"{'Last Seen':<12} {'Weather':<20} {'Samples'}"
        )
        print("-" * 180)

        sorted_live_trips = sorted(
            live_trips.values(),
            key=lambda live_trip: live_trip.trip.route.id,
        )
        city = observatory.get_city("Rome")
        for live_trip in sorted_live_trips:
            self._print_live_trip_summary_row(live_trip, city)
        print("-" * 180)

        geocoding = self.context.geocoding_service
        if geocoding:
            resolved = geocoding.get_and_reset_resolved_count()
            pending = geocoding.get_queue_size()
            if resolved > 0 or pending > 0:
                logging.info(
                    "Geocoding: %s resolved this cycle, %s pending",
                    resolved,
                    pending,
                )

    def _print_live_trip_summary_row(self, live_trip, city):
        """Print one tracking summary row."""
        trip = live_trip.trip
        headsign = trip.direction_name
        if headsign and len(headsign) > 20:
            headsign = headsign[:18] + ".."

        delta_min = int((time.time() - live_trip.last_seen_timestamp) / 60)
        last_seen = f"{delta_min}m ago" if delta_min > 0 else "Just now"
        status = (
            "DEPOSIT"
            if city and city.is_live_trip_in_deposit(live_trip.id)
            else "ACTIVE"
        )

        vehicle_type = "Unknown"
        if live_trip.vehicle_type:
            vehicle_type = live_trip.vehicle_type.name
            if len(vehicle_type) > 13:
                vehicle_type = vehicle_type[:11] + ".."

        location = live_trip.get_location_name() or "Resolving..."
        if len(location) > 23:
            location = location[:21] + ".."

        gps = live_trip.get_gps_data()
        speed = gps.speed if gps and gps.speed else live_trip.derived_speed
        weather = self._weather_summary(live_trip)

        print(
            f"{live_trip.vehicle.label:<10} {vehicle_type:<15} "
            f"{trip.id:<15} {trip.route.id:<10} {str(headsign):<20} "
            f"{location:<25} {speed:.1f}km/h  {status:<10} "
            f"{last_seen:<12} {weather:<20} {len(live_trip.measurements)}"
        )

    @staticmethod
    def _weather_summary(live_trip) -> str:
        """Return a compact weather summary for a live trip."""
        try:
            measurement = live_trip.get_last_measurement()
            if measurement and measurement.weather:
                weather = measurement.weather
                return f"{weather.description} ({weather.precip_intensity}mm/h)"
        except Exception:
            pass
        return "N/A"

    def _geocoding_is_async(self) -> bool:
        """Return whether the configured geocoder needs a background worker."""
        from application.domain.map_info import AsyncGeocodingService

        return isinstance(self.context.geocoding_service, AsyncGeocodingService)

    def _timing(self, key: str, default: int) -> int:
        """Read one timing value from config."""
        return int(self.context.config.get("timings", {}).get(key, default))

    def _should_stop(self) -> bool:
        """Return True when service threads should exit."""
        return self.context.shutdown_event.is_set() or self.context.stop_event.is_set()
