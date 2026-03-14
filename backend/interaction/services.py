"""
Services - Background threads for data collection, saving, and uptime.

Subscribes to events from the event bus for start/stop control.
"""

import logging
import threading
import time
from application.live import data
from application.domain.internal_events import domain_events, DIARY_FINISHED
from persistence import saving_loop, saving_parquet, log_uptime
from .events import console_events

COLLECTION_THREAD = None
SAVING_THREAD = None
UPTIME_THREAD = None
WEATHER_THREAD = None
GEOCODING_THREAD = None
TRAFFIC_THREAD = None
GUI_THREAD = None

# Validator threads
VALIDATOR_THREAD = None
LIVE_VALIDATOR_THREAD = None
LIVE_VALIDATOR_SESSION = None

UPDATE_TIME = 900
TRAFFIC_UPDATE_TIME = 900  # 15 minutes

# State interface for GUI (set by main.py)
_state_interface = None


def uptime_loop(stop_event):
    logging.info(" > Uptime Logger Started.")
    while not stop_event.is_set():
        log_uptime()
        for _ in range(60):
            if stop_event.is_set():
                break
            time.sleep(1)
    logging.info(" > Uptime Logger Stopped.")




def start_services(observatory=None, config: dict = None):
    """Start all background services."""
    global \
        COLLECTION_THREAD, \
        SAVING_THREAD, \
        UPTIME_THREAD, \
        WEATHER_THREAD, \
        GEOCODING_THREAD, \
        TRAFFIC_THREAD, \
        GUI_THREAD, \
        _state_interface

    logging.info("Starting Services...")
    data.STOP_COLLECTION_EVENT.clear()

    config = config or {}
    timings = config.get("timings", {})
    services_cfg = config.get("services", {})

    update_time = int(timings.get("update_interval", 900))
    traffic_update_time = int(timings.get("traffic_update_interval", 900))
    gui_port = int(services_cfg.get("debug_gui_port", 8050))

    # Subscribe to domain events
    domain_events.subscribe(DIARY_FINISHED, _on_diary_finished)

    if COLLECTION_THREAD is None or not COLLECTION_THREAD.is_alive():
        COLLECTION_THREAD = threading.Thread(
            target=data.run_collection_loop,
            daemon=True,
        )
        COLLECTION_THREAD.start()
        logging.info(" > Collection Thread Started.")
    else:
        logging.info(" > Collection Thread already running.")

    if SAVING_THREAD is None or not SAVING_THREAD.is_alive():
        SAVING_THREAD = threading.Thread(
            target=saving_loop,
            args=(
                data.OBSERVATORY,
                saving_parquet(),
                data.STOP_COLLECTION_EVENT,
                data.CACHE_STRATEGY,  # Pass cache strategy for city cache saving
            ),
        )
        SAVING_THREAD.start()
        logging.info(" > Saving Thread Started.")
    else:
        logging.info(" > Saving Thread already running.")

    if UPTIME_THREAD is None or not UPTIME_THREAD.is_alive():
        UPTIME_THREAD = threading.Thread(
            target=uptime_loop, args=(data.STOP_COLLECTION_EVENT,), daemon=True
        )
        UPTIME_THREAD.start()
    else:
        logging.info(" > Uptime Thread already running.")

    if WEATHER_THREAD is None or not WEATHER_THREAD.is_alive():
        WEATHER_THREAD = threading.Thread(
            target=data.run_weather_loop,
            args=(update_time,),
            daemon=True,
        )
        WEATHER_THREAD.start()
    else:
        logging.info(" > Weather Thread already running.")

    # Only start geocoding thread for async mode (rate-limited servers)
    from application.domain.map_info import AsyncGeocodingService

    if data.OBSERVATORY and isinstance(
        data.OBSERVATORY._geocoding, AsyncGeocodingService
    ):
        if GEOCODING_THREAD is None or not GEOCODING_THREAD.is_alive():
            GEOCODING_THREAD = threading.Thread(
                target=data.run_geocoding_loop,
                daemon=True,
            )
            GEOCODING_THREAD.start()
            logging.info(" > Geocoding Thread Started (async mode).")
        else:
            logging.info(" > Geocoding Thread already running.")
    else:
        logging.info(" > Geocoding Thread NOT needed (sync mode).")

    # Start traffic thread if traffic service is configured
    if data.TRAFFIC_SERVICE:
        if TRAFFIC_THREAD is None or not TRAFFIC_THREAD.is_alive():
            TRAFFIC_THREAD = threading.Thread(
                target=data.run_traffic_loop,
                args=(traffic_update_time,),
                daemon=True,
            )
            TRAFFIC_THREAD.start()
            logging.info(" > Traffic Thread Started.")
        else:
            logging.info(" > Traffic Thread already running.")
    else:
        logging.info(" > Traffic Thread NOT started (no API key).")

    # Start Debug GUI if state interface is available
    if _state_interface is not None:
        if GUI_THREAD is None or not GUI_THREAD.is_alive():
            from . import debug_gui

            GUI_THREAD = debug_gui.start_gui(_state_interface, port=gui_port)
        else:
            logging.info(" > Debug GUI already running.")


def set_state_interface(state_interface):
    """Set the state interface for the GUI (called from main.py)."""
    global _state_interface
    _state_interface = state_interface


def start_batch_validation(date_str, predictor, observatory):
    """Start batch validation in a background thread."""
    global VALIDATOR_THREAD

    if VALIDATOR_THREAD is not None and VALIDATOR_THREAD.is_alive():
        logging.warning("Batch validation already running.")
        return

    def _run():
        from application.services.validator import Validator
        try:
            validator = Validator(predictor, observatory)
            report = validator.validate_date(date_str)
            logging.info(
                f"Batch validation complete: {report.total_trips_validated} trips, "
                f"median RMSE={report.median_rmse:.2f}s"
            )
        except Exception as e:
            logging.error(f"Batch validation failed: {e}")

    VALIDATOR_THREAD = threading.Thread(target=_run, daemon=True)
    VALIDATOR_THREAD.start()
    logging.info(f"Batch validation started for {date_str}")


def start_live_validation(date_str, predictor, observatory, bus_type_predictor=None):
    """Start live validation in a background thread with its own event loop."""
    global LIVE_VALIDATOR_THREAD, LIVE_VALIDATOR_SESSION
    import asyncio
    import uuid

    if LIVE_VALIDATOR_THREAD is not None and LIVE_VALIDATOR_THREAD.is_alive():
        logging.warning("Live validation already running.")
        return

    from application.services.live_validator import LiveValidationSession

    session_id = str(uuid.uuid4())
    session = LiveValidationSession(
        session_id=session_id,
        target_date=date_str,
        predictor=predictor,
        observatory=observatory,
        bus_type_predictor=bus_type_predictor,
    )
    LIVE_VALIDATOR_SESSION = session

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            started = loop.run_until_complete(session.start())
            if started:
                loop.run_until_complete(_wait_for_session(session))
            else:
                logging.warning(f"Live validation session failed to start for {date_str}")
        except Exception as e:
            logging.error(f"Live validation error: {e}")
        finally:
            loop.close()

    async def _wait_for_session(session):
        """Wait until the session completes or is stopped."""
        while session.status in ("predicting", "monitoring"):
            await asyncio.sleep(1)

    LIVE_VALIDATOR_THREAD = threading.Thread(target=_run, daemon=True)
    LIVE_VALIDATOR_THREAD.start()
    logging.info(f"Live validation started for {date_str} (session {session_id[:8]})")


def stop_live_validation():
    """Stop the active live validation session."""
    global LIVE_VALIDATOR_SESSION
    import asyncio

    if LIVE_VALIDATOR_SESSION is None:
        logging.info("No live validation session running.")
        return

    session = LIVE_VALIDATOR_SESSION
    loop = session._loop

    if loop is not None and not loop.is_closed():
        future = asyncio.run_coroutine_threadsafe(session.stop(), loop)
        try:
            future.result(timeout=10)
        except Exception as e:
            logging.error(f"Error stopping live validation: {e}")

    LIVE_VALIDATOR_SESSION = None
    logging.info("Live validation stopped.")


def get_validation_status():
    """Return status info for both validators."""
    status = {}

    if VALIDATOR_THREAD is not None and VALIDATOR_THREAD.is_alive():
        status["batch"] = "running"
    else:
        status["batch"] = "idle"

    if LIVE_VALIDATOR_SESSION is not None and LIVE_VALIDATOR_THREAD is not None and LIVE_VALIDATOR_THREAD.is_alive():
        session = LIVE_VALIDATOR_SESSION
        status["live"] = session.status
        status["live_validated"] = len(session.validated_trips)
        status["live_pending"] = len(session.pending_trip_ids)
        status["live_predicted"] = len(session.predicted_trips)
        status["live_discarded"] = len(session.discarded_trip_ids)
        live_status = session.get_status()
        status["live_median_rmse"] = live_status.median_rmse
    else:
        status["live"] = "idle"

    return status


def stop_services():
    """Stop all background services gracefully."""
    logging.info("Stopping Services (Collection & Auto-Save)...")
    data.STOP_COLLECTION_EVENT.set()

    # Stop validators
    stop_live_validation()

    # Unsubscribe
    domain_events.unsubscribe(DIARY_FINISHED, _on_diary_finished)

    # Stop GUI
    try:
        from . import debug_gui

        debug_gui.stop_gui()
    except Exception:
        pass


# ============================================================
# Event Subscriptions
# ============================================================

def _on_diary_finished(event_data: dict):
    """Handler: Records measurements in Historical Ledger and vehicle summary
    in Vehicle Ledger when a diary completes.
    """
    # Record raw measurements in Historical Ledger
    _record_historical(event_data)

    # Record trip summary in Vehicle Ledger
    _record_vehicle_trip(event_data)


def _record_historical(event_data: dict):
    """Extract measurement records from diary and store in the Historical Ledger."""
    from application.domain.ledgers import extract_measurements_from_diary

    diary = event_data.get("diary")
    route_id = event_data.get("route_id")
    observatory = event_data.get("observatory")
    if not diary or not observatory or not route_id:
        return

    trip = observatory.search_trip(diary.trip_id)
    if not trip:
        return

    try:
        records = extract_measurements_from_diary(diary, trip, route_id)
        observatory.historical.record_measurements(records)
        if records:
            logging.info(f"Recorded {len(records)} measurements for trip {diary.trip_id}")
    except Exception as e:
        logging.error(f"Failed to record historical measurements: {e}")


def _record_vehicle_trip(event_data: dict):
    """Summarize diary into a vehicle trip record and store in VehicleLedger."""
    from application.domain.ledgers import summarize_diary_for_vehicle

    diary = event_data.get("diary")
    route_id = event_data.get("route_id")
    observatory = event_data.get("observatory")
    vehicle_type_name = event_data.get("vehicle_type_name", "Unknown")
    if not diary or not observatory or not route_id:
        return

    trip = observatory.search_trip(diary.trip_id)
    direction_id = trip.direction_id if trip else 0

    try:
        record = summarize_diary_for_vehicle(
            diary, route_id, direction_id, vehicle_type_name
        )
        if record:
            observatory.vehicle_ledger.record_trip(record)
    except Exception as e:
        logging.error(f"Failed to record vehicle trip: {e}")


def _on_services_start(event_data):
    """Handler for services_start event."""
    start_services()


def _on_services_stop(event_data):
    """Handler for services_stop event."""
    stop_services()


def _on_shutdown(event_data):
    """Handler for shutdown_requested event."""
    data.SHUTDOWN_EVENT.set()
    stop_services()


# Register handlers
console_events.subscribe("services_start", _on_services_start)
console_events.subscribe("services_stop", _on_services_stop)
console_events.subscribe("shutdown_requested", _on_shutdown)
