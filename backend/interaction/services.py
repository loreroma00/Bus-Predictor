"""
Services - Background threads for data collection, saving, and uptime.

Subscribes to events from the event bus for start/stop control.
"""

import logging
from application.domain.internal_events import (
    LIVE_TRIP_FINISHED,
    SERVICES_START,
    SERVICES_STOP,
    SHUTDOWN_REQUESTED,
    console_events,
    domain_events,
)

# State interface for GUI (set by main.py)
_state_interface = None
_runtime_context = None
_thread_loader = None
_validation_controller = None


def start_services(context=None):
    """Start all background services."""
    global _state_interface, _runtime_context, _thread_loader

    logging.info("Starting Services...")
    if context is not None:
        _runtime_context = context

    if _runtime_context is None:
        logging.error("Cannot start services: runtime context is missing.")
        return

    if _state_interface is not None:
        _runtime_context.state_interface = _state_interface
        if hasattr(_state_interface, "set_context"):
            _state_interface.set_context(_runtime_context)

    # Subscribe to domain events
    domain_events.subscribe(LIVE_TRIP_FINISHED, _on_live_trip_finished)

    from thread_loader import ThreadLoader

    if _thread_loader is None or _thread_loader.context is not _runtime_context:
        _thread_loader = ThreadLoader(_runtime_context)
        _runtime_context.thread_loader = _thread_loader

    _thread_loader.start()


def set_state_interface(state_interface):
    """Set the state interface for the GUI (called from main.py)."""
    global _state_interface
    _state_interface = state_interface
    if _runtime_context is not None and hasattr(state_interface, "set_context"):
        state_interface.set_context(_runtime_context)


def _get_validation_controller():
    """Return the process-wide validation controller."""
    global _validation_controller

    from application.services.validator import ValidationController

    context = _runtime_context
    if context is not None:
        controller = getattr(context, "validation_controller", None)
        if controller is None:
            controller = ValidationController(
                persistence_gateway=getattr(context, "persistence_gateway", None),
            )
            context.validation_controller = controller
        return controller

    if _validation_controller is None:
        _validation_controller = ValidationController()
    return _validation_controller


def start_batch_validation(date_str, predictor, observatory):
    """Start batch validation through the validation controller."""
    controller = _get_validation_controller()
    controller.start_historical(date_str, predictor, observatory)


def start_live_validation(date_str, predictor, observatory, bus_type_predictor=None):
    """Start live validation through the validation controller."""
    controller = _get_validation_controller()
    controller.start_live(
        date_str,
        predictor,
        observatory,
        bus_type_predictor=bus_type_predictor,
    )


def stop_live_validation():
    """Stop the active live validation session."""
    controller = _get_validation_controller()
    controller.stop_live()


def get_validation_status():
    """Return status info for both validators."""
    controller = _get_validation_controller()
    return controller.get_status()


def generate_trip_validation_chart(trip_id, observatory, predictor=None):
    """Generate a validation chart for one trip."""
    controller = _get_validation_controller()
    return controller.generate_trip_validation_chart(
        trip_id=trip_id,
        observatory=observatory,
        predictor=predictor,
    )


def render_training_loss(log_path, output_path=None):
    """Render a training-loss chart through the validation controller."""
    controller = _get_validation_controller()
    return controller.render_training_loss(log_path, output_path)


def stop_services():
    """Stop all background services gracefully."""
    global _thread_loader
    logging.info("Stopping Services (Collection & Auto-Save)...")

    # Stop validators
    stop_live_validation()

    # Unsubscribe
    domain_events.unsubscribe(LIVE_TRIP_FINISHED, _on_live_trip_finished)
    if _thread_loader is not None:
        _thread_loader.stop()
    elif _runtime_context is not None and _runtime_context.stop_event is not None:
        _runtime_context.stop_event.set()


def join_core_threads(timeout: float = 5):
    """Wait briefly for core service threads."""
    if _thread_loader is not None:
        _thread_loader.join_core(timeout=timeout)


# ============================================================
# Event Subscriptions
# ============================================================

def _on_live_trip_finished(event_data: dict):
    """Record measurements and vehicle summary when a live trip completes."""
    _record_historical(event_data)
    _record_vehicle_trip(event_data)


def _record_historical(event_data: dict):
    """Extract measurement records from LiveTrip and store them."""
    from application.domain.virtual_entities import extract_measurements_from_live_trip

    live_trip = event_data.get("live_trip")
    route_id = event_data.get("route_id")
    observatory = _runtime_context.observatory if _runtime_context is not None else None
    if not live_trip or not observatory or not route_id:
        return

    try:
        records = extract_measurements_from_live_trip(live_trip, route_id)
        observatory.historical.record_measurements(records)
        observatory.historical.push_to_db()
        if records:
            logging.info(f"Recorded {len(records)} measurements for trip {live_trip.trip_id}")
    except Exception as e:
        logging.error(f"Failed to record historical measurements: {e}")


def _record_vehicle_trip(event_data: dict):
    """Summarize LiveTrip into vehicle history."""
    from application.domain.static_data import summarize_live_trip_for_vehicle

    live_trip = event_data.get("live_trip")
    route_id = event_data.get("route_id")
    vehicle_type_name = event_data.get("vehicle_type_name", "Unknown")
    if not live_trip or not route_id:
        return

    try:
        record = summarize_live_trip_for_vehicle(
            live_trip,
            route_id=route_id,
            vehicle_type_name=vehicle_type_name,
        )
        if record:
            live_trip.vehicle.record_trip(record)
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
    if _runtime_context is not None:
        _runtime_context.shutdown_event.set()
    stop_services()


# Register handlers
console_events.subscribe(SERVICES_START, _on_services_start)
console_events.subscribe(SERVICES_STOP, _on_services_stop)
console_events.subscribe(SHUTDOWN_REQUESTED, _on_shutdown)
